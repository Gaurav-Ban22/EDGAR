from torch import nn
from sklearn.preprocessing import StandardScaler
import torch
from deep_dynamics.model.build_network import build_network, string_to_torch, create_module
import yaml
import pickle
import numpy as np
from abc import abstractmethod


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


# ---------------------------------------------------------------------------
# Physics helper functions (normal forces, aero, load transfer, traction)
# ---------------------------------------------------------------------------

def compute_aero_forces(vx, vy, rho, A_f, A_r, C_lf, C_lr):
    v_sq = vx ** 2 + vy ** 2
    F_aero_f = 0.5 * rho * v_sq * A_f * C_lf
    F_aero_r = 0.5 * rho * v_sq * A_r * C_lr
    return F_aero_f, F_aero_r


def compute_load_transfer(h, L, m, a_x):
    return (h / L) * m * a_x


def compute_normal_forces(l_f, l_r, L, m, g, delta_W, F_aero_f, F_aero_r):
    F_fz_static = (l_r / L) * m * g
    F_rz_static = (l_f / L) * m * g
    F_fz = F_fz_static - delta_W + F_aero_f
    F_rz = F_rz_static + delta_W + F_aero_r
    return F_fz, F_rz


def compute_traction_limited_force(F_rx_desired, mu, F_rz):
    return torch.minimum(F_rx_desired, mu * F_rz)

class DatasetBase(torch.utils.data.Dataset):
    def __init__(self, features, labels, scaler=None):
        # features = features[1300:]
        # labels = labels[1300:]
        self.X_data = torch.from_numpy(features).float().to(device)
        self.y_data = torch.from_numpy(labels).float().to(device)
        self.X_norm = torch.zeros(features.shape)
        num_instances, num_time_steps, num_features = features.shape
        train_data = features.reshape((-1, num_features))
        if scaler is None:
            self.scaler = StandardScaler()
            norm_train_data = self.scaler.fit_transform(train_data)
            self.X_norm = torch.from_numpy(norm_train_data.reshape((num_instances, num_time_steps, num_features))).float().to(device)
        else:
            self.scaler = scaler
            norm_train_data = self.scaler.transform(train_data)
            self.X_norm = torch.from_numpy(norm_train_data.reshape((num_instances, num_time_steps, num_features))).float().to(device)
    def __len__(self):
        return(self.X_data.shape[0])
    def __getitem__(self, idx):
        x = self.X_data[idx]
        y = self.y_data[idx]
        x_norm = self.X_norm[idx]
        return x, y, x_norm
    def split(self, percent):
        split_id = int(len(self)* percent)
        torch.manual_seed(0)
        return torch.utils.data.random_split(self, [split_id, (len(self) - split_id)])

class DeepDynamicsDataset(DatasetBase):
    def __init__(self, features, labels, scalers=None):
        super().__init__(features[:,:,:7], labels, scalers)
    
class DeepPacejkaDataset(DatasetBase):
    def __init__(self, features, labels, scalers=None):
        features = np.delete(features, [3,5], axis=2)
        super().__init__(features, labels, scalers)

class ModelBase(nn.Module):
    def __init__(self, param_dict, output_module, eval=False):
        super().__init__()
        self.param_dict = param_dict
        layers = build_network(self.param_dict)
        self.batch_size = self.param_dict["MODEL"]["OPTIMIZATION"]["BATCH_SIZE"]
        if self.param_dict["MODEL"]["LAYERS"][0].get("LAYERS"):
            self.is_rnn = True
            self.rnn_n_layers = self.param_dict["MODEL"]["LAYERS"][0].get("LAYERS")
            self.rnn_hiden_dim = self.param_dict["MODEL"]["HORIZON"]
            layers.insert(1, nn.Flatten())
        else:
            self.is_rnn = False
        self.horizon = self.param_dict["MODEL"]["HORIZON"]
        layers.extend(output_module)
        self.feed_forward = nn.ModuleList(layers)
        if eval:
            self.loss_function = string_to_torch[self.param_dict["MODEL"]["OPTIMIZATION"]["LOSS"]](reduction='none')
        else:
            self.loss_function = string_to_torch[self.param_dict["MODEL"]["OPTIMIZATION"]["LOSS"]]()
        self.optimizer = string_to_torch[self.param_dict["MODEL"]["OPTIMIZATION"]["OPTIMIZER"]](self.parameters(), lr=self.param_dict["MODEL"]["OPTIMIZATION"]["LR"])
        self.epochs = self.param_dict["MODEL"]["OPTIMIZATION"]["NUM_EPOCHS"]
        self.state = list(self.param_dict["STATE"])
        self.actions = list(self.param_dict["ACTIONS"])
        self.sys_params = list([*(list(p.keys())[0] for p in self.param_dict["PARAMETERS"])])
        self.vehicle_specs = self.param_dict["VEHICLE_SPECS"]

    @abstractmethod
    def differential_equation(self, x, output):
        pass

    def forward(self, x, x_norm, h0=None):
        for i in range(len(self.feed_forward)):
            if i == 0:
                if isinstance(self.feed_forward[i], torch.nn.RNNBase):
                    ff, h0 = self.feed_forward[0](x_norm, h0)
                else:
                    ff = self.feed_forward[i](torch.reshape(x_norm, (len(x), -1)))
            else:
                if isinstance(self.feed_forward[i], torch.nn.RNNBase):
                    ff, h0 = self.feed_forward[0](ff, h0)
                else:
                    ff = self.feed_forward[i](ff)
        o = self.differential_equation(x, ff)
        return o, h0, ff
    
    def test_sys_params(self, x, Ts=0.02):
        _, sys_param_dict = self.unpack_sys_params(torch.zeros((1, len(self.sys_params))))
        state_action_dict = self.unpack_state_actions(x)
        steering = state_action_dict["STEERING_FB"] + state_action_dict["STEERING_CMD"]
        throttle = state_action_dict["THROTTLE_FB"] + state_action_dict["THROTTLE_CMD"]
        alphaf = steering - torch.atan2(self.vehicle_specs["lf"]*state_action_dict["YAW_RATE"] + state_action_dict["VY"], torch.abs(state_action_dict["VX"])) + sys_param_dict["Shf"]
        alphar = torch.atan2((self.vehicle_specs["lr"]*state_action_dict["YAW_RATE"] - state_action_dict["VY"]), torch.abs(state_action_dict["VX"])) + sys_param_dict["Shr"]
        Frx = (sys_param_dict["Cm1"]-sys_param_dict["Cm2"]*state_action_dict["VX"])*throttle - sys_param_dict["Cr0"] - sys_param_dict["Cr2"]*(state_action_dict["VX"]**2)
        Ffy = sys_param_dict["Svf"] + sys_param_dict["Df"] * torch.sin(sys_param_dict["Cf"] * torch.atan(sys_param_dict["Bf"] * alphaf - sys_param_dict["Ef"] * (sys_param_dict["Bf"] * alphaf - torch.atan(sys_param_dict["Bf"] * alphaf))))
        Fry = sys_param_dict["Svr"] +sys_param_dict["Dr"] * torch.sin(sys_param_dict["Cr"] * torch.atan(sys_param_dict["Br"] * alphar - sys_param_dict["Er"] * (sys_param_dict["Br"] * alphar - torch.atan(sys_param_dict["Br"] * alphar))))
        dxdt = torch.zeros(len(x), 3).to(device)
        dxdt[:,0] = 1/self.vehicle_specs["mass"] * (Frx - Ffy*torch.sin(steering)) + state_action_dict["VY"]*state_action_dict["YAW_RATE"]
        dxdt[:,1] = 1/self.vehicle_specs["mass"] * (Fry + Ffy*torch.cos(steering)) - state_action_dict["VX"]*state_action_dict["YAW_RATE"]
        dxdt[:,2] = 1/sys_param_dict["Iz"] * (Ffy*self.vehicle_specs["lf"]*torch.cos(steering) - Fry*self.vehicle_specs["lr"])
        dxdt *= Ts
        return x[:,-1,:3] + dxdt


    def unpack_sys_params(self, o):
        sys_params_dict = dict()
        for i in range(len(self.sys_params)):
            sys_params_dict[self.sys_params[i]] = o[:,i]
        ground_truth_dict =  dict()
        for p in self.param_dict["PARAMETERS"]:
            ground_truth_dict.update(p)
        return sys_params_dict, ground_truth_dict

    def unpack_state_actions(self, x):
        state_action_dict = dict()
        global_index = 0
        for i in range(len(self.state)):
            state_action_dict[self.state[i]] = x[:,-1, global_index]
            global_index += 1
        for i in range(len(self.actions)):
            state_action_dict[self.actions[i]] = x[:,-1, global_index]
            global_index += 1
        return state_action_dict

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.rnn_n_layers, batch_size, self.rnn_hiden_dim).zero_().to(device)
        return hidden
    
    def weighted_mse_loss(self, input, target, weight):
        return (weight * (input - target) ** 2)

    
class DeepDynamicsModel(ModelBase):
    def __init__(self, param_dict, eval=False):

        class GuardLayer(nn.Module):
            def __init__(self, param_dict):
                super().__init__()
                guard_output = create_module("DENSE", param_dict["MODEL"]["LAYERS"][-1]["OUT_FEATURES"], param_dict["MODEL"]["HORIZON"], len(param_dict["PARAMETERS"]), activation="Sigmoid")
                self.guard_dense = guard_output[0]
                self.guard_activation = guard_output[1]
                self.coefficient_ranges = torch.zeros(len(param_dict["PARAMETERS"])).to(device)
                self.coefficient_mins = torch.zeros(len(param_dict["PARAMETERS"])).to(device)
                for i in range(len(param_dict["PARAMETERS"])):
                    self.coefficient_ranges[i] = param_dict["PARAMETERS"][i]["Max"]- param_dict["PARAMETERS"][i]["Min"]
                    self.coefficient_mins[i] = param_dict["PARAMETERS"][i]["Min"]

            def forward(self, x):
                guard_output = self.guard_dense(x)
                guard_output = self.guard_activation(guard_output) * self.coefficient_ranges + self.coefficient_mins
                return guard_output

        
        super().__init__(param_dict, [GuardLayer(param_dict)], eval)

    def differential_equation(self, x, output, Ts=0.02):
        sys_param_dict, _ = self.unpack_sys_params(output)
        state_action_dict = self.unpack_state_actions(x)
        steering = state_action_dict["STEERING_FB"] + state_action_dict["STEERING_CMD"]
        throttle = state_action_dict["THROTTLE_FB"] + state_action_dict["THROTTLE_CMD"]
        alphaf = steering - torch.atan2(self.vehicle_specs["lf"]*state_action_dict["YAW_RATE"] + state_action_dict["VY"], torch.abs(state_action_dict["VX"])) + sys_param_dict["Shf"]
        alphar = torch.atan2((self.vehicle_specs["lr"]*state_action_dict["YAW_RATE"] - state_action_dict["VY"]), torch.abs(state_action_dict["VX"])) + sys_param_dict["Shr"]
        Frx = (sys_param_dict["Cm1"]-sys_param_dict["Cm2"]*state_action_dict["VX"])*throttle - sys_param_dict["Cr0"] - sys_param_dict["Cr2"]*(state_action_dict["VX"]**2)
        Ffy = sys_param_dict["Svf"] + sys_param_dict["Df"] * torch.sin(sys_param_dict["Cf"] * torch.atan(sys_param_dict["Bf"] * alphaf - sys_param_dict["Ef"] * (sys_param_dict["Bf"] * alphaf - torch.atan(sys_param_dict["Bf"] * alphaf))))
        Fry = sys_param_dict["Svr"] +sys_param_dict["Dr"] * torch.sin(sys_param_dict["Cr"] * torch.atan(sys_param_dict["Br"] * alphar - sys_param_dict["Er"] * (sys_param_dict["Br"] * alphar - torch.atan(sys_param_dict["Br"] * alphar))))
        dxdt = torch.zeros(len(x), 3).to(device)
        dxdt[:,0] = 1/self.vehicle_specs["mass"] * (Frx - Ffy*torch.sin(steering)) + state_action_dict["VY"]*state_action_dict["YAW_RATE"]
        dxdt[:,1] = 1/self.vehicle_specs["mass"] * (Fry + Ffy*torch.cos(steering)) - state_action_dict["VX"]*state_action_dict["YAW_RATE"]
        dxdt[:,2] = 1/sys_param_dict["Iz"] * (Ffy*self.vehicle_specs["lf"]*torch.cos(steering) - Fry*self.vehicle_specs["lr"])
        dxdt *= Ts
        return x[:,-1,:3] + dxdt


class DeepPacejkaModel(ModelBase):
    def __init__(self, param_dict, eval=False):
        output_module = create_module("DENSE", param_dict["MODEL"]["LAYERS"][-1]["OUT_FEATURES"], param_dict["MODEL"]["HORIZON"], len(param_dict["PARAMETERS"]), activation=None)
        super().__init__(param_dict, output_module, eval)

    def differential_equation(self, x, output, Ts=0.02):
        sys_param_dict, _ = self.unpack_sys_params(output)
        state_action_dict = self.unpack_state_actions(x)
        steering = state_action_dict["STEERING_FB"] + state_action_dict["STEERING_CMD"]
        alphaf = steering - torch.atan2(self.vehicle_specs["lf"]*state_action_dict["YAW_RATE"] + state_action_dict["VY"], torch.abs(state_action_dict["VX"]))
        alphar = torch.atan2((self.vehicle_specs["lr"]*state_action_dict["YAW_RATE"] - state_action_dict["VY"]), torch.abs(state_action_dict["VX"]))
        Ffy = sys_param_dict["Df"] * torch.sin(sys_param_dict["Cf"] * torch.atan(sys_param_dict["Bf"] * alphaf - sys_param_dict["Ef"] * (sys_param_dict["Bf"] * alphaf - torch.atan(sys_param_dict["Bf"] * alphaf))))
        Fry = sys_param_dict["Dr"] * torch.sin(sys_param_dict["Cr"] * torch.atan(sys_param_dict["Br"] * alphar - sys_param_dict["Er"] * (sys_param_dict["Br"] * alphar - torch.atan(sys_param_dict["Br"] * alphar))))
        dxdt = torch.zeros(len(x), 3).to(device)
        dxdt[:,0] = 1/self.vehicle_specs["mass"] * (sys_param_dict["Frx"] - Ffy*torch.sin(steering)) + state_action_dict["VY"]*state_action_dict["YAW_RATE"]
        dxdt[:,1] = 1/self.vehicle_specs["mass"] * (Fry + Ffy*torch.cos(steering)) - state_action_dict["VX"]*state_action_dict["YAW_RATE"]
        dxdt[:,2] = 1/self.vehicle_specs["Iz"] * (Ffy*self.vehicle_specs["lf"]*torch.cos(steering) - Fry*self.vehicle_specs["lr"])
        dxdt *= Ts
        return x[:,-1,:3] + dxdt
    
class DeepDynamicsModelIAC(ModelBase):
    def __init__(self, param_dict, eval=False):

        class GuardLayer(nn.Module):
            def __init__(self, param_dict):
                super().__init__()
                guard_output = create_module("DENSE", param_dict["MODEL"]["LAYERS"][-1]["OUT_FEATURES"], param_dict["MODEL"]["HORIZON"], len(param_dict["PARAMETERS"]), activation="Sigmoid")
                self.guard_dense = guard_output[0]
                self.guard_activation = guard_output[1]
                self.coefficient_ranges = torch.zeros(len(param_dict["PARAMETERS"])).to(device)
                self.coefficient_mins = torch.zeros(len(param_dict["PARAMETERS"])).to(device)
                for i in range(len(param_dict["PARAMETERS"])):
                    self.coefficient_ranges[i] = param_dict["PARAMETERS"][i]["Max"]- param_dict["PARAMETERS"][i]["Min"]
                    self.coefficient_mins[i] = param_dict["PARAMETERS"][i]["Min"]

            def forward(self, x):
                guard_output = self.guard_dense(x)
                guard_output = self.guard_activation(guard_output) * self.coefficient_ranges + self.coefficient_mins
                return guard_output

        
        super().__init__(param_dict, [GuardLayer(param_dict)], eval)

    def differential_equation(self, x, output, Ts=0.04):
        sys_param_dict, _ = self.unpack_sys_params(output)
        state_action_dict = self.unpack_state_actions(x)
        steering = state_action_dict["STEERING_FB"] + state_action_dict["STEERING_CMD"]
        throttle = state_action_dict["THROTTLE_FB"] + state_action_dict["THROTTLE_CMD"]
        alphaf = steering - torch.atan2(self.vehicle_specs["lf"]*state_action_dict["YAW_RATE"] + state_action_dict["VY"], torch.abs(state_action_dict["VX"])) + sys_param_dict["Shf"]
        alphar = torch.atan2((self.vehicle_specs["lr"]*state_action_dict["YAW_RATE"] - state_action_dict["VY"]), torch.abs(state_action_dict["VX"])) + sys_param_dict["Shr"]
        Frx = (sys_param_dict["Cm1"]-sys_param_dict["Cm2"]*state_action_dict["VX"])*throttle - sys_param_dict["Cr0"] - sys_param_dict["Cr2"]*(state_action_dict["VX"]**2)
        Ffy = sys_param_dict["Svf"] + sys_param_dict["Df"] * torch.sin(sys_param_dict["Cf"] * torch.atan(sys_param_dict["Bf"] * alphaf - sys_param_dict["Ef"] * (sys_param_dict["Bf"] * alphaf - torch.atan(sys_param_dict["Bf"] * alphaf))))
        Fry = sys_param_dict["Svr"] + sys_param_dict["Dr"] * torch.sin(sys_param_dict["Cr"] * torch.atan(sys_param_dict["Br"] * alphar - sys_param_dict["Er"] * (sys_param_dict["Br"] * alphar - torch.atan(sys_param_dict["Br"] * alphar))))
        dxdt = torch.zeros(len(x), 3).to(device)
        dxdt[:,0] = 1/self.vehicle_specs["mass"] * (Frx - Ffy*torch.sin(steering)) + state_action_dict["VY"]*state_action_dict["YAW_RATE"]
        dxdt[:,1] = 1/self.vehicle_specs["mass"] * (Fry + Ffy*torch.cos(steering)) - state_action_dict["VX"]*state_action_dict["YAW_RATE"]
        dxdt[:,2] = 1/sys_param_dict["Iz"] * (Ffy*self.vehicle_specs["lf"]*torch.cos(steering) - Fry*self.vehicle_specs["lr"])
        dxdt *= Ts
        return x[:,-1,:3] + dxdt
    

class DeepPacejkaModelIAC(ModelBase):
    def __init__(self, param_dict, eval=False):
        output_module = create_module("DENSE", param_dict["MODEL"]["LAYERS"][-1]["OUT_FEATURES"], param_dict["MODEL"]["HORIZON"], len(param_dict["PARAMETERS"]), activation=None)
        super().__init__(param_dict, output_module, eval)

    def differential_equation(self, x, output, Ts=0.04):
        sys_param_dict, _ = self.unpack_sys_params(output)
        state_action_dict = self.unpack_state_actions(x)
        steering = state_action_dict["STEERING_FB"] + state_action_dict["STEERING_CMD"]
        alphaf = steering - torch.atan2(self.vehicle_specs["lf"]*state_action_dict["YAW_RATE"] + state_action_dict["VY"], torch.abs(state_action_dict["VX"]))
        alphar = torch.atan2((self.vehicle_specs["lr"]*state_action_dict["YAW_RATE"] - state_action_dict["VY"]), torch.abs(state_action_dict["VX"]))
        Ffy = sys_param_dict["Df"] * torch.sin(sys_param_dict["Cf"] * torch.atan(sys_param_dict["Bf"] * alphaf - sys_param_dict["Ef"] * (sys_param_dict["Bf"] * alphaf - torch.atan(sys_param_dict["Bf"] * alphaf))))
        Fry = sys_param_dict["Dr"] * torch.sin(sys_param_dict["Cr"] * torch.atan(sys_param_dict["Br"] * alphar - sys_param_dict["Er"] * (sys_param_dict["Br"] * alphar - torch.atan(sys_param_dict["Br"] * alphar))))
        dxdt = torch.zeros(len(x), 3).to(device)
        dxdt[:,0] = 1/self.vehicle_specs["mass"] * (sys_param_dict["Frx"] - Ffy*torch.sin(steering)) + state_action_dict["VY"]*state_action_dict["YAW_RATE"]
        dxdt[:,1] = 1/self.vehicle_specs["mass"] * (Fry + Ffy*torch.cos(steering)) - state_action_dict["VX"]*state_action_dict["YAW_RATE"]
        dxdt[:,2] = 1/self.vehicle_specs["Iz"] * (Ffy*self.vehicle_specs["lf"]*torch.cos(steering) - Fry*self.vehicle_specs["lr"])
        dxdt *= Ts
        return x[:,-1,:3] + dxdt


class DeepDynamicsPINN(nn.Module):
    """Physics-Informed Neural Network variant of DeepDynamicsModel.

    Instead of using a GuardLayer + physics equations structurally in the
    forward pass, the NN directly predicts state changes (dVX, dVY, dYAW_RATE).
    Physics compliance is enforced via a residual loss term using learnable
    Pacejka tire model parameters with full longitudinal force modeling.
    """
    def __init__(self, param_dict, eval=False):
        super().__init__()
        self.param_dict = param_dict
        self.batch_size = param_dict["MODEL"]["OPTIMIZATION"]["BATCH_SIZE"]
        self.horizon = param_dict["MODEL"]["HORIZON"]
        self.epochs = param_dict["MODEL"]["OPTIMIZATION"]["NUM_EPOCHS"]
        self.state = list(param_dict["STATE"])
        self.actions = list(param_dict["ACTIONS"])
        self.vehicle_specs = param_dict["VEHICLE_SPECS"]
        self.physics_weight = param_dict["MODEL"]["OPTIMIZATION"].get("PHYSICS_LOSS_WEIGHT", 0.1)

        # Build the shared network layers (GRU + Dense hidden layers)
        layers = build_network(param_dict)
        if param_dict["MODEL"]["LAYERS"][0].get("LAYERS"):
            self.is_rnn = True
            self.rnn_n_layers = param_dict["MODEL"]["LAYERS"][0].get("LAYERS")
            self.rnn_hiden_dim = param_dict["MODEL"]["HORIZON"]
            layers.insert(1, nn.Flatten())
        else:
            self.is_rnn = False

        # Final output layer: predict 3 state deltas (dVX, dVY, dYAW_RATE)
        last_hidden_size = param_dict["MODEL"]["LAYERS"][-1]["OUT_FEATURES"]
        layers.append(nn.Linear(last_hidden_size, 3))

        self.feed_forward = nn.ModuleList(layers)

        # Learnable Pacejka tire model parameters (15 total, optimized via backprop)
        # NOTE: Df and Dr are NO LONGER learnable — they are computed analytically
        #       as D = mu * Fz, where Fz includes aero downforce + load transfer.
        physics_params = param_dict.get("PHYSICS_PARAMS", {})
        # Pacejka lateral force shape params (no Df/Dr — those are analytical now)
        self.Bf = nn.Parameter(torch.tensor(float(physics_params.get("Bf", 5.579))))
        self.Cf = nn.Parameter(torch.tensor(float(physics_params.get("Cf", 1.2))))
        self.Ef = nn.Parameter(torch.tensor(float(physics_params.get("Ef", -0.083))))
        self.Br = nn.Parameter(torch.tensor(float(physics_params.get("Br", 5.3852))))
        self.Cr = nn.Parameter(torch.tensor(float(physics_params.get("Cr", 1.2691))))
        self.Er = nn.Parameter(torch.tensor(float(physics_params.get("Er", -0.019))))
        # Longitudinal force params
        self.Cm1 = nn.Parameter(torch.tensor(float(physics_params.get("Cm1", 0.287))))
        self.Cm2 = nn.Parameter(torch.tensor(float(physics_params.get("Cm2", 0.0545))))
        self.Cr0 = nn.Parameter(torch.tensor(float(physics_params.get("Cr0", 0.0518))))
        self.Cr2 = nn.Parameter(torch.tensor(float(physics_params.get("Cr2", 0.00035))))
        # Slip angle shift params
        self.Shf = nn.Parameter(torch.tensor(float(physics_params.get("Shf", -0.0013))))
        self.Shr = nn.Parameter(torch.tensor(float(physics_params.get("Shr", -0.00376))))
        # Vertical force shift params
        self.Svf = nn.Parameter(torch.tensor(float(physics_params.get("Svf", 0.00043))))
        self.Svr = nn.Parameter(torch.tensor(float(physics_params.get("Svr", 0.00091))))
        # Iz from learnable param (fallback) or vehicle_specs (preferred)
        self.Iz = nn.Parameter(torch.tensor(float(physics_params.get("Iz", self.vehicle_specs.get("Iz", 27.8e-6)))))

        # Loss and optimizer
        if eval:
            self.loss_function = string_to_torch[param_dict["MODEL"]["OPTIMIZATION"]["LOSS"]](reduction='none')
        else:
            self.loss_function = string_to_torch[param_dict["MODEL"]["OPTIMIZATION"]["LOSS"]]()
        self.optimizer = string_to_torch[param_dict["MODEL"]["OPTIMIZATION"]["OPTIMIZER"]](self.parameters(), lr=param_dict["MODEL"]["OPTIMIZATION"]["LR"])

    def forward(self, x, x_norm, h0=None, Ts=0.02):
        for i in range(len(self.feed_forward)):
            if i == 0:
                if isinstance(self.feed_forward[i], torch.nn.RNNBase):
                    ff, h0 = self.feed_forward[0](x_norm, h0)
                else:
                    ff = self.feed_forward[i](torch.reshape(x_norm, (len(x), -1)))
            else:
                if isinstance(self.feed_forward[i], torch.nn.RNNBase):
                    ff, h0 = self.feed_forward[0](ff, h0)
                else:
                    ff = self.feed_forward[i](ff)
        # ff is the raw NN output: 3 state deltas
        predictions = x[:, -1, :3] + ff * Ts
        return predictions, h0, ff

    def physics_residual(self, x, predictions, Ts=0.02):
        """Compute the PINN ODE residual loss.

        Standard PINN formulation: penalizes the residual between the NN's
        predicted state derivatives and the derivatives computed by the
        governing differential equations (Pacejka tire model + bicycle
        dynamics), i.e.  loss = || dstate/dt_NN - f(x, u, theta) ||^2

        Enhanced with modifiedmodelpacejka physics:
          - Longitudinal load transfer
          - Per-axle aerodynamic downforce
          - Analytical Df/Dr = mu * Fz (not learned)
          - Traction-limited longitudinal force
        """
        state_action_dict = self._unpack_state_actions(x)
        steering = state_action_dict["STEERING_FB"] + state_action_dict["STEERING_CMD"]
        throttle = state_action_dict["THROTTLE_FB"] + state_action_dict["THROTTLE_CMD"]
        vx = state_action_dict["VX"]
        vy = state_action_dict["VY"]
        omega = state_action_dict["YAW_RATE"]

        # Vehicle geometry from config
        mass = self.vehicle_specs["mass"]
        lf = self.vehicle_specs["lf"]
        lr = self.vehicle_specs["lr"]
        L = self.vehicle_specs["L"]
        h = self.vehicle_specs["h"]
        mu = self.vehicle_specs["mu"]
        g = self.vehicle_specs["g"]

        # --- NN-predicted state derivatives ---
        nn_dstate_dt = (predictions - x[:, -1, :3]) / Ts

        # --- Physics-predicted state derivatives (governing ODEs) ---

        # Slip angles with shift parameters
        alphaf = steering - torch.atan2(lf * omega + vy, torch.abs(vx)) + self.Shf
        alphar = torch.atan2(lr * omega - vy, torch.abs(vx)) + self.Shr

        # Desired longitudinal force from drivetrain model
        F_rx_desired = ((self.Cm1 - self.Cm2 * vx) * throttle
                        - self.Cr0 - self.Cr2 * (vx ** 2))

        # Longitudinal load transfer: delta_W = (h / L) * m * a_x
        a_x = F_rx_desired / mass
        delta_W = compute_load_transfer(h, L, mass, a_x)

        # Per-axle aerodynamic downforce
        F_aero_f, F_aero_r = compute_aero_forces(
            vx, vy, self.vehicle_specs["rho"],
            self.vehicle_specs["A_f"], self.vehicle_specs["A_r"],
            self.vehicle_specs["C_lf"], self.vehicle_specs["C_lr"])

        # Normal forces (static + load transfer + aero)
        F_fz, F_rz = compute_normal_forces(lf, lr, L, mass, g, delta_W, F_aero_f, F_aero_r)

        # Analytical Df, Dr from normal forces (NOT learned by NN)
        Df = mu * F_fz
        Dr = mu * F_rz

        # Traction-limited longitudinal force
        Frx = compute_traction_limited_force(F_rx_desired, mu, F_rz)

        # Pacejka magic formula for lateral tire forces (with vertical shifts)
        Ffy = self.Svf + Df * torch.sin(self.Cf * torch.atan(
            self.Bf * alphaf - self.Ef * (self.Bf * alphaf - torch.atan(self.Bf * alphaf))
        ))
        Fry = self.Svr + Dr * torch.sin(self.Cr * torch.atan(
            self.Br * alphar - self.Er * (self.Br * alphar - torch.atan(self.Br * alphar))
        ))

        # Bicycle model dynamics (governing ODEs)
        physics_dstate_dt = torch.zeros(len(x), 3).to(x.device)
        physics_dstate_dt[:, 0] = (1 / mass * (Frx - Ffy * torch.sin(steering))
                                    + vy * omega)
        physics_dstate_dt[:, 1] = (1 / mass * (Fry + Ffy * torch.cos(steering))
                                    - vx * omega)
        physics_dstate_dt[:, 2] = (1 / self.Iz * (
            Ffy * lf * torch.cos(steering) - Fry * lr
        ))

        # ODE residual: || dstate/dt_NN - f(x, u, theta) ||^2
        return torch.nn.functional.mse_loss(nn_dstate_dt, physics_dstate_dt)

    def _unpack_state_actions(self, x):
        """Unpack state and action variables from input tensor."""
        state_action_dict = dict()
        global_index = 0
        for i in range(len(self.state)):
            state_action_dict[self.state[i]] = x[:, -1, global_index]
            global_index += 1
        for i in range(len(self.actions)):
            state_action_dict[self.actions[i]] = x[:, -1, global_index]
            global_index += 1
        return state_action_dict

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.rnn_n_layers, batch_size, self.rnn_hiden_dim).zero_().to(weight.device)
        return hidden

    def weighted_mse_loss(self, input, target, weight):
        return (weight * (input - target) ** 2)


string_to_model = {
    "DeepDynamics" : DeepDynamicsModel,
    "DeepPacejka" : DeepPacejkaModel,
    "DeepDynamicsIAC" : DeepDynamicsModelIAC,
    "DeepPacejkaIAC" : DeepPacejkaModelIAC,
    "DeepDynamicsPINN" : DeepDynamicsPINN
}

string_to_dataset = {
    "DeepDynamics" : DeepDynamicsDataset,
    "DeepPacejka" : DeepPacejkaDataset,
    "DeepDynamicsIAC" : DeepDynamicsDataset,
    "DeepPacejkaIAC" : DeepPacejkaDataset,
    "DeepDynamicsPINN" : DeepDynamicsDataset
}
