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
        
        # Explicit Physics Additions: Load Transfer + Aero Downforce for PCNN Baseline
        accel_x_approx = (throttle * sys_param_dict["Cm1"] - sys_param_dict["Cr0"]) / self.vehicle_specs["mass"]
        F_downforce_f, F_downforce_r = compute_aero_forces(
            state_action_dict["VX"], state_action_dict["VY"], self.vehicle_specs["rho"],
            self.vehicle_specs["A_f"], self.vehicle_specs["A_r"],
            self.vehicle_specs["C_lf"], self.vehicle_specs["C_lr"]
        )
        delta_Fz = compute_load_transfer(
            self.vehicle_specs["h"], self.vehicle_specs["L"], 
            self.vehicle_specs["mass"], accel_x_approx
        )
        F_zf, F_zr = compute_normal_forces(
            self.vehicle_specs["mass"], self.vehicle_specs["g"],
            self.vehicle_specs["lf"], self.vehicle_specs["lr"],
            self.vehicle_specs["L"], delta_Fz, F_downforce_f, F_downforce_r
        )
        # Scale the NN predicted peak friction by the analytical load shift
        Ffy_max = sys_param_dict["Df"] * F_zf / (self.vehicle_specs["mass"] * self.vehicle_specs["g"] * self.vehicle_specs["lr"] / self.vehicle_specs["L"])
        Fry_max = sys_param_dict["Dr"] * F_zr / (self.vehicle_specs["mass"] * self.vehicle_specs["g"] * self.vehicle_specs["lf"] / self.vehicle_specs["L"])

        alphaf = steering - torch.atan2(self.vehicle_specs["lf"]*state_action_dict["YAW_RATE"] + state_action_dict["VY"], torch.abs(state_action_dict["VX"])) + sys_param_dict["Shf"]
        alphar = torch.atan2((self.vehicle_specs["lr"]*state_action_dict["YAW_RATE"] - state_action_dict["VY"]), torch.abs(state_action_dict["VX"])) + sys_param_dict["Shr"]
        
        Frx = (sys_param_dict["Cm1"]-sys_param_dict["Cm2"]*state_action_dict["VX"])*throttle - sys_param_dict["Cr0"] - sys_param_dict["Cr2"]*(state_action_dict["VX"]**2)
        Ffy = sys_param_dict["Svf"] + Ffy_max * torch.sin(sys_param_dict["Cf"] * torch.atan(sys_param_dict["Bf"] * alphaf - sys_param_dict["Ef"] * (sys_param_dict["Bf"] * alphaf - torch.atan(sys_param_dict["Bf"] * alphaf))))
        Fry = sys_param_dict["Svr"] + Fry_max * torch.sin(sys_param_dict["Cr"] * torch.atan(sys_param_dict["Br"] * alphar - sys_param_dict["Er"] * (sys_param_dict["Br"] * alphar - torch.atan(sys_param_dict["Br"] * alphar))))
        
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


class DeepDynamicsPINN(ModelBase):
    def __init__(self, param_dict, eval=False):

        class DualOutputLayer(nn.Module):
            def __init__(self, param_dict):
                super().__init__()
                in_features = param_dict["MODEL"]["LAYERS"][-1]["OUT_FEATURES"]
                horizon = param_dict["MODEL"]["HORIZON"]
                # 1. Output 3 raw states (dxdt): vx, vy, yaw_rate
                dxdt_output = create_module("DENSE", in_features, horizon, 3, activation=None)
                self.dxdt_dense = dxdt_output[0]
                
                # 2. Output 17 abstract coefficients natively without a GuardLayer constraint
                coeff_output = create_module("DENSE", in_features, horizon, len(param_dict["PARAMETERS"]), activation=None)
                self.coeff_dense = coeff_output[0]

            def forward(self, x):
                dxdt = self.dxdt_dense(x)
                coeffs = self.coeff_dense(x)
                return torch.cat((dxdt, coeffs), dim=-1)

        super().__init__(param_dict, [DualOutputLayer(param_dict)], eval)
        self.physics_weight = param_dict["MODEL"]["OPTIMIZATION"].get("PHYSICS_LOSS_WEIGHT", 1.0)

    def differential_equation(self, x, output, Ts=0.02):
        # The Neural Network directly predicted the dxdt derivatives independently of Pacejka!
        dxdt_pred = output[:,:3] * Ts 
        return x[:,-1,:3] + dxdt_pred

    def physics_residual(self, x, output, Ts=0.02):
        """
        Computes the ODE Physics Loss (MSE) + Taylor Coefficient LASSO Penalties.
        This forces the unconstrained neural network to learn the physical mapping algebraically
        without having to structurally conform to the matrix.
        """
        # Network abstract outputs
        nn_dxdt = output[:,:3] * Ts
        
        # Unpack the 17 coefficients from the neural network's unbounded linear layer
        sys_param_dict, _ = self.unpack_sys_params(output[:, 3:]) 
        state_action_dict = self.unpack_state_actions(x)
        steering = state_action_dict["STEERING_FB"] + state_action_dict["STEERING_CMD"]
        throttle = state_action_dict["THROTTLE_FB"] + state_action_dict["THROTTLE_CMD"]

        # --- 1. Compute Foundational Mathematical Load Shift & Aero Baselines ---
        accel_x_approx = (throttle * sys_param_dict["Cm1"] - sys_param_dict["Cr0"]) / self.vehicle_specs["mass"]

        F_downforce_f, F_downforce_r = compute_aero_forces(
            state_action_dict["VX"], state_action_dict["VY"], self.vehicle_specs["rho"],
            self.vehicle_specs["A_f"], self.vehicle_specs["A_r"],
            self.vehicle_specs["C_lf"], self.vehicle_specs["C_lr"]
        )
        
        delta_Fz = compute_load_transfer(
            self.vehicle_specs["h"], self.vehicle_specs["L"], 
            self.vehicle_specs["mass"], accel_x_approx
        )
        
        F_zf, F_zr = compute_normal_forces(
            self.vehicle_specs["mass"], self.vehicle_specs["g"],
            self.vehicle_specs["lf"], self.vehicle_specs["lr"],
            self.vehicle_specs["L"], delta_Fz, F_downforce_f, F_downforce_r
        )
        
        # Determine strict Newtonian scale for peak friction using load shift
        Ffy_max = sys_param_dict["Df"] * F_zf / (self.vehicle_specs["mass"] * self.vehicle_specs["g"] * self.vehicle_specs["lr"] / self.vehicle_specs["L"])
        Fry_max = sys_param_dict["Dr"] * F_zr / (self.vehicle_specs["mass"] * self.vehicle_specs["g"] * self.vehicle_specs["lf"] / self.vehicle_specs["L"])

        # --- 2. Calculate the Newtonian Differential Equation (Theoretical Pacejka Dxdt) ---
        alphaf = steering - torch.atan2(self.vehicle_specs["lf"]*state_action_dict["YAW_RATE"] + state_action_dict["VY"], torch.abs(state_action_dict["VX"])) + sys_param_dict["Shf"]
        alphar = torch.atan2((self.vehicle_specs["lr"]*state_action_dict["YAW_RATE"] - state_action_dict["VY"]), torch.abs(state_action_dict["VX"])) + sys_param_dict["Shr"]
        
        Frx = (sys_param_dict["Cm1"]-sys_param_dict["Cm2"]*state_action_dict["VX"])*throttle - sys_param_dict["Cr0"] - sys_param_dict["Cr2"]*(state_action_dict["VX"]**2)
        Ffy = sys_param_dict["Svf"] + Ffy_max * torch.sin(sys_param_dict["Cf"] * torch.atan(sys_param_dict["Bf"] * alphaf - sys_param_dict["Ef"] * (sys_param_dict["Bf"] * alphaf - torch.atan(sys_param_dict["Bf"] * alphaf))))
        Fry = sys_param_dict["Svr"] + Fry_max * torch.sin(sys_param_dict["Cr"] * torch.atan(sys_param_dict["Br"] * alphar - sys_param_dict["Er"] * (sys_param_dict["Br"] * alphar - torch.atan(sys_param_dict["Br"] * alphar))))
        
        ode_dxdt = torch.zeros(len(x), 3).to(device)
        ode_dxdt[:,0] = 1/self.vehicle_specs["mass"] * (Frx - Ffy*torch.sin(steering)) + state_action_dict["VY"]*state_action_dict["YAW_RATE"]
        ode_dxdt[:,1] = 1/self.vehicle_specs["mass"] * (Fry + Ffy*torch.cos(steering)) - state_action_dict["VX"]*state_action_dict["YAW_RATE"]
        ode_dxdt[:,2] = 1/sys_param_dict["Iz"] * (Ffy*self.vehicle_specs["lf"]*torch.cos(steering) - Fry*self.vehicle_specs["lr"])
        ode_dxdt *= Ts
        
        # --- 3. Calculate Dual Loss Objectives ---
        # Objective A: Enforce the NN Dxdt to align with the continuous mathematical ODE Dxdt
        ode_loss = torch.mean((nn_dxdt - ode_dxdt)**2)
        
        # Objective B: Enforce the unconstrained abstract Peak Frictions to align with Taylor Load Shift Baselines
        baseline_Df = self.vehicle_specs["mu"] * F_zf
        baseline_Dr = self.vehicle_specs["mu"] * F_zr
        lasso_Df = torch.mean(torch.abs(sys_param_dict["Df"] - baseline_Df))
        lasso_Dr = torch.mean(torch.abs(sys_param_dict["Dr"] - baseline_Dr))
        
        # Objective C: Sparsify unknown higher-order geometric boundaries
        lasso_Cm2 = torch.mean(torch.abs(sys_param_dict["Cm2"])) 
        lasso_Cr2 = torch.mean(torch.abs(sys_param_dict["Cr2"])) 
        lasso_Shf = torch.mean(torch.abs(sys_param_dict["Shf"])) 
        lasso_Shr = torch.mean(torch.abs(sys_param_dict["Shr"]))
        lasso_Svf = torch.mean(torch.abs(sys_param_dict["Svf"]))
        lasso_Svr = torch.mean(torch.abs(sys_param_dict["Svr"]))

        # Total unified physics loss bounds the entirely naked neural network predictions
        total_physics_loss = ode_loss + lasso_Df + lasso_Dr + lasso_Cm2 + lasso_Cr2 + lasso_Shf + lasso_Shr + lasso_Svf + lasso_Svr
        return total_physics_loss


class DeepDynamicsPCNNPINN(ModelBase):
    """Hybrid PCNN + PINN Architecture.
    
    This model utilizes the GuardLayer from the PCNN to ensure outputs remain strictly within 
    defined bounds, and computes internal state natively. However, it additionally applies the 
    LASSO (L1) loss penalty to gently pull the Neural Network's predicted coefficients toward 
    the fundamental theoretical physical mathematical equations (Taylor Baselines).
    """
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
        self.physics_weight = param_dict["MODEL"]["OPTIMIZATION"].get("PHYSICS_LOSS_WEIGHT", 1.0)

    def differential_equation(self, x, output, Ts=0.02):
        # Calculate state updates using the dynamically predicted coefficients
        sys_param_dict, _ = self.unpack_sys_params(output)
        state_action_dict = self.unpack_state_actions(x)
        steering = state_action_dict["STEERING_FB"] + state_action_dict["STEERING_CMD"]
        throttle = state_action_dict["THROTTLE_FB"] + state_action_dict["THROTTLE_CMD"]
        
        # Explicit Physics Additions: Load Transfer + Aero Downforce
        accel_x_approx = (throttle * sys_param_dict["Cm1"] - sys_param_dict["Cr0"]) / self.vehicle_specs["mass"]
        F_downforce_f, F_downforce_r = compute_aero_forces(
            state_action_dict["VX"], state_action_dict["VY"], self.vehicle_specs["rho"],
            self.vehicle_specs["A_f"], self.vehicle_specs["A_r"],
            self.vehicle_specs["C_lf"], self.vehicle_specs["C_lr"]
        )
        delta_Fz = compute_load_transfer(
            self.vehicle_specs["h"], self.vehicle_specs["L"], 
            self.vehicle_specs["mass"], accel_x_approx
        )
        F_zf, F_zr = compute_normal_forces(
            self.vehicle_specs["mass"], self.vehicle_specs["g"],
            self.vehicle_specs["lf"], self.vehicle_specs["lr"],
            self.vehicle_specs["L"], delta_Fz, F_downforce_f, F_downforce_r
        )
        
        # The Peak Friction parameters are explicitly scaled by load transfers prior to the tire curve calculations
        Ffy_max = sys_param_dict["Df"] * F_zf / (self.vehicle_specs["mass"] * self.vehicle_specs["g"] * self.vehicle_specs["lr"] / self.vehicle_specs["L"])
        Fry_max = sys_param_dict["Dr"] * F_zr / (self.vehicle_specs["mass"] * self.vehicle_specs["g"] * self.vehicle_specs["lf"] / self.vehicle_specs["L"])
        
        alphaf = steering - torch.atan2(self.vehicle_specs["lf"]*state_action_dict["YAW_RATE"] + state_action_dict["VY"], torch.abs(state_action_dict["VX"])) + sys_param_dict["Shf"]
        alphar = torch.atan2((self.vehicle_specs["lr"]*state_action_dict["YAW_RATE"] - state_action_dict["VY"]), torch.abs(state_action_dict["VX"])) + sys_param_dict["Shr"]
        
        Frx = (sys_param_dict["Cm1"]-sys_param_dict["Cm2"]*state_action_dict["VX"])*throttle - sys_param_dict["Cr0"] - sys_param_dict["Cr2"]*(state_action_dict["VX"]**2)
        Ffy = sys_param_dict["Svf"] + Ffy_max * torch.sin(sys_param_dict["Cf"] * torch.atan(sys_param_dict["Bf"] * alphaf - sys_param_dict["Ef"] * (sys_param_dict["Bf"] * alphaf - torch.atan(sys_param_dict["Bf"] * alphaf))))
        Fry = sys_param_dict["Svr"] + Fry_max * torch.sin(sys_param_dict["Cr"] * torch.atan(sys_param_dict["Br"] * alphar - sys_param_dict["Er"] * (sys_param_dict["Br"] * alphar - torch.atan(sys_param_dict["Br"] * alphar))))
        
        dxdt = torch.zeros(len(x), 3).to(device)
        dxdt[:,0] = 1/self.vehicle_specs["mass"] * (Frx - Ffy*torch.sin(steering)) + state_action_dict["VY"]*state_action_dict["YAW_RATE"]
        dxdt[:,1] = 1/self.vehicle_specs["mass"] * (Fry + Ffy*torch.cos(steering)) - state_action_dict["VX"]*state_action_dict["YAW_RATE"]
        dxdt[:,2] = 1/sys_param_dict["Iz"] * (Ffy*self.vehicle_specs["lf"]*torch.cos(steering) - Fry*self.vehicle_specs["lr"])
        dxdt *= Ts
        return x[:,-1,:3] + dxdt

    def physics_residual(self, x, output, Ts=0.02):
        """
        Computes the LASSO (L1) penalty between the PCNN's bounded dynamic coefficients 
        and the fundamental physical Taylor series mathematical baselines.
        """
        sys_param_dict, _ = self.unpack_sys_params(output)
        state_action_dict = self.unpack_state_actions(x)
        steering = state_action_dict["STEERING_FB"] + state_action_dict["STEERING_CMD"]

        accel_x_approx = (state_action_dict["THROTTLE_CMD"] * sys_param_dict["Cm1"] - sys_param_dict["Cr0"]) / self.vehicle_specs["mass"]

        F_downforce_f, F_downforce_r = compute_aero_forces(
            state_action_dict["VX"], state_action_dict["VY"], self.vehicle_specs["rho"],
            self.vehicle_specs["A_f"], self.vehicle_specs["A_r"],
            self.vehicle_specs["C_lf"], self.vehicle_specs["C_lr"]
        )
        
        delta_Fz = compute_load_transfer(
            self.vehicle_specs["h"], self.vehicle_specs["L"], 
            self.vehicle_specs["mass"], accel_x_approx
        )
        
        F_zf, F_zr = compute_normal_forces(
            self.vehicle_specs["mass"], self.vehicle_specs["g"], 
            self.vehicle_specs["lf"], self.vehicle_specs["lr"], self.vehicle_specs["L"],
            delta_Fz, F_downforce_f, F_downforce_r
        )
        
        baseline_Df = self.vehicle_specs["mu"] * F_zf
        baseline_Dr = self.vehicle_specs["mu"] * F_zr

        lasso_Df = torch.mean(torch.abs(sys_param_dict["Df"] - baseline_Df))
        lasso_Dr = torch.mean(torch.abs(sys_param_dict["Dr"] - baseline_Dr))
        
        lasso_Cm2 = torch.mean(torch.abs(sys_param_dict["Cm2"]))
        lasso_Cr2 = torch.mean(torch.abs(sys_param_dict["Cr2"]))
        lasso_Shf = torch.mean(torch.abs(sys_param_dict["Shf"]))
        lasso_Shr = torch.mean(torch.abs(sys_param_dict["Shr"]))
        lasso_Svf = torch.mean(torch.abs(sys_param_dict["Svf"]))
        lasso_Svr = torch.mean(torch.abs(sys_param_dict["Svr"]))

        lasso_loss = lasso_Df + lasso_Dr + lasso_Cm2 + lasso_Cr2 + lasso_Shf + lasso_Shr + lasso_Svf + lasso_Svr
        return lasso_loss


string_to_model = {
    "DeepDynamics" : DeepDynamicsModel,
    "DeepPacejka" : DeepPacejkaModel,
    "DeepDynamicsIAC" : DeepDynamicsModelIAC,
    "DeepPacejkaIAC" : DeepPacejkaModelIAC,
    "DeepDynamicsPINN" : DeepDynamicsPINN,
    "DeepDynamicsPCNNPINN" : DeepDynamicsPCNNPINN
}

string_to_dataset = {
    "DeepDynamics" : DeepDynamicsDataset,
    "DeepPacejka" : DeepPacejkaDataset,
    "DeepDynamicsIAC" : DeepDynamicsDataset,
    "DeepPacejkaIAC" : DeepPacejkaDataset,
    "DeepDynamicsPINN" : DeepDynamicsDataset,
    "DeepDynamicsPCNNPINN" : DeepDynamicsDataset
}
