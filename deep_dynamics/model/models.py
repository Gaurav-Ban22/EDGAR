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
else:
    device = torch.device("cpu")

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



# =============================================================================
# DeepBlue.AI V2 Models - Aerodynamic Downforce + Load Transfer + Friction Clip
# =============================================================================
# Changes from original DeepDynamicsModel:
#   1. Df and Dr are NO LONGER predicted by the neural network.
#      Instead, they are computed analytically as: D = mu * F_z
#      where F_z is velocity-dependent (includes downforce + load transfer).
#   2. F_z is no longer constant. It now equals:
#      F_z,f = (l_r / L) * m * g + 0.5 * F_downforce
#      F_z,r = (l_f / L) * m * g + 0.5 * F_downforce
#      where F_downforce = 0.5 * rho * v_x^2 * A_wing * Cl
#   3. F_rx (drivetrain longitudinal force) is clamped at mu * F_z,r
#      to prevent exceeding the friction limit (no wheel spin).
#   4. New VEHICLE_SPECS fields required in config:
#      mu, rho, Cl, A_wing, h_cog
# =============================================================================


class DeepDynamicsModelV2(ModelBase):
    """Modified Deep Dynamics for simulation data (Ts=0.02).
    Removes Df/Dr from NN output, computes them from downforce physics."""

    def __init__(self, param_dict, eval=False):

        class GuardLayer(nn.Module):
            def __init__(self, param_dict):
                super().__init__()
                guard_output = create_module(
                    "DENSE",
                    param_dict["MODEL"]["LAYERS"][-1]["OUT_FEATURES"],
                    param_dict["MODEL"]["HORIZON"],
                    len(param_dict["PARAMETERS"]),
                    activation="Sigmoid"
                )
                self.guard_dense = guard_output[0]
                self.guard_activation = guard_output[1]
                self.coefficient_ranges = torch.zeros(
                    len(param_dict["PARAMETERS"])
                ).to(device)
                self.coefficient_mins = torch.zeros(
                    len(param_dict["PARAMETERS"])
                ).to(device)
                for i in range(len(param_dict["PARAMETERS"])):
                    self.coefficient_ranges[i] = (
                        param_dict["PARAMETERS"][i]["Max"]
                        - param_dict["PARAMETERS"][i]["Min"]
                    )
                    self.coefficient_mins[i] = param_dict["PARAMETERS"][i]["Min"]

            def forward(self, x):
                guard_output = self.guard_dense(x)
                guard_output = (
                    self.guard_activation(guard_output)
                    * self.coefficient_ranges
                    + self.coefficient_mins
                )
                return guard_output

        super().__init__(param_dict, [GuardLayer(param_dict)], eval)

    def differential_equation(self, x, output, Ts=0.02):
        sys_param_dict, _ = self.unpack_sys_params(output)
        state_action_dict = self.unpack_state_actions(x)

        steering = state_action_dict["STEERING_FB"] + state_action_dict["STEERING_CMD"]
        throttle = state_action_dict["THROTTLE_FB"] + state_action_dict["THROTTLE_CMD"]
        vx = state_action_dict["VX"]
        vy = state_action_dict["VY"]
        yaw_rate = state_action_dict["YAW_RATE"]

        # Vehicle geometry and constants from config
        mass = self.vehicle_specs["mass"]
        lf = self.vehicle_specs["lf"]
        lr = self.vehicle_specs["lr"]
        L = lf + lr
        mu = self.vehicle_specs["mu"]
        rho = self.vehicle_specs["rho"]
        Cl = self.vehicle_specs["Cl"]
        A_wing = self.vehicle_specs["A_wing"]

        # ---- Aerodynamic downforce (velocity-dependent) ----
        # F_downforce = 0.5 * rho * vx^2 * A_wing * Cl
        F_downforce = 0.5 * rho * (vx ** 2) * A_wing * Cl

        # ---- Normal forces on front and rear axles ----
        # Static weight split by lever arm ratio, plus downforce (split 50/50 for now)
        Fz_f = (lr / L) * mass * 9.81 + 0.5 * F_downforce
        Fz_r = (lf / L) * mass * 9.81 + 0.5 * F_downforce

        # ---- Compute Df, Dr analytically (NOT from NN) ----
        Df = mu * Fz_f
        Dr = mu * Fz_r

        # ---- Slip angles (identical to original) ----
        alphaf = (
            steering
            - torch.atan2(lf * yaw_rate + vy, torch.abs(vx))
            + sys_param_dict["Shf"]
        )
        alphar = (
            torch.atan2(lr * yaw_rate - vy, torch.abs(vx))
            + sys_param_dict["Shr"]
        )

        # ---- Drivetrain longitudinal force ----
        Frx = (
            (sys_param_dict["Cm1"] - sys_param_dict["Cm2"] * vx) * throttle
            - sys_param_dict["Cr0"]
            - sys_param_dict["Cr2"] * (vx ** 2)
        )

        # ---- Friction clipping: cap Frx at mu * Fz_r ----
        Frx = torch.clamp(Frx, max=float(mu * Fz_r) if not isinstance(Fz_r, torch.Tensor) else None)
        if isinstance(Fz_r, torch.Tensor):
            Frx = torch.min(Frx, mu * Fz_r)

        # ---- Pacejka lateral tire forces (using analytical Df, Dr) ----
        Ffy = (
            sys_param_dict["Svf"]
            + Df * torch.sin(
                sys_param_dict["Cf"] * torch.atan(
                    sys_param_dict["Bf"] * alphaf
                    - sys_param_dict["Ef"] * (
                        sys_param_dict["Bf"] * alphaf
                        - torch.atan(sys_param_dict["Bf"] * alphaf)
                    )
                )
            )
        )
        Fry = (
            sys_param_dict["Svr"]
            + Dr * torch.sin(
                sys_param_dict["Cr"] * torch.atan(
                    sys_param_dict["Br"] * alphar
                    - sys_param_dict["Er"] * (
                        sys_param_dict["Br"] * alphar
                        - torch.atan(sys_param_dict["Br"] * alphar)
                    )
                )
            )
        )

        # ---- State derivatives (identical structure to original) ----
        dxdt = torch.zeros(len(x), 3).to(device)
        dxdt[:, 0] = (
            1.0 / mass * (Frx - Ffy * torch.sin(steering))
            + vy * yaw_rate
        )
        dxdt[:, 1] = (
            1.0 / mass * (Fry + Ffy * torch.cos(steering))
            - vx * yaw_rate
        )
        dxdt[:, 2] = (
            1.0 / sys_param_dict["Iz"]
            * (Ffy * lf * torch.cos(steering) - Fry * lr)
        )
        dxdt *= Ts
        return x[:, -1, :3] + dxdt


class DeepDynamicsModelIACV2(ModelBase):
    """Modified Deep Dynamics for IAC real-world data (Ts=0.04).
    Identical physics to DeepDynamicsModelV2, different default timestep."""

    def __init__(self, param_dict, eval=False):

        class GuardLayer(nn.Module):
            def __init__(self, param_dict):
                super().__init__()
                guard_output = create_module(
                    "DENSE",
                    param_dict["MODEL"]["LAYERS"][-1]["OUT_FEATURES"],
                    param_dict["MODEL"]["HORIZON"],
                    len(param_dict["PARAMETERS"]),
                    activation="Sigmoid"
                )
                self.guard_dense = guard_output[0]
                self.guard_activation = guard_output[1]
                self.coefficient_ranges = torch.zeros(
                    len(param_dict["PARAMETERS"])
                ).to(device)
                self.coefficient_mins = torch.zeros(
                    len(param_dict["PARAMETERS"])
                ).to(device)
                for i in range(len(param_dict["PARAMETERS"])):
                    self.coefficient_ranges[i] = (
                        param_dict["PARAMETERS"][i]["Max"]
                        - param_dict["PARAMETERS"][i]["Min"]
                    )
                    self.coefficient_mins[i] = param_dict["PARAMETERS"][i]["Min"]

            def forward(self, x):
                guard_output = self.guard_dense(x)
                guard_output = (
                    self.guard_activation(guard_output)
                    * self.coefficient_ranges
                    + self.coefficient_mins
                )
                return guard_output

        super().__init__(param_dict, [GuardLayer(param_dict)], eval)

    def differential_equation(self, x, output, Ts=0.04):
        sys_param_dict, _ = self.unpack_sys_params(output)
        state_action_dict = self.unpack_state_actions(x)

        steering = state_action_dict["STEERING_FB"] + state_action_dict["STEERING_CMD"]
        throttle = state_action_dict["THROTTLE_FB"] + state_action_dict["THROTTLE_CMD"]
        vx = state_action_dict["VX"]
        vy = state_action_dict["VY"]
        yaw_rate = state_action_dict["YAW_RATE"]

        mass = self.vehicle_specs["mass"]
        lf = self.vehicle_specs["lf"]
        lr = self.vehicle_specs["lr"]
        L = lf + lr
        mu = self.vehicle_specs["mu"]
        rho = self.vehicle_specs["rho"]
        Cl = self.vehicle_specs["Cl"]
        A_wing = self.vehicle_specs["A_wing"]

        F_downforce = 0.5 * rho * (vx ** 2) * A_wing * Cl

        Fz_f = (lr / L) * mass * 9.81 + 0.5 * F_downforce
        Fz_r = (lf / L) * mass * 9.81 + 0.5 * F_downforce

        Df = mu * Fz_f
        Dr = mu * Fz_r

        alphaf = (
            steering
            - torch.atan2(lf * yaw_rate + vy, torch.abs(vx))
            + sys_param_dict["Shf"]
        )
        alphar = (
            torch.atan2(lr * yaw_rate - vy, torch.abs(vx))
            + sys_param_dict["Shr"]
        )

        Frx = (
            (sys_param_dict["Cm1"] - sys_param_dict["Cm2"] * vx) * throttle
            - sys_param_dict["Cr0"]
            - sys_param_dict["Cr2"] * (vx ** 2)
        )

        Frx = torch.clamp(Frx, max=float(mu * Fz_r) if not isinstance(Fz_r, torch.Tensor) else None)
        if isinstance(Fz_r, torch.Tensor):
            Frx = torch.min(Frx, mu * Fz_r)

        Ffy = (
            sys_param_dict["Svf"]
            + Df * torch.sin(
                sys_param_dict["Cf"] * torch.atan(
                    sys_param_dict["Bf"] * alphaf
                    - sys_param_dict["Ef"] * (
                        sys_param_dict["Bf"] * alphaf
                        - torch.atan(sys_param_dict["Bf"] * alphaf)
                    )
                )
            )
        )
        Fry = (
            sys_param_dict["Svr"]
            + Dr * torch.sin(
                sys_param_dict["Cr"] * torch.atan(
                    sys_param_dict["Br"] * alphar
                    - sys_param_dict["Er"] * (
                        sys_param_dict["Br"] * alphar
                        - torch.atan(sys_param_dict["Br"] * alphar)
                    )
                )
            )
        )

        dxdt = torch.zeros(len(x), 3).to(device)
        dxdt[:, 0] = (
            1.0 / mass * (Frx - Ffy * torch.sin(steering))
            + vy * yaw_rate
        )
        dxdt[:, 1] = (
            1.0 / mass * (Fry + Ffy * torch.cos(steering))
            - vx * yaw_rate
        )
        dxdt[:, 2] = (
            1.0 / sys_param_dict["Iz"]
            * (Ffy * lf * torch.cos(steering) - Fry * lr)
        )
        dxdt *= Ts
        return x[:, -1, :3] + dxdt


string_to_model = {
    "DeepDynamics" : DeepDynamicsModel,
    "DeepPacejka" : DeepPacejkaModel,
    "DeepDynamicsIAC" : DeepDynamicsModelIAC,
    "DeepPacejkaIAC" : DeepPacejkaModelIAC,
    "DeepDynamicsV2" : DeepDynamicsModelV2,
    "DeepDynamicsIACV2" : DeepDynamicsModelIACV2
}

string_to_dataset = {
    "DeepDynamics" : DeepDynamicsDataset,
    "DeepPacejka" : DeepPacejkaDataset,
    "DeepDynamicsIAC" : DeepDynamicsDataset,
    "DeepPacejkaIAC" : DeepPacejkaDataset,
    "DeepDynamicsV2" : DeepDynamicsDataset,
    "DeepDynamicsIACV2" : DeepDynamicsDataset,
}
