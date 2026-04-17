from torch import nn
from sklearn.preprocessing import StandardScaler
import torch
from deep_dynamics.model.build_network import build_network, string_to_torch, create_module
import yaml
import pickle
import numpy as np
from abc import abstractmethod


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
        vx = state_action_dict["VX"]
        vy = state_action_dict["VY"]
        omega = state_action_dict["YAW_RATE"]
        m = self.vehicle_specs["mass"]
        lf = self.vehicle_specs["lf"]
        lr = self.vehicle_specs["lr"]
        L = self.vehicle_specs["L"]
        h = self.vehicle_specs["h"]

        alphaf = steering - torch.atan2(lf * omega + vy, torch.abs(vx)) + sys_param_dict["Shf"]
        alphar = torch.atan2(lr * omega - vy, torch.abs(vx)) + sys_param_dict["Shr"]

        F_rx_desired = (sys_param_dict["Cm1"] - sys_param_dict["Cm2"] * vx) * throttle \
                       - sys_param_dict["Cr0"] - sys_param_dict["Cr2"] * (vx ** 2)

        a_x = F_rx_desired / m
        delta_W = compute_load_transfer(h, L, m, a_x)
        F_aero_f, F_aero_r = compute_aero_forces(
            vx, vy, self.vehicle_specs["rho"],
            self.vehicle_specs["A_f"], self.vehicle_specs["A_r"],
            self.vehicle_specs["C_lf"], self.vehicle_specs["C_lr"])
        F_fz, F_rz = compute_normal_forces(lf, lr, L, m, self.vehicle_specs["g"], delta_W, F_aero_f, F_aero_r)

        mu = self.vehicle_specs["mu"]
        D_f = mu * F_fz
        D_r = mu * F_rz

        Ffy = sys_param_dict["Svf"] + D_f * torch.sin(sys_param_dict["Cf"] * torch.atan(sys_param_dict["Bf"] * alphaf - sys_param_dict["Ef"] * (sys_param_dict["Bf"] * alphaf - torch.atan(sys_param_dict["Bf"] * alphaf))))
        Fry = sys_param_dict["Svr"] + D_r * torch.sin(sys_param_dict["Cr"] * torch.atan(sys_param_dict["Br"] * alphar - sys_param_dict["Er"] * (sys_param_dict["Br"] * alphar - torch.atan(sys_param_dict["Br"] * alphar))))

        Frx = compute_traction_limited_force(F_rx_desired, mu, F_rz)

        dxdt = torch.zeros(len(x), 3).to(device)
        dxdt[:,0] = 1/m * (Frx - Ffy*torch.sin(steering)) + vy*omega
        dxdt[:,1] = 1/m * (Fry + Ffy*torch.cos(steering)) - vx*omega
        dxdt[:,2] = 1/sys_param_dict["Iz"] * (Ffy*lf*torch.cos(steering) - Fry*lr)
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
        vx = state_action_dict["VX"]
        vy = state_action_dict["VY"]
        omega = state_action_dict["YAW_RATE"]
        m = self.vehicle_specs["mass"]
        lf = self.vehicle_specs["lf"]
        lr = self.vehicle_specs["lr"]
        L = self.vehicle_specs["L"]
        h = self.vehicle_specs["h"]

        alphaf = steering - torch.atan2(lf * omega + vy, torch.abs(vx)) + sys_param_dict["Shf"]
        alphar = torch.atan2(lr * omega - vy, torch.abs(vx)) + sys_param_dict["Shr"]

        F_rx_desired = (sys_param_dict["Cm1"] - sys_param_dict["Cm2"] * vx) * throttle \
                       - sys_param_dict["Cr0"] - sys_param_dict["Cr2"] * (vx ** 2)

        a_x = F_rx_desired / m
        delta_W = compute_load_transfer(h, L, m, a_x)
        F_aero_f, F_aero_r = compute_aero_forces(
            vx, vy, self.vehicle_specs["rho"],
            self.vehicle_specs["A_f"], self.vehicle_specs["A_r"],
            self.vehicle_specs["C_lf"], self.vehicle_specs["C_lr"])
        F_fz, F_rz = compute_normal_forces(lf, lr, L, m, self.vehicle_specs["g"], delta_W, F_aero_f, F_aero_r)

        mu = self.vehicle_specs["mu"]
        D_f = mu * F_fz
        D_r = mu * F_rz

        Ffy = sys_param_dict["Svf"] + D_f * torch.sin(sys_param_dict["Cf"] * torch.atan(sys_param_dict["Bf"] * alphaf - sys_param_dict["Ef"] * (sys_param_dict["Bf"] * alphaf - torch.atan(sys_param_dict["Bf"] * alphaf))))
        Fry = sys_param_dict["Svr"] + D_r * torch.sin(sys_param_dict["Cr"] * torch.atan(sys_param_dict["Br"] * alphar - sys_param_dict["Er"] * (sys_param_dict["Br"] * alphar - torch.atan(sys_param_dict["Br"] * alphar))))

        Frx = compute_traction_limited_force(F_rx_desired, mu, F_rz)

        dxdt = torch.zeros(len(x), 3).to(device)
        dxdt[:,0] = 1/m * (Frx - Ffy*torch.sin(steering)) + vy*omega
        dxdt[:,1] = 1/m * (Fry + Ffy*torch.cos(steering)) - vx*omega
        dxdt[:,2] = 1/sys_param_dict["Iz"] * (Ffy*lf*torch.cos(steering) - Fry*lr)
        dxdt *= Ts
        return x[:,-1,:3] + dxdt


class DeepDynamicsPINNModel(DeepDynamicsModel):
    """
    Deep Dynamics model with PINN-style composite loss function.
    Same architecture as DeepDynamicsModel (GRU + dense + guard layer + physics ODE)
    but the loss includes ODE residuals at interior timesteps, energy conservation,
    and parameter regularization in addition to the standard data fidelity term.
    """

    def __init__(self, param_dict, eval=False):
        super().__init__(param_dict, eval)
        pinn_cfg = param_dict.get("PINN", {})
        self.lambda_data = pinn_cfg.get("LAMBDA_DATA", 1.0)
        self.lambda_ode = pinn_cfg.get("LAMBDA_ODE", 0.01)
        self.lambda_energy = pinn_cfg.get("LAMBDA_ENERGY", 0.001)
        self.lambda_param_reg = pinn_cfg.get("LAMBDA_PARAM_REG", 0.0001)
        self._Ts = pinn_cfg.get("TS", 0.02)
        self._warmup_start = pinn_cfg.get("WARMUP_START", 50)
        self._warmup_end = pinn_cfg.get("WARMUP_END", 150)

        self._state_idx = {name: i for i, name in enumerate(self.state)}
        self._action_idx = {name: len(self.state) + i for i, name in enumerate(self.actions)}

        self._nominal_params = {}
        for p in self.param_dict["PARAMETERS"]:
            for key, val in p.items():
                if key not in ("Min", "Max") and val is not None:
                    self._nominal_params[key] = float(val)

    def _feat(self, x, t, name):
        idx = self._state_idx.get(name, self._action_idx.get(name))
        return x[:, t, idx]

    def _physics_derivs(self, vx, vy, omega, steering, throttle, sp):
        """Evaluate the vehicle dynamics ODE for arbitrary state/action values."""
        m = self.vehicle_specs["mass"]
        lf = self.vehicle_specs["lf"]
        lr = self.vehicle_specs["lr"]
        L = self.vehicle_specs["L"]
        h = self.vehicle_specs["h"]

        alphaf = steering - torch.atan2(lf * omega + vy, torch.abs(vx)) + sp["Shf"]
        alphar = torch.atan2(lr * omega - vy, torch.abs(vx)) + sp["Shr"]

        F_rx_desired = (sp["Cm1"] - sp["Cm2"] * vx) * throttle \
                       - sp["Cr0"] - sp["Cr2"] * (vx ** 2)

        a_x = F_rx_desired / m
        delta_W = compute_load_transfer(h, L, m, a_x)
        F_aero_f, F_aero_r = compute_aero_forces(
            vx, vy, self.vehicle_specs["rho"],
            self.vehicle_specs["A_f"], self.vehicle_specs["A_r"],
            self.vehicle_specs["C_lf"], self.vehicle_specs["C_lr"])
        F_fz, F_rz = compute_normal_forces(lf, lr, L, m, self.vehicle_specs["g"],
                                            delta_W, F_aero_f, F_aero_r)

        mu = self.vehicle_specs["mu"]
        D_f = mu * F_fz
        D_r = mu * F_rz

        Ffy = sp["Svf"] + D_f * torch.sin(sp["Cf"] * torch.atan(
            sp["Bf"] * alphaf - sp["Ef"] * (sp["Bf"] * alphaf - torch.atan(sp["Bf"] * alphaf))))
        Fry = sp["Svr"] + D_r * torch.sin(sp["Cr"] * torch.atan(
            sp["Br"] * alphar - sp["Er"] * (sp["Br"] * alphar - torch.atan(sp["Br"] * alphar))))
        Frx = compute_traction_limited_force(F_rx_desired, mu, F_rz)

        dvx = 1 / m * (Frx - Ffy * torch.sin(steering)) + vy * omega
        dvy = 1 / m * (Fry + Ffy * torch.cos(steering)) - vx * omega
        domega = 1 / sp["Iz"] * (Ffy * lf * torch.cos(steering) - Fry * lr)
        return dvx, dvy, domega, Frx, Ffy, Fry

    def compute_pinn_loss(self, x, predictions, labels, ff_output, epoch=0):
        sp, _ = self.unpack_sys_params(ff_output)
        Ts = self._Ts
        m = self.vehicle_specs["mass"]

        if epoch < self._warmup_start:
            physics_weight = 0.0
        elif epoch >= self._warmup_end:
            physics_weight = 1.0
        else:
            physics_weight = (epoch - self._warmup_start) / (self._warmup_end - self._warmup_start)

        # 1. Data fidelity
        L_data = ((predictions - labels) ** 2).mean()

        # 2. ODE residual at interior timesteps of the input window.
        #    The learned parameters should be consistent with observed
        #    state transitions at every timestep, not only the last.
        L_ode = torch.tensor(0.0, device=x.device)
        n_interior = x.shape[1] - 1
        for t in range(n_interior):
            vx_t = self._feat(x, t, "VX")
            vy_t = self._feat(x, t, "VY")
            omega_t = self._feat(x, t, "YAW_RATE")
            steer_t = self._feat(x, t, "STEERING_FB") + self._feat(x, t, "STEERING_CMD")
            thr_t = self._feat(x, t, "THROTTLE_FB") + self._feat(x, t, "THROTTLE_CMD")

            obs_delta_vx = self._feat(x, t + 1, "VX") - vx_t
            obs_delta_vy = self._feat(x, t + 1, "VY") - vy_t
            obs_delta_om = self._feat(x, t + 1, "YAW_RATE") - omega_t

            p_dvx, p_dvy, p_dom, _, _, _ = self._physics_derivs(
                vx_t, vy_t, omega_t, steer_t, thr_t, sp)

            L_ode = L_ode + ((p_dvx * Ts - obs_delta_vx) ** 2).mean()
            L_ode = L_ode + ((p_dvy * Ts - obs_delta_vy) ** 2).mean()
            L_ode = L_ode + ((p_dom * Ts - obs_delta_om) ** 2).mean()
        L_ode = L_ode / max(n_interior, 1)

        # 3. Energy conservation: dKE should equal work done by forces.
        vx = self._feat(x, -1, "VX")
        vy = self._feat(x, -1, "VY")
        omega = self._feat(x, -1, "YAW_RATE")
        steer = self._feat(x, -1, "STEERING_FB") + self._feat(x, -1, "STEERING_CMD")
        thr = self._feat(x, -1, "THROTTLE_FB") + self._feat(x, -1, "THROTTLE_CMD")

        _, _, _, Frx, Ffy, Fry = self._physics_derivs(vx, vy, omega, steer, thr, sp)

        KE_curr = 0.5 * m * (vx ** 2 + vy ** 2)
        KE_next = 0.5 * m * (predictions[:, 0] ** 2 + predictions[:, 1] ** 2)
        P_forces = (Frx * vx + Fry * vy
                    + Ffy * (vy * torch.cos(steer) - vx * torch.sin(steer)))
        energy_residual = (KE_next - KE_curr) - P_forces * Ts
        L_energy = (energy_residual ** 2).mean()

        # 4. Soft regularization toward nominal parameter values.
        L_param = torch.tensor(0.0, device=x.device)
        n_reg = 0
        for key, nominal in self._nominal_params.items():
            if key in sp:
                L_param = L_param + ((sp[key] - nominal) ** 2).mean()
                n_reg += 1
        if n_reg > 0:
            L_param = L_param / n_reg

        total = (self.lambda_data * L_data
                 + physics_weight * (self.lambda_ode * L_ode
                                     + self.lambda_energy * L_energy
                                     + self.lambda_param_reg * L_param))

        return total, {
            "total": total.item(),
            "data": L_data.item(),
            "ode": L_ode.item(),
            "energy": L_energy.item(),
            "param_reg": L_param.item(),
        }


class DeepPacejkaModel(ModelBase):
    def __init__(self, param_dict, eval=False):
        output_module = create_module("DENSE", param_dict["MODEL"]["LAYERS"][-1]["OUT_FEATURES"], param_dict["MODEL"]["HORIZON"], len(param_dict["PARAMETERS"]), activation=None)
        super().__init__(param_dict, output_module, eval)

    def differential_equation(self, x, output, Ts=0.02):
        sys_param_dict, _ = self.unpack_sys_params(output)
        state_action_dict = self.unpack_state_actions(x)
        steering = state_action_dict["STEERING_FB"] + state_action_dict["STEERING_CMD"]
        vx = state_action_dict["VX"]
        vy = state_action_dict["VY"]
        omega = state_action_dict["YAW_RATE"]
        m = self.vehicle_specs["mass"]
        lf = self.vehicle_specs["lf"]
        lr = self.vehicle_specs["lr"]
        L = self.vehicle_specs["L"]
        h = self.vehicle_specs["h"]

        alphaf = steering - torch.atan2(lf * omega + vy, torch.abs(vx))
        alphar = torch.atan2(lr * omega - vy, torch.abs(vx))

        F_rx_predicted = sys_param_dict["Frx"]
        a_x = F_rx_predicted / m
        delta_W = compute_load_transfer(h, L, m, a_x)
        F_aero_f, F_aero_r = compute_aero_forces(
            vx, vy, self.vehicle_specs["rho"],
            self.vehicle_specs["A_f"], self.vehicle_specs["A_r"],
            self.vehicle_specs["C_lf"], self.vehicle_specs["C_lr"])
        F_fz, F_rz = compute_normal_forces(lf, lr, L, m, self.vehicle_specs["g"], delta_W, F_aero_f, F_aero_r)

        mu = self.vehicle_specs["mu"]
        D_f = mu * F_fz
        D_r = mu * F_rz

        Ffy = D_f * torch.sin(sys_param_dict["Cf"] * torch.atan(sys_param_dict["Bf"] * alphaf - sys_param_dict["Ef"] * (sys_param_dict["Bf"] * alphaf - torch.atan(sys_param_dict["Bf"] * alphaf))))
        Fry = D_r * torch.sin(sys_param_dict["Cr"] * torch.atan(sys_param_dict["Br"] * alphar - sys_param_dict["Er"] * (sys_param_dict["Br"] * alphar - torch.atan(sys_param_dict["Br"] * alphar))))

        Frx = compute_traction_limited_force(F_rx_predicted, mu, F_rz)

        dxdt = torch.zeros(len(x), 3).to(device)
        dxdt[:,0] = 1/m * (Frx - Ffy*torch.sin(steering)) + vy*omega
        dxdt[:,1] = 1/m * (Fry + Ffy*torch.cos(steering)) - vx*omega
        dxdt[:,2] = 1/self.vehicle_specs["Iz"] * (Ffy*lf*torch.cos(steering) - Fry*lr)
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
        vx = state_action_dict["VX"]
        vy = state_action_dict["VY"]
        omega = state_action_dict["YAW_RATE"]
        m = self.vehicle_specs["mass"]
        lf = self.vehicle_specs["lf"]
        lr = self.vehicle_specs["lr"]
        L = self.vehicle_specs["L"]
        h = self.vehicle_specs["h"]

        alphaf = steering - torch.atan2(lf * omega + vy, torch.abs(vx)) + sys_param_dict["Shf"]
        alphar = torch.atan2(lr * omega - vy, torch.abs(vx)) + sys_param_dict["Shr"]

        F_rx_desired = (sys_param_dict["Cm1"] - sys_param_dict["Cm2"] * vx) * throttle \
                       - sys_param_dict["Cr0"] - sys_param_dict["Cr2"] * (vx ** 2)

        a_x = F_rx_desired / m
        delta_W = compute_load_transfer(h, L, m, a_x)
        F_aero_f, F_aero_r = compute_aero_forces(
            vx, vy, self.vehicle_specs["rho"],
            self.vehicle_specs["A_f"], self.vehicle_specs["A_r"],
            self.vehicle_specs["C_lf"], self.vehicle_specs["C_lr"])
        F_fz, F_rz = compute_normal_forces(lf, lr, L, m, self.vehicle_specs["g"], delta_W, F_aero_f, F_aero_r)

        mu = self.vehicle_specs["mu"]
        D_f = mu * F_fz
        D_r = mu * F_rz

        Ffy = sys_param_dict["Svf"] + D_f * torch.sin(sys_param_dict["Cf"] * torch.atan(sys_param_dict["Bf"] * alphaf - sys_param_dict["Ef"] * (sys_param_dict["Bf"] * alphaf - torch.atan(sys_param_dict["Bf"] * alphaf))))
        Fry = sys_param_dict["Svr"] + D_r * torch.sin(sys_param_dict["Cr"] * torch.atan(sys_param_dict["Br"] * alphar - sys_param_dict["Er"] * (sys_param_dict["Br"] * alphar - torch.atan(sys_param_dict["Br"] * alphar))))

        Frx = compute_traction_limited_force(F_rx_desired, mu, F_rz)

        dxdt = torch.zeros(len(x), 3).to(device)
        dxdt[:,0] = 1/m * (Frx - Ffy*torch.sin(steering)) + vy*omega
        dxdt[:,1] = 1/m * (Fry + Ffy*torch.cos(steering)) - vx*omega
        dxdt[:,2] = 1/sys_param_dict["Iz"] * (Ffy*lf*torch.cos(steering) - Fry*lr)
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
        vx = state_action_dict["VX"]
        vy = state_action_dict["VY"]
        omega = state_action_dict["YAW_RATE"]
        m = self.vehicle_specs["mass"]
        lf = self.vehicle_specs["lf"]
        lr = self.vehicle_specs["lr"]
        L = self.vehicle_specs["L"]
        h = self.vehicle_specs["h"]

        alphaf = steering - torch.atan2(lf * omega + vy, torch.abs(vx))
        alphar = torch.atan2(lr * omega - vy, torch.abs(vx))

        F_rx_predicted = sys_param_dict["Frx"]
        a_x = F_rx_predicted / m
        delta_W = compute_load_transfer(h, L, m, a_x)
        F_aero_f, F_aero_r = compute_aero_forces(
            vx, vy, self.vehicle_specs["rho"],
            self.vehicle_specs["A_f"], self.vehicle_specs["A_r"],
            self.vehicle_specs["C_lf"], self.vehicle_specs["C_lr"])
        F_fz, F_rz = compute_normal_forces(lf, lr, L, m, self.vehicle_specs["g"], delta_W, F_aero_f, F_aero_r)

        mu = self.vehicle_specs["mu"]
        D_f = mu * F_fz
        D_r = mu * F_rz

        Ffy = D_f * torch.sin(sys_param_dict["Cf"] * torch.atan(sys_param_dict["Bf"] * alphaf - sys_param_dict["Ef"] * (sys_param_dict["Bf"] * alphaf - torch.atan(sys_param_dict["Bf"] * alphaf))))
        Fry = D_r * torch.sin(sys_param_dict["Cr"] * torch.atan(sys_param_dict["Br"] * alphar - sys_param_dict["Er"] * (sys_param_dict["Br"] * alphar - torch.atan(sys_param_dict["Br"] * alphar))))

        Frx = compute_traction_limited_force(F_rx_predicted, mu, F_rz)

        dxdt = torch.zeros(len(x), 3).to(device)
        dxdt[:,0] = 1/m * (Frx - Ffy*torch.sin(steering)) + vy*omega
        dxdt[:,1] = 1/m * (Fry + Ffy*torch.cos(steering)) - vx*omega
        dxdt[:,2] = 1/self.vehicle_specs["Iz"] * (Ffy*lf*torch.cos(steering) - Fry*lr)
        dxdt *= Ts
        return x[:,-1,:3] + dxdt


string_to_model = {
    "DeepDynamics" : DeepDynamicsModel,
    "DeepDynamicsPINN" : DeepDynamicsPINNModel,
    "DeepPacejka" : DeepPacejkaModel,
    "DeepDynamicsIAC" : DeepDynamicsModelIAC,
    "DeepPacejkaIAC" : DeepPacejkaModelIAC,
}

string_to_dataset = {
    "DeepDynamics" : DeepDynamicsDataset,
    "DeepDynamicsPINN" : DeepDynamicsDataset,
    "DeepPacejka" : DeepPacejkaDataset,
    "DeepDynamicsIAC" : DeepDynamicsDataset,
    "DeepPacejkaIAC" : DeepPacejkaDataset,
}
