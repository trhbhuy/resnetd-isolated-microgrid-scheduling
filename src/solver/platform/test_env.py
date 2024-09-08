import gymnasium as gym
from gymnasium import spaces
import numpy as np
import gurobipy as gp
from gurobipy import GRB

from .. import config as cfg
from ..methods.data_loader import load_data
from .util import scaler_loader, check_boundary_constraint, check_ramp_constraint, check_setpoint, generate_pla_points, calculate_F_deg, calculate_F_ess

class MicrogridEnv(gym.Env):
    def __init__(self):
        """Initialize the microgrid environment."""
        # Load and initialize the network data, parameters, and PLA points
        self._init_network_data()
        self._init_params()
        self._initialize_pla_points()
    
        # Load the simulation data
        self.data = load_data(is_train=False)
        self.num_scenarios = len(self.data['p_if']) // self.T_num
        print(f"Number of scenarios: {self.num_scenarios}")

        # Load state and action scalers
        self.state_scaler, self.action_scaler = scaler_loader()

        # Define observation space (normalized to [0, 1])
        observation_dim = 3
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(observation_dim,), dtype=np.float32)

        # Define action space (normalized to [0, 1])
        action_dim = 1
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(action_dim,), dtype=np.float32)

    def reset(self, seed=0):
        """Reset the environment to an initial state."""
        self.scenario_seed = int(seed)
        self.time_step = 0

        # Calculate the index for the scenario data
        index = self._get_index(self.scenario_seed, self.time_step)

        # Initialize state
        initial_state = np.array([
            self.time_step,
            self.data['p_if'][index] + self.data['p_fl_1'][index] + self.data['p_fl_2'][index] - 
            (self.data['p_pv_max'][index] + self.data['p_wg_max'][index]),
            self.soc_ess_setpoint
        ], dtype=np.float32)

        # Initialize the previous DEG power to zero
        self.p_deg_tempt = 0

        # Normalize the initial state
        self.state = self.state_scaler.transform([initial_state])[0].astype(np.float32)

        return self.state, {}

    def step(self, action):
        """Take an action and return the next state, reward, and termination status."""
        # Inverse transform the state from normalized form
        current_state = self.state_scaler.inverse_transform([self.state])[0].astype(np.float32)

        # Decompose the state into individual variables
        time_step, p_net, soc_ess_tempt = current_state
        time_step = int(np.round(time_step))

        # Fetch the data for the current time step
        base_idx = self._get_index(self.scenario_seed, time_step)

        # Inverse transform the action using the scaler
        action_pred = self.action_scaler.inverse_transform(action.reshape(1, -1))[0]

        # Clip action to be within its respective limit
        action = np.clip(action_pred, -self.p_ess_dch_max, self.p_ess_ch_max)[0]
        p_ess_ch, p_ess_dch, soc_ess = self._update_ess(time_step, action, soc_ess_tempt)

        # Solve the MILP optimization problem
        p_deg, u_deg, p_pv, p_wg, p_ls_1, p_ls_2, reward = self.optim(self.data['p_pv_max'][base_idx], 
                                                                      self.data['p_wg_max'][base_idx], 
                                                                      self.data['p_if'][base_idx], 
                                                                      self.data['p_fl_1'][base_idx], 
                                                                      self.data['p_fl_2'][base_idx], 
                                                                      p_ess_ch, p_ess_dch)

        # Calculate the reward and penalties
        cumulative_penalty = self._get_penalty(time_step, p_deg, u_deg, self.p_deg_tempt, self.r_deg, soc_ess)
        reward += cumulative_penalty * self.penalty_coefficient

        # Prepare next state
        next_state, terminated = self._get_obs(time_step, soc_ess)

        # Update the state for the next step
        self.state = self.state_scaler.transform([next_state])[0].astype(np.float32)
        self.p_deg_tempt = p_deg if time_step < self.T_num else 0

        return self.state, reward, terminated, False, {
            "p_deg": p_deg,
            "u_deg": u_deg,
            "p_pv": p_pv,
            "p_wg": p_wg,
            "p_ls_1": p_ls_1,
            "p_ls_2": p_ls_2,
            "p_ess_ch": p_ess_ch,
            "p_ess_dch": p_ess_dch,
            "soc_ess": soc_ess
        }

    def _get_obs(self, time_step, soc_ess):
        """Prepare the next state and determine if the episode has terminated."""
        # Increment the time step.
        time_step += 1

        # Determine if the episode has terminated.
        terminated = time_step >= self.T_num

        # Prepare the next state if the episode is ongoing.
        if not terminated:
            base_idx = self.scenario_seed * self.T_num + time_step
            next_state = np.array([
                time_step,
                self.data['p_if'][base_idx] + self.data['p_fl_1'][base_idx] + self.data['p_fl_2'][base_idx] - 
                (self.data['p_pv_max'][base_idx] + self.data['p_wg_max'][base_idx]),
                soc_ess
            ], dtype=np.float32)
        else:
            next_state = np.array([time_step, 0, 0], dtype=np.float32)

        return next_state, terminated

    def _update_ess(self, time_step, action, soc_ess_tempt):
        """Update the ESS state based on the action taken."""
        # Initialize charge and discharge power
        p_ess_ch, p_ess_dch = 0.0, 0.0

        if time_step == 0:
            # No ESS action allowed at the first timestep
            pass
        elif time_step == self.T_num - 1:
            # Limit the charge power to not exceed the SOC setpoint
            p_ess_ch = min((self.soc_ess_setpoint - soc_ess_tempt) / (self.n_ess_ch * self.delta_t), self.p_ess_ch_max)
        else:
            # Charge or discharge ESS based on action
            p_ess_ch = max(action, 0)
            p_ess_dch = -min(action, 0)

            # Calculate potential SOC after applying action
            soc_ess = soc_ess_tempt + self.delta_t * (p_ess_ch * self.n_ess_ch - p_ess_dch / self.n_ess_dch)

            # Postprocess action to meet all ESS bound constraints
            p_ess_ch, p_ess_dch = self._postprocess_bound(p_ess_ch, p_ess_dch, soc_ess, soc_ess_tempt, self.p_ess_ch_max, self.p_ess_dch_max, self.n_ess_ch, self.n_ess_dch, self.soc_ess_max, self.soc_ess_min)

            # # Special condition for the second-to-last timestep
            # if time_step == self.T_num - 2:
            #     p_ess_ch, p_ess_dch = self._postprocess_ess_setpoint(p_ess_ch, p_ess_dch, soc_ess, soc_ess_tempt)

        # Update SOC based on adjusted powers
        soc_ess = soc_ess_tempt + self.delta_t * (p_ess_ch * self.n_ess_ch - p_ess_dch / self.n_ess_dch)

        return p_ess_ch, p_ess_dch, soc_ess

    def _postprocess_bound(self, p_ch, p_dch, soc, soc_tempt, p_ch_max, p_dch_max, n_ch, n_dch, soc_max, soc_min):
        """Adjust charging and discharging powers based on SOC constraints."""
        # SOC is above the maximum limit.
        if soc > soc_max:
            p_ch = min((soc_max - soc_tempt) / (n_ch * self.delta_t), p_ch_max)
            p_dch = 0
        # SOC is below the minimum limit.
        elif soc < soc_min:
            p_ch = 0
            p_dch = min((soc_tempt - soc_min) * n_dch / self.delta_t, p_dch_max)
        else:
            p_ch = min(p_ch, p_ch_max)
            p_dch = min(p_dch, p_dch_max)

        return p_ch, p_dch

    def _postprocess_ess_setpoint(self, p_ess_ch, p_ess_dch, soc_ess, soc_ess_tempt):
        """Handle special conditions for ESS at the second-to-last timestep."""
        if soc_ess < self.soc_ess_threshold:
            if soc_ess_tempt < self.soc_ess_threshold:
                p_ess_ch = min((self.soc_ess_threshold - soc_ess_tempt) / (self.n_ess_ch * self.delta_t), self.p_ess_ch_max)
                p_ess_dch = 0
            elif soc_ess_tempt > self.soc_ess_threshold:
                p_ess_ch = 0
                p_ess_dch = min((soc_ess_tempt - self.soc_ess_threshold) * self.n_ess_dch / self.delta_t, self.p_ess_dch_max)

        return p_ess_ch, p_ess_dch

    def _get_penalty(self, time_step, p_deg, u_deg, p_deg_tempt, r_deg, soc_ess):
        """Calculate penalties for boundary and ramp rate violations for the generator and ESS."""
        # DEG penalties
        deg_penalty = 0
        deg_penalty += check_boundary_constraint(p_deg, self.p_deg_min * u_deg, self.p_deg_max)
        if time_step > 0:
            deg_penalty += check_ramp_constraint(p_deg, p_deg_tempt, r_deg)

        # ESS penalties
        ess_penalty = 0
        ess_penalty += check_boundary_constraint(soc_ess, self.soc_ess_min, self.soc_ess_max)

        if time_step == 0 or time_step == (self.T_num - 1):
            ess_penalty += check_setpoint(soc_ess, self.soc_ess_setpoint)

        return deg_penalty + ess_penalty

    def optim(self, p_pv_max, p_wg_max, p_if, p_fl_1, p_fl_2, p_ess_ch, p_ess_dch):
        """Optimization method for the microgrid."""
        # Create a new model
        model = gp.Model()
        model.ModelSense = GRB.MINIMIZE
        model.Params.LogToConsole = 0

        ## Diesel engine generator (DEG)
        p_deg = model.addVar(vtype=GRB.CONTINUOUS, name="p_deg")
        pla_deg = model.addVar(vtype=GRB.CONTINUOUS, name="pla_deg")
        F_deg = model.addVar(vtype=GRB.CONTINUOUS, name="F_deg")
        u_deg = model.addVar(vtype=GRB.BINARY, name="u_deg")

        # DEG Constraints
        model.addConstr(p_deg <= self.p_deg_max * u_deg)
        model.addConstr(p_deg >= self.p_deg_min * u_deg)

        model.addGenConstrPWL(p_deg, pla_deg, self.ptu_deg, self.ptf_deg)
        model.addConstr(F_deg == pla_deg * u_deg)

        ## Solar PV scheduled
        p_pv = model.addVar(ub=p_pv_max, vtype=GRB.CONTINUOUS, name="p_pv")

        ## Wind generator scheduled
        p_wg = model.addVar(ub=p_wg_max, vtype=GRB.CONTINUOUS, name="p_wg")

        # Operation and Maintenance (OM) cost
        F_OM = p_pv * self.phi_pv + p_wg * self.phi_wg + ((p_ess_ch + p_ess_dch) ** 2) * self.phi_ess

        ## Flexible load modeling
        p_ls_1 = model.addVar(lb=self.ls_setting * p_fl_1, ub=p_fl_1, vtype=GRB.CONTINUOUS, name="p_ls_1")
        p_ls_2 = model.addVar(lb=self.ls_setting * p_fl_2, ub=p_fl_2, vtype=GRB.CONTINUOUS, name="p_ls_2")

        # Demand Response (DR) cost
        F_DR = (p_fl_1 - p_ls_1) * self.phi_ls_1 + (p_fl_2 - p_ls_2) * self.phi_ls_2

        ## Define network variables
        p_it = model.addVars(self.B_set, vtype=GRB.CONTINUOUS, name="p_it")
        q_it = model.addVars(self.B_set, vtype=GRB.CONTINUOUS, name="q_it")
        l_P_it = model.addVars(self.B_set, vtype=GRB.CONTINUOUS, name="l_P_it")
        l_Q_it = model.addVars(self.B_set, vtype=GRB.CONTINUOUS, name="l_Q_it")

        P_ijt = model.addVars(self.branch_ij, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="P_ijt")
        Q_ijt = model.addVars(self.branch_ij, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="Q_ijt")
        L_ijt = model.addVars(self.branch_ij, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="L_ijt")
        v_it = model.addVars(self.B_set, lb=self.v_min**2, ub=self.v_max**2, vtype=GRB.CONTINUOUS, name="v_it")

        ## Assign values to network variables
        # Node 0 - Inflexible load demand
        model.addConstr(p_it[self.if_node] == 0)
        model.addConstr(q_it[self.if_node] == 0)
        model.addConstr(l_P_it[self.if_node] == p_if / self.baseMVA)
        model.addConstr(l_Q_it[self.if_node] == 0.3 * l_P_it[self.if_node])

        # Node 1 - Flexible load demands 1 and 2
        model.addConstr(p_it[self.fl_node] == 0)
        model.addConstr(q_it[self.fl_node] == 0)
        model.addConstr(l_P_it[self.fl_node] == (p_ls_1 + p_ls_2) / self.baseMVA)
        model.addConstr(l_Q_it[self.fl_node] == 0.3 * l_P_it[self.fl_node])

        # Node 2 - ESS
        model.addConstr(p_it[self.ess_node] == p_ess_dch / self.baseMVA)
        model.addConstr(q_it[self.ess_node] == 0)
        model.addConstr(l_P_it[self.ess_node] == p_ess_ch / self.baseMVA)
        model.addConstr(l_Q_it[self.ess_node] == 0)

        # Node 3 - PV
        model.addConstr(p_it[self.pv_node] == p_pv / self.baseMVA)
        model.addConstr(q_it[self.pv_node] <= p_it[self.pv_node])
        model.addConstr(l_P_it[self.pv_node] == 0)
        model.addConstr(l_Q_it[self.pv_node] == 0)

        # Node 4 - WG
        model.addConstr(p_it[self.wg_node] == p_wg / self.baseMVA)
        model.addConstr(q_it[self.wg_node] <= p_it[self.wg_node])
        model.addConstr(l_P_it[self.wg_node] == 0)
        model.addConstr(l_Q_it[self.wg_node] == 0)

        # Node 5 - DEG
        model.addConstr(p_it[self.deg_node] == p_deg / self.baseMVA)
        model.addConstr(q_it[self.deg_node] <= p_it[self.deg_node])
        model.addConstr(l_P_it[self.deg_node] == 0)
        model.addConstr(l_Q_it[self.deg_node] == 0)

        # Squared line current limits
        for (ii, jj) in self.branch_ij:
            model.addConstr(L_ijt[ii, jj] <= self.I_max[ii, jj]**2)

        # Power flow equations
        for jj in self.B_set:
            if jj == 0:
                model.addConstr(0 == l_P_it[jj] - p_it[jj] + gp.quicksum(P_ijt[jj, kk] for kk in self.Ninsert_set[jj]))
                model.addConstr(0 == l_Q_it[jj] - q_it[jj] + gp.quicksum(Q_ijt[jj, kk] for kk in self.Ninsert_set[jj]))
            else:
                ii = self.Nout_set[jj][0]  # Parent node

                # Line Power Flows
                model.addConstr(P_ijt[ii, jj] == l_P_it[jj] - p_it[jj] + self.r_ij[ii, jj] * L_ijt[ii, jj] + gp.quicksum(P_ijt[jj, kk] for kk in self.Ninsert_set[jj]))
                model.addConstr(Q_ijt[ii, jj] == l_Q_it[jj] - q_it[jj] + self.x_ij[ii, jj] * L_ijt[ii, jj] + gp.quicksum(Q_ijt[jj, kk] for kk in self.Ninsert_set[jj]))

        for (ii, jj) in self.branch_ij:
            # Nodal voltage
            model.addConstr(v_it[jj] == v_it[ii] + (self.r_ij[ii, jj]**2 + self.x_ij[ii, jj]**2) * L_ijt[ii, jj] - 2 * (self.r_ij[ii, jj] * P_ijt[ii, jj] + self.x_ij[ii, jj] * Q_ijt[ii, jj]))

            # Squared current magnitude on lines
            model.addConstr(L_ijt[ii, jj] * v_it[ii] >= (P_ijt[ii, jj]**2 + Q_ijt[ii, jj]**2))

        # Define problem and solve
        model.setObjective(F_deg + F_OM + F_DR)
        model.optimize()

        # return results
        return p_deg.X, u_deg.X, p_pv.X, p_wg.X, p_ls_1.X, p_ls_2.X, model.ObjVal

    def _init_network_data(self):
        """Load network data and parameters from the scenario configuration."""
        self.bus_data = cfg.bus_data
        self.branch_data = cfg.branch_data
        self.baseMVA = cfg.BASE_MVA
        self.B_set = cfg.B_SET
        self.T_num = cfg.T_NUM
        self.T_set = cfg.T_SET
        self.delta_t = cfg.DELTA_T
        self.branch_ij = cfg.BRANCH_IJ
        self.r_ij = cfg.R_IJ
        self.x_ij = cfg.X_IJ
        self.I_max = cfg.I_MAX
        self.v_min = cfg.V_MIN
        self.v_max = cfg.V_MAX
        self.Ninsert_set = cfg.NINSERT_SET
        self.Nout_set = cfg.NOUT_SET
        self.if_node = cfg.IF_NODE
        self.fl_node = cfg.FL_NODE
        self.ess_node = cfg.ESS_NODE
        self.pv_node = cfg.PV_NODE
        self.wg_node = cfg.WG_NODE
        self.deg_node = cfg.DEG_NODE

    def _init_params(self):
        """Initialize constants and parameters from the scenario configuration."""
        # Parameters for Generation Units
        self.p_pv_rate = cfg.P_PV_RATE
        self.n_pv = cfg.N_PV
        self.phi_pv = cfg.PHI_PV
        
        self.p_wg_rate = cfg.P_WG_RATE
        self.n_wg = cfg.N_WG
        self.phi_wg = cfg.PHI_WG

        # Diesel engine generator (DEG) Parameters
        self.p_deg_max = cfg.P_DEG_MAX
        self.p_deg_min = cfg.P_DEG_MIN
        self.r_deg = cfg.R_DEG
        self.w1_deg = cfg.W1_DEG
        self.w2_deg = cfg.W2_DEG
        self.w3_deg = cfg.W3_DEG

        # Energy Storage System (ESS) Parameters
        self.p_ess_ch_max = cfg.P_ESS_CH_MAX
        self.p_ess_dch_max = cfg.P_ESS_DCH_MAX
        self.ess_dod = cfg.ESS_DOD
        self.soc_ess_max = cfg.SOC_ESS_MAX
        self.n_ess_ch = cfg.N_ESS_CH
        self.n_ess_dch = cfg.N_ESS_DCH
        self.soc_ess_min = cfg.SOC_ESS_MIN
        self.soc_ess_setpoint = cfg.SOC_ESS_SETPOINT
        self.phi_ess = cfg.PHI_ESS
        self.soc_ess_threshold = cfg.SOC_ESS_THRESHOLD
        self.penalty_coefficient = cfg.PENALTY_COEFFICIENT

        # Flexible Load (FL) Parameters
        self.ls_setting = cfg.LS_SETTING
        self.phi_ls_1 = cfg.PHI_LS_1
        self.phi_ls_2 = cfg.PHI_LS_2

    def _initialize_pla_points(self, npts: int = 101):
        """Initialize the piecewise linear approximation (PLA) points for DEG and ESS."""
        # Create PLA points for DEG
        self.ptu_deg, self.ptf_deg = generate_pla_points(self.p_deg_min, self.p_deg_max, lambda p: calculate_F_deg(p, self.w1_deg, self.w2_deg, self.w3_deg), npts)

        # Create PLA points for ESS
        self.ptu_ess, self.ptf_ess = generate_pla_points(0, self.p_ess_ch_max, calculate_F_ess)

    def _get_index(self, scenario: int, time_step: int) -> int:
        """Get index for scenario data based on time step and scenario seed."""
        return scenario * self.T_num + time_step
