import numpy as np
import gurobipy as gp
from gurobipy import GRB
from typing import Dict, Optional
from .. import config as cfg

from .util import extract_results
from .components.diesel_generator import DEG
from .components.renewables import PV, WG
from .components.energy_storage import ESS
from .components.load import FlexibleLoad
from .components.distflow import NetworkConstraints

class Microgrid:
    def __init__(self):
        # Time Horizon
        self.T_num = cfg.T_NUM
        self.T_set = cfg.T_SET
        # self.delta_t = cfg.DELTA_T

        # Initialize the components of the microgrid
        self.deg = DEG(cfg.P_DEG_MAX, cfg.P_DEG_MIN, cfg.R_DEG, cfg.W1_DEG, cfg.W2_DEG, cfg.W3_DEG, cfg.T_SET)

        self.pv = PV(cfg.T_SET, cfg.DELTA_T, cfg.P_PV_RATE, cfg.N_PV, cfg.PHI_PV)
        self.wg = WG(cfg.T_SET, cfg.DELTA_T, cfg.P_WG_RATE, cfg.N_WG, cfg.PHI_WG)
        self.ess = ESS(cfg.T_NUM, cfg.T_SET, cfg.DELTA_T, cfg.P_ESS_CH_MAX, cfg.P_ESS_DCH_MAX, cfg.N_ESS_CH, cfg.N_ESS_DCH, cfg.SOC_ESS_MAX, cfg.SOC_ESS_MIN, cfg.SOC_ESS_SETPOINT, enable_cost_modeling=True, phi_ess=cfg.PHI_ESS)
        self.flexible_load_1 = FlexibleLoad(cfg.T_SET, cfg.LS_SETTING, cfg.PHI_LS_1)
        self.flexible_load_2 = FlexibleLoad(cfg.T_SET, cfg.LS_SETTING, cfg.PHI_LS_2)

        self.network_constraints = NetworkConstraints(cfg.B_SET, cfg.bus_data, cfg.branch_data, cfg.BASE_MVA, cfg.BRANCH_IJ, cfg.R_IJ, cfg.X_IJ, cfg.I_MAX, cfg.V_MIN, cfg.V_MAX, cfg.NINSERT_SET, cfg.NOUT_SET, cfg.IF_NODE, cfg.FL_NODE, cfg.ESS_NODE, cfg.PV_NODE, cfg.WG_NODE, cfg.DEG_NODE, cfg.T_SET)

    def optim(self, p_pv_max: np.ndarray, p_wg_max: np.ndarray, p_if: np.ndarray, p_fl_1: np.ndarray, p_fl_2: np.ndarray) -> Dict[str, np.ndarray]:
        """Optimization method for the microgrid."""
        model = gp.Model()
        model.ModelSense = GRB.MINIMIZE
        model.Params.LogToConsole = 0

        # Initialize variables for each component
        p_deg, pla_deg, F_deg, u_deg = self.deg.add_variables(model)
        p_ess_ch, p_ess_dch, u_ess_ch, u_ess_dch, soc_ess, p_ess, F_ess = self.ess.add_variables(model)
        p_pv = self.pv.add_variables(model, p_pv_max)
        p_wg = self.wg.add_variables(model, p_wg_max)
        p_ls_1 = self.flexible_load_1.add_variables(model, p_fl_1, 'p_ls_1')
        p_ls_2 = self.flexible_load_2.add_variables(model, p_fl_2, 'p_ls_2')

        p_it, q_it, l_P_it, l_Q_it, P_ijt, Q_ijt, L_ijt, v_it = self.network_constraints.add_variables(model)

        # Add constraints for each component
        self.deg.add_constraints(model, p_deg, pla_deg, F_deg, u_deg)
        self.ess.add_constraints(model, p_ess_ch, p_ess_dch, u_ess_ch, u_ess_dch, soc_ess, p_ess, F_ess)

        self.network_constraints.add_constraints(model, p_it, q_it, l_P_it, l_Q_it, P_ijt, Q_ijt, L_ijt, v_it)
        self.network_constraints.add_pq_constraints(model, p_it, q_it, l_P_it, l_Q_it, p_if, p_ls_1, p_ls_2, p_ess_ch, p_ess_dch, p_pv, p_wg, p_deg)

        # Operation and maintenance cost
        F_OM = self.pv.get_cost(p_pv) + self.wg.get_cost(p_wg) + self.ess.get_cost(F_ess)

        # Demand response cost
        F_DR = self.flexible_load_1.get_cost(p_fl_1, p_ls_1) + self.flexible_load_2.get_cost(p_fl_2, p_ls_2)

        # Define problem and solve
        model.setObjective(F_deg.sum() + F_OM + F_DR)
        model.optimize()

        if model.status == GRB.OPTIMAL:
            results = {
                'ObjVal': model.ObjVal,
                'p_deg': extract_results(p_deg, self.T_set),
                'u_deg': extract_results(u_deg, self.T_set),
                'p_pv': extract_results(p_pv, self.T_set),
                'p_wg': extract_results(p_wg, self.T_set),
                'p_ess_ch': extract_results(p_ess_ch, self.T_set),
                'p_ess_dch': extract_results(p_ess_dch, self.T_set),
                'u_ess_ch': extract_results(u_ess_ch, self.T_set),
                'u_ess_dch': extract_results(u_ess_dch, self.T_set),
                'soc_ess': extract_results(soc_ess, self.T_set),
                'p_ls_1': extract_results(p_ls_1, self.T_set),
                'p_ls_2': extract_results(p_ls_2, self.T_set),
                'p_it': extract_results(p_it, self.T_set, self.network_constraints.B_set),
                'q_it': extract_results(q_it, self.T_set, self.network_constraints.B_set),
                'l_P_it': extract_results(l_P_it, self.T_set, self.network_constraints.B_set),
                'l_Q_it': extract_results(l_Q_it, self.T_set, self.network_constraints.B_set),
                'v_it': extract_results(v_it, self.T_set, self.network_constraints.B_set),
            }

        else:
            raise RuntimeError(f"Optimization was unsuccessful. Model status: {model.status}")

        return results