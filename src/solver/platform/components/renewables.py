import numpy as np
import gurobipy as gp
from gurobipy import GRB

class PV:
    def __init__(self, T_set, delta_t, p_pv_rate, n_pv, phi_pv=None):
        """Initialize parameters."""
        self.T_set = T_set
        self.delta_t = delta_t

        self.p_pv_rate = p_pv_rate
        self.n_pv = n_pv
        self.phi_pv = phi_pv

    def add_variables(self, model, p_pv_max):
        """Add variables to the model."""
        p_pv = model.addVars(self.T_set, ub=p_pv_max, vtype=GRB.CONTINUOUS, name="p_pv")
        return p_pv

    def get_pv_output(self, ghi, theta_air_out):
        """ Calculate the PV generation data based on the given GHI and outdoor temperature."""
        p_pv = self.p_pv_rate * (ghi * 0.25 + ghi * theta_air_out * 0.03 + (1.01 - 1.13 * self.n_pv) * np.square(ghi))
        return np.clip(p_pv, 0, self.p_pv_rate)

    def get_cost(self, p_pv):
        """Calculate the OM cost of PV."""
        return gp.quicksum(p_pv[t] * self.phi_pv for t in self.T_set)
    
class WG:
    def __init__(self, T_set, delta_t, p_wg_rate, n_wg, phi_wg):
        """Initialize parameters."""
        self.T_set = T_set
        self.delta_t = delta_t

        self.p_wg_rate = p_wg_rate
        self.n_wg = n_wg
        self.phi_wg = phi_wg

    def add_variables(self, model, p_wg_max):
        """Add variables to the model."""
        p_wg = model.addVars(self.T_set, ub=p_wg_max, vtype=GRB.CONTINUOUS, name="p_wg")
        return p_wg
    
    def get_cost(self, p_wg):
        """Calculate the OM cost of WG."""
        return gp.quicksum(p_wg[t] * self.phi_wg for t in self.T_set)
