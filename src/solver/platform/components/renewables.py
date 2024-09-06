import gurobipy as gp
from gurobipy import GRB

class PV:
    def __init__(self, p_pv_rate, n_pv, phi_pv, T_set):
        self.p_pv_rate = p_pv_rate
        self.n_pv = n_pv
        self.phi_pv = phi_pv

        self.T_set = T_set

    def add_variables(self, model, p_pv_max):
        p_pv = model.addVars(self.T_set, ub=p_pv_max, vtype=GRB.CONTINUOUS, name="p_pv")
        return p_pv
    
    def get_cost(self, p_pv):
        return gp.quicksum(p_pv[t] * self.phi_pv for t in self.T_set)

class WG:
    def __init__(self, p_wg_rate, n_wg, phi_wg, T_set):
        self.p_wg_rate = p_wg_rate
        self.n_wg = n_wg
        self.phi_wg = phi_wg

        self.T_set = T_set

    def add_variables(self, model, p_wg_max):
        p_wg = model.addVars(self.T_set, ub=p_wg_max, vtype=GRB.CONTINUOUS, name="p_wg")
        return p_wg
    
    def get_cost(self, p_wg):
        return gp.quicksum(p_wg[t] * self.phi_wg for t in self.T_set)
