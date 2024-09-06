import numpy as np
import gurobipy as gp
from gurobipy import GRB

class FlexibleLoad:
    def __init__(self, ls_setting, phi_ls, T_set):
        self.ls_setting = ls_setting
        self.phi_ls = phi_ls
        self.T_set = T_set

    def add_variables(self, model, p_fl):
        p_ls = model.addVars(self.T_set, lb=self.ls_setting*p_fl, ub=p_fl, vtype=GRB.CONTINUOUS, name="p_ls_1")
        return p_ls
    
    def get_cost(self, p_fl, p_ls):
        return gp.quicksum(((p_fl[t] - p_ls[t]) * self.phi_ls) for t in self.T_set)