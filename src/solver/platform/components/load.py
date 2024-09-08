import gurobipy as gp
from gurobipy import GRB

class FlexibleLoad:
    def __init__(self, T_set, ls_setting, phi_ls):
        """Initialize parameters."""
        self.T_set = T_set

        self.ls_setting = ls_setting
        self.phi_ls = phi_ls

    def add_variables(self, model, p_fl, var_name='p_ls'):
        """Add variables to the model."""
        p_ls = model.addVars(self.T_set, lb=self.ls_setting*p_fl, ub=p_fl, vtype=GRB.CONTINUOUS, name=var_name)
        return p_ls
    
    def get_cost(self, p_fl, p_ls):
        """Calculate the DR cost of flexible load."""
        return gp.quicksum(((p_fl[t] - p_ls[t]) * self.phi_ls) for t in self.T_set)