import numpy as np
import gurobipy as gp
from gurobipy import GRB

class ESS:
    def __init__(self, T_num, T_set, delta_t, p_ess_ch_max, p_ess_dch_max, n_ess_ch, n_ess_dch, soc_ess_max, soc_ess_min, soc_ess_setpoint, enable_cost_modeling=False, phi_ess=None):
        """Initialize parameters."""
        self.T_num = T_num
        self.T_set = T_set
        self.delta_t = delta_t

        self.p_ess_ch_max = p_ess_ch_max
        self.p_ess_dch_max = p_ess_dch_max
        self.n_ess_ch = n_ess_ch
        self.n_ess_dch = n_ess_dch
        self.soc_ess_max = soc_ess_max
        self.soc_ess_min = soc_ess_min
        self.soc_ess_setpoint = soc_ess_setpoint
        self.enable_cost_modeling = enable_cost_modeling

        if self.enable_cost_modeling:
            self.phi_ess = phi_ess if phi_ess is not None else 1e-6  # Set default value if not provided
            # Generate PLA points if optimization flag is true
            self.ptu, self.ptf = self.generate_pla_points(0, self.p_ess_ch_max, self.get_F_ess)

    def add_variables(self, model):
        """Add variables to the model."""
        p_ess_ch = model.addVars(self.T_set, vtype=GRB.CONTINUOUS, name="p_ess_ch")
        p_ess_dch = model.addVars(self.T_set, vtype=GRB.CONTINUOUS, name="p_ess_dch")
        u_ess_ch = model.addVars(self.T_set, vtype=GRB.BINARY, name="u_ess_ch")
        u_ess_dch = model.addVars(self.T_set, vtype=GRB.BINARY, name="u_ess_dch")
        soc_ess = model.addVars(self.T_set, lb=self.soc_ess_min, ub=self.soc_ess_max, vtype=GRB.CONTINUOUS, name="soc_ess")

        # Add optional optimization variables if `enable_cost_modeling` is True
        p_ess, F_ess = None, None
        if self.enable_cost_modeling:
            p_ess = model.addVars(self.T_set, vtype=GRB.CONTINUOUS, name="p_ess")
            F_ess = model.addVars(self.T_set, vtype=GRB.CONTINUOUS, name="F_ess")

        return p_ess_ch, p_ess_dch, u_ess_ch, u_ess_dch, soc_ess, p_ess, F_ess

    def add_constraints(self, model, p_ess_ch, p_ess_dch, u_ess_ch, u_ess_dch, soc_ess, p_ess=None, F_ess=None):
        """Add constraints to the model."""
        for t in self.T_set:
            model.addConstr(p_ess_ch[t] <= self.p_ess_ch_max * u_ess_ch[t])
            model.addConstr(p_ess_dch[t] <= self.p_ess_dch_max * u_ess_dch[t])
            model.addConstr(u_ess_ch[t] + u_ess_dch[t] >= 0)
            model.addConstr(u_ess_ch[t] + u_ess_dch[t] <= 1)
            
            if t == 0:
                model.addConstr(soc_ess[t] == self.soc_ess_setpoint + self.delta_t * (p_ess_ch[t] * self.n_ess_ch - p_ess_dch[t] / self.n_ess_dch))
            else:
                model.addConstr(soc_ess[t] == soc_ess[t-1] + self.delta_t * (p_ess_ch[t] * self.n_ess_ch - p_ess_dch[t] / self.n_ess_dch))

        model.addConstr(soc_ess[0] == self.soc_ess_setpoint)
        model.addConstr(soc_ess[self.T_set[-1]] == self.soc_ess_setpoint)

        # Add optional constraints if `enable_cost_modeling` is True
        if self.enable_cost_modeling:
            for t in self.T_set:
                model.addConstr(p_ess[t] == p_ess_ch[t] + p_ess_dch[t])
                model.addGenConstrPWL(p_ess[t], F_ess[t], self.ptu, self.ptf)

    def get_cost(self, F_ess):
        """Calculate the operation & maintenance (OM) cost for ESS."""
        return gp.quicksum(F_ess[t] * self.phi_ess for t in self.T_set)

    def generate_pla_points(self, lb: float, ub: float, func: callable, npts: int = 101) -> tuple:
        """Generate piecewise linear approximation (PLA) points."""
        ptu = np.linspace(lb, ub, npts)
        ptf = np.array([func(u) for u in ptu])
        return ptu, ptf

    def get_F_ess(self, p_ess: float) -> float:
        """Calculate the operation cost for the ESS."""
        return p_ess**2