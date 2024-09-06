import numpy as np
import gurobipy as gp
from gurobipy import GRB

# from ..util import generate_pla_points, calculate_F_deg

class DEG:
    def __init__(self, p_deg_max, p_deg_min, r_deg, w1, w2, w3, T_set):
        self.p_deg_max = p_deg_max
        self.p_deg_min = p_deg_min
        self.r_deg = r_deg
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.T_set = T_set
    
        # Generate PLA points using the method in this class
        self.ptu, self.ptf = self.generate_pla_points(self.p_deg_min, self.p_deg_max, self.get_F_deg)

    def add_variables(self, model):
        """
        Add variables to the optimization model for the DEG.
        """
        p_deg = model.addVars(self.T_set, vtype=GRB.CONTINUOUS, name="p_deg")
        pla_deg = model.addVars(self.T_set, vtype=GRB.CONTINUOUS, name="pla_deg")
        F_deg = model.addVars(self.T_set, vtype=GRB.CONTINUOUS, name="F_deg")
        u_deg = model.addVars(self.T_set, vtype=GRB.BINARY, name="u_deg")
        return p_deg, pla_deg, F_deg, u_deg

    def add_constraints(self, model, p_deg, pla_deg, F_deg, u_deg):
        """
        Add constraints to the optimization model for the DEG.
        """
        for t in self.T_set:
            model.addConstr(p_deg[t] <= self.p_deg_max * u_deg[t])
            model.addConstr(p_deg[t] >= self.p_deg_min * u_deg[t])
            model.addGenConstrPWL(p_deg[t], pla_deg[t], self.ptu, self.ptf)
            model.addConstr(F_deg[t] == pla_deg[t] * u_deg[t])
            if t >= 1:
                model.addConstr(p_deg[t] >= p_deg[t-1] - self.r_deg)
                model.addConstr(p_deg[t] <= p_deg[t-1] + self.r_deg)

    def generate_pla_points(self, lb: float, ub: float, func: callable, npts: int = 101) -> tuple:
        """
        Generate piecewise linear approximation (PLA) points for a given function.
        """
        ptu = np.linspace(lb, ub, npts)
        ptf = np.array([func(u) for u in ptu])
        return ptu, ptf

    def get_F_deg(self, p_deg: float) -> float:
        """
        Calculate the fuel consumption for the Diesel Engine Generator (DEG).
        """
        return self.w3 * p_deg**2 + self.w2 * p_deg + self.w1
