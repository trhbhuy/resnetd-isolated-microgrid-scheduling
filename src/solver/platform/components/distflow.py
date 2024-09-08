import gurobipy as gp
from gurobipy import GRB

class NetworkConstraints:
    def __init__(self, T_set, B_set, bus_data, branch_data, baseMVA, branch_ij, r_ij, x_ij, I_max, v_min, v_max, Ninsert_set, Nout_set, if_node, fl_node, ess_node, pv_node, wg_node, deg_node):
        """Initialize parameters."""
        self.T_set = T_set
        self.B_set = B_set
        self.bus_data = bus_data
        self.branch_data = branch_data
        self.baseMVA = baseMVA
        self.branch_ij = branch_ij
        self.r_ij = r_ij
        self.x_ij = x_ij
        self.I_max = I_max
        self.v_min = v_min
        self.v_max = v_max
        self.Ninsert_set = Ninsert_set
        self.Nout_set = Nout_set

        self.if_node = if_node
        self.fl_node = fl_node
        self.ess_node = ess_node
        self.pv_node = pv_node
        self.wg_node = wg_node
        self.deg_node = deg_node

    def add_variables(self, model):
        """Add variables to the model."""
        p_it = model.addVars(self.B_set, self.T_set, vtype=GRB.CONTINUOUS, name="p_it")
        q_it = model.addVars(self.B_set, self.T_set, vtype=GRB.CONTINUOUS, name="q_it")
        l_P_it = model.addVars(self.B_set, self.T_set, vtype=GRB.CONTINUOUS, name="l_P_it")
        l_Q_it = model.addVars(self.B_set, self.T_set, vtype=GRB.CONTINUOUS, name="l_Q_it")

        P_ijt = model.addVars(self.branch_ij, self.T_set, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="P_ijt")
        Q_ijt = model.addVars(self.branch_ij, self.T_set, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="Q_ijt")
        L_ijt = model.addVars(self.branch_ij, self.T_set, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="L_ijt")
        v_it = model.addVars(self.B_set, self.T_set, lb=self.v_min**2, ub=self.v_max**2, vtype=GRB.CONTINUOUS, name="v_it")

        return p_it, q_it, l_P_it, l_Q_it, P_ijt, Q_ijt, L_ijt, v_it

    def add_constraints(self, model, p_it, q_it, l_P_it, l_Q_it, P_ijt, Q_ijt, L_ijt, v_it):
        """Add constraints to the model."""
        # Squared line current limits
        for tt in self.T_set:
            for (ii, jj) in self.branch_ij:
                model.addConstr(L_ijt[ii, jj, tt] <= self.I_max[ii, jj]**2)

        # Power flow equations
        for tt in self.T_set:
            for jj in self.B_set:
                if jj == 0:
                    model.addConstr(0 == l_P_it[jj, tt] - p_it[jj, tt] + gp.quicksum(P_ijt[jj, kk, tt] for kk in self.Ninsert_set[jj]))
                    model.addConstr(0 == l_Q_it[jj, tt] - q_it[jj, tt] + gp.quicksum(Q_ijt[jj, kk, tt] for kk in self.Ninsert_set[jj]))
                else:
                    ii = self.Nout_set[jj][0]  # Parent node

                    # Line Power Flows
                    model.addConstr(P_ijt[ii, jj, tt] == l_P_it[jj, tt] - p_it[jj, tt] + self.r_ij[ii, jj] * L_ijt[ii, jj, tt] + gp.quicksum(P_ijt[jj, kk, tt] for kk in self.Ninsert_set[jj]))
                    model.addConstr(Q_ijt[ii, jj, tt] == l_Q_it[jj, tt] - q_it[jj, tt] + self.x_ij[ii, jj] * L_ijt[ii, jj, tt] + gp.quicksum(Q_ijt[jj, kk, tt] for kk in self.Ninsert_set[jj]))

            for (ii, jj) in self.branch_ij:
                # Nodal voltage
                model.addConstr(v_it[jj, tt] == v_it[ii, tt] + (self.r_ij[ii, jj]**2 + self.x_ij[ii, jj]**2) * L_ijt[ii, jj, tt] - 2 * (self.r_ij[ii, jj] * P_ijt[ii, jj, tt] + self.x_ij[ii, jj] * Q_ijt[ii, jj, tt]))

                # Squared current magnitude on lines
                model.addConstr(L_ijt[ii, jj, tt] * v_it[ii, tt] >= (P_ijt[ii, jj, tt]**2 + Q_ijt[ii, jj, tt]**2))

    def add_pq_constraints(self, model, p_it, q_it, l_P_it, l_Q_it, p_if, p_ls_1, p_ls_2, p_ess_ch, p_ess_dch, p_pv, p_wg, p_deg):
        """Add gen and load constraints to the model."""
        for tt in self.T_set:
            # Node 0 - Inflexible load demand
            model.addConstr(p_it[self.if_node, tt] == 0)
            model.addConstr(q_it[self.if_node, tt] == 0)
            model.addConstr(l_P_it[self.if_node, tt] == p_if[tt] / self.baseMVA)
            model.addConstr(l_Q_it[self.if_node, tt] == 0.3 * l_P_it[self.if_node, tt])

            # Node 1 - Flexible load demands 1 and 2
            model.addConstr(p_it[self.fl_node, tt] == 0)
            model.addConstr(q_it[self.fl_node, tt] == 0)
            model.addConstr(l_P_it[self.fl_node, tt] == (p_ls_1[tt] + p_ls_2[tt]) / self.baseMVA)
            model.addConstr(l_Q_it[self.fl_node, tt] == 0.3 * l_P_it[self.fl_node, tt])

            # Node 2 - ESS
            model.addConstr(p_it[self.ess_node, tt] == p_ess_dch[tt] / self.baseMVA)
            model.addConstr(q_it[self.ess_node, tt] == 0)
            model.addConstr(l_P_it[self.ess_node, tt] == p_ess_ch[tt] / self.baseMVA)
            model.addConstr(l_Q_it[self.ess_node, tt] == 0)

            # Node 3 - PV
            model.addConstr(p_it[self.pv_node, tt] == p_pv[tt] / self.baseMVA)
            model.addConstr(q_it[self.pv_node, tt] <= p_it[self.pv_node, tt])
            model.addConstr(l_P_it[self.pv_node, tt] == 0)
            model.addConstr(l_Q_it[self.pv_node, tt] == 0)

            # Node 4 - WG
            model.addConstr(p_it[self.wg_node, tt] == p_wg[tt] / self.baseMVA)
            model.addConstr(q_it[self.wg_node, tt] <= p_it[self.wg_node, tt])
            model.addConstr(l_P_it[self.wg_node, tt] == 0)
            model.addConstr(l_Q_it[self.wg_node, tt] == 0)

            # Node 5 - DEG
            model.addConstr(p_it[self.deg_node, tt] == p_deg[tt] / self.baseMVA)
            model.addConstr(q_it[self.deg_node, tt] <= p_it[self.deg_node, tt])
            model.addConstr(l_P_it[self.deg_node, tt] == 0)
            model.addConstr(l_Q_it[self.deg_node, tt] == 0)
