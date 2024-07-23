import gurobipy as gp
import numpy as np
from dmpcpwa.agents.mld_agent import MldAgent
from dmpcpwa.mpc.mpc_mld import MpcMld
from gymnasium.wrappers import TimeLimit
from mpcrl.wrappers.envs import MonitorEpisodes
from scipy.linalg import block_diag

from system.env import Network
from system.model import Model
from utils.plotting import plot_system

class CentralizedMPC(MpcMld):
    """A centralized MPC controller that solved for the control input with a mixed-integer program."""

    def __init__(self, model: Model, N: int) -> None:
        """Initialize the centralized MPC controller.
        
        Parameters
        ----------
        model : Model
            The model of the system.
        N : int
            The horizon length.
        """
        system = model.get_cent_system()
        super().__init__(system, N, verbose=True)

        Q_x_l, Q_u_l = model.get_cost_matrices()
        P_l = model.terminal_costs
        Q_x = block_diag(*[Q_x_l] * model.n)
        Q_u = block_diag(*[Q_u_l] * model.n)
        A, b = model.get_inv_set()
        P = block_diag(*P_l)

        obj = 0
        for k in range(N):
            obj += (
                self.x[:, k] @ Q_x @ self.x[:, [k]]
                + self.u[:, k] @ Q_u @ self.u[:, [k]]
            )
        obj += self.x[:, N] @ P @ self.x[:, [N]]
        self.mpc_model.setObjective(obj, gp.GRB.MINIMIZE)

        # terminal constraint
        self.mpc_model.addConstrs(
            A @ self.x[i : i + 2, [N]] <= b for i in range(0, 2 * model.n, 2)
        )
