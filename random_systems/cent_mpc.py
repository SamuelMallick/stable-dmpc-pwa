import gurobipy as gp
from dmpcpwa.mpc.mpc_mld import MpcMld
from scipy.linalg import block_diag
import numpy as np


class CentralizedMPC(MpcMld):
    """A centralized MPC controller that solved for the control input with a mixed-integer program."""

    def __init__(self, system: dict, N: int) -> None:
        """Initialize the centralized MPC controller."""
        super().__init__(system, N, verbose=False)

        Q_x = np.eye(4)
        Q_u = np.eye(2)

        obj = 0
        for k in range(N):
            obj += (
                self.x[:, k] @ Q_x @ self.x[:, [k]]
                + self.u[:, k] @ Q_u @ self.u[:, [k]]
            )
        obj += self.x[:, N] @ Q_x @ self.x[:, [N]]
        self.mpc_model.setObjective(obj, gp.GRB.MINIMIZE)
