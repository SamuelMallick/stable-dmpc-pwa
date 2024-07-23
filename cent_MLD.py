import gurobipy as gp
import numpy as np
from dmpcpwa.agents.mld_agent import MldAgent
from dmpcpwa.mpc.mpc_mld import MpcMld
from gymnasium.wrappers import TimeLimit
from mpcrl.wrappers.envs import MonitorEpisodes
from scipy.linalg import block_diag

from system.env import Network
from system.model import get_cent_system, get_cost_matrices, get_inv_set, get_terminal_costs
from utils.plotting import plot_system

N = 5
n = 3  # num sub-systems
ep_len = 30


class Cent_MPC(MpcMld):
    Q_x_l, Q_u_l = get_cost_matrices()
    P_l = get_terminal_costs()
    Q_x = block_diag(*[Q_x_l] * n)
    Q_u = block_diag(*[Q_u_l] * n)
    A, b = get_inv_set()
    P = block_diag(*P_l)

    def __init__(self, system: dict, N: int) -> None:
        # dynamics, state, and input constraints built in here with MLD model conversion
        super().__init__(system, N, verbose=True)

        obj = 0
        for k in range(N):
            obj += (
                self.x[:, k] @ self.Q_x @ self.x[:, [k]]
                + self.u[:, k] @ self.Q_u @ self.u[:, [k]]
            )
        obj += self.x[:, N] @ self.P @ self.x[:, [N]]
        self.mpc_model.setObjective(obj, gp.GRB.MINIMIZE)

        # terminal constraint
        self.mpc_model.addConstrs(
            self.A @ self.x[i : i + 2, [N]] <= self.b for i in range(0, 2 * n, 2)
        )


# controller
mpc = Cent_MPC(get_cent_system(), N)

# environment
env = MonitorEpisodes(
    TimeLimit(
        Network(),
        max_episode_steps=ep_len,
    )
)

# agent
agent = MldAgent(mpc)
agent.evaluate(env=env, episodes=1, seed=1)

if len(env.observations) > 0:
    X = env.observations[0].squeeze()
    U = env.actions[0].squeeze()
    R = env.rewards[0]
else:
    X = np.squeeze(env.ep_observations)
    U = np.squeeze(env.ep_actions)
    R = np.squeeze(env.ep_rewards)

print(f"cost = {sum(R)}")
plot_system(X, U)
