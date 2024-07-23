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
from mpcs.cent_mpc import CentralizedMPC

N = 5   # horizon
ep_len = 30
model = Model(strong_coupling=False)

# controller
mpc = CentralizedMPC(model, N)

# environment
env = MonitorEpisodes(
    TimeLimit(
        Network(model),
        max_episode_steps=ep_len,
    )
)

# agent
agent = MldAgent(mpc)
agent.evaluate(env=env, episodes=1, seed=1)

# get data from env
X = np.squeeze(env.observations)
U = np.squeeze(env.actions)
R = np.squeeze(env.rewards)

print(f"cost = {sum(R)}")
plot_system(X, U)
