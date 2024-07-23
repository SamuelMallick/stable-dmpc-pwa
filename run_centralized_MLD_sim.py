import pickle

from gymnasium import Env
import numpy as np
from dmpcpwa.agents.mld_agent import MldAgent
from gymnasium.wrappers import TimeLimit
from mpcrl.wrappers.envs import MonitorEpisodes
import pandas as pd

from mpcs.cent_mpc import CentralizedMPC
from system.env import Network
from system.model import Model
from utils.plotting import plot_system

PLOT = False
SAVE = True
MULTIPLE_ICS = True
STRONG_COUPLING = False

N = 5  # horizon
ep_len = 30
model = Model(strong_coupling=STRONG_COUPLING)

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
class TimedMldAgent(MldAgent):
    def reset(self, seed=None) -> None:
        self.solve_times: list[float] = []
        return super().reset(seed)
    
    def on_timestep_end(self, env: Env, episode: int, timestep: int) -> None:
        self.solve_times.append(self.run_time)
        return super().on_timestep_end(env, episode, timestep)
    
agent = TimedMldAgent(mpc)
agent.evaluate(env=env, episodes=1, seed=1)

# get data from env
X = np.squeeze(env.observations)
U = np.squeeze(env.actions)
R = np.squeeze(env.rewards)

if MULTIPLE_ICS:
    df = pd.read_csv("utils/ICs.csv", header=None)
    X0s = df.values
    for i in range(X0s.shape[0]):
        env.set_IC(X0s[[i], :].T)
        env.reset()
        agent.evaluate(env=env, episodes=1, seed=1)
        if len(env.observations) > 0:
            X = env.observations[i].squeeze()
            U = env.actions[i].squeeze()
            R = env.rewards[i]
        else:
            X = np.squeeze(env.ep_observations)
            U = np.squeeze(env.ep_actions)
            R = np.squeeze(env.ep_rewards)

        print(f"cost = {sum(R)}")
        if PLOT:
            plot_system(X, U)

        id = i + 1
        if SAVE:
            with open(
                f"cent_mld_{id}" + ".pkl",
                "wb",
            ) as file:
                pickle.dump(X, file)
                pickle.dump(U, file)
                pickle.dump(R, file)
                pickle.dump(agent.solve_times, file)
else:
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

    id = "unstab" if STRONG_COUPLING else "stab"
    if SAVE:
        with open(
            f"cent_mld_{id}" + ".pkl",
            "wb",
        ) as file:
            pickle.dump(X, file)
            pickle.dump(U, file)
            pickle.dump(R, file)
            pickle.dump(agent.solve_times, file)

