import logging
import pickle

import numpy as np
import pandas as pd
from dmpcpwa.agents.no_control_agent import NoControlAgent
from dmpcrl.core.admm import g_map
from gymnasium.wrappers import TimeLimit
from mpcrl.wrappers.agents import Log
from mpcrl.wrappers.envs import MonitorEpisodes

from agents.pwa_agent_terminal import PwaAgentTerminal
from agents.stable_switching_admm_coordinator import StableSwitchingAdmmCoordinator
from mpcs.local_mpc import LocalMpc
from system.env import Network
from system.model import Model
from utils.plotting import plot_system

PLOT = False
SAVE = True

STRONG_COUPLING = False

CONTROL = True  # if False the open loop uncontrolled system runs
MULTIPLE_ICS = True  # if True the simulation is run for all ICs in the ICs.csv file

N = 5
n = 3
ep_len = 30
model = Model(strong_coupling=STRONG_COUPLING)

# environment
env = MonitorEpisodes(
    TimeLimit(
        Network(model),
        max_episode_steps=ep_len,
    )
)

if STRONG_COUPLING:
    admm_iters = 75
    switching_iters = 50
    rho = 5.0
else:
    admm_iters = 50
    switching_iters = 50
    rho = 0.5

systems = model.get_local_coupled_systems()
Q_x, Q_u = model.get_cost_matrices()
G_map = g_map(model.adj)
P = model.terminal_costs
X_t = model.get_inv_set()
# distributed mpcs and params
local_mpcs: list[LocalMpc] = []
local_fixed_dist_parameters: list[dict] = []
for i in range(n):
    local_mpcs.append(
        LocalMpc(
            system=systems[i],
            N=N,
            Q=(Q_x, Q_u),
            num_neighbours=len(G_map[i]) - 1,
            my_index=G_map[i].index(i),
            P=P[i],
            X_t=X_t,
            rho=rho,
            solver="osqp",
        )
    )
    local_fixed_dist_parameters.append(local_mpcs[i].fixed_pars_init)

# coordinator
if CONTROL:
    agent = Log(
        StableSwitchingAdmmCoordinator(
            local_mpcs=local_mpcs,
            local_fixed_parameters=local_fixed_dist_parameters,
            model=model,
            G=G_map,
            Adj=model.adj,
            rho=rho,
            N=N,
            X_t=X_t,
            cent_mpc=None,
            admm_iters=admm_iters,
            switching_iters=switching_iters,
            agent_class=PwaAgentTerminal,
            debug_plot=False,
        ),
        level=logging.DEBUG,
        log_frequencies={"on_timestep_end": 1},
    )
else:
    agent = NoControlAgent(3, local_mpcs[0])

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
        print(
            f"av solve time = {sum([t for t in agent.solve_times if t != 0])/len([t for t in agent.solve_times if t != 0])}"
        )
        if PLOT:
            plot_system(X, U)

        id = i + 1
        if SAVE:
            with open(
                f"gadmm_{id}" + ".pkl",
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
    print(
        f"av solve time = {sum([t for t in agent.solve_times if t != 0])/len([t for t in agent.solve_times if t != 0])}"
    )
    # print(f"av solve time = {np.mean(agent.solve_times)}")
    plot_system(X, U)

    id = "unstab" if STRONG_COUPLING else "stab"
    if SAVE:
        with open(
            f"gadmm_{id}" + ".pkl",
            "wb",
        ) as file:
            pickle.dump(X, file)
            pickle.dump(U, file)
            pickle.dump(R, file)
            pickle.dump(agent.solve_times, file)
