import logging
import pickle

import casadi as cs
from csnlp.wrappers import Mpc
import gurobipy as gp
import numpy as np
import pandas as pd
from csnlp import Nlp
from dmpcpwa.agents.g_admm_coordinator import GAdmmCoordinator, PwaAgent
from dmpcpwa.agents.no_control_agent import NoControlAgent
from dmpcpwa.mpc.mpc_mld import MpcMld
from dmpcpwa.mpc.mpc_switching import MpcSwitching
from dmpcrl.core.admm import g_map
from dmpcrl.mpc.mpc_admm import MpcAdmm
from gymnasium.wrappers import TimeLimit
from mpcrl.util.seeding import RngType
from mpcrl.wrappers.agents import Log
from mpcrl.wrappers.envs import MonitorEpisodes
from scipy.linalg import block_diag

from system.env import Network
from system.model import (
    get_adj,
    get_cent_system,
    get_cost_matrices,
    get_inv_set,
    get_local_coupled_systems,
    get_local_system,
    get_terminal_costs,
    get_terminal_K,
    get_warm_start,
)
from utils.plotting import plot_system

PLOT = True
SAVE = False

CONTROL = True  # if False the open loop uncontrolled system runs
MULTIPLE_ICS = False  # if True the simulation is run for all ICs in the ICs.csv file
SAVE_WARM_START = (
    False  # if True the warm_start calculated in the first step is saved in u.pkl file
)
CENT_WARM_START = False  # if True a centralized MLD controller is used to find an initial feasible solution in the first time step
MODEL_WARM_START = False  # if True an initial feasible solution, saved in the model.py file is used in the first time step
USE_TERM_CONTROLLER = False  # if True local systems switch to their terminal controllers when locally in terminal set, even if other sub-systems are out of the terminal set
N = 5
n = 3
nx_l = 2
nu_l = 1
Q_x_l, Q_u_l = get_cost_matrices()
A, b = get_inv_set()
P = get_terminal_costs()

ep_len = 30

Adj = get_adj()
G_map = g_map(Adj)

system = get_local_system()
systems = get_local_coupled_systems()


class LocalMpc(MpcSwitching):
    rho = 5  # ADMM penalty parameter
    horizon = N

    def __init__(self, num_neighbours, my_index, P) -> None:
        nlp = Nlp[cs.SX]()
        super().__init__(nlp, N)
        self.nx_l = nx_l
        self.nu_l = nu_l

        x, x_c = self.augmented_state(num_neighbours, my_index, size=nx_l)

        u, _ = self.action(
            "u",
            nu_l,
        )

        x_c_list = (
            []
        )  # store the bits of x that are couplings in a list for ease of access
        for i in range(num_neighbours):
            x_c_list.append(x_c[nx_l * i : nx_l * (i + 1), :])

        r = system["T"][0].shape[0]  # number of conditions when constraining a region
        self.set_dynamics(nx_l, nu_l, r, x, u, x_c_list)

        # normal constraints
        for k in range(N + 1):
            self.constraint(f"state_{k}", system["D"] @ x[:, [k]], "<=", system["E"])
            for x_n in x_c_list:
                self.constraint(
                    f"state_{x_n}_{k}", system["D"] @ x_n[:, [k]], "<=", system["E"]
                )

        for k in range(N):
            self.constraint(f"control_{k}", system["F"] @ u[:, [k]], "<=", system["G"])

        # terminal constraint
        self.constraint(f"terminal", A @ x[:, [N]], "<=", b)

        self.set_local_cost(
            sum(
                x[:, k].T @ Q_x_l @ x[:, k] + u[:, k].T @ Q_u_l @ u[:, k]
                for k in range(N)
            )
            + x[:, N].T @ P @ x[:, N]
        )

        # parameters to add constraints enforcing terminal switching controllers
        self.K = [self.parameter(f"K_{k}", (nu_l, nx_l)) for k in range(N)]
        for k in range(N):
            self.fixed_pars_init[f"K_{k}"] = np.zeros((nu_l, nx_l))
        self.u = u
        self.x = x

        solver = "ipopt"
        if solver == "ipopt":
            opts = {
                "expand": True,
                "show_eval_warnings": True,
                "warn_initial_bounds": True,
                "print_time": False,
                "record_time": True,
                "bound_consistency": True,
                "calc_lam_x": True,
                "calc_lam_p": False,
                # "jit": True,
                # "jit_cleanup": True,
                "ipopt": {
                    # "linear_solver": "ma97",
                    # "linear_system_scaling": "mc19",
                    # "nlp_scaling_method": "equilibration-based",
                    "max_iter": 500,
                    "sb": "yes",
                    "print_level": 0,
                },
            }
        elif solver == "qrqp":
            opts = {
                "expand": True,
                "print_time": False,
                "record_time": True,
                "error_on_fail": False,
                "print_info": False,
                "print_iter": False,
                "print_header": False,
                "max_iter": 2000,
            }
        elif solver == "qpoases":
            opts = {
                "print_time": False,
                "record_time": True,
                "error_on_fail": True,
                "printLevel": "none",
            }
        else:
            raise RuntimeError("No solver type defined.")

        self.init_solver(opts, solver=solver)


# centralized controller used to get warm_start
class Cent_MPC(MpcMld):
    Q_x = block_diag(*[Q_x_l] * n)
    Q_u = block_diag(*[Q_u_l] * n)

    def __init__(self, system: dict, N: int) -> None:
        super().__init__(system, N, verbose=True)

        obj = 0

        # leave objective as zero to only search for a feasible solution

        # for k in range(N):
        #    obj += (
        #        self.x[:, k] @ self.Q_x @ self.x[:, [k]]
        #        + self.u[:, k] @ self.Q_u @ self.u[:, [k]]
        #    )
        # obj += self.x[:, N] @ self.Q_x @ self.x[:, [N]]

        self.mpc_model.addConstrs(
            A @ self.x[i * nx_l : (i + 1) * nx_l, [N]] <= b for i in range(n)
        )
        self.mpc_model.setObjective(obj, gp.GRB.MINIMIZE)

        # so that Gurobi only searches for a feasbile sol
        self.mpc_model.setParam("SolutionLimit", 1)


class StableGAdmmCoordinator(GAdmmCoordinator):
    K = get_terminal_K()
    term_flags = [
        False for i in range(n)
    ]  # flags become True when sub-systems enter terminal sets
    prev_x = [
        np.zeros((nx_l, N)) for i in range(n)
    ]  # previous sol's state trajectory for constructing warm_starts
    first_step = True
    solve_times = []  # store solution times accounting for parallel computation

    def __init__(
        self,
        local_mpcs: list[MpcAdmm],
        local_fixed_parameters: list[dict],
        systems: list[dict],
        G: list[list[int]],
        Adj,
        rho: float,
        cent_mpc,
        admm_iters=50,
        switching_iters=float("inf"),
        agent_class=PwaAgent,
        debug_plot = False
    ) -> None:
        super().__init__(
            local_mpcs,
            local_fixed_parameters,
            systems,
            G,
            Adj,
            rho,
            admm_iters=admm_iters,
            switching_iters=switching_iters,
            agent_class=agent_class,
            debug_plot=debug_plot
        )
        self.cent_mpc = cent_mpc
        for i in range(n):
            self.agents[i].set_K(self.K)

    def reset(self, seed: RngType = None) -> None:
        self.term_flags = [False for i in range(n)]
        self.prev_x = [np.zeros((nx_l, N)) for i in range(n)]
        self.first_step = True
        self.solve_times = []
        return super().reset(seed)

    def on_timestep_end(self, env, episode: int, timestep: int) -> None:
        # add latest solution time at end of time step
        if all(self.term_flags):
            self.solve_times.append(0.0)
        else:
            self.solve_times.append(self.prev_sol_time)
        return super().on_timestep_end(env, episode, timestep)

    def g_admm_control(self, state, warm_start=None):
        x = [
            state[self.nx_l * i : self.nx_l * (i + 1), :] for i in range(self.n)
        ]  # break global state into local pieces

        # set terminal flags
        for i in range(n):
            if all(A @ x[i] <= b):
                if not self.term_flags[i]:
                    self.term_flags[i] = True

                    if USE_TERM_CONTROLLER:
                        # set linear control constraints
                        for k in range(N):
                            self.agents[i].V.constraint(
                                f"term_cntrl_{k}",
                                self.agents[i].V.u[:, [k]],
                                "==",
                                self.agents[i].V.K[k] @ self.agents[i].V.x[:, [k]],
                            )

        # if all sub-systems in terminal set, use linear switching controllers
        if all(self.term_flags):
            action_list = []
            for i in range(n):
                regions = self.agents[i].identify_regions(x[i])
                action_list.append(self.K[regions[0]] @ x[i])
            return cs.DM(action_list), None, None, None

        # otherwise, generate warm_start for switching ADMM procedure

        # warm start for first time_step
        if self.first_step or self.prev_sol is None:
            if CENT_WARM_START:
                u, info = self.cent_mpc.solve_mpc(state)
                warm_start = [info["u"][[i], :] for i in range(n)]
                if SAVE_WARM_START:
                    with open("examples/small_stable/u.pkl", "wb") as file:
                        pickle.dump(warm_start, file)
            elif MODEL_WARM_START:
                warm_start = get_warm_start(N)
            else:
                warm_start = None

            self.first_step = False

        # generate warm_start from either shifted solution or terminal controller
        else:
            prev_final_x = [self.prev_traj[i][:, [-1]] for i in range(n)]
            prev_final_u = [self.prev_sol[i][:, [-1]] for i in range(n)]
            warm_start = []
            for i in range(n):
                regions = self.agents[i].identify_regions(prev_final_x[i])
                warm_start.append(
                    np.hstack(
                        (self.prev_sol[i][:, 1:], self.K[regions[0]] @ prev_final_x[i])
                    )
                )

            if USE_TERM_CONTROLLER:
                # if any of the systems are now using terminal controllers, we generate a warm start that is u = Kx.
                # to do this we have to rollout the states using the other systems' warm_starts to ensure consistency
                if any(self.term_flags):
                    x_temp = [np.zeros((self.nx_l, self.N)) for i in range(self.n)]
                    for i in range(self.n):
                        x_temp[i][:, [0]] = state[
                            self.nx_l * i : self.nx_l * (i + 1), [0]
                        ]  # add the first known states to the temp
                        if self.term_flags[i]:
                            regions = self.agents[i].identify_regions(x_temp[i][:, [0]])
                            warm_start[i][:, [0]] = (
                                self.K[regions[0]] @ x_temp[i][:, [0]]
                            )

                    for k in range(1, self.N):
                        for i in range(self.n):
                            xc_temp = []
                            for j in range(self.n):
                                if self.Adj[i, j] == 1:
                                    xc_temp.append(x_temp[j][:, [k - 1]])
                            x_temp[i][:, [k]] = self.agents[i].next_state(
                                x_temp[i][:, [k - 1]],
                                warm_start[i][:, [k - 1]],
                                xc_temp,
                            )
                            if self.term_flags[i]:
                                regions = self.agents[i].identify_regions(
                                    x_temp[i][:, [k]]
                                )
                                warm_start[i][:, [k]] = (
                                    self.K[regions[0]] @ x_temp[i][:, [k]]
                                )

        # run switching admm procedure with warm_start
        return super().g_admm_control(state, warm_start)


# override PWA agent class to add in a terminal controller that switches with regions and uses final region in sequence
class PwaAgentTerminal(PwaAgent):

    def __init__(
        self,
        mpc: Mpc,
        fixed_parameters: dict,
        pwa_system: dict,
        use_terminal_sequence: bool = False,
    ) -> None:
        super().__init__(mpc, fixed_parameters, pwa_system, use_terminal_sequence=True)

    K: list[np.ndarray] = []  # terminal controllers

    def set_K(self, K):
        self.K = K

    def set_sequence(self, s: list[int]):
        if len(self.K) == 0:
            raise RuntimeError("Linear controller must be set before sequence is set.")
        for i in range(len(s)):
            self.fixed_parameters[f"K_{i}"] = self.K[s[i]]
        return super().set_sequence(s)


# environment
env = MonitorEpisodes(
    TimeLimit(
        Network(),
        max_episode_steps=ep_len,
    )
)
# cent mpc for initial feasible guess
cent_mpc = Cent_MPC(get_cent_system(), N)

# distributed mpcs and params
local_mpcs: list[LocalMpc] = []
local_fixed_dist_parameters: list[dict] = []
for i in range(n):
    local_mpcs.append(
        LocalMpc(num_neighbours=len(G_map[i]) - 1, my_index=G_map[i].index(i), P=P[i])
    )
    local_fixed_dist_parameters.append(local_mpcs[i].fixed_pars_init)

# coordinator
iters = 75
if CONTROL:
    agent = Log(
        StableGAdmmCoordinator(
            local_mpcs,
            local_fixed_dist_parameters,
            systems,
            G_map,
            Adj,
            local_mpcs[0].rho,
            cent_mpc,
            admm_iters=iters,
            switching_iters=50,
            agent_class=PwaAgentTerminal,
            debug_plot=True
        ),
        level=logging.DEBUG,
        log_frequencies={"on_timestep_end": 1},
    )
else:
    agent = NoControlAgent(3, local_mpcs[0])

if MULTIPLE_ICS:
    df = pd.read_csv("ICs.csv", header=None)
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
    plot_system(X, U)

    id = "unstab"
    if SAVE:
        with open(
            f"gadmm_{id}" + ".pkl",
            "wb",
        ) as file:
            pickle.dump(X, file)
            pickle.dump(U, file)
            pickle.dump(R, file)
            pickle.dump(agent.solve_times, file)
