import pickle

import casadi as cs
import numpy as np
from dmpcpwa.agents.g_admm_coordinator import GAdmmCoordinator, PwaAgent
from dmpcpwa.mpc.mpc_mld import MpcMld
from dmpcrl.mpc.mpc_admm import MpcAdmm
from mpcrl.util.seeding import RngType

from system.model import Model


class StableSwitchingAdmmCoordinator(GAdmmCoordinator):
    """A switching ADMM coordinator with additional mechanisms for terminal controllers."""

    def __init__(
        self,
        local_mpcs: list[MpcAdmm],
        local_fixed_parameters: list[dict],
        model: Model,
        G: list[list[int]],
        Adj,
        rho: float,
        N: int,
        X_t: tuple[np.ndarray, np.ndarray],
        cent_mpc: MpcMld | None = None,
        admm_iters=50,
        switching_iters=float("inf"),
        agent_class=PwaAgent,
        centralized_warm_start: bool = False,
        model_warm_start: bool = False,
        use_term_controller: bool = False,
        debug_plot=False,
    ) -> None:
        # TODO docstring
        self.centralized_warm_start = centralized_warm_start
        self.model_warm_start = model_warm_start
        self.use_term_controller = use_term_controller

        self.model = model
        systems = model.get_local_coupled_systems()
        self.N = N
        self.n = len(systems)
        self.nx_l = systems[0]["A"][0].shape[1]  # state dimension of local systems
        self.nu_l = systems[0]["B"][0].shape[1]  # control dimension of local systems
        self.A, self.b = X_t[0], X_t[1]  # terminal constraint Ax <= b

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
            debug_plot=debug_plot,
        )

        self.term_flags = [
            False for i in range(self.n)
        ]  # flags become True when sub-systems enter terminal sets
        self.prev_x = [
            np.zeros((self.nx_l, N)) for i in range(self.n)
        ]  # previous sol's state trajectory for constructing warm_starts
        self.first_step = True  # flag for first time step
        self.solve_times: list[float] = (
            []
        )  # store solution times accounting for parallel computation

        # terminal controllers
        self.K = model.terminal_controllers
        for i in range(self.n):
            self.agents[i].set_K(self.K)

    def reset(self, seed: RngType = None) -> None:
        """Reset the coordinator for new simulations."""
        self.term_flags = [False for i in range(self.n)]
        self.prev_x = [np.zeros((self.nx_l, self.N)) for i in range(self.n)]
        self.first_step = True
        self.solve_times = []
        return super().reset(seed)

    def on_timestep_end(self, env, episode: int, timestep: int) -> None:
        # add latest solution time at end of time step
        if all(self.term_flags):
            self.solve_times.append(0.0)  # zero time incurred by terminal controllers
        else:
            self.solve_times.append(self.prev_sol_time)
        return super().on_timestep_end(env, episode, timestep)

    def g_admm_control(self, state: np.ndarray, warm_start=None):
        """Run the switching ADMM procedure to generate control inputs"""
        x = np.split(state, self.n)  # break global state into local pieces

        # set terminal flags if agents are in terminal sets
        for i in range(self.n):
            if all(self.A @ x[i] <= self.b):
                if not self.term_flags[i]:
                    self.term_flags[i] = True

                    if self.use_term_controller:
                        # set linear control constraints
                        for k in range(self.N):
                            self.agents[i].V.constraint(
                                f"term_cntrl_{k}",
                                self.agents[i].V.u[:, [k]],
                                "==",
                                self.agents[i].V.K[k] @ self.agents[i].V.x[:, [k]],
                            )

        # if all sub-systems in terminal set, use linear switching controllers
        if all(self.term_flags):
            action_list = []
            for i in range(self.n):
                regions = self.agents[i].identify_regions(x[i])
                action_list.append(self.K[regions[0]] @ x[i])
            return cs.DM(action_list), None, None, None

        # otherwise, generate warm_start for switching ADMM procedure

        # warm start for first time_step
        if self.first_step or self.prev_sol is None:
            if self.centralized_warm_start:
                raise RuntimeError("CENT_WARM_START not implemented.")
                u, info = self.cent_mpc.solve_mpc(state)
                warm_start = [info["u"][[i], :] for i in range(self.n)]
                if SAVE_WARM_START:
                    with open("examples/small_stable/u.pkl", "wb") as file:
                        pickle.dump(warm_start, file)
            elif self.model_warm_start:
                warm_start = self.model.get_warm_start(self.N)
            else:
                warm_start = None

            self.first_step = False

        # generate warm_start from either shifted solution or terminal controller
        else:
            prev_final_x = [self.prev_traj[i][:, [-1]] for i in range(self.n)]
            prev_final_u = [self.prev_sol[i][:, [-1]] for i in range(self.n)]
            warm_start = []
            for i in range(self.n):
                regions = self.agents[i].identify_regions(prev_final_x[i])
                warm_start.append(
                    np.hstack(
                        (self.prev_sol[i][:, 1:], self.K[regions[0]] @ prev_final_x[i])
                    )
                )

            if self.use_term_controller:
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
