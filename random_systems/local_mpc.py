from typing import Literal

import casadi as cs
import numpy as np
from csnlp import Nlp
from dmpcpwa.mpc.mpc_switching import MpcSwitching


class LocalMpc(MpcSwitching):
    """A local MPC scheme within the switching ADMM framework."""

    def __init__(
        self,
        system: dict,
        N: int,
        num_neighbours: int,
        my_index: int,
        rho: float = 0.5,
        solver: Literal["ipopt", "qrqp", "qpoases", "gurobi"] = "ipopt",
    ) -> None:
        """Initialize the local MPC scheme."""
        nlp = Nlp[cs.SX]()
        super().__init__(nlp, N)
        nx_l = system["A"][0].shape[1]  # state dimension
        nu_l = system["B"][0].shape[1]  # control dimension
        Q_x = np.eye(2)
        Q_u = np.eye(1)

        # the following four class variables are required in the functions called from super class
        self.horizon = N
        self.rho = rho
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

        self.set_local_cost(
            sum(x[:, k].T @ Q_x @ x[:, k] + u[:, k].T @ Q_u @ u[:, k] for k in range(N))
            + x[:, N].T @ Q_x @ x[:, N]
        )

        self.u = u
        self.x = x

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
                "jit": True,
            }
        elif solver == "osqp":
            opts = {
                "print_time": False,
                "record_time": True,
                "error_on_fail": True,
                "printLevel": "none",
            }
        elif solver == "gurobi":
            opts = {
                "print_time": False,
                "record_time": True,
                "error_on_fail": True,
                "gurobi": {
                    "OutputFlag": 0,
                    "LogToConsole": 0,
                    # "BarConvTol": 1e-3,
                },
            }
        else:
            raise RuntimeError("No solver type defined.")

        self.init_solver(opts, solver=solver)
