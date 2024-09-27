import pickle

import numpy as np
from cent_mpc import CentralizedMPC
from dmpcpwa.agents.g_admm_coordinator import GAdmmCoordinator
from dmpcpwa.utils.pwa_models import evalulate_cost, evalulate_cost_distributed, cent_switching_sequence_from_distributed
from dmpcrl.core.admm import g_map
from local_mpc import LocalMpc
import matplotlib.pyplot as plt

from system import RandomNetwork

np_random = np.random.default_rng(0)

N = 5
rho = 0.25

sys = RandomNetwork(np_random)

# cent setup
cent_sys = sys.get_cent_system()
cent_mpc = CentralizedMPC(cent_sys, 5)

# dist setup
systems = sys.get_local_coupled_systems()
G_map = g_map(sys.adj)
local_mpcs: list[LocalMpc] = []
local_fixed_dist_parameters: list[dict] = []
for i in range(2):
    local_mpcs.append(
        LocalMpc(
            system=systems[i],
            N=N,
            num_neighbours=len(G_map[i]) - 1,
            my_index=G_map[i].index(i),
            rho=0.5,
            solver="ipopt",
        )
    )
    local_fixed_dist_parameters.append(local_mpcs[i].fixed_pars_init)
admm_iters = 1000
res_tol = 0.001
agent = GAdmmCoordinator(
    local_mpcs=local_mpcs,
    local_fixed_parameters=local_fixed_dist_parameters,
    systems=systems,
    G=G_map,
    Adj=sys.adj,
    rho=rho,
    admm_iters=admm_iters,
    switching_iters=100,
    residual_tol=res_tol,
    debug_plot=False,
)

costs = []
states = []
bad_found = False
count = 0
while count < 1000:
    x = np_random.uniform(-10, 10, size=(4, 1))
    sys.reset()
    try:
        for rho in [0.5]:
            agent.reset()
            _, info = cent_mpc.solve_mpc(x)
            cent_u = info["u"]
            seq = list(info["delta"].argmax(axis=0))
            # cent_cost = evalulate_cost(x, cent_u, cent_sys, np.eye(4), np.eye(2))
            cent_cost, cent_x_traj = evalulate_cost(x, cent_u, cent_sys, np.eye(4), np.eye(2), seq)
            cent_cost_pred = info["cost"]
            ud, _, _, info = agent.g_admm_control(x)
            sol_list = info["sol_list"]
            seqs = info["seqs"]
            iters = info["iter"]
            print(f"Iterations {iters} for dist")

            if info["res"] > res_tol:
                print("Residual too high, didn't converge due to poorly tuned rho, or violation of assumption 1")
            else:
                dist_u = np.vstack([sol.vals["u"] for sol in sol_list])
                # dist_cost = evalulate_cost_distributed(x, dist_u, systems, sys.adj, np.eye(2), np.eye(1))
                dist_cost, dist_x_traj = evalulate_cost_distributed(
                    x, dist_u, systems, sys.adj, np.eye(2), np.eye(1), seqs
                )
                dist_cost_pred = sum([sol.f for sol in sol_list])

                costs.append((cent_cost, dist_cost))
                count += 1
                states.append(x)
                diff = 100 * (dist_cost - cent_cost) / cent_cost
                print(f"Cent cost: {cent_cost}, Dist cost: {dist_cost}")
                print(f"Percent diff: {diff}")
                if diff > np.inf:
                    bad_found = True
                break

        if bad_found:
            break
    except:
        print('infeas initial condition error')

with open("costs.pkl", "wb") as f:
    pickle.dump({"costs": costs, "states": states}, f)
