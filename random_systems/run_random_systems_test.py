from system import RandomNetwork
from local_mpc import LocalMpc
from cent_mpc import CentralizedMPC
from dmpcpwa.agents.g_admm_coordinator import GAdmmCoordinator
from dmpcpwa.utils.pwa_models import evalulate_cost, evalulate_cost_distributed
import numpy as np
from dmpcrl.core.admm import g_map
import pickle

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
agent = GAdmmCoordinator(
    local_mpcs=local_mpcs,
    local_fixed_parameters=local_fixed_dist_parameters,
    systems=systems,
    G=G_map,
    Adj=sys.adj,
    rho=rho,
    admm_iters=2000,
    switching_iters=100,
    residual_tol=0.0001,
    debug_plot=False,
)

costs = []
for t in range(1000):
    for rho in [0.1, 0.25, 0.5, 2, 5]:
        agent.reset()
        try:
            x = np_random.uniform(-10, 10, size=(4, 1))
            # x = np.array([[-0.1456704 ],
            #     [-5.12893994],
            #     [ 0.52697479],
            #     [-1.51106445]])
            # x = np.array([[4.9644855], [-8.26637354], [-1.48287519], [-2.06496226]])
            _, info = cent_mpc.solve_mpc(x)
            cent_u = info["u"]
            seq = list(info["delta"].argmax(axis=0))
            # cent_cost = evalulate_cost(x, cent_u, cent_sys, np.eye(4), np.eye(2))
            cent_cost = evalulate_cost(x, cent_u, cent_sys, np.eye(4), np.eye(2), seq)
            cent_cost_pred = info["cost"]
            ud, _, _, info = agent.g_admm_control(x)
            sol_list = info["sol_list"]
            seqs = info["seqs"]
            iters = info["iter"]
            print(f"Iterations {iters} for dist")
            dist_u = np.vstack([sol.vals["u"] for sol in sol_list])
            # dist_cost = evalulate_cost_distributed(x, dist_u, systems, sys.adj, np.eye(2), np.eye(1))
            dist_cost = evalulate_cost_distributed(
                x, dist_u, systems, sys.adj, np.eye(2), np.eye(1), seqs
            )
            dist_cost_pred = sum([sol.f for sol in sol_list])

            costs.append((cent_cost, dist_cost))
            print(f"Cent cost: {cent_cost}, Dist cost: {dist_cost}")
            sys.reset()
            break
        except:
            print("Error")
            continue
    sys.reset()

with open("costs_hard.pkl", "wb") as f:
    pickle.dump(costs, f)
