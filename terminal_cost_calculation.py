import itertools

import cvxpy as cp
import numpy as np
from scipy.linalg import block_diag

from model import (
    get_A_c,
    get_adj,
    get_cost_matrices,
    get_local_coupled_systems,
    get_terminal_K,
)

systems = get_local_coupled_systems()
Q_l, R_l = get_cost_matrices()
K_term = get_terminal_K()
Adj = get_adj()
A_c = get_A_c()
M = len(systems)  # number of systems M
n = systems[0]["A"][0].shape[0]  # state dimension of systems
m = systems[0]["B"][0].shape[1]  # input dimension of systems
L = len(systems[0]["A"])  # number of PWA regions

# cvxpy variables for terminal cost matrix of each system
P_l = []
P_l = [
    [cp.Variable((n, n), PSD=True) if j == i else np.zeros((n, n)) for j in range(M)]
    for i in range(M)
]
P = cp.bmat(P_l)

# Q and R matrices irrespective of PWA regions
Q = block_diag(*[Q_l] * M)
R = block_diag(*[R_l] * M)

# cycle through all possible combinations of local PWA regions
elements = [i for i in range(L)]
combs = [p for p in itertools.product(elements, repeat=M)]
constraints = []  # LMI conditions in a list

for comb in combs:
    # build K matrix
    K_l = [
        [K_term[comb[i]] if j == i else np.zeros((m, n)) for j in range(M)]
        for i in range(M)
    ]
    K = np.bmat(K_l)

    # build global closed loop A matrix
    A_g_l = [
        [
            systems[i]["A"][comb[i]] + systems[i]["B"][comb[i]] @ K_term[comb[i]]
            if j == i
            else A_c
            if Adj[i, j] == 1
            else np.zeros((n, n))
            for j in range(M)
        ]
        for i in range(M)
    ]
    A_g = np.bmat(A_g_l)

    constraints.append(P - A_g.T @ P @ A_g - Q - K.T @ R @ K >> 0)

prob = cp.Problem(cp.Minimize(0), constraints)
prob.solve()
if prob.status != cp.OPTIMAL:
    raise RuntimeError("Solve failed for lyapunov function.")
P = P.value
for i in range(M):
    P_l[i] = P[n * i : n * (i + 1), n * i : n * (i + 1)]

print(f"Terminal cost matrix is {P}")
