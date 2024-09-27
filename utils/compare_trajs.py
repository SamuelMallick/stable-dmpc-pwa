import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from dmpcpwa.utils.tikz import save2tikz

sys.path.append(os.getcwd())
from system.model import Model

model = Model()

plt.rc("text", usetex=True)
plt.rc("font", size=14)
plt.style.use("bmh")

data = loadmat("data/data_cplex.mat")
data = data["data"][0]

num_ICs = 100

data_swa = {}
data_rc = {}
data_cent = {}

# get data from matlab files for rc approach
names = ["cplex", "mosek", "gurobi"]
for name in names:
    data = loadmat(f"data/data_{name}.mat")
    data = data["data"][0]
    data_rc[name] = {"X": [], "U": [], "T": [], "J": []}
    for i in range(num_ICs):
        data_rc[name]["X"].append(data[i]["X"][0, 0])
        data_rc[name]["U"].append(data[i]["U"][0, 0])
        data_rc[name]["T"].append(np.max(data[i]["T"][0, 0], axis=0))
        data_rc[name]["J"].append(data[i]["J"][0, 0].item())

# get data from pickles for swa approach
names = ["gurobi", "osqp", "qpoases", "qrqp"]
for name in names:
    data_swa[name] = {"X": [], "U": [], "T": [], "J": []}
    for i in range(num_ICs):
        with open(
            f"data/{name}/gadmm_{i+1}.pkl",
            "rb",
        ) as file:
            data_swa[name]["X"].append(pickle.load(file))
            data_swa[name]["U"].append(pickle.load(file))
            data_swa[name]["J"].append(
                pickle.load(file).sum()
            )  # sum timestep costs to get total cost
            data_swa[name]["T"].append(np.array(pickle.load(file)))

# get data from pickles for cent approach
names = ["gurobi"]
for name in names:
    data_cent[name] = {"X": [], "U": [], "T": [], "J": []}
    for i in range(num_ICs):
        with open(
            f"data/cent/cent_mld_{i+1}.pkl",
            "rb",
        ) as file:
            data_cent[name]["X"].append(pickle.load(file))
            data_cent[name]["U"].append(pickle.load(file))
            data_cent[name]["J"].append(
                pickle.load(file).sum()
            )  # sum timestep costs to get total cost
            data_cent[name]["T"].append(np.array(pickle.load(file)))

_, axs = plt.subplots(1, 1, constrained_layout=True, sharex=True)
lw = 0.5
name_swa = "qrqp"
name_rc = "gurobi"
name_cent = "gurobi"
axs.bar(
    [i for i in range(1, num_ICs + 1)],
    [
        data_rc[name_rc]["J"][i] / data_cent[name_cent]["J"][i] - 1
        for i in range(num_ICs)
    ],
    bottom=1,
    color="C1",
)
axs.bar(
    [i for i in range(1, num_ICs + 1)],
    [
        data_swa[name_swa]["J"][i] / data_cent[name_cent]["J"][i] - 1
        for i in range(num_ICs)
    ],
    bottom=1,
    color="C2",
)
axs.set_ylabel(r"$J/J_{MLD}$")
axs.legend(["RC-Sets", "Sw-ADMM"], loc="upper left", ncol=2)
# save2tikz(plt.gcf())

_, axs = plt.subplots(1, 1, constrained_layout=True, sharex=True)
axs.boxplot([np.array([
        data_swa[name_rc]["J"][i] / data_cent[name_cent]["J"][i] - 1
        for i in range(num_ICs)
    ]),
    np.array([data_rc[name_rc]["J"][i] / data_cent[name_cent]["J"][i] - 1
        for i in range(num_ICs)])]
    )


_, axs = plt.subplots(1, 2, constrained_layout=True, sharey=True)
time_data_sw_admm = [
    np.concatenate(data_swa[name_swa]["T"])[
        np.concatenate(data_swa[name_swa]["T"]) != 0
    ]
    for name_swa in data_swa.keys()
] 
axs[0].boxplot(time_data_sw_admm, labels=['gurobi', 'osqp', 'qpoases', 'qrqp'])
time_data_rc_sets = [
    np.concatenate(data_rc[name_rc]["T"])[np.concatenate(data_rc[name_rc]["T"]) != 0]
    for name_rc in data_rc.keys()
]
axs[1].boxplot(time_data_rc_sets, labels=['cplex', 'mosek', 'gurobi'])
#+ [
#     np.concatenate(data_cent[name_cent]["T"])[
#         np.concatenate(data_cent[name_cent]["T"]) != 0
#     ]
#     for name_cent in data_cent.keys()
# ]

# axs.text(2, axs.get_ylim()[1] + 0.5, 'Main Label', ha='center', fontsize=12)
axs[0].set_yscale("log")
axs[0].set_title("Sw-ADMM")
axs[1].set_title("RC-Sets")
axs[0].set_ylabel(r"Time (s)")
axs[1].set_ylabel(r"Time (s)")
save2tikz(plt.gcf())

# axs[1].set_ylabel(r"Time (s)")
# axs[1].set_xlabel("Different initial states.")
# # axs[1].set_ylim([0.001, 200])
# axs[1].legend(["RC-Sets", "Sw-ADMM", "Cent-MLD"], loc="upper center", ncol=2)
# save2tikz(plt.gcf())

# _, axs = plt.subplots(1, 1, constrained_layout=True, sharex=True)
# IC = 0
# X = X_our[IC]
# for i in range(3):
#     axs.plot(X[0, 2 * i], X[0, 2 * i + 1], "o", color=f"C{i}", label="_nolegend_")
#     axs.plot(
#         X[:, 2 * i],
#         X[:, 2 * i + 1],
#         linestyle="-",
#         linewidth=1,
#         color=f"C{i}",
#         label="_nolegend_",
#     )
#     axs.plot(X[-1, 2 * i], X[-1, 2 * i + 1], "x", color=f"C{i}", label="_nolegend_")
# X = X_opt[IC]
# for i in range(3):
#     axs.plot(X[0, 2 * i], X[0, 2 * i + 1], "o", color=f"C{i}", label="_nolegend_")
#     axs.plot(
#         X[:, 2 * i],
#         X[:, 2 * i + 1],
#         linestyle="-.",
#         linewidth=1,
#         color=f"C{i}",
#         label="_nolegend_",
#     )
#     axs.plot(X[-1, 2 * i], X[-1, 2 * i + 1], "x", color=f"C{i}", label="_nolegend_")
# X = X_other[IC]
# for i in range(3):
#     axs.plot(X[2 * i, 0], X[2 * i + 1, 0], "o", color=f"C{i}", label="_nolegend_")
#     axs.plot(
#         X[2 * i, :],
#         X[2 * i + 1, :],
#         linestyle="--",
#         linewidth=1,
#         color=f"C{i}",
#         label="_nolegend_",
#     )
#     axs.plot(X[2 * i, -1], X[2 * i + 1, -1], "x", color=f"C{i}", label="_nolegend_")
# axs.set_xlim([-20, 20])
# axs.set_ylim([-20, 20])

# # shade PWA regions
# x = np.linspace(0, 20, 100)
# axs.fill_between(x, x, -x, color="gray", alpha=0.3, label="_nolegend_")
# x = np.linspace(-20, 0, 100)
# axs.fill_between(x, -x, x, color="gray", alpha=0.3, label="_nolegend_")

# # terminal set
# X0 = model.get_inv_set_vertices()
# p = Polygon(X0, facecolor="r", alpha=0.3, label="_nolegend_")
# axs.add_patch(p)

# # make fake legend
# axs.plot(-100, 100, "-", color="k")
# axs.plot(-100, 100, "--", color="k")
# axs.plot(-100, 100, "-.", color="k")
# axs.legend(["Sw-ADMM", "RC-Sets", "Cent-MLD"])

# # label things
# plt.text(18, 0, r"$P_1$", fontsize=20, color="black")
# plt.text(-2, -19, r"$P_2$", fontsize=20, color="black")
# plt.text(-19, 0, r"$P_3$", fontsize=20, color="black")
# plt.text(-2, 18, r"$P_4$", fontsize=20, color="black")
# plt.text(-3, 2, r"$\mathcal{X}_{T}$", fontsize=20, color="black")
# # save2tikz(plt.gcf())

plt.show()
