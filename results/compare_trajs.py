import pickle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from scipy.io import loadmat

plt.rc("text", usetex=True)
plt.rc("font", size=14)
plt.style.use("bmh")

data = loadmat("examples/small_stable/data/data.mat")
data = data["data"][0]

num_ICs = 100

J_other = []
T_av_other = []
T_ma_other = []
T_mi_other = []
J_ours = []
T_av_our = []
T_ma_our = []
T_mi_our = []
X_other = []
X_our = []
for i in range(num_ICs):
    X_other.append(data[i]["X"][0, 0])
    U = data[i]["U"][0, 0]
    T = data[i]["T"][0, 0]
    T = [max(T[:, t]) for t in range(T.shape[1])]
    T_av_other.append(sum([t for t in T if t != 0]) / len([t for t in T if t != 0]))
    T_ma_other.append(max([t for t in T if t != 0]))
    T_mi_other.append(min([t for t in T if t != 0]))
    J_other.append(data[i]["J"][0, 0][0])

for i in range(num_ICs):
    with open(
        f"examples/small_stable/data/gadmm_{i+1}.pkl",
        "rb",
    ) as file:
        X_our.append(pickle.load(file))
        U = pickle.load(file)
        R = pickle.load(file)
        solve_times = pickle.load(file)
        T_av_our.append(
            sum([t for t in solve_times if t != 0])
            / len([t for t in solve_times if t != 0])
        )
        T_ma_our.append(max([t for t in solve_times if t != 0]))
        T_mi_our.append(min([t for t in solve_times if t != 0]))
        R = R.squeeze()
        J_ours.append(sum(R))

_, axs = plt.subplots(2, 1, constrained_layout=True, sharex=True)
lw = 0.5
axs[0].plot(
    [i for i in range(1, num_ICs + 1)],
    [J_other[i] - J_ours[i] for i in range(num_ICs)],
    marker="*",
    linestyle="None",
    color="C0",
)
axs[0].set_ylabel(r"$\Delta J$")

axs[1].plot(
    [i for i in range(1, num_ICs + 1)],
    T_av_other,
    marker="o",
    linestyle="None",
    color="C1",
    markerfacecolor="None",
)
axs[1].plot(
    [i for i in range(1, num_ICs + 1)],
    T_ma_other,
    marker="v",
    linestyle="None",
    color="C1",
    label="_nolegend_",
    markerfacecolor="None",
)
axs[1].plot(
    [i for i in range(1, num_ICs + 1)],
    T_mi_other,
    marker="^",
    linestyle="None",
    color="C1",
    label="_nolegend_",
    markerfacecolor="None",
)
for i in range(num_ICs):
    axs[1].plot(
        [i + 1, i + 1],
        [T_ma_other[i], T_mi_other[i]],
        linewidth=lw,
        color="C1",
        label="_nolegend_",
    )

axs[1].plot(
    [i for i in range(1, num_ICs + 1)],
    T_av_our,
    marker="o",
    linestyle="None",
    color="C2",
    markerfacecolor="None",
)
axs[1].plot(
    [i for i in range(1, num_ICs + 1)],
    T_ma_our,
    marker="v",
    linestyle="None",
    color="C2",
    label="_nolegend_",
    markerfacecolor="None",
)
axs[1].plot(
    [i for i in range(1, num_ICs + 1)],
    T_mi_our,
    marker="^",
    linestyle="None",
    color="C2",
    label="_nolegend_",
    markerfacecolor="None",
)
for i in range(num_ICs):
    axs[1].plot(
        [i + 1, i + 1],
        [T_ma_our[i], T_mi_our[i]],
        linewidth=lw,
        color="C2",
        label="_nolegend_",
    )
axs[1].set_yscale("log")

axs[1].set_ylabel(r"Time (s)")
axs[1].set_xlabel("Different initial states.")
axs[1].set_ylim([0.001, 0.8])
axs[1].legend(["RC-Sets", "Sw-ADMM"], loc="upper center", ncol=2)
# save2tikz(plt.gcf())

_, axs = plt.subplots(1, 1, constrained_layout=True, sharex=True)
IC = 0
X = X_our[IC]
for i in range(3):
    axs.plot(X[0, 2 * i], X[0, 2 * i + 1], "o", color=f"C{i}", label="_nolegend_")
    axs.plot(
        X[:, 2 * i],
        X[:, 2 * i + 1],
        linestyle="-",
        linewidth=1,
        color=f"C{i}",
        label="_nolegend_",
    )
    axs.plot(X[-1, 2 * i], X[-1, 2 * i + 1], "x", color=f"C{i}", label="_nolegend_")
X = X_other[IC]
for i in range(3):
    axs.plot(X[2 * i, 0], X[2 * i + 1, 0], "o", color=f"C{i}", label="_nolegend_")
    axs.plot(
        X[2 * i, :],
        X[2 * i + 1, :],
        linestyle="--",
        linewidth=1,
        color=f"C{i}",
        label="_nolegend_",
    )
    axs.plot(X[2 * i, -1], X[2 * i + 1, -1], "x", color=f"C{i}", label="_nolegend_")
axs.set_xlim([-20, 20])
axs.set_ylim([-20, 20])

# shade PWA regions
x = np.linspace(0, 20, 100)
axs.fill_between(x, x, -x, color="gray", alpha=0.3, label="_nolegend_")
x = np.linspace(-20, 0, 100)
axs.fill_between(x, -x, x, color="gray", alpha=0.3, label="_nolegend_")

# terminal set
X0 = np.array(
    [
        [-5.7352, 0.3567],
        [3.4420, -5.5259],
        [3.4420, -5.5259],
        [3.7696, -4.7875],
        [5.7352, -0.3567],
        [-3.4420, 5.5259],
        [-3.5319, 5.3232],
        [-4.1510, 3.9276],
        [-5.7352, 0.3567],
    ]
)
p = Polygon(X0, facecolor="r", alpha=0.3, label="_nolegend_")
axs.add_patch(p)

# make fake legend
axs.plot(-100, 100, "-", color="k")
axs.plot(-100, 100, "--", color="k")
axs.legend(["Sw-ADMM", "RC-Sets"])

# label things
plt.text(18, 0, r"$P_1$", fontsize=20, color="black")
plt.text(-2, -19, r"$P_2$", fontsize=20, color="black")
plt.text(-19, 0, r"$P_3$", fontsize=20, color="black")
plt.text(-2, 18, r"$P_4$", fontsize=20, color="black")
plt.text(-3, 2, r"$\mathcal{X}_{T}$", fontsize=20, color="black")
# save2tikz(plt.gcf())

plt.show()
