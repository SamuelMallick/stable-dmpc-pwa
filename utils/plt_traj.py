import pickle

import matplotlib.pyplot as plt
import numpy as np
from dmpcpwa.utils.tikz import save2tikz
from matplotlib.patches import Polygon

from system.model import get_inv_set_vertices

plt.rc("text", usetex=True)
plt.rc("font", size=14)
plt.style.use("bmh")

with open(
    f"data/gadmm_unstab.pkl",
    "rb",
) as file:
    X = pickle.load(file)
    U = pickle.load(file)
    R = pickle.load(file)
    solve_times = pickle.load(file)
    R = R.squeeze()

_, axs = plt.subplots(1, 1, constrained_layout=True, sharex=True)
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
axs.set_xlim([-20, 20])
axs.set_ylim([-20, 20])

# shade PWA regions
x = np.linspace(0, 20, 100)
axs.fill_between(x, x, -x, color="gray", alpha=0.3, label="_nolegend_")
x = np.linspace(-20, 0, 100)
axs.fill_between(x, -x, x, color="gray", alpha=0.3, label="_nolegend_")

# terminal set
X0 = get_inv_set_vertices()
p = Polygon(X0, facecolor="r", alpha=0.3, label="_nolegend_")
axs.add_patch(p)

# make fake legend
axs.plot(-100, 100, "-", color="k")
axs.legend(["Sw-ADMM"])

# label things
plt.text(18, 0, r"$P_1$", fontsize=20, color="black")
plt.text(-2, -19, r"$P_2$", fontsize=20, color="black")
plt.text(-19, 0, r"$P_3$", fontsize=20, color="black")
plt.text(-2, 18, r"$P_4$", fontsize=20, color="black")
plt.text(-3, 2, r"$\mathcal{X}_{T}$", fontsize=20, color="black")
save2tikz(plt.gcf())

plt.show()
