import pickle
import numpy as np
import matplotlib.pyplot as plt

with open("random_systems/costs_hard.pkl", "rb") as f:
    costs = pickle.load(f)

diffs = [
    100 * (dist_cost.item() - cent_cost.item()) / cent_cost.item()
    for cent_cost, dist_cost in costs
]
print(f"Mean difference: {np.mean(diffs)}")
print(f"Max difference: {np.max(diffs)}")
print(f"Min difference: {np.min(diffs)}")
print(f"Std difference: {np.std(diffs)}")

plt.plot(diffs)
plt.show()
