import pickle

import matplotlib.pyplot as plt
import numpy as np

with open("random_systems/costs.pkl", "rb") as f:
    data = pickle.load(f)
    costs = data["costs"]
    states = data["states"]

diffs = [
    100 * (dist_cost.item() - cent_cost.item()) / cent_cost.item()
    for cent_cost, dist_cost in costs
]
# diffs = np.delete(diffs, [447, 126])
print(f"Mean difference: {np.mean(diffs)}")
print(f"Median difference: {np.median(diffs)}")
print(f"Max difference: {np.max(diffs)}")
print(f"Min difference: {np.min(diffs)}")
print(f"Std difference: {np.std(diffs)}")

plt.plot(diffs)
plt.show()
