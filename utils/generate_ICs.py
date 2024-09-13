import pickle

import numpy as np
import pandas as pd

np.random.seed(15)

num_ICs = 100
up_bound = 20
low_bound = -20

X0 = []
# Generate ICs
for i in range(100):
    with open(
        f"data/gadmm_{i+1}.pkl",
        "rb",
    ) as file:
        X = pickle.load(file)
        X0.append(X[0, :])
data = np.array(X0)
# data = np.random.uniform(low=low_bound, high=up_bound, size=(num_ICs, 6))

# Specify the CSV file path
csv_file_path = "ICs.csv"
df = pd.DataFrame(data)
df.to_csv(csv_file_path, header=False, index=False)
print(f"CSV file '{csv_file_path}' has been generated.")
