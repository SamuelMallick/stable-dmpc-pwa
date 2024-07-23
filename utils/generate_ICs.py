import numpy as np
import pandas as pd

np.random.seed(15)

num_ICs = 100
up_bound = 20
low_bound = -20

# Generate ICs
data = np.random.uniform(low=low_bound, high=up_bound, size=(num_ICs, 6))

# Specify the CSV file path
csv_file_path = "ICs.csv"
df = pd.DataFrame(data)
df.to_csv(csv_file_path, header=False, index=False)
print(f"CSV file '{csv_file_path}' has been generated.")
