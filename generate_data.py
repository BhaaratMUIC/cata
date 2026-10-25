import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

## Generate Random Numbers
## XOR each value (toward 1/0)
## Add noise to create clusters
## 80/20 split

def generate_data(num_samples=1000, d=2, noise_std=0.2, test_size=0.2):
	X_base = np.random.choice([-1.0, 1.0], size=(num_samples, d))
	y = (np.prod(X_base, axis=1) > 0)
	noise = np.random.normal(loc=0.0, scale=noise_std, size=(num_samples, d))
	X = (X_base + noise)

	df = pd.DataFrame(X, columns=[f"feature_{i+1}" for i in range(d)])
	df['label'] = y

	train_df, test_df = train_test_split(df, test_size=test_size)
	train_df.to_csv("xor_train.csv", index=False)
	test_df.to_csv("xor_test.csv", index=False)

if __name__ == "__main__":
	generate_data(num_samples=1000, d=2)
	##
	## train_read = pd.read_csv("xor_train.csv")
	## test_read = pd.read_csv("xor_test.csv")
	
	## print(train_read.head())
	## print(test_read.head())