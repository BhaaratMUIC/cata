import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

## Generate Random Numbers
## XOR each value (toward 1/0)
## Add noise to create clusters
## 80/20 split


def create_task1_data(num_samples=1000, d=4, noise_std=0.2, test_size=0.2):
	X_active = np.random.choice([-1.0, 1.0], size=(num_samples, 2))
	X_inactive = np.random.uniform(low=-1.0, high=1.0, size=(num_samples, 2))
	
	y = (np.prod(X_active, axis=1) > 0).astype(np.float32)
	
	X_base = np.hstack((X_active, X_inactive))
	noise = np.random.normal(loc=0.0, scale=noise_std, size=(num_samples, d))
	X = (X_base + noise).astype(np.float32)

	df = pd.DataFrame(X, columns=[f"feature_{i+1}" for i in range(d)])
	df['label'] = y

	train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
	train_df.to_csv("xor_train.csv", index=False)
	test_df.to_csv("xor_test.csv", index=False)

def create_task2_data(num_samples=1000, d=4, noise_std=0.2, test_size=0.2):
	X_inactive = np.random.uniform(low=-1.0, high=1.0, size=(num_samples, 2))
	X_active = np.random.choice([-1.0, 1.0], size=(num_samples, 2))
	
	y = (np.prod(X_active, axis=1) > 0).astype(np.float32)
	
	X_base = np.hstack((X_inactive, X_active))
	noise = np.random.normal(loc=0.0, scale=noise_std, size=(num_samples, d))
	X = (X_base + noise).astype(np.float32)

	df = pd.DataFrame(X, columns=[f"feature_{i+1}" for i in range(d)])
	df['label'] = y

	train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
	train_df.to_csv("xor_train_task2.csv", index=False)
	test_df.to_csv("xor_test_task2.csv", index=False)

def create_complex_task_data(task_id, num_samples=2000, d=4, noise_std=0.8, flip_ratio=0.25, test_size=0.2):
	X_base = np.random.uniform(low=-2.0, high=2.0, size=(num_samples, d))

	if task_id == 1:
		y = (X_base[:, 0] * X_base[:, 1] > 0).astype(np.float32)
	else:
		y = (X_base[:, 2] * X_base[:, 3] > 0).astype(np.float32)

	noise = np.random.normal(loc=0.0, scale=noise_std, size=(num_samples, d))
	X = (X_base + noise).astype(np.float32)
	
	flip_indices = np.random.choice(num_samples, size=int(num_samples * flip_ratio), replace=False)
	y[flip_indices] = 1.0 - y[flip_indices] 

	df = pd.DataFrame(X, columns=[f"feature_{i+1}" for i in range(d)])
	df['label'] = y

	train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
	
	suffix = "" if task_id == 1 else "_task2"
	train_df.to_csv(f"xor_train{suffix}.csv", index=False)
	test_df.to_csv(f"xor_test{suffix}.csv", index=False)

if __name__ == "__main__":
	create_complex_task_data(task_id=1)
	create_complex_task_data(task_id=2)
