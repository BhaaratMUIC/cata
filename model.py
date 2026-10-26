import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class XORDataset(Dataset):
	def __init__(self, csv_file):
		self.data = pd.read_csv(csv_file)
		self.X = torch.tensor(self.data.iloc[:, :-1].values, dtype=torch.float32)
		self.y = torch.tensor(self.data.iloc[:, -1].values, dtype=torch.float32).unsqueeze(1)
		
	def __len__(self):
		return len(self.data)
	
	def __getitem__(self, idx):
		return self.X[idx], self.y[idx]

class BaseContinualNet(nn.Module):
	def __init__(self, input_dim=2, hidden_dim=16):
		super(BaseContinualNet, self).__init__()
		self.network = nn.Sequential(
			nn.Linear(input_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, 1),
			nn.Sigmoid()
		)
		
	def forward(self, x):
		return self.network(x)


if __name__ == "__main__":
	train_dataset = XORDataset("xor_train.csv")
	train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
	model = BaseContinualNet(input_dim=2, hidden_dim=16)

	features, labels = next(iter(train_loader))
	outputs = model(features)
