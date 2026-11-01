import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class HingeLoss(nn.Module):
	def forward(self, preds, labels):
		return torch.mean(F.relu(1 - preds * labels))

class XORDataset(Dataset):
	def __init__(self, csv_file):
		self.data = pd.read_csv(csv_file)
		self.X = torch.tensor(self.data.iloc[:, :-1].values, dtype=torch.float32)
		y_raw = torch.tensor(self.data.iloc[:, -1].values, dtype=torch.float32).unsqueeze(1)
		self.y = torch.where(y_raw > 0, 1.0, -1.0)
		
	def __len__(self):
		return len(self.data)
	
	def __getitem__(self, idx):
		return self.X[idx], self.y[idx]

class BiasControlledNet(nn.Module):
	def __init__(self, input_dim=4, hidden_dim=16):
		super(BiasControlledNet, self).__init__()
		self.network = nn.Sequential(
			nn.Linear(input_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, 1)
		)
		self.anchors = {}

	def forward(self, x):
		return self.network(x)

	def set_anchors(self):
		for name, param in self.named_parameters():
			self.anchors[name] = param.clone().detach()

def train_bias_controlled(model, dataloader, criterion, optimizer, epochs=30, lambda_reg=0.01):
	model.train()
	for epoch in range(epochs):
		total_loss = 0
		for features, labels in dataloader:
			optimizer.zero_grad()
			outputs = model(features)
			loss = criterion(outputs, labels)
			
			if model.anchors:
				bias_penalty = 0
				for name, param in model.named_parameters():
					if name in model.anchors:
						bias_penalty += torch.sum((param - model.anchors[name]) ** 2)
				loss += lambda_reg * bias_penalty
			
			loss.backward()
			optimizer.step()
			total_loss += loss.item()
		if (epoch + 1) % 5 == 0:
			print(f"Epoch {epoch+1}/{epochs} | Average Loss: {total_loss/len(dataloader):.4f}")

def evaluate_model(model, dataloader):
	model.eval()
	correct = 0
	total = 0
	with torch.no_grad():
		for features, labels in dataloader:
			outputs = model(features)
			predictions = torch.sign(outputs)
			predictions[predictions == 0] = 1.0 
			correct += (predictions == labels).sum().item()
			total += labels.size(0)
	accuracy = (correct / total) * 100
	print(f"Test Accuracy: {accuracy:.2f}%")
	return accuracy

if __name__ == "__main__":
	train_dataset = XORDataset("xor_train.csv")
	train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
	test_dataset = XORDataset("xor_test.csv")
	test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

	model = BiasControlledNet(input_dim=4, hidden_dim=16)
	criterion = HingeLoss()
	optimizer = optim.SGD(model.parameters(), lr=0.05)

	print("Training on Task 1...")
	train_bias_controlled(model, train_loader, criterion, optimizer, epochs=30)
	print("\nEvaluating Task 1:")
	evaluate_model(model, test_loader)

	model.set_anchors()

	train_dataset_2 = XORDataset("xor_train_task2.csv")
	train_loader_2 = DataLoader(train_dataset_2, batch_size=32, shuffle=True)
	test_dataset_2 = XORDataset("xor_test_task2.csv")
	test_loader_2 = DataLoader(test_dataset_2, batch_size=32, shuffle=False)

	print("\nTraining on Task 2 (With Bias Control)...")
	train_bias_controlled(model, train_loader_2, criterion, optimizer, epochs=30, lambda_reg=0.01)

	print("\nEvaluating Task 2:")
	evaluate_model(model, test_loader_2)
	print("\nRe-evaluating Task 1 (Checking Retention):")
	evaluate_model(model, test_loader)