import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset

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
	def __init__(self, input_dim=4, hidden_dim=32):
		super(BiasControlledNet, self).__init__()
		self.network = nn.Sequential(
			nn.Linear(input_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, 1)
		)
		self.anchors = {}
		self.fisher = {}

	def forward(self, x):
		return self.network(x)

	def set_anchors(self):
		self.anchors = {}
		for name, param in self.named_parameters():
			self.anchors[name] = param.clone().detach()

	def compute_fisher(self, dataloader, criterion):
		self.fisher = {}
		for name, param in self.named_parameters():
			self.fisher[name] = torch.zeros_like(param)
		
		self.eval()
		for features, labels in dataloader:
			self.zero_grad()
			outputs = self(features)
			loss = criterion(outputs, labels)
			loss.backward()
			for name, param in self.named_parameters():
				if param.grad is not None:
					self.fisher[name] += param.grad.detach() ** 2
		
		for name in self.fisher:
			self.fisher[name] /= len(dataloader)

class QuadraticNet(BiasControlledNet):
	def __init__(self, input_dim=4, hidden_dim=32):
		super(BiasControlledNet, self).__init__()
		self.fc1 = nn.Linear(input_dim, hidden_dim)
		self.fc2 = nn.Linear(hidden_dim, 1, bias=False)
		self.anchors = {}
		self.fisher = {}

	def forward(self, x):
		h = torch.pow(self.fc1(x), 2)
		return self.fc2(h)

def train_bias_controlled(model, dataloader, criterion, optimizer, epochs=30, lambda_reg=0.01, verbose=True):
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
						fisher_weight = model.fisher.get(name, 1.0)
						bias_penalty += torch.sum(fisher_weight * (param - model.anchors[name]) ** 2)
				loss += lambda_reg * bias_penalty
			
			loss.backward()
			optimizer.step()
			total_loss += loss.item()
		if verbose and (epoch + 1) % 5 == 0:
			print(f"Epoch {epoch+1}/{epochs} | Average Loss: {total_loss/len(dataloader):.4f}")

def evaluate_model(model, dataloader, verbose=True):
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
	if verbose:
		print(f"Test Accuracy: {accuracy:.2f}%")
	return accuracy

if __name__ == "__main__":
	train_dataset = XORDataset("xor_train.csv")
	test_dataset = XORDataset("xor_test.csv")
	train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
	test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

	train_dataset_2 = XORDataset("xor_train_task2.csv")
	test_dataset_2 = XORDataset("xor_test_task2.csv")
	train_loader_2 = DataLoader(train_dataset_2, batch_size=32, shuffle=True)
	test_loader_2 = DataLoader(test_dataset_2, batch_size=32, shuffle=False)

	print("\n" + "="*50)
	print("BASE EXPERIMENT: NAIVE LAMBDA SEARCH (Adam Opt)")
	print("="*50)
	lambdas_to_test = [0.001, 0.005, 0.01, 0.05, 0.1]
	experiment_results = {}
	best_lambda = 0.01

	for l_reg in lambdas_to_test:
		model = BiasControlledNet(input_dim=4, hidden_dim=32)
		criterion = HingeLoss()
		optimizer = optim.Adam(model.parameters(), lr=0.001)
		
		train_bias_controlled(model, train_loader, criterion, optimizer, epochs=30, lambda_reg=0.0, verbose=False)
		model.set_anchors()
		train_bias_controlled(model, train_loader_2, criterion, optimizer, epochs=30, lambda_reg=l_reg, verbose=False)

		t1_acc = evaluate_model(model, test_loader, verbose=False)
		t2_acc = evaluate_model(model, test_loader_2, verbose=False)
		experiment_results[l_reg] = (t1_acc, t2_acc)
		
		if t1_acc > 65.0 and t2_acc > 65.0:
			best_lambda = l_reg

	print(f"{'Lambda':<10} | {'T1 Retention':<15} | {'T2 Acquisition'}")
	print("-" * 50)
	for l_reg, (t1, t2) in experiment_results.items():
		print(f"{l_reg:<10} | {t1:>10.2f}%      | {t2:>10.2f}%")

	print("\n" + "="*50)
	print(f"EXPERIMENT 1: NETWORK CAPACITY (Lambda={best_lambda})")
	print("="*50)
	hidden_dims = [8, 16, 32, 64, 128]
	
	for h_dim in hidden_dims:
		model = BiasControlledNet(input_dim=4, hidden_dim=h_dim)
		criterion = HingeLoss()
		optimizer = optim.Adam(model.parameters(), lr=0.001)
		
		train_bias_controlled(model, train_loader, criterion, optimizer, epochs=30, lambda_reg=0.0, verbose=False)
		model.set_anchors()
		train_bias_controlled(model, train_loader_2, criterion, optimizer, epochs=30, lambda_reg=best_lambda, verbose=False)
		
		t1_acc = evaluate_model(model, test_loader, verbose=False)
		t2_acc = evaluate_model(model, test_loader_2, verbose=False)
		print(f"Hidden Dim: {h_dim:<4} | Task 1: {t1_acc:>6.2f}% | Task 2: {t2_acc:>6.2f}%")

	print("\n" + "="*50)
	print("EXPERIMENT 2: LAYER FREEZING (Feature Extraction)")
	print("="*50)
	model_frozen = BiasControlledNet(input_dim=4, hidden_dim=32)
	criterion = HingeLoss()
	optimizer_full = optim.Adam(model_frozen.parameters(), lr=0.001)
	
	train_bias_controlled(model_frozen, train_loader, criterion, optimizer_full, epochs=30, lambda_reg=0.0, verbose=False)
	
	for name, param in model_frozen.named_parameters():
		if 'network.6' not in name:
			param.requires_grad = False
			
	optimizer_frozen = optim.Adam(filter(lambda p: p.requires_grad, model_frozen.parameters()), lr=0.001)
	
	train_bias_controlled(model_frozen, train_loader_2, criterion, optimizer_frozen, epochs=30, lambda_reg=0.0, verbose=False)
	
	t1_acc = evaluate_model(model_frozen, test_loader, verbose=False)
	t2_acc = evaluate_model(model_frozen, test_loader_2, verbose=False)
	print(f"Layer Freezing         | Task 1: {t1_acc:>6.2f}% | Task 2: {t2_acc:>6.2f}%")

	print("\n" + "="*50)
	print("EXPERIMENT 3: THEORETICAL BOUNDS (Adam Opt)")
	print("="*50)
	model_lower = BiasControlledNet(input_dim=4, hidden_dim=32)
	opt_lower1 = optim.Adam(model_lower.parameters(), lr=0.001)
	train_bias_controlled(model_lower, train_loader, HingeLoss(), opt_lower1, epochs=30, lambda_reg=0.0, verbose=False)
	opt_lower2 = optim.Adam(model_lower.parameters(), lr=0.001)
	train_bias_controlled(model_lower, train_loader_2, HingeLoss(), opt_lower2, epochs=30, lambda_reg=0.0, verbose=False)
	
	lb_t1 = evaluate_model(model_lower, test_loader, verbose=False)
	lb_t2 = evaluate_model(model_lower, test_loader_2, verbose=False)
	print(f"Lower Bound (Naive Seq)  | Task 1: {lb_t1:>6.2f}% | Task 2: {lb_t2:>6.2f}%")
	
	model_upper = BiasControlledNet(input_dim=4, hidden_dim=32)
	opt_upper = optim.Adam(model_upper.parameters(), lr=0.001)
	joint_train_data = ConcatDataset([train_dataset, train_dataset_2])
	joint_loader = DataLoader(joint_train_data, batch_size=32, shuffle=True)
	train_bias_controlled(model_upper, joint_loader, HingeLoss(), opt_upper, epochs=30, lambda_reg=0.0, verbose=False)
	
	ub_t1 = evaluate_model(model_upper, test_loader, verbose=False)
	ub_t2 = evaluate_model(model_upper, test_loader_2, verbose=False)
	print(f"Upper Bound (Joint Train)| Task 1: {ub_t1:>6.2f}% | Task 2: {ub_t2:>6.2f}%")

	print("\n" + "="*50)
	print("EXPERIMENT 4: EXPERIENCE REPLAY ABLATION")
	print("="*50)
	buffer_percentages = [0.01, 0.05, 0.10, 0.25]
	
	for pct in buffer_percentages:
		model_replay = BiasControlledNet(input_dim=4, hidden_dim=32)
		optimizer = optim.Adam(model_replay.parameters(), lr=0.001)
		
		train_bias_controlled(model_replay, train_loader, HingeLoss(), optimizer, epochs=30, lambda_reg=0.0, verbose=False)
		
		buffer_size = int(pct * len(train_dataset))
		indices = np.random.choice(len(train_dataset), buffer_size, replace=False)
		replay_subset = Subset(train_dataset, indices)
		combined_loader = DataLoader(ConcatDataset([train_dataset_2, replay_subset]), batch_size=32, shuffle=True)
		
		train_bias_controlled(model_replay, combined_loader, HingeLoss(), optimizer, epochs=30, lambda_reg=0.0, verbose=False)
		
		t1_acc = evaluate_model(model_replay, test_loader, verbose=False)
		t2_acc = evaluate_model(model_replay, test_loader_2, verbose=False)
		print(f"Buffer Size: {int(pct*100):>2}%     | Task 1: {t1_acc:>6.2f}% | Task 2: {t2_acc:>6.2f}%")

	print("\n" + "="*50)
	print("EXPERIMENT 5: ELASTIC WEIGHT CONSOLIDATION (EWC)")
	print("="*50)
	ewc_lambdas = [1000, 5000, 10000] 
	
	for ewc_l in ewc_lambdas:
		model_ewc = BiasControlledNet(input_dim=4, hidden_dim=32)
		criterion = HingeLoss()
		optimizer = optim.Adam(model_ewc.parameters(), lr=0.001)

		train_bias_controlled(model_ewc, train_loader, criterion, optimizer, epochs=30, lambda_reg=0.0, verbose=False)
		
		model_ewc.set_anchors()
		model_ewc.compute_fisher(train_loader, criterion)
		
		train_bias_controlled(model_ewc, train_loader_2, criterion, optimizer, epochs=30, lambda_reg=ewc_l, verbose=False)
		
		t1_acc = evaluate_model(model_ewc, test_loader, verbose=False)
		t2_acc = evaluate_model(model_ewc, test_loader_2, verbose=False)
		print(f"EWC Lambda: {ewc_l:<10} | Task 1: {t1_acc:>6.2f}% | Task 2: {t2_acc:>6.2f}%")

	print("\n" + "="*50)
	print("EXPERIMENT 6: ARCHITECTURE DYNAMICS (EWC Lambda=5000)")
	print("="*50)
	architectures = {
		"Deep ReLU (Current)": BiasControlledNet(input_dim=4, hidden_dim=32),
		"2-Layer Quadratic": QuadraticNet(input_dim=4, hidden_dim=32)
	}
	
	for name, net in architectures.items():
		optimizer = optim.Adam(net.parameters(), lr=0.001)
		
		train_bias_controlled(net, train_loader, HingeLoss(), optimizer, epochs=30, lambda_reg=0.0, verbose=False)
		net.set_anchors()
		net.compute_fisher(train_loader, HingeLoss())
		train_bias_controlled(net, train_loader_2, HingeLoss(), optimizer, epochs=30, lambda_reg=5000, verbose=False)
		
		t1_acc = evaluate_model(net, test_loader, verbose=False)
		t2_acc = evaluate_model(net, test_loader_2, verbose=False)
		print(f"{name:<19} | Task 1: {t1_acc:>6.2f}% | Task 2: {t2_acc:>6.2f}%")