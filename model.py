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

    ## Base Experiment: Naive Lambda Search
    ## Test a range of penalty weights to find the baseline anchor balance
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


    ## Experiment 1: Network Capacity
    ## Vary the hidden dimensions to see if overparameterization prevents forgetting
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


    ## Experiment 2: Standard L2 Regularization
    ## Compare the custom anchors against traditional weight decay
    print("\n" + "="*50)
    print("EXPERIMENT 2: STANDARD L2 REGULARIZATION vs CUSTOM ANCHORS")
    print("="*50)
    weight_decays = [1e-4, 1e-3, 1e-2]
    
    for wd in weight_decays:
        model = BiasControlledNet(input_dim=4, hidden_dim=32)
        criterion = HingeLoss()
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        train_bias_controlled(model, train_loader, criterion, optimizer, epochs=30, lambda_reg=0.0, verbose=False)
        
        optimizer2 = optim.Adam(model.parameters(), lr=0.001, weight_decay=wd)
        train_bias_controlled(model, train_loader_2, criterion, optimizer2, epochs=30, lambda_reg=0.0, verbose=False)
        
        t1_acc = evaluate_model(model, test_loader, verbose=False)
        t2_acc = evaluate_model(model, test_loader_2, verbose=False)
        print(f"Weight Decay: {wd:<6} | Task 1: {t1_acc:>6.2f}% | Task 2: {t2_acc:>6.2f}%")


    ## Experiment 3: Asymmetric Learning Rates
    ## Reduce the learning rate on the second task to protect previous weights
    print("\n" + "="*50)
    print("EXPERIMENT 3: TASK 2 LEARNING RATE DECAY")
    print("="*50)
    lrs_task2 = [0.001, 0.0005, 0.0001]
    
    for lr2 in lrs_task2:
        model = BiasControlledNet(input_dim=4, hidden_dim=32)
        
        opt1 = optim.Adam(model.parameters(), lr=0.001)
        train_bias_controlled(model, train_loader, HingeLoss(), opt1, epochs=30, lambda_reg=0.0, verbose=False)
        model.set_anchors()
        
        opt2 = optim.Adam(model.parameters(), lr=lr2)
        train_bias_controlled(model, train_loader_2, HingeLoss(), opt2, epochs=30, lambda_reg=best_lambda, verbose=False)
        
        t1_acc = evaluate_model(model, test_loader, verbose=False)
        t2_acc = evaluate_model(model, test_loader_2, verbose=False)
        print(f"T2 LR: {lr2:<6.4f} | Task 1: {t1_acc:>6.2f}% | Task 2: {t2_acc:>6.2f}%")


    ## Experiment 4: Theoretical Bounds
    ## Establish the lower bound (naive sequential) and upper bound (joint training)
    print("\n" + "="*50)
    print("EXPERIMENT 4: THEORETICAL BOUNDS (Adam Opt)")
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


    ## Experiment 5: Experience Replay
    ## Mix a small 10% buffer of Task 1 data into Task 2 training
    print("\n" + "="*50)
    print("EXPERIMENT 5: EXPERIENCE REPLAY (10% Buffer)")
    print("="*50)
    model_replay = BiasControlledNet(input_dim=4, hidden_dim=32)
    criterion = HingeLoss()
    optimizer = optim.Adam(model_replay.parameters(), lr=0.001)

    train_bias_controlled(model_replay, train_loader, criterion, optimizer, epochs=30, lambda_reg=0.0, verbose=False)

    buffer_size = int(0.10 * len(train_dataset))
    indices = np.random.choice(len(train_dataset), buffer_size, replace=False)
    replay_subset = Subset(train_dataset, indices)

    combined_dataset = ConcatDataset([train_dataset_2, replay_subset])
    combined_loader = DataLoader(combined_dataset, batch_size=32, shuffle=True)

    train_bias_controlled(model_replay, combined_loader, criterion, optimizer, epochs=30, lambda_reg=0.0, verbose=False)

    t1_acc = evaluate_model(model_replay, test_loader, verbose=False)
    t2_acc = evaluate_model(model_replay, test_loader_2, verbose=False)
    print(f"Replay Buffer (10%)      | Task 1: {t1_acc:>6.2f}% | Task 2: {t2_acc:>6.2f}%")
    

    ## Experiment 6: Elastic Weight Consolidation (EWC)
    ## Use Fisher Information to heavily penalize only the weights critical to Task 1
    print("\n" + "="*50)
    print("EXPERIMENT 6: ELASTIC WEIGHT CONSOLIDATION (EWC)")
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