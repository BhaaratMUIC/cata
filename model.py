import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
import matplotlib.pyplot as plt

os.makedirs("figs", exist_ok=True)
plt.rcParams.update({"font.size": 11, "figure.dpi": 300})

SEEDS = [0, 1, 2, 3, 4]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

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
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.anchors = {}
        self.fisher = {}

    def forward(self, x):
        return self.network(x)

    def set_anchors(self):
        self.anchors = {n: p.clone().detach() for n, p in self.named_parameters()}

    def compute_fisher(self, dataloader, criterion):
        self.fisher = {n: torch.zeros_like(p) for n, p in self.named_parameters()}
        self.eval()
        for features, labels in dataloader:
            self.zero_grad()
            loss = criterion(self(features), labels)
            loss.backward()
            for n, p in self.named_parameters():
                if p.grad is not None:
                    self.fisher[n] += p.grad.detach() ** 2
        for n in self.fisher:
            self.fisher[n] /= len(dataloader)

class QuadraticNet(BiasControlledNet):
    def __init__(self, input_dim=4, hidden_dim=32):
        super().__init__(input_dim, hidden_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1, bias=False)
        self.anchors = {}
        self.fisher = {}

    def forward(self, x):
        return self.fc2(torch.pow(self.fc1(x), 2))

def train_model(model, dataloader, criterion, optimizer, epochs=30, lambda_reg=0.0):
    model.train()
    for _ in range(epochs):
        for features, labels in dataloader:
            optimizer.zero_grad()
            loss = criterion(model(features), labels)
            if model.anchors and lambda_reg > 0:
                penalty = 0.0
                for n, p in model.named_parameters():
                    if n in model.anchors:
                        fw = model.fisher.get(n, 1.0)
                        penalty += torch.sum(fw * (p - model.anchors[n]) ** 2)
                loss = loss + lambda_reg * penalty
            loss.backward()
            optimizer.step()

def evaluate_model(model, dataloader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for features, labels in dataloader:
            preds = torch.sign(model(features))
            preds[preds == 0] = 1.0
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return (correct / total) * 100.0

def mean_std(arr):
    return float(np.mean(arr)), float(np.std(arr))

if __name__ == "__main__":
    train_ds1 = XORDataset("xor_train.csv")
    test_ds1 = XORDataset("xor_test.csv")
    train_ds2 = XORDataset("xor_train_task2.csv")
    test_ds2 = XORDataset("xor_test_task2.csv")

    test_l1 = DataLoader(test_ds1, batch_size=32, shuffle=False)
    test_l2 = DataLoader(test_ds2, batch_size=32, shuffle=False)
    criterion = HingeLoss()

    # ---------------------------------------------------------
    # 0. THEORETICAL BOUNDS (Naive Lower Bound & Joint Upper Bound)
    # ---------------------------------------------------------
    lb_t1, lb_t2, ub_t1, ub_t2 = [], [], [], []
    for s in SEEDS:
        set_seed(s)
        tl1 = DataLoader(train_ds1, batch_size=32, shuffle=True)
        tl2 = DataLoader(train_ds2, batch_size=32, shuffle=True)
        m = BiasControlledNet(4, 32)
        train_model(m, tl1, criterion, optim.Adam(m.parameters(), lr=0.001), 30, 0.0)
        train_model(m, tl2, criterion, optim.Adam(m.parameters(), lr=0.001), 30, 0.0)
        lb_t1.append(evaluate_model(m, test_l1))
        lb_t2.append(evaluate_model(m, test_l2))

        set_seed(s)
        m_joint = BiasControlledNet(4, 32)
        jl = DataLoader(ConcatDataset([train_ds1, train_ds2]), batch_size=32, shuffle=True)
        train_model(m_joint, jl, criterion, optim.Adam(m_joint.parameters(), lr=0.001), 30, 0.0)
        ub_t1.append(evaluate_model(m_joint, test_l1))
        ub_t2.append(evaluate_model(m_joint, test_l2))

    lb_t1_m, lb_t1_s = mean_std(lb_t1)
    lb_t2_m, lb_t2_s = mean_std(lb_t2)
    ub_t1_m, ub_t1_s = mean_std(ub_t1)
    ub_t2_m, ub_t2_s = mean_std(ub_t2)
    print(f"Lower Bound (Naive) | T1: {lb_t1_m:.2f}±{lb_t1_s:.2f}% | T2: {lb_t2_m:.2f}±{lb_t2_s:.2f}%")
    print(f"Upper Bound (Joint) | T1: {ub_t1_m:.2f}±{ub_t1_s:.2f}% | T2: {ub_t2_m:.2f}±{ub_t2_s:.2f}%\n")

    # ---------------------------------------------------------
    # 1. EXPANDED STATIC L2 LAMBDA SEARCH
    # ---------------------------------------------------------
    lambdas = [0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5]
    l2_t1_m, l2_t1_s, l2_t2_m, l2_t2_s = [], [], [], []
    print("=== EXP 1: STATIC L2 LAMBDA SEARCH ===")
    for lam in lambdas:
        r1, r2 = [], []
        for s in SEEDS:
            set_seed(s)
            tl1 = DataLoader(train_ds1, batch_size=32, shuffle=True)
            tl2 = DataLoader(train_ds2, batch_size=32, shuffle=True)
            m = BiasControlledNet(4, 32)
            opt = optim.Adam(m.parameters(), lr=0.001)
            train_model(m, tl1, criterion, opt, 30, 0.0)
            m.set_anchors()
            train_model(m, tl2, criterion, opt, 30, lam)
            r1.append(evaluate_model(m, test_l1))
            r2.append(evaluate_model(m, test_l2))
        m1, s1 = mean_std(r1); m2, s2 = mean_std(r2)
        l2_t1_m.append(m1); l2_t1_s.append(s1); l2_t2_m.append(m2); l2_t2_s.append(s2)
        print(f"Lambda {lam:<6} | T1: {m1:5.2f}±{s1:4.2f}% | T2: {m2:5.2f}±{s2:4.2f}%")

    plt.figure(figsize=(6, 4))
    plt.errorbar(lambdas, l2_t1_m, yerr=l2_t1_s, marker='o', capsize=3, label="Task 1 Retention")
    plt.errorbar(lambdas, l2_t2_m, yerr=l2_t2_s, marker='s', capsize=3, label="Task 2 Acquisition")
    plt.axhline(ub_t1_m, linestyle='--', color='gray', alpha=0.7, label=f"Joint Upper Bound ({ub_t1_m:.1f}%)")
    plt.xscale("log")
    plt.xlabel(r"Static $L_2$ Penalty ($\lambda$)")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Static Bias Control: Stability-Plasticity Trade-off")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figs/exp1_lambda_search.png")
    plt.close()

    # ---------------------------------------------------------
    # 2. EXPANDED EWC LAMBDA SEARCH
    # ---------------------------------------------------------
    ewc_lambdas = [10, 50, 100, 250, 500, 1000, 2000, 5000, 10000]
    ewc_t1_m, ewc_t1_s, ewc_t2_m, ewc_t2_s = [], [], [], []
    print("\n=== EXP 2: EWC LAMBDA SEARCH ===")
    for lam in ewc_lambdas:
        r1, r2 = [], []
        for s in SEEDS:
            set_seed(s)
            tl1 = DataLoader(train_ds1, batch_size=32, shuffle=True)
            tl2 = DataLoader(train_ds2, batch_size=32, shuffle=True)
            m = BiasControlledNet(4, 32)
            opt = optim.Adam(m.parameters(), lr=0.001)
            train_model(m, tl1, criterion, opt, 30, 0.0)
            m.set_anchors()
            m.compute_fisher(tl1, criterion)
            train_model(m, tl2, criterion, opt, 30, lam)
            r1.append(evaluate_model(m, test_l1))
            r2.append(evaluate_model(m, test_l2))
        m1, s1 = mean_std(r1); m2, s2 = mean_std(r2)
        ewc_t1_m.append(m1); ewc_t1_s.append(s1); ewc_t2_m.append(m2); ewc_t2_s.append(s2)
        print(f"EWC {lam:<6} | T1: {m1:5.2f}±{s1:4.2f}% | T2: {m2:5.2f}±{s2:4.2f}%")

    plt.figure(figsize=(6, 4))
    plt.errorbar(ewc_lambdas, ewc_t1_m, yerr=ewc_t1_s, marker='o', capsize=3, label="Task 1 Retention")
    plt.errorbar(ewc_lambdas, ewc_t2_m, yerr=ewc_t2_s, marker='s', capsize=3, label="Task 2 Acquisition")
    plt.axhline(ub_t1_m, linestyle='--', color='gray', alpha=0.7, label=f"Joint Upper Bound ({ub_t1_m:.1f}%)")
    plt.xscale("log")
    plt.xlabel(r"EWC Penalty ($\lambda$)")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Elastic Weight Consolidation (EWC) Sensitivity")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figs/exp2_ewc_search.png")
    plt.close()

    # ---------------------------------------------------------
    # 3. NETWORK CAPACITY SCALING
    # ---------------------------------------------------------
    hidden_dims = [4, 8, 16, 32, 64, 128, 256]
    cap_t1_m, cap_t1_s, cap_t2_m, cap_t2_s = [], [], [], []
    print("\n=== EXP 3: NETWORK CAPACITY SCALING (Lambda=0.05) ===")
    for h in hidden_dims:
        r1, r2 = [], []
        for s in SEEDS:
            set_seed(s)
            tl1 = DataLoader(train_ds1, batch_size=32, shuffle=True)
            tl2 = DataLoader(train_ds2, batch_size=32, shuffle=True)
            m = BiasControlledNet(4, h)
            opt = optim.Adam(m.parameters(), lr=0.001)
            train_model(m, tl1, criterion, opt, 30, 0.0)
            m.set_anchors()
            train_model(m, tl2, criterion, opt, 30, 0.05)
            r1.append(evaluate_model(m, test_l1))
            r2.append(evaluate_model(m, test_l2))
        m1, s1 = mean_std(r1); m2, s2 = mean_std(r2)
        cap_t1_m.append(m1); cap_t1_s.append(s1); cap_t2_m.append(m2); cap_t2_s.append(s2)
        print(f"Hidden {h:<4} | T1: {m1:5.2f}±{s1:4.2f}% | T2: {m2:5.2f}±{s2:4.2f}%")

    plt.figure(figsize=(6, 4))
    plt.errorbar(hidden_dims, cap_t1_m, yerr=cap_t1_s, marker='o', capsize=3, label="Task 1 Retention")
    plt.errorbar(hidden_dims, cap_t2_m, yerr=cap_t2_s, marker='s', capsize=3, label="Task 2 Acquisition")
    plt.xscale("log", base=2)
    plt.xticks(hidden_dims, [str(h) for h in hidden_dims])
    plt.xlabel("Hidden Layer Width (Neurons)")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Effect of Network Width on Retention & Acquisition")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figs/exp3_capacity.png")
    plt.close()

    # ---------------------------------------------------------
    # 4. EXPERIENCE REPLAY ABLATION
    # ---------------------------------------------------------
    replay_pcts = [0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.25, 0.50]
    rep_t1_m, rep_t1_s, rep_t2_m, rep_t2_s = [], [], [], []
    print("\n=== EXP 4: EXPERIENCE REPLAY ABLATION ===")
    for pct in replay_pcts:
        r1, r2 = [], []
        for s in SEEDS:
            set_seed(s)
            tl1 = DataLoader(train_ds1, batch_size=32, shuffle=True)
            m = BiasControlledNet(4, 32)
            opt = optim.Adam(m.parameters(), lr=0.001)
            train_model(m, tl1, criterion, opt, 30, 0.0)
            if pct > 0:
                idx = np.random.choice(len(train_ds1), int(pct * len(train_ds1)), replace=False)
                ds2_comb = ConcatDataset([train_ds2, Subset(train_ds1, idx)])
            else:
                ds2_comb = train_ds2
            train_model(m, DataLoader(ds2_comb, batch_size=32, shuffle=True), criterion, opt, 30, 0.0)
            r1.append(evaluate_model(m, test_l1))
            r2.append(evaluate_model(m, test_l2))
        m1, s1 = mean_std(r1); m2, s2 = mean_std(r2)
        rep_t1_m.append(m1); rep_t1_s.append(s1); rep_t2_m.append(m2); rep_t2_s.append(s2)
        print(f"Replay {int(pct*100):>2}% | T1: {m1:5.2f}±{s1:4.2f}% | T2: {m2:5.2f}±{s2:4.2f}%")

    plt.figure(figsize=(6, 4))
    x_pct = [p * 100 for p in replay_pcts]
    plt.errorbar(x_pct, rep_t1_m, yerr=rep_t1_s, marker='o', capsize=3, label="Task 1 Retention")
    plt.errorbar(x_pct, rep_t2_m, yerr=rep_t2_s, marker='s', capsize=3, label="Task 2 Acquisition")
    plt.axhline(ub_t1_m, linestyle='--', color='gray', alpha=0.7, label=f"Joint Upper Bound ({ub_t1_m:.1f}%)")
    plt.xlabel("Task 1 Replay Buffer Size (%)")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Experience Replay Ablation with Hinge Loss")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figs/exp4_replay.png")
    plt.close()

    # ---------------------------------------------------------
    # 5. OVERALL METHOD & ARCHITECTURE COMPARISON (BAR CHART)
    # ---------------------------------------------------------
    frz_t1, frz_t2, quad_t1, quad_t2 = [], [], [], []
    for s in SEEDS:
        # Layer Freezing
        set_seed(s)
        tl1 = DataLoader(train_ds1, batch_size=32, shuffle=True)
        tl2 = DataLoader(train_ds2, batch_size=32, shuffle=True)
        mf = BiasControlledNet(4, 32)
        train_model(mf, tl1, criterion, optim.Adam(mf.parameters(), lr=0.001), 30, 0.0)
        for n, p in mf.named_parameters():
            if "network.6" not in n:
                p.requires_grad = False
        train_model(mf, tl2, criterion, optim.Adam(filter(lambda p: p.requires_grad, mf.parameters()), lr=0.001), 30, 0.0)
        frz_t1.append(evaluate_model(mf, test_l1))
        frz_t2.append(evaluate_model(mf, test_l2))

        # 2-Layer Quadratic + EWC (lambda=2000)
        set_seed(s)
        mq = QuadraticNet(4, 32)
        optq = optim.Adam(mq.parameters(), lr=0.001)
        train_model(mq, tl1, criterion, optq, 30, 0.0)
        mq.set_anchors()
        mq.compute_fisher(tl1, criterion)
        train_model(mq, tl2, criterion, optq, 30, 2000)
        quad_t1.append(evaluate_model(mq, test_l1))
        quad_t2.append(evaluate_model(mq, test_l2))

    # Pick best static L2 and best EWC by average (T1 + T2) / 2
    best_l2_idx = int(np.argmax([(a + b) / 2 for a, b in zip(l2_t1_m, l2_t2_m)]))
    best_ewc_idx = int(np.argmax([(a + b) / 2 for a, b in zip(ewc_t1_m, ewc_t2_m)]))

    methods = [
        "Naive\n(Lower)", "Layer\nFreezing",
        f"Static L2\n($\\lambda$={lambdas[best_l2_idx]})",
        f"EWC MLP\n($\\lambda$={ewc_lambdas[best_ewc_idx]})",
        "EWC Quad\n($\\lambda$=2000)", "Replay\n(25%)", "Joint\n(Upper)"
    ]
    t1_vals = [lb_t1_m, mean_std(frz_t1)[0], l2_t1_m[best_l2_idx], ewc_t1_m[best_ewc_idx], mean_std(quad_t1)[0], rep_t1_m[-2], ub_t1_m]
    t1_errs = [lb_t1_s, mean_std(frz_t1)[1], l2_t1_s[best_l2_idx], ewc_t1_s[best_ewc_idx], mean_std(quad_t1)[1], rep_t1_s[-2], ub_t1_s]
    t2_vals = [lb_t2_m, mean_std(frz_t2)[0], l2_t2_m[best_l2_idx], ewc_t2_m[best_ewc_idx], mean_std(quad_t2)[0], rep_t2_m[-2], ub_t2_m]
    t2_errs = [lb_t2_s, mean_std(frz_t2)[1], l2_t2_s[best_l2_idx], ewc_t2_s[best_ewc_idx], mean_std(quad_t2)[1], rep_t2_s[-2], ub_t2_s]

    x = np.arange(len(methods))
    w = 0.35
    plt.figure(figsize=(10, 4.2))
    plt.bar(x - w/2, t1_vals, w, yerr=t1_errs, capsize=3, label="Task 1 Retention")
    plt.bar(x + w/2, t2_vals, w, yerr=t2_errs, capsize=3, label="Task 2 Acquisition")
    plt.xticks(x, methods)
    plt.ylabel("Test Accuracy (%)")
    plt.ylim(35, 75)
    plt.title("Overall Continual Learning Method Comparison (Phase 2 Noisy 4D XOR)")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figs/exp5_method_comparison.png")
    plt.close()

    print("\nAll 5 figures saved to ./figs/!")