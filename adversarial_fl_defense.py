# adversarial_fl_defense.py
# Demonstrates model poisoning attack and robust aggregation defense
# Objective 4: Robust distributed method against adversarial attacks

import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import copy
import json
import os

# ============================================================
# LOCK ALL RANDOM SEEDS FOR REPRODUCIBILITY
# ============================================================
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

NUM_CLIENTS = 5
POISONED_CLIENTS = [1, 2]
FL_ROUNDS = 20
LOCAL_EPOCHS = 3
BATCH_SIZE = 64
LEARNING_RATE = 0.001
TRIM_RATIO = 0.2
POISON_MULTIPLIER = 5.0

os.makedirs('results', exist_ok=True)

class IntrusionDetector(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.network(x)

print("[*] Loading dataset...")
df = pd.read_csv('ngn_traffic_data.csv')
X = df.drop(['label', 'attack_type'], axis=1).values.astype(np.float32)
y = df['label'].values.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

client_data = []
indices = np.arange(len(X_train))
np.random.shuffle(indices)
chunk_size = len(X_train) // NUM_CLIENTS
for i in range(NUM_CLIENTS):
    start = i * chunk_size
    end = start + chunk_size if i < NUM_CLIENTS - 1 else len(X_train)
    client_idx = indices[start:end]
    client_data.append({
        'X': torch.tensor(X_train[client_idx], dtype=torch.float32),
        'y': torch.tensor(y_train[client_idx], dtype=torch.float32).unsqueeze(1)
    })

X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=BATCH_SIZE, shuffle=False)

input_dim = X_train.shape[1]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def federated_averaging(client_states):
    avg = {}
    for key in client_states[0].keys():
        avg[key] = sum([s[key] for s in client_states]) / len(client_states)
    return avg

def trimmed_mean_aggregation(client_states, trim_ratio=TRIM_RATIO):
    num_trim = max(1, int(len(client_states) * trim_ratio))
    robust_state = {}
    for key in client_states[0].keys():
        stacked = torch.stack([s[key] for s in client_states])
        sorted_vals, _ = torch.sort(stacked, dim=0)
        if 2 * num_trim < len(client_states):
            trimmed = sorted_vals[num_trim:-num_trim]
        else:
            trimmed = sorted_vals
        robust_state[key] = trimmed.mean(dim=0)
    return robust_state

def train_client_normal(model, data, epochs, device):
    model.train()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loader = DataLoader(TensorDataset(data['X'], data['y']), batch_size=BATCH_SIZE, shuffle=True)
    for _ in range(epochs):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
    return copy.deepcopy(model.state_dict())

def train_client_poisoned(model, data, epochs, device):
    """Poisoned client: trains on flipped labels AND amplifies attack gradient."""
    model.train()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    poisoned_y = torch.zeros_like(data['y'])
    loader = DataLoader(TensorDataset(data['X'], poisoned_y), batch_size=BATCH_SIZE, shuffle=True)
    for _ in range(epochs):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            with torch.no_grad():
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad *= POISON_MULTIPLIER
            optimizer.step()
    return copy.deepcopy(model.state_dict())

def evaluate_model(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            outputs = model(bx)
            pred = (outputs >= 0.5).float()
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(by.cpu().numpy())
    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()
    return accuracy_score(all_labels, all_preds), f1_score(all_labels, all_preds)

# SCENARIO 1: Poisoning + Standard FedAvg
print(f"\n[{'='*60}]")
print(f"SCENARIO 1: Model Poisoning + Standard FedAvg")
print(f"Poisoned clients: {[p+1 for p in POISONED_CLIENTS]} of {NUM_CLIENTS}")
print(f"Poison multiplier: {POISON_MULTIPLIER}x")
print(f"[{'='*60}]")

global_model_vuln = IntrusionDetector(input_dim).to(device)
vuln_accuracies, vuln_f1s = [], []

for round_num in range(1, FL_ROUNDS + 1):
    client_states = []
    for client_id in range(NUM_CLIENTS):
        local_model = IntrusionDetector(input_dim).to(device)
        local_model.load_state_dict(global_model_vuln.state_dict())
        if client_id in POISONED_CLIENTS:
            state = train_client_poisoned(local_model, client_data[client_id], LOCAL_EPOCHS, device)
        else:
            state = train_client_normal(local_model, client_data[client_id], LOCAL_EPOCHS, device)
        client_states.append(state)
    new_state = federated_averaging(client_states)
    global_model_vuln.load_state_dict(new_state)
    acc, f1 = evaluate_model(global_model_vuln, test_loader, device)
    vuln_accuracies.append(acc)
    vuln_f1s.append(f1)
    if round_num % 5 == 0:
        print(f"  Round {round_num}: Accuracy={acc:.4f}, F1={f1:.4f}")

vuln_acc, vuln_f1 = evaluate_model(global_model_vuln, test_loader, device)
print(f"\n  VULNERABLE FINAL: Accuracy={vuln_acc:.4f}, F1={vuln_f1:.4f}")

# SCENARIO 2: Poisoning + Trimmed Mean
print(f"\n[{'='*60}]")
print(f"SCENARIO 2: Model Poisoning + Trimmed Mean Defense")
print(f"Trim ratio: {TRIM_RATIO} ({int(TRIM_RATIO*100)}% trimmed from each tail)")
print(f"[{'='*60}]")

global_model_robust = IntrusionDetector(input_dim).to(device)
robust_accuracies, robust_f1s = [], []

for round_num in range(1, FL_ROUNDS + 1):
    client_states = []
    for client_id in range(NUM_CLIENTS):
        local_model = IntrusionDetector(input_dim).to(device)
        local_model.load_state_dict(global_model_robust.state_dict())
        if client_id in POISONED_CLIENTS:
            state = train_client_poisoned(local_model, client_data[client_id], LOCAL_EPOCHS, device)
        else:
            state = train_client_normal(local_model, client_data[client_id], LOCAL_EPOCHS, device)
        client_states.append(state)
    new_state = trimmed_mean_aggregation(client_states, TRIM_RATIO)
    global_model_robust.load_state_dict(new_state)
    acc, f1 = evaluate_model(global_model_robust, test_loader, device)
    robust_accuracies.append(acc)
    robust_f1s.append(f1)
    if round_num % 5 == 0:
        print(f"  Round {round_num}: Accuracy={acc:.4f}, F1={f1:.4f}")

robust_acc, robust_f1 = evaluate_model(global_model_robust, test_loader, device)
print(f"\n  ROBUST FINAL: Accuracy={robust_acc:.4f}, F1={robust_f1:.4f}")

# Summary
print(f"\n[{'='*60}]")
print(f"OBJECTIVE 4 RESULTS SUMMARY")
print(f"[{'='*60}]")
print(f"{'Metric':<30} {'Vulnerable (FedAvg)':<25} {'Robust (Trimmed Mean)':<25}")
print(f"{'-'*80}")
print(f"{'Final Accuracy':<30} {vuln_acc:<25.4f} {robust_acc:<25.4f}")
print(f"{'Final F1-Score':<30} {vuln_f1:<25.4f} {robust_f1:<25.4f}")
print(f"{'Defense Improvement':<30} {robust_acc - vuln_acc:<25.4f} {robust_f1 - vuln_f1:<25.4f}")
poison_success = vuln_f1 < 0.75
print(f"\nPoison attack succeeded in vulnerable model: {'YES' if poison_success else 'Partial'}")
print(f"Robust defense mitigated attack: {'YES' if robust_f1 > vuln_f1 + 0.05 else 'Partial'}")

results = {
    'scenario': 'poisoning_attack_vs_defense',
    'num_clients': NUM_CLIENTS,
    'poisoned_clients': [p+1 for p in POISONED_CLIENTS],
    'fl_rounds': FL_ROUNDS,
    'trim_ratio': TRIM_RATIO,
    'vulnerable': {
        'final_accuracy': float(vuln_acc), 'final_f1': float(vuln_f1),
        'round_accuracies': [float(a) for a in vuln_accuracies],
        'round_f1s': [float(f) for f in vuln_f1s]
    },
    'robust': {
        'final_accuracy': float(robust_acc), 'final_f1': float(robust_f1),
        'round_accuracies': [float(a) for a in robust_accuracies],
        'round_f1s': [float(f) for f in robust_f1s]
    },
    'defense_improvement': {
        'accuracy_gain': float(robust_acc - vuln_acc),
        'f1_gain': float(robust_f1 - vuln_f1)
    }
}

with open('results/adversarial_defense.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to results/adversarial_defense.json")
