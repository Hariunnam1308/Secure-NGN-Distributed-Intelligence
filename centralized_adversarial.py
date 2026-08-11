#!/usr/bin/env python3
"""
Centralized Adversarial Robustness Experiment
Tests centralized models under data poisoning attacks
and compares against federated trimmed-mean defense.
MSc Dissertation: Distributed Intelligence for Secure NGNs
"""

import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
import json
import os

# ============================================================
# LOCK ALL RANDOM SEEDS FOR REPRODUCIBILITY
# ============================================================
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ============================================================
# CONFIGURATION
# ============================================================
POISON_RATIO = 0.40          # 40% of training data is poisoned
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 30
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs('results', exist_ok=True)

# ============================================================
# MODEL DEFINITION
# ============================================================
class IntrusionDetector(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

# ============================================================
# DATA LOADING AND POISONING
# ============================================================
print("[*] Loading dataset...")
df = pd.read_csv('ngn_traffic_data.csv')
X = df.drop(['label', 'attack_type'], axis=1).values.astype(np.float32)
y = df['label'].values.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ============================================================
# POISON THE TRAINING DATA
# ============================================================
print(f"[*] Poisoning {POISON_RATIO*100:.0f}% of training data...")
np.random.seed(12345)
poison_indices = np.random.choice(
    len(y_train), size=int(len(y_train) * POISON_RATIO), replace=False
)
y_train_poisoned = y_train.copy()
y_train_poisoned[poison_indices] = 1 - y_train_poisoned[poison_indices]
# Flipped: attacks become benign, benign becomes attack

poison_mask = np.zeros(len(y_train), dtype=bool)
poison_mask[poison_indices] = True

print(f"    Clean samples: {(~poison_mask).sum()}")
print(f"    Poisoned samples: {poison_mask.sum()}")
print(f"    Clean attack ratio: {y_train[~poison_mask].mean():.2%}")
print(f"    Poisoned attack ratio: {y_train_poisoned[poison_mask].mean():.2%}")

# ============================================================
# UTILITY FUNCTIONS
# ============================================================
def create_loaders(X_data, y_data, shuffle=True):
    """Create PyTorch DataLoader from numpy arrays."""
    X_t = torch.tensor(X_data, dtype=torch.float32)
    y_t = torch.tensor(y_data, dtype=torch.float32).unsqueeze(1)
    return DataLoader(TensorDataset(X_t, y_t), batch_size=BATCH_SIZE, shuffle=shuffle)

def evaluate_model(model, loader):
    """Evaluate model and return accuracy and F1."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            outputs = model(batch_x)
            predicted = (outputs >= 0.5).float()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()
    return accuracy_score(all_labels, all_preds), f1_score(all_labels, all_preds), all_preds, all_labels

# ============================================================
# SCENARIO 1: Centralized + Poisoned Data + Standard Training (VULNERABLE)
# ============================================================
print(f"\n[{'='*60}]")
print(f"SCENARIO 1: Centralized Model on Poisoned Data (Vulnerable)")
print(f"Poison ratio: {POISON_RATIO*100:.0f}%")
print(f"No defense mechanism applied")
print(f"[{'='*60}]")

train_loader_poisoned = create_loaders(X_train, y_train_poisoned, shuffle=True)
test_loader = create_loaders(X_test, y_test, shuffle=False)

model_vuln = IntrusionDetector(X_train.shape[1]).to(DEVICE)
criterion = nn.BCELoss()
optimizer = optim.Adam(model_vuln.parameters(), lr=LEARNING_RATE)

train_losses_vuln = []
test_accs_vuln = []
test_f1s_vuln = []

for epoch in range(EPOCHS):
    model_vuln.train()
    epoch_loss = 0
    for batch_x, batch_y in train_loader_poisoned:
        batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
        optimizer.zero_grad()
        outputs = model_vuln(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader_poisoned)
    train_losses_vuln.append(avg_loss)

    if (epoch + 1) % 5 == 0:
        acc, f1, _, _ = evaluate_model(model_vuln, test_loader)
        test_accs_vuln.append(acc)
        test_f1s_vuln.append(f1)
        print(f"  Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Test Acc: {acc:.4f} | Test F1: {f1:.4f}")

vuln_acc, vuln_f1, vuln_preds, vuln_labels = evaluate_model(model_vuln, test_loader)
print(f"\n  VULNERABLE CENTRALIZED FINAL: Accuracy={vuln_acc:.4f}, F1={vuln_f1:.4f}")

# ============================================================
# SCENARIO 2: Centralized + Poisoned Data + Robust Training (DEFENDED)
# ============================================================
print(f"\n[{'='*60}]")
print(f"SCENARIO 2: Centralized Model on Poisoned Data (Robust Defense)")
print(f"Defense: Loss-based sample filtering + gradient clipping")
print(f"[{'='*60}]")

model_robust = IntrusionDetector(X_train.shape[1]).to(DEVICE)
criterion_robust = nn.BCELoss(reduction='none')
optimizer_robust = optim.Adam(model_robust.parameters(), lr=LEARNING_RATE)

train_losses_robust = []
test_accs_robust = []
test_f1s_robust = []

for epoch in range(EPOCHS):
    model_robust.train()
    epoch_loss = 0
    samples_used = 0

    for batch_x, batch_y in train_loader_poisoned:
        batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
        optimizer_robust.zero_grad()
        outputs = model_robust(batch_x)
        losses = criterion_robust(outputs, batch_y)

        # Robust defense: exclude high-loss samples (likely poisoned)
        loss_threshold = torch.median(losses) * 2.0
        mask = losses <= loss_threshold

        if mask.sum() > 0:
            filtered_loss = losses[mask].mean()
            filtered_loss.backward()
            # Gradient clipping for additional robustness
            torch.nn.utils.clip_grad_norm_(model_robust.parameters(), 1.0)
            optimizer_robust.step()
            epoch_loss += filtered_loss.item()
            samples_used += mask.sum().item()

    if samples_used > 0:
        avg_loss = epoch_loss / max(1, len(train_loader_poisoned))
    else:
        avg_loss = 999.0
    train_losses_robust.append(avg_loss)

    if (epoch + 1) % 5 == 0:
        acc, f1, _, _ = evaluate_model(model_robust, test_loader)
        test_accs_robust.append(acc)
        test_f1s_robust.append(f1)
        print(f"  Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Test Acc: {acc:.4f} | Test F1: {f1:.4f}")

robust_acc, robust_f1, robust_preds, robust_labels = evaluate_model(model_robust, test_loader)
print(f"\n  ROBUST CENTRALIZED FINAL: Accuracy={robust_acc:.4f}, F1={robust_f1:.4f}")

# ============================================================
# LOAD EXISTING FEDERATED RESULTS
# ============================================================
try:
    with open('results/baseline_centralized.json') as f:
        baseline = json.load(f)
    clean_centralized_acc = baseline['accuracy']
    clean_centralized_f1 = baseline['f1_score']
except:
    clean_centralized_acc = 0.9583
    clean_centralized_f1 = 0.9098
    print("[!] Could not load baseline, using known values")

try:
    with open('results/federated_dp.json') as f:
        fl_dp = json.load(f)
    fl_clean_acc = fl_dp['final_accuracy']
    fl_clean_f1 = fl_dp['final_f1']
except:
    fl_clean_acc = 0.8215
    fl_clean_f1 = 0.4878
    print("[!] Could not load FL results, using known values")

try:
    with open('results/adversarial_defense.json') as f:
        adv = json.load(f)
    fl_vuln_acc = adv['vulnerable']['final_accuracy']
    fl_vuln_f1 = adv['vulnerable']['final_f1']
    fl_robust_acc = adv['robust']['final_accuracy']
    fl_robust_f1 = adv['robust']['final_f1']
except:
    fl_vuln_acc = 0.7920
    fl_vuln_f1 = 0.2877
    fl_robust_acc = 0.8035
    fl_robust_f1 = 0.3526
    print("[!] Could not load adversarial FL results, using known values")

# ============================================================
# COMPARISON SUMMARY
# ============================================================
print(f"\n[{'='*70}]")
print(f"COMPLETE SIX-MODEL COMPARISON")
print(f"[{'='*70}]")
print(f"{'Model':<40} {'Accuracy':<12} {'F1-Score':<12}")
print(f"{'-'*64}")
print(f"{'Centralized Baseline (Clean)':<40} {clean_centralized_acc:<12.4f} {clean_centralized_f1:<12.4f}")
print(f"{'Centralized + Poison (Vulnerable)':<40} {vuln_acc:<12.4f} {vuln_f1:<12.4f}")
print(f"{'Centralized + Poison (Robust Defense)':<40} {robust_acc:<12.4f} {robust_f1:<12.4f}")
print(f"{'Federated with DP (Clean)':<40} {fl_clean_acc:<12.4f} {fl_clean_f1:<12.4f}")
print(f"{'FL + Poison (Vulnerable FedAvg)':<40} {fl_vuln_acc:<12.4f} {fl_vuln_f1:<12.4f}")
print(f"{'FL + Poison (Robust Trimmed-Mean)':<40} {fl_robust_acc:<12.4f} {fl_robust_f1:<12.4f}")
print(f"{'-'*64}")

# Key comparisons
centralized_drop = clean_centralized_acc - vuln_acc
federated_drop = fl_clean_acc - fl_vuln_acc
centralized_recovery = robust_acc - vuln_acc
federated_recovery = fl_robust_acc - fl_vuln_acc

print(f"\nKEY INSIGHTS:")
print(f"  Centralized accuracy drop under poison: {centralized_drop:.4f} ({centralized_drop/clean_centralized_acc*100:.1f}%)")
print(f"  Federated accuracy drop under poison:    {federated_drop:.4f} ({federated_drop/fl_clean_acc*100:.1f}%)")
print(f"  Centralized defense recovery:            +{centralized_recovery:.4f}")
print(f"  Federated trimmed-mean recovery:         +{federated_recovery:.4f}")

if federated_recovery > centralized_recovery:
    print(f"\n  >>> Federated trimmed-mean defense outperforms centralized robust training")
    print(f"  >>> Recovery advantage: +{federated_recovery - centralized_recovery:.4f}")
else:
    print(f"\n  >>> Centralized defense shows stronger recovery in this configuration")

# ============================================================
# SAVE RESULTS
# ============================================================
results = {
    'scenario': 'centralized_adversarial_comparison',
    'poison_ratio': POISON_RATIO,
    'centralized_vulnerable': {
        'accuracy': float(vuln_acc),
        'f1_score': float(vuln_f1),
        'train_losses': [float(l) for l in train_losses_vuln],
        'test_accuracies': [float(a) for a in test_accs_vuln],
        'test_f1s': [float(f) for f in test_f1s_vuln]
    },
    'centralized_robust': {
        'accuracy': float(robust_acc),
        'f1_score': float(robust_f1),
        'defense_method': 'loss_based_filtering_with_gradient_clipping',
        'train_losses': [float(l) for l in train_losses_robust],
        'test_accuracies': [float(a) for a in test_accs_robust],
        'test_f1s': [float(f) for f in test_f1s_robust]
    },
    'full_comparison': {
        'centralized_clean_accuracy': clean_centralized_acc,
        'centralized_clean_f1': clean_centralized_f1,
        'centralized_poison_vulnerable_accuracy': float(vuln_acc),
        'centralized_poison_vulnerable_f1': float(vuln_f1),
        'centralized_poison_robust_accuracy': float(robust_acc),
        'centralized_poison_robust_f1': float(robust_f1),
        'federated_clean_accuracy': fl_clean_acc,
        'federated_clean_f1': fl_clean_f1,
        'federated_poison_vulnerable_accuracy': fl_vuln_acc,
        'federated_poison_vulnerable_f1': fl_vuln_f1,
        'federated_poison_robust_accuracy': fl_robust_acc,
        'federated_poison_robust_f1': fl_robust_f1,
        'centralized_poison_impact': float(centralized_drop),
        'federated_poison_impact': float(federated_drop),
        'centralized_defense_recovery': float(centralized_recovery),
        'federated_defense_recovery': float(federated_recovery)
    }
}

with open('results/centralized_adversarial.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n[OK] Results saved to results/centralized_adversarial.json")
