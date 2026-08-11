# federated_learning_simulation.py
# Simulates 5 edge nodes training collaboratively via Federated Learning
# with Differential Privacy for privacy-preserving threat intelligence sharing

import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import copy
import json
import os

# ============================================================
# LOCK ALL RANDOM SEEDS FOR REPRODUCIBILITY
# ============================================================
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# --- Configuration ---
NUM_CLIENTS = 5
FL_ROUNDS = 20
LOCAL_EPOCHS = 3
BATCH_SIZE = 64
LEARNING_RATE = 0.001
DP_EPSILON = 8.0
DP_DELTA = 1e-5
NOISE_MULTIPLIER = 1.0
CLIP_NORM = 1.0

os.makedirs('results', exist_ok=True)

# --- Model definition ---
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

# --- Load and partition data ---
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
    print(f"  Client {i+1}: {len(client_idx)} samples, "
          f"attack ratio: {y_train[client_idx].mean():.2%}")

X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=BATCH_SIZE, shuffle=False)

# --- Differential Privacy utilities ---
def add_dp_noise(gradients, noise_multiplier, clip_norm):
    noisy_grads = {}
    for name, grad in gradients.items():
        grad_norm = torch.norm(grad)
        if grad_norm > clip_norm:
            grad = grad * (clip_norm / grad_norm)
        noise = torch.normal(0, noise_multiplier * clip_norm, grad.shape).to(grad.device)
        noisy_grads[name] = grad + noise
    return noisy_grads

def compute_privacy_spent(noise_multiplier, steps, delta):
    epsilon = np.sqrt(2 * np.log(1.25 / delta)) / noise_multiplier * np.sqrt(steps)
    return epsilon

# --- Federated Averaging ---
def federated_averaging(client_models):
    avg_state = {}
    for key in client_models[0].keys():
        avg_state[key] = sum([m[key] for m in client_models]) / len(client_models)
    return avg_state

# --- Training function for a single client ---
def train_client(model, data, epochs, device, add_privacy=False):
    model.train()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    loader = DataLoader(TensorDataset(data['X'], data['y']), batch_size=BATCH_SIZE, shuffle=True)

    for _ in range(epochs):
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()

            if add_privacy:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
                for param in model.parameters():
                    if param.grad is not None:
                        noise = torch.normal(0, NOISE_MULTIPLIER * CLIP_NORM, param.grad.shape).to(device)
                        param.grad += noise

            optimizer.step()

    return copy.deepcopy(model.state_dict())

# --- Evaluation on global test set ---
def evaluate(model, test_loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            predicted = (outputs >= 0.5).float()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()
    return accuracy_score(all_labels, all_preds), f1_score(all_labels, all_preds)

# --- Main FL loop ---
print(f"\n[{'='*50}]")
print(f"FEDERATED LEARNING WITH DIFFERENTIAL PRIVACY")
print(f"Clients: {NUM_CLIENTS} | FL Rounds: {FL_ROUNDS} | Local Epochs: {LOCAL_EPOCHS}")
print(f"DP: epsilon≈{DP_EPSILON}, noise_multiplier={NOISE_MULTIPLIER}")
print(f"[{'='*50}]\n")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

input_dim = X_train.shape[1]
global_model = IntrusionDetector(input_dim).to(device)

fl_round_accuracies = []
fl_round_f1s = []

for round_num in range(1, FL_ROUNDS + 1):
    print(f"--- FL Round {round_num}/{FL_ROUNDS} ---")

    client_states = []

    for client_id in range(NUM_CLIENTS):
        local_model = IntrusionDetector(input_dim).to(device)
        local_model.load_state_dict(global_model.state_dict())

        state = train_client(local_model, client_data[client_id],
                            LOCAL_EPOCHS, device, add_privacy=True)
        client_states.append(state)

    new_global_state = federated_averaging(client_states)
    global_model.load_state_dict(new_global_state)

    acc, f1 = evaluate(global_model, test_loader, device)
    fl_round_accuracies.append(acc)
    fl_round_f1s.append(f1)
    print(f"  Global Model -> Accuracy: {acc:.4f}, F1: {f1:.4f}")

# --- Final evaluation ---
final_acc, final_f1 = evaluate(global_model, test_loader, device)

print(f"\n[{'='*50}]")
print(f"FEDERATED LEARNING RESULTS (With Differential Privacy)")
print(f"[{'='*50}]")
print(f"Final Test Accuracy: {final_acc:.4f}")
print(f"Final Test F1-Score: {final_f1:.4f}")
print(f"Privacy budget (approx epsilon): {compute_privacy_spent(NOISE_MULTIPLIER, FL_ROUNDS * LOCAL_EPOCHS * NUM_CLIENTS, DP_DELTA):.2f}")

global_model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        outputs = global_model(batch_x)
        predicted = (outputs >= 0.5).float()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())

all_preds = np.array(all_preds).flatten()
all_labels = np.array(all_labels).flatten()
print(f"\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=['BENIGN', 'ATTACK']))

results = {
    'model': 'federated_with_dp',
    'num_clients': NUM_CLIENTS,
    'fl_rounds': FL_ROUNDS,
    'local_epochs': LOCAL_EPOCHS,
    'dp_epsilon_approx': float(compute_privacy_spent(NOISE_MULTIPLIER, FL_ROUNDS * LOCAL_EPOCHS * NUM_CLIENTS, DP_DELTA)),
    'noise_multiplier': NOISE_MULTIPLIER,
    'clip_norm': CLIP_NORM,
    'final_accuracy': float(final_acc),
    'final_f1': float(final_f1),
    'round_accuracies': [float(a) for a in fl_round_accuracies],
    'round_f1s': [float(f) for f in fl_round_f1s]
}

with open('results/federated_dp.json', 'w') as f:
    json.dump(results, f, indent=2)

torch.save(global_model.state_dict(), 'results/fl_model_dp.pth')
print(f"\nResults saved to results/federated_dp.json")
