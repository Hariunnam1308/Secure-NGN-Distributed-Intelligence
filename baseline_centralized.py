# baseline_centralized.py
# Centralized intrusion detection model: baseline for FL comparison

import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import json
import os
import joblib

# ============================================================
# LOCK ALL RANDOM SEEDS FOR REPRODUCIBILITY
# ============================================================
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# --- Load data ---
df = pd.read_csv('ngn_traffic_data.csv')
X = df.drop(['label', 'attack_type'], axis=1).values.astype(np.float32)
y = df['label'].values.astype(np.float32)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save scaler for later use in FL
joblib.dump(scaler, 'scaler.pkl')

# Convert to PyTorch tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=64, shuffle=False)

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

# --- Training ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = IntrusionDetector(X_train.shape[1]).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 30
train_losses = []
test_accuracies = []

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)

    # Evaluate
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            predicted = (outputs >= 0.5).float()
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)

    acc = correct / total
    test_accuracies.append(acc)

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Test Accuracy: {acc:.4f}")

# --- Final evaluation ---
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

final_acc = accuracy_score(all_labels, all_preds)
final_f1 = f1_score(all_labels, all_preds)

print(f"\n{'='*50}")
print(f"CENTRALIZED BASELINE RESULTS")
print(f"{'='*50}")
print(f"Test Accuracy: {final_acc:.4f}")
print(f"Test F1-Score: {final_f1:.4f}")
print(f"\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=['BENIGN', 'ATTACK']))
print(f"\nConfusion Matrix:")
print(confusion_matrix(all_labels, all_preds))

# Save results
results = {
    'model': 'centralized_baseline',
    'accuracy': float(final_acc),
    'f1_score': float(final_f1),
    'epochs': epochs,
    'final_loss': float(train_losses[-1]),
    'train_losses': [float(l) for l in train_losses],
    'test_accuracies': [float(a) for a in test_accuracies]
}

os.makedirs('results', exist_ok=True)
with open('results/baseline_centralized.json', 'w') as f:
    json.dump(results, f, indent=2)

torch.save(model.state_dict(), 'results/baseline_model.pth')
print(f"\nResults saved to results/baseline_centralized.json")
print(f"Model saved to results/baseline_model.pth")
