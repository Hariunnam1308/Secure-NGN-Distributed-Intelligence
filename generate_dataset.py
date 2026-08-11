# generate_dataset.py
# Creates a realistic network traffic dataset with class overlap
# for the NGN Federated Learning proof-of-concept

import numpy as np
import pandas as pd

np.random.seed(42)

N_SAMPLES = 20000
N_FEATURES = 30

feature_names = [
    'flow_duration', 'total_fwd_packets', 'total_bwd_packets',
    'fwd_packet_length_max', 'fwd_packet_length_min', 'fwd_packet_length_mean',
    'bwd_packet_length_max', 'bwd_packet_length_min', 'bwd_packet_length_mean',
    'flow_bytes_per_sec', 'flow_packets_per_sec', 'flow_iat_mean',
    'flow_iat_std', 'flow_iat_max', 'flow_iat_min',
    'fwd_iat_total', 'fwd_iat_mean', 'fwd_iat_std',
    'bwd_iat_total', 'bwd_iat_mean', 'bwd_iat_std',
    'fwd_psh_flags', 'bwd_psh_flags', 'fwd_urg_flags',
    'bwd_urg_flags', 'fwd_header_length', 'bwd_header_length',
    'packet_length_mean', 'packet_length_std', 'packet_length_variance'
]

# Benign traffic: broad Gaussian cluster
benign_count = int(N_SAMPLES * 0.75)
benign = np.random.multivariate_normal(
    mean=np.random.uniform(20, 60, N_FEATURES),
    cov=np.eye(N_FEATURES) * np.random.uniform(15, 40, N_FEATURES),
    size=benign_count
)

# Attack traffic: shifted distributions, but with OVERLAP to prevent trivial separation
attack_types = ['DDoS', 'Botnet', 'BruteForce', 'Infiltration', 'PortScan']
attack_counts = [1600, 1400, 1000, 600, 400]

attacks = []
for i, (attack_type, count) in enumerate(zip(attack_types, attack_counts)):
    mean_shift = np.random.uniform(15, 40, N_FEATURES) * (i + 1) * 0.4
    cov_scale = np.random.uniform(3, 8)
    attack_data = np.random.multivariate_normal(
        mean=benign.mean(axis=0) + mean_shift,
        cov=np.eye(N_FEATURES) * cov_scale * 8,
        size=count
    )
    # Add overlap: mix in some benign-like samples within each attack class
    overlap_idx = np.random.choice(count, size=int(count * 0.15), replace=False)
    attack_data[overlap_idx] = np.random.multivariate_normal(
        mean=benign.mean(axis=0),
        cov=np.eye(N_FEATURES) * 20,
        size=len(overlap_idx)
    )
    attacks.append(attack_data)

attack_data = np.vstack(attacks)
attack_labels_raw = np.concatenate([[at] * c for at, c in zip(attack_types, attack_counts)])

# Combine
X = np.vstack([benign, attack_data])
y_raw = np.array(['BENIGN'] * benign_count + list(attack_labels_raw))

# Add realistic noise
X += np.random.normal(0, 5, X.shape)
X = np.abs(X)

# Make some benign samples look slightly suspicious (false positive challenge)
ambiguous_idx = np.random.choice(benign_count, size=int(benign_count * 0.08), replace=False)
X[ambiguous_idx] += np.random.normal(25, 10, (len(ambiguous_idx), N_FEATURES))
X = np.abs(X)

y = (y_raw != 'BENIGN').astype(int)

df = pd.DataFrame(X, columns=feature_names)
df['label'] = y
df['attack_type'] = y_raw
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv('ngn_traffic_data.csv', index=False)

print(f"Dataset created: {df.shape[0]} samples, {df.shape[1]-2} features")
print(f"Benign: {(df['label']==0).sum()} ({((df['label']==0).sum()/len(df)*100):.1f}%)")
print(f"Attack: {(df['label']==1).sum()} ({((df['label']==1).sum()/len(df)*100):.1f}%)")
print(f"Attack types: {df['attack_type'].value_counts().to_dict()}")
print("(Added ~15% class overlap and ~8% ambiguous benign samples for realism)")
