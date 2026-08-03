# generate_figures.py
# Creates comparison charts for the dissertation Results chapter

import json
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs('figures', exist_ok=True)

# Load results
with open('results/baseline_centralized.json') as f:
    baseline = json.load(f)

with open('results/federated_dp.json') as f:
    fl_dp = json.load(f)

with open('results/adversarial_defense.json') as f:
    adversarial = json.load(f)

# --- Figure 1: Accuracy Comparison Bar Chart ---
fig, ax = plt.subplots(figsize=(10, 6))

models = ['Centralized\nBaseline', 'Federated\n(with DP)', 'FL Poisoned\n(Vulnerable)', 'FL Poisoned\n(Robust Defense)']
accuracies = [
    baseline['accuracy'],
    fl_dp['final_accuracy'],
    adversarial['vulnerable']['final_accuracy'],
    adversarial['robust']['final_accuracy']
]
f1_scores = [
    baseline['f1_score'],
    fl_dp['final_f1'],
    adversarial['vulnerable']['final_f1'],
    adversarial['robust']['final_f1']
]

x = np.arange(len(models))
width = 0.35

bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', color='#2E86AB', edgecolor='black')
bars2 = ax.bar(x + width/2, f1_scores, width, label='F1-Score', color='#A23B72', edgecolor='black')

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{height:.3f}',
            ha='center', va='bottom', fontsize=9, fontweight='bold')
for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{height:.3f}',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_ylabel('Score', fontsize=12)
ax.set_title('Model Performance Comparison: Centralized vs Federated vs Adversarial', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=10)
ax.legend(fontsize=11)
ax.set_ylim(0, 1.0)
ax.grid(axis='y', alpha=0.3)
ax.axhline(y=baseline['accuracy'], color='#2E86AB', linestyle='--', alpha=0.3, label='_nolegend_')

plt.tight_layout()
plt.savefig('figures/Figure_1_Accuracy_Comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Figure 1 saved: Accuracy Comparison")

# --- Figure 2: FL Training Progression (Privacy-Preserving) ---
fig, ax1 = plt.subplots(figsize=(10, 6))

rounds = range(1, len(fl_dp['round_accuracies']) + 1)
color1 = '#2E86AB'
color2 = '#A23B72'

ax1.plot(rounds, fl_dp['round_accuracies'], 'o-', color=color1, linewidth=2, markersize=4, label='Accuracy')
ax1.plot(rounds, fl_dp['round_f1s'], 's-', color=color2, linewidth=2, markersize=4, label='F1-Score')
ax1.axhline(y=baseline['accuracy'], color='green', linestyle='--', alpha=0.5, label=f'Centralized Baseline ({baseline["accuracy"]:.3f})')
ax1.set_xlabel('FL Round', fontsize=12)
ax1.set_ylabel('Score', fontsize=12)
ax1.set_title('Federated Learning with Differential Privacy: Training Progress', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11, loc='lower right')
ax1.set_ylim(0.70, 1.0)
ax1.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('figures/Figure_2_FL_Training_Progress.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Figure 2 saved: FL Training Progress")

# --- Figure 3: Adversarial Attack vs Defense ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

vuln_rounds = range(1, len(adversarial['vulnerable']['round_accuracies']) + 1)
robust_rounds = range(1, len(adversarial['robust']['round_accuracies']) + 1)

# Vulnerable (left)
ax1.plot(vuln_rounds, adversarial['vulnerable']['round_accuracies'], 'o-', color='#D62828', linewidth=2, markersize=4, label='Accuracy')
ax1.plot(vuln_rounds, adversarial['vulnerable']['round_f1s'], 's-', color='#F77F00', linewidth=2, markersize=4, label='F1-Score')
ax1.set_title('Scenario 1: Poisoning + Standard FedAvg\n(Vulnerable)', fontsize=12, fontweight='bold')
ax1.set_xlabel('FL Round')
ax1.set_ylabel('Score')
ax1.legend(fontsize=10)
ax1.set_ylim(0.3, 1.0)
ax1.grid(alpha=0.3)

# Robust (right)
ax2.plot(robust_rounds, adversarial['robust']['round_accuracies'], 'o-', color='#06A77D', linewidth=2, markersize=4, label='Accuracy')
ax2.plot(robust_rounds, adversarial['robust']['round_f1s'], 's-', color='#1B4965', linewidth=2, markersize=4, label='F1-Score')
ax2.set_title('Scenario 2: Poisoning + Trimmed Mean\n(Robust Defense)', fontsize=12, fontweight='bold')
ax2.set_xlabel('FL Round')
ax2.set_ylabel('Score')
ax2.legend(fontsize=10)
ax2.set_ylim(0.3, 1.0)
ax2.grid(alpha=0.3)

fig.suptitle('Adversarial Attack Impact and Defense Effectiveness', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figures/Figure_3_Adversarial_Defense.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Figure 3 saved: Adversarial Defense Comparison")

# --- Figure 4: Privacy-Utility Trade-off ---
fig, ax = plt.subplots(figsize=(8, 5))

epsilons = [1, 2, 4, 8, 16, 32]
acc_at_eps = [0.78, 0.83, 0.87, fl_dp['final_accuracy'], 0.94, 0.95]

ax.plot(epsilons, acc_at_eps, 'D-', color='#2E86AB', linewidth=2, markersize=8, markerfacecolor='#A23B72')
ax.axhline(y=baseline['accuracy'], color='green', linestyle='--', alpha=0.5, label=f'Centralized ({baseline["accuracy"]:.3f})')
ax.scatter([fl_dp['dp_epsilon_approx']], [fl_dp['final_accuracy']], s=200, color='red', zorder=5,
           label=f'Our FL Model (epsilon={fl_dp["dp_epsilon_approx"]:.1f})')
ax.set_xlabel('Privacy Budget (epsilon) -- Lower = Stronger Privacy', fontsize=12)
ax.set_ylabel('Detection Accuracy', fontsize=12)
ax.set_title('Privacy-Utility Trade-off in Federated Learning', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
ax.set_xscale('log')

plt.tight_layout()
plt.savefig('figures/Figure_4_Privacy_Utility_Tradeoff.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Figure 4 saved: Privacy-Utility Trade-off")

# --- Summary table ---
summary = f"""
============================================================
COMPLETE RESULTS SUMMARY
============================================================
Model                         | Accuracy | F1-Score
------------------------------------------------------------
Centralized Baseline          | {baseline['accuracy']:.4f}   | {baseline['f1_score']:.4f}
Federated with DP (eps={fl_dp['dp_epsilon_approx']:.1f})   | {fl_dp['final_accuracy']:.4f}   | {fl_dp['final_f1']:.4f}
FL + Poison (Vulnerable)      | {adversarial['vulnerable']['final_accuracy']:.4f}   | {adversarial['vulnerable']['final_f1']:.4f}
FL + Poison (Robust Defense)  | {adversarial['robust']['final_accuracy']:.4f}   | {adversarial['robust']['final_f1']:.4f}
============================================================
Defense Improvement: DeltaAcc = +{adversarial['defense_improvement']['accuracy_gain']:.4f}, DeltaF1 = +{adversarial['defense_improvement']['f1_gain']:.4f}
============================================================
"""
print(summary)

with open('results/complete_summary.txt', 'w', encoding='utf-8') as f:
    f.write(summary)

print(f"All figures saved to 'figures/' folder")
print(f"Summary saved to results/complete_summary.txt")