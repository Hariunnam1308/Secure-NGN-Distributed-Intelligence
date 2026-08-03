
# Distributed Intelligence for Secure and Trustworthy Next-Generation Networks

This repository contains the implementation of a proof-of-concept framework for securing Next-Generation Networks (NGNs) using **Privacy-Preserving Federated Learning** and **Robust Adversarial Defenses**.

## 🛡️ Project Overview
This research investigates how Distributed Intelligence (DI) can address the scalability and privacy limitations of centralized security models. It implements:
- **Federated Learning (FL):** Collaborative training without raw data sharing.
- **Differential Privacy (DP):** Protecting client data through gradient clipping and noise.
- **Robust Aggregation:** Using Trimmed Mean to mitigate model poisoning attacks.

## 📂 Repository Structure
- `generate_dataset.py`: Creates a synthetic NGN traffic dataset.
- `baseline_centralized.py`: Establishes a performance benchmark.
- `federated_learning_simulation.py`: Implements FL with Differential Privacy (Objective 3).
- `adversarial_fl_defense.py`: Implements robust defense against poisoning (Objective 4).
- `generate_figures.py`: Generates visualization charts for evaluation.

## 🚀 How to Run
1. **Install Dependencies:**
   ```bash
   pip install numpy pandas scikit-learn torch matplotlib seaborn joblib
   ```
2. **Generate Data:**
   ```bash
   python generate_dataset.py
   ```
3. **Run Experiments:**
   Execute the scripts in order to see the results for centralized, federated, and adversarial scenarios.

## 📊 Key Findings
The study successfully demonstrates that **Trimmed Mean Aggregation** significantly improves the resilience of FL models against poisoning attacks, while **Differential Privacy** provides a quantifiable privacy-utility balance for NGN edge nodes.

## 🎓 Academic Context
This project was completed as part of an MSc Research Dissertation (June - August 2026). Full implementation details and evaluation can be found in the accompanying thesis chapters.
```
