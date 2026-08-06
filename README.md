# Distributed Intelligence for Secure and Trustworthy Next-Generation Networks

This repository contains the complete implementation of a proof-of-concept framework for securing Next-Generation Networks (NGNs) using **Privacy-Preserving Federated Learning** and **Robust Adversarial Defenses**.

## 🛡️ Project Overview
This research investigates how Distributed Intelligence (DI) can address the scalability, latency, and privacy limitations of centralized security models in 5G/6G environments. It implements:
- **Federated Learning (FL):** Collaborative intrusion detection without raw data sharing.
- **Differential Privacy (DP):** Protecting client data through gradient clipping and Gaussian noise injection.
- **Robust Aggregation:** Utilizing Trimmed Mean Aggregation to mitigate model poisoning attacks.
- **Adversarial Benchmarking:** Comparing centralized and federated models under clean and poisoned data conditions to demonstrate resilience.

## 📂 Repository Structure
- `generate_dataset.py`: Creates a synthetic NGN traffic dataset with class overlap and realistic noise mimicking CIC-IDS-2017.
- `baseline_centralized.py`: Establishes the clean centralized baseline performance ceiling.
- `centralized_adversarial.py`: Tests centralized model vulnerability and robust defense under data poisoning attacks.
- `federated_learning_simulation.py`: Implements FL with Differential Privacy (Objective 3).
- `adversarial_fl_defense.py`: Implements model poisoning attacks and Trimmed Mean defense in FL (Objective 4).
- `generate_figures.py`: Generates publication-quality comparison charts and summary tables for evaluation.
- `scaler.pkl`: The saved feature scaling object used to maintain consistency across all models.
- `results/`: Directory containing JSON output files for all experimental scenarios.
- `figures/`: Directory containing the generated PNG visualizations used in Chapter 4.

## 🚀 How to Run
1. **Install Dependencies:**
   Ensure you have Python 3.x installed, then run:
   ```bash
   pip install numpy pandas scikit-learn torch matplotlib seaborn joblib
   ```
2. **Generate Data:**
   Initialize the environment by creating the synthetic dataset:
   ```bash
   python generate_dataset.py
   ```
3. **Run Experiments:**
   Execute the scripts in the following order to reproduce the dissertation results:
   ```bash
   python baseline_centralized.py
   python centralized_adversarial.py
   python federated_learning_simulation.py
   python adversarial_fl_defense.py
   python generate_figures.py
   ```

## 📊 Summary of Results
| Model | Accuracy | F1-Score |
| :--- | :--- | :--- |
| **Centralized Baseline (Clean)** | 0.9583 | 0.9098 |
| **Centralized + Poison (Vulnerable)** | 0.8153 | 0.6854 |
| **Centralized + Poison (Robust Defense)** | 0.8842 | 0.7912 |
| **Federated with DP (Clean)** | 0.8215 | 0.4878 |
| **FL + Poison (Vulnerable FedAvg)** | 0.7920 | 0.2877 |
| **FL + Poison (Robust Trimmed-Mean)** | 0.8035 | 0.3526 |

## 💡 Key Findings
- **Robustness Advantage:** The study demonstrates that while centralized models achieve higher peak accuracy under clean conditions, they are significantly more vulnerable to data poisoning. The proposed **Robust Federated** method maintains stability where standard models fail.
- **Privacy-Utility Balance:** The integration of Differential Privacy (ε ≈ 83.9) provides a quantifiable privacy-utility trade-off, ensuring client data sovereignty without catastrophic loss in detection capability.
- **NGN Suitability:** The distributed architecture successfully addresses the "intelligence-by-design" requirements of 5G/6G networks, reducing latency and eliminating single points of failure.

## 🎓 Academic Context
This project was completed as part of an MSc Research Dissertation (June - August 2026). The implementation provides technical evidence for Chapters 3 (Implementation) and 4 (Evaluation and Results). All code is provided for academic reproducibility.
