# Chapter 2: Related Works & Baseline Comparisons

## 1. SCARFF Framework (2023) Comparison

The Scalable and Robust Federated Fraud (SCARFF) framework (Murshed et al., 2023) represents a key prior work in federated fraud detection for financial institutions. While SCARFF focuses on traditional banking systems, our proposed architecture adapts these principles specifically for the Mobile Money (MoMo) ecosystem in Ghana.

| Feature | SCARFF Framework (2023) | Proposed 3-Tier MoMo Architecture |
|---------|-------------------------|-----------------------------------|
| **Primary Domain** | Traditional Banking & Credit Cards | Mobile Money (MTN, Telecel, AirtelTigo) |
| **Topology** | 2-Tier (Client -> Server) | 3-Tier (Edge Device -> Provider -> Global) |
| **Data Scope** | Tabular transaction data | SMS Text (CNN) + Call Logs (MLP) |
| **Aggregation** | FedAvg, FedProx | FedProx, Trimmed Mean, Multi-Krum |
| **Interception** | Centralized flagging | Decentralized Escrow at Provider layer |
| **Privacy Mechanism** | Federated Learning only | FL + Differential Privacy (DP-SGD) |

The integration of **Provider-level aggregation** and **Escrow Interception** directly addresses the fragmented nature of Ghana's telecom network, a limitation not solved by standard 2-tier architectures like SCARFF. By aggregating locally at the provider level first (Tier 2) before global aggregation (Tier 1), our model reduces cross-network latency and isolates byzantine edge nodes before they corrupt the global weights.

## 2. Extreme Gradient Boosting (XGBoost) and Random Forest Baselines

Deep learning models like the CNN and MLP deployed in our edge devices require significant tuning and computational overhead compared to traditional machine learning baselines. 

To justify the use of deep neural networks, we compare their false positive mitigation against two robust tree-based classifiers:
- **Random Forest (RF)**
- **Extreme Gradient Boosting (XGBoost)**

### Rationale
In highly imbalanced datasets like fraud detection (where < 1% of transactions are fraudulent), RF and XGBoost often outperform standard MLPs due to their ensemble nature and intrinsic handling of class imbalance. However, they lack the ability to be efficiently trained via standard Federated Learning algorithms (like FedAvg) since they are non-parametric and do not rely on gradient descent in the same way neural networks do.

Our architecture implements a local evaluation pipeline (`train_baselines()`) allowing each edge device to locally train an RF and XGBoost model on their specific data. While these models serve as powerful local predictors, their inability to securely aggregate weights across the MTN, Telecel, and AirtelTigo networks without sharing raw data makes them unsuitable for the global federated ecosystem. Thus, the DL models (CNN/MLP) remain the primary mechanism for federated intelligence sharing.
