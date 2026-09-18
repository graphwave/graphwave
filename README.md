
# GraphWave: A Dynamic Context-Adaptive Multimodal Feature Fusion Framework for Threat Detection

## Introduction

Accurate network traffic classification is critical for defending against evolving cyber threats. However, mainstream methods relying solely on intra-flow features fail to distinguish malicious traffic with highly similar legitimate patterns, lacking contextual modeling and robustness to adversarial evasion. To address these limitations, this paper proposes **GraphWave**, a dynamic context-adaptive multimodal framework built on a novel heterogeneous Graph2Seq paradigm. It constructs maximal connected subgraphs leveraging attacker, target, and temporal contexts to capture key contextual correlations of attack chains. It integrates wavelet-enhanced dynamic graph attention networks for spatial context learning and Transformer encoders for long-range intra-flow temporal modeling. A multimodal cross-attention fusion mechanism aligns spatial and temporal representations to enhance discriminative feature integration. Extensive evaluations on six real-world datasets show GraphWave achieves a **99.13% F1-score**, outperforming state-of-the-art methods by **7.62%** on average. Theoretical analysis and empirical results validate its strong robustness against traffic obfuscation, temporal confusion, high intra-flow similarity, and low-and-slow evasion.

![image](https://github.com/graphwave/graphwave/blob/main/fig/framework.png)

## References
- [Towards Context-Aware Traffic Classification via Time-Wavelet Fusion Network](https://dl.acm.org/doi/10.1145/3690624.3709315), Ziming Zhao, Zhuoxue Song, Xiaofei Xie, Zhaoxuan Li, et al. - KDD 2025
- [Detecting Unknown Encrypted Malicious Traffic in Real Time via Flow Interaction Graph Analysis](https://www.ndss-symposium.org/ndss-paper/detecting-unknown-encrypted-malicious-traffic-in-real-time-via-flow-interaction-graph-analysis/), Chuanpu Fu, Qi Li, Ke Xu - NDSS 2023
- [Realtime Robust Malicious Traffic Detection via Frequency Domain Analysis](https://dl.acm.org/doi/10.1145/3460120.3484585), Chuanpu Fu, Qi Li, Meng Shen, Ke Xu - CCS 2021 (Whisper)
- [ET-BERT: A Contextualized Datagram Representation with Pre-training Transformers for Encrypted Traffic Classification](https://dl.acm.org/doi/10.1145/3485447.3512217), Xinjie Lin, et al. - WWW 2022
- [FlowLens: Enabling Efficient Flow Classification for ML-based Network Security Applications](https://www.ndss-symposium.org/ndss-paper/flowlens-enabling-efficient-flow-classification-for-ml-based-network-security-applications/), Diogo Barradas, et al. - NDSS 2021
- [FlowPrint: Semi-Supervised Mobile-App Fingerprinting on Encrypted Network Traffic](https://www.ndss-symposium.org/ndss-paper/flowprint-semi-supervised-mobile-app-fingerprinting-on-encrypted-network-traffic/), Thijs van Ede, et al. - NDSS 2020

> **Note**: For better readability and code organization, this repository has been reformatted and cleaned up with the assistance of a large language model (GLM-5.2-Flash). The core logic faithfully follows the original implementation used in the paper; if you encounter any issues, please open an issue on GitHub.
