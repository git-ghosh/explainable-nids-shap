# Explainable Network Intrusion Detection Using LightGBM and TreeSHAP

Official code, experimental configuration, and benchmark results for the paper:  
**"Explainable Network Intrusion Detection Using LightGBM and TreeSHAP: Accuracy, Explanation Faithfulness, and Real-Time Deployment Feasibility"**

**Author:** Antor Ghosh (Department of Computer Science and Engineering, Daffodil International University)

---

## Overview

This repository provides a reproducible, leakage-aware evaluation pipeline for explainable network intrusion detection systems (NIDS). The framework combines a class-balanced multiclass **LightGBM** detector with **TreeSHAP** in a tiered **explain-on-alert** architecture:

- **Leakage-Aware Protocol:** Cleaning, de-duplication, conflicting-label resolution, and removal of shortcut identifier attributes (such as `Destination Port` in CIC-IDS2017) prior to stratified splitting.
- **Tiered Explain-on-Alert Design:** Every network flow is scored in sub-millisecond time (`0.39 ms` median latency), while exact Shapley explanations are computed only on flagged alerts (`3.29 ms/alert`, **108× faster** than LIME).
- **SHAP-Guided Feature Reduction:** Selects compact feature subsets strictly from validation SHAP rankings, retaining **96.97% macro-F1** on CIC-IDS2017 using only **15 of 76 features** (80.26% feature reduction).
- **Quantitative XAI Evaluation:** Measures cross-split ranking stability (Kendall's $\tau$), agreement with split-gain and permutation importance, and top-$k$ deletion faithfulness across five stratified splits.

---

## Repository Structure

```text
explainable-nids-shap/
├── run_experiments.py      # Full reproducible training, evaluation, XAI & latency pipeline
├── results_cic/            # Metrics, tables, and figures for CIC-IDS2017
├── results_unsw/           # Metrics, tables, and figures for UNSW-NB15
└── README.md
