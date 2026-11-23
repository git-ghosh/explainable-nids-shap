cat << 'EOF' > README.md
# Explainable Network Intrusion Detection Using LightGBM and TreeSHAP

Reproducible implementation for the paper:
> **Explainable Network Intrusion Detection Using LightGBM and TreeSHAP: Accuracy, Explanation Faithfulness, and Real-Time Deployment Feasibility**

## Key Highlights
- **Leakage-Free Data Sanitization:** Removes non-finite records, exact duplicates, and feature-identical rows with conflicting class labels *prior* to stratified splitting, and excludes `Destination Port` by default to prevent port-number shortcut learning.
- **Multi-Model Benchmarking:** Evaluates class-weighted **LightGBM** against Logistic Regression, Decision Tree, Random Forest, MLP, and XGBoost across 5 stratified splits on **CIC-IDS2017** (`2,519,357` cleaned records, 11 classes, 76 features) and **UNSW-NB15** (`136,871` cleaned records, 10 classes, 34 features).
- **Verified Explainability:** Evaluates exact **TreeSHAP** attributions via multi-split deletion faithfulness curves (on stratified 25,000-flow test subsets using validation-only feature rankings), cross-split Kendall's $\tau$ and top-10 Jaccard stability, and validation-guided feature reduction.
- **Real-Time SOC Feasibility:** Single-thread median inference latency of `0.39–0.56 ms` (`2,020–2,942 flows/s`) and `3.61–7.65 ms` per-alert TreeSHAP attribution (`42x–126x` faster than LIME).

## Summary of Results (5-Split Mean ± Std)

| Dataset | Model | Macro-F1 (%) | MCC (%) | ROC-AUC (%) | Median Latency | TreeSHAP / Alert |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **CIC-IDS2017** (76 feat.) | **LightGBM** | **97.55 ± 0.07** | **99.71 ± 0.01** | **99.99 ± 0.01** | 0.56 ms | 4.15 ms |
| **CIC-IDS2017** (Top-15) | **LightGBM** | **97.17 ± 0.25** | **99.64 ± 0.02** | — | **0.48 ms** | **3.61 ms** |
| **UNSW-NB15** (34 feat.) | **LightGBM** | **67.48 ± 1.19** | **74.42 ± 0.19** | **98.27 ± 0.07** | **0.39 ms** | 7.65 ms |
| **UNSW-NB15** (Top-30) | **LightGBM** | **67.39 ± 1.03** | **74.45 ± 0.18** | — | 0.41 ms | **7.36 ms** |

## Installation & Usage

```bash
python -m venv nids-env
source nids-env/bin/activate  # On Windows: .\nids-env\Scripts\Activate.ps1
pip install lightgbm shap xgboost scikit-learn pandas numpy scipy matplotlib lime
