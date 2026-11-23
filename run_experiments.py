#!/usr/bin/env python3
"""
Reproducible experiments for the paper
  "Explainable Network Intrusion Detection Using LightGBM and TreeSHAP:
   Accuracy, Explanation Faithfulness, and Real-Time Deployment Feasibility"
"""
import argparse
import json
import pickle
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import kendalltau, loguniform, randint, uniform
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             matthews_corrcoef,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                     train_test_split)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

import lightgbm as lgb
import shap
import xgboost as xgb

warnings.filterwarnings("ignore")

def _xgb_dev():
    try:
        xgb.XGBClassifier(tree_method="hist", device="cuda", n_estimators=2).fit([[0,1],[1,0]], [0,1])
        return "cuda"
    except Exception:
        return "cpu"

XGB_DEV = _xgb_dev()

CIC_ID_COLS = ["Flow ID", "Source IP", "Source Port", "Destination IP", "Timestamp"]
DST_PORT = "Destination Port"


# --------------------------------------------------------------------------- #
# Data loading and cleaning (leakage-aware: clean + de-duplicate BEFORE splitting)
# --------------------------------------------------------------------------- #
def load_cicids(path, min_count, keep_port):
    files = sorted(Path(path).glob("*.csv"))
    if not files:
        sys.exit(f"No CSV files found in {path}")
    df = pd.concat([pd.read_csv(f, encoding="latin1", low_memory=False) for f in files],
                   ignore_index=True)
    stats = {"raw_records": int(len(df))}
    df.columns = df.columns.str.strip()
    # CICFlowMeter writes 'Fwd Header Length' twice; pandas renames the copy to '.1'
    for c in [c for c in df.columns if c.endswith(".1")]:
        if c[:-2] in df.columns and df[c].equals(df[c[:-2]]):
            df = df.drop(columns=c)
    label = df["Label"].astype(str).str.strip()
    label = label.where(~label.str.contains("Web Attack", na=False), "Web Attack")
    df = df.drop(columns=["Label"] + [c for c in CIC_ID_COLS if c in df.columns])
    df = df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    df["__label__"] = label.values

    n_before_nan = len(df)
    df = df.dropna()
    stats["dropped_nan_inf_rows"] = int(n_before_nan - len(df))
    stats["after_nan_inf_removal"] = int(len(df))

    n_before_dup = len(df)
    df = df.drop_duplicates()
    stats["dropped_exact_duplicates"] = int(n_before_dup - len(df))
    stats["after_deduplication"] = int(len(df))

    # Remove feature-identical rows that carry conflicting class labels
    feats = [c for c in df.columns if c != "__label__"]
    conflict_mask = df.duplicated(subset=feats, keep=False)
    stats["conflicting_label_rows"] = int(conflict_mask.sum())
    df = df[~conflict_mask].reset_index(drop=True)
    stats["after_conflict_removal"] = int(len(df))

    counts = df["__label__"].value_counts()
    dropped = counts[counts < min_count].index.tolist()
    stats["dropped_min_count_rows"] = int(df["__label__"].isin(dropped).sum())
    df = df[~df["__label__"].isin(dropped)]
    stats["classes_dropped_min_count"] = dropped
    stats["final_cleaned_records"] = int(len(df))

    y = df["__label__"].values
    X = df.drop(columns="__label__").astype(np.float32).reset_index(drop=True)
    if not keep_port and DST_PORT in X.columns:
        stats["note"] = "Destination Port excluded from default feature set"
    return X, y, "BENIGN", stats


def load_unsw(path, min_count):
    files = sorted(Path(path).glob("*training-set*.csv")) + sorted(Path(path).glob("*testing-set*.csv"))
    if not files:
        sys.exit(f"No UNSW-NB15 training/testing-set CSV files found in {path}")
    df = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)
    stats = {"raw_records": int(len(df))}
    df.columns = df.columns.str.strip()
    y = df["attack_cat"].astype(str).str.strip()
    df = df.drop(columns=[c for c in ["id", "label", "attack_cat"] if c in df.columns])
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    if cat_cols:  # label-free ordinal encoding of proto / service / state
        df[cat_cols] = OrdinalEncoder(handle_unknown="use_encoded_value",
                                      unknown_value=-1).fit_transform(df[cat_cols].astype(str))
    df = df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    df["__label__"] = y.values

    n_before_nan = len(df)
    df = df.dropna()
    stats["dropped_nan_inf_rows"] = int(n_before_nan - len(df))
    stats["after_nan_inf_removal"] = int(len(df))

    n_before_dup = len(df)
    df = df.drop_duplicates()
    stats["dropped_exact_duplicates"] = int(n_before_dup - len(df))
    stats["after_deduplication"] = int(len(df))

    # Remove feature-identical rows that carry conflicting class labels
    feats = [c for c in df.columns if c != "__label__"]
    conflict_mask = df.duplicated(subset=feats, keep=False)
    stats["conflicting_label_rows"] = int(conflict_mask.sum())
    df = df[~conflict_mask].reset_index(drop=True)
    stats["after_conflict_removal"] = int(len(df))

    counts = df["__label__"].value_counts()
    dropped = counts[counts < min_count].index.tolist()
    stats["dropped_min_count_rows"] = int(df["__label__"].isin(dropped).sum())
    df = df[~df["__label__"].isin(dropped)]
    stats["classes_dropped_min_count"] = dropped
    stats["final_cleaned_records"] = int(len(df))

    return (df.drop(columns="__label__").astype(np.float32).reset_index(drop=True),
            df["__label__"].values, "Normal", stats)


def stratified_sample(X, y, n, seed=0, floor=200):
    """Proportional stratified sample of ~n rows; every class keeps >= floor rows (or all of it)."""
    if n is None or len(X) <= n:
        return X, y
    rng = np.random.RandomState(seed)
    frac, idx = n / len(X), []
    for k in np.unique(y):
        ii = np.where(y == k)[0]
        m = max(int(len(ii) * frac), min(len(ii), floor))
        idx.append(rng.choice(ii, m, replace=False))
    idx = np.sort(np.concatenate(idx))
    return X.iloc[idx].reset_index(drop=True), y[idx]


# --------------------------------------------------------------------------- #
# Models and metrics
# --------------------------------------------------------------------------- #
def lgbm(best, seed, n_jobs, **over):
    params = dict(class_weight="balanced", n_jobs=n_jobs, verbose=-1, random_state=seed)
    params.update(best or {})
    params.update(over)
    return lgb.LGBMClassifier(**params)


def model_zoo(seed, n_jobs):
    """(estimator, search space). Identical tuning budget for every model."""
    return {
        "LogReg": (make_pipeline(StandardScaler(),
                                 LogisticRegression(max_iter=300, class_weight="balanced")),
                   {"logisticregression__C": loguniform(1e-3, 1e2)}),
        "DecisionTree": (DecisionTreeClassifier(class_weight="balanced", random_state=seed),
                         {"max_depth": [10, 20, 30, None], "min_samples_leaf": [1, 2, 5, 10]}),
        "RandomForest": (RandomForestClassifier(class_weight="balanced", n_jobs=n_jobs, random_state=seed),
                         {"n_estimators": [100, 200, 300], "max_depth": [None, 20, 40],
                          "min_samples_leaf": [1, 2, 4], "max_features": ["sqrt", 0.5]}),
        "XGBoost": (xgb.XGBClassifier(tree_method="hist", device=XGB_DEV, n_jobs=n_jobs, random_state=seed),
                    {"n_estimators": randint(200, 501), "max_depth": randint(4, 13),
                     "learning_rate": loguniform(0.02, 0.3), "subsample": uniform(0.6, 0.4),
                     "colsample_bytree": uniform(0.5, 0.5)}),
        "MLP": (make_pipeline(StandardScaler(),
                              MLPClassifier(early_stopping=True, max_iter=100, random_state=seed)),
                {"mlpclassifier__hidden_layer_sizes": [(64,), (128,), (128, 64)],
                 "mlpclassifier__alpha": loguniform(1e-5, 1e-2),
                 "mlpclassifier__learning_rate_init": loguniform(1e-4, 1e-2)}),
        "LightGBM": (lgbm(None, seed, n_jobs),
                     {"num_leaves": randint(31, 256), "learning_rate": loguniform(0.02, 0.3),
                      "n_estimators": randint(200, 601), "min_child_samples": randint(10, 101),
                      "subsample": uniform(0.6, 0.4), "subsample_freq": [1],
                      "colsample_bytree": uniform(0.5, 0.5), "reg_lambda": uniform(0, 10)}),
    }


def fit_model(name, est, X, y):
    kw = {"sample_weight": compute_sample_weight("balanced", y)} if name == "XGBoost" else {}
    t = time.perf_counter()
    est.fit(X, y, **kw)
    return time.perf_counter() - t


def evaluate(y_true, y_pred, y_proba, K, benign_idx):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=list(range(K)),
                                                 average="macro", zero_division=0)
    res = dict(accuracy=accuracy_score(y_true, y_pred), macro_precision=p, macro_recall=r,
               macro_f1=f, weighted_f1=f1_score(y_true, y_pred, average="weighted", zero_division=0),
               mcc=matthews_corrcoef(y_true, y_pred))
    yt, yp = y_true != benign_idx, y_pred != benign_idx
    tp, fn = (yt & yp).sum(), (yt & ~yp).sum()
    fp, tn = (~yt & yp).sum(), (~yt & ~yp).sum()
    res["detection_rate"] = tp / max(tp + fn, 1)
    res["false_alarm_rate"] = fp / max(fp + tn, 1)
    try:
        res["roc_auc_ovr"] = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro",
                                           labels=list(range(K)))
    except Exception:
        res["roc_auc_ovr"] = np.nan
    return res


def pct(m, s=None):
    return f"{100 * m:.2f}" if s is None else f"{100 * m:.2f} ± {100 * s:.2f}"


# --------------------------------------------------------------------------- #
# SHAP helpers
# --------------------------------------------------------------------------- #
def shap_class_first(sv):
    """Return SHAP values as (classes, samples, features) for any shap version."""
    if isinstance(sv, list):
        return np.stack(sv, axis=0)
    sv = np.asarray(sv)
    return np.transpose(sv, (2, 0, 1)) if sv.ndim == 3 else sv[None]


def shap_values(model, X):
    expl = shap.TreeExplainer(model)
    return expl, shap_class_first(expl.shap_values(X, check_additivity=False))


def global_importance(sv):
    return np.abs(sv).mean(axis=(0, 1))


def save_bar(imp, cols, path, title, top=15):
    order = np.argsort(-imp)[:top][::-1]
    plt.figure(figsize=(3.4, 3.4))
    plt.barh(np.array(cols)[order], imp[order], color="#1f4e79")
    plt.xlabel("mean |SHAP value| (all classes)")
    plt.title(title, fontsize=8)
    plt.tick_params(labelsize=6)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def save_class_heatmap(sv, cols, classes, path, top=12):
    per_class = np.abs(sv).mean(axis=1)  # (K, F)
    order = np.argsort(-per_class.sum(0))[:top]
    plt.figure(figsize=(3.6, 3.6))
    plt.imshow(per_class[:, order].T, aspect="auto", cmap="viridis")
    plt.yticks(range(top), np.array(cols)[order], fontsize=6)
    plt.xticks(range(len(classes)), classes, rotation=60, ha="right", fontsize=6)
    plt.colorbar(label="mean |SHAP|")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def save_waterfall(expl, sv, Xs, i, c, cols, path):
    ev = np.atleast_1d(expl.expected_value)
    e = shap.Explanation(values=sv[c][i], base_values=float(ev[c if len(ev) > 1 else 0]),
                         data=Xs.iloc[i].values, feature_names=list(cols))
    plt.figure()
    shap.plots.waterfall(e, max_display=10, show=False)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close("all")


# --------------------------------------------------------------------------- #
# Main experiment
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cicids2017", "unsw"], required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--params-json", default=None)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--n-iter", type=int, default=20, help="random-search iterations per model")
    ap.add_argument("--tune-rows", type=int, default=100000)
    ap.add_argument("--shap-n", type=int, default=5000)
    ap.add_argument("--faith-n", type=int, default=25000, help="proportional stratified test flows for deletion test")
    ap.add_argument("--min-class-count", type=int, default=50)
    ap.add_argument("--keep-dst-port", action="store_true")
    ap.add_argument("--eps", type=float, default=0.005, help="tolerance for choosing k*")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args()
    if a.fast:
        a.max_rows, a.seeds, a.n_iter, a.tune_rows, a.shap_n, a.faith_n = 30000, 2, 3, 10000, 500, 5000
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tick = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

    # ---- data ------------------------------------------------------------- #
    tick("loading and cleaning data")
    if a.dataset == "cicids2017":
        X_all, y_str, benign, stats = load_cicids(a.data, a.min_class_count, a.keep_dst_port)
    else:
        X_all, y_str, benign, stats = load_unsw(a.data, a.min_class_count)
    le = LabelEncoder()
    y_all = le.fit_transform(y_str)
    classes, K = list(le.classes_), len(le.classes_)
    benign_idx = classes.index(benign)
    X_all, y_all = stratified_sample(X_all, y_all, a.max_rows, seed=0)
    use_port = a.keep_dst_port or a.dataset != "cicids2017"
    feats = [c for c in X_all.columns if c != DST_PORT or use_port]
    X = X_all[feats]
    stats.update(n_used=int(len(X)), n_features=len(feats), classes=classes,
                 class_counts={c: int((y_all == i).sum()) for i, c in enumerate(classes)})
    json.dump(stats, open(out / "dataset_stats.json", "w"), indent=2, default=str)
    pd.Series(stats["class_counts"], name="records").to_csv(out / "table_class_counts.csv")
    tick(f"Cleaning summary: {stats}")
    tick(f"{len(X)} records, {len(feats)} features, {K} classes: {classes}")

    # ---- hyper-parameter search on seed-0 training data (equal budget) ------ #
    if a.params_json and Path(a.params_json).exists():
        best = json.load(open(a.params_json))
        tick(f"loaded pre-tuned params from {a.params_json} (XGB device: {XGB_DEV})")
    else:
        tick("hyper-parameter search")
        Xtr0, _, ytr0, _ = train_test_split(X, y_all, test_size=0.2, stratify=y_all, random_state=0)
        Xt, yt = stratified_sample(Xtr0.reset_index(drop=True), ytr0, a.tune_rows, seed=0, floor=60)
        best = {}
        for name, (est, space) in model_zoo(0, a.n_jobs).items():
            kw = {"sample_weight": compute_sample_weight("balanced", yt)} if name == "XGBoost" else {}
            rs = RandomizedSearchCV(est, space, n_iter=a.n_iter, scoring="f1_macro", refit=False,
                                    cv=StratifiedKFold(3, shuffle=True, random_state=0), random_state=0)
            rs.fit(Xt, yt, **kw)
            best[name] = rs.best_params_
            tick(f"  {name}: cv macro-F1 {rs.best_score_:.4f}")
    json.dump(best, open(out / "best_params.json", "w"), indent=2, default=str)

    # ---- main loop over stratified splits ----------------------------------- #
    det_rows, cls_rows, abl_rows, fs_rows, faith_rows, rank_val, kstars = [], [], [], [], [], [], []
    ks_grid = [k for k in ([5, 10] if a.fast else [5, 10, 15, 20, 30]) if k < len(feats)]
    ks_f = [k for k in [1, 2, 3, 5, 8, 10, 15, 20] if k <= len(feats)]
    final = {}  # seed-0 artefacts reused by the explanation analyses

    for seed in range(a.seeds):
        tick(f"split seed {seed}")
        Xtr, Xte, ytr, yte = train_test_split(X, y_all, test_size=0.2, stratify=y_all, random_state=seed)
        share = np.bincount(ytr, minlength=K) / len(ytr)
        minority = [k for k in range(K) if share[k] < 0.01]

        # 1) detection performance of all models
        lgb_model = None
        for name, (est, _) in model_zoo(seed, a.n_jobs).items():
            m = clone(est).set_params(**best[name])
            ft = fit_model(name, m, Xtr, ytr)
            pred, proba = m.predict(Xte), m.predict_proba(Xte)
            det_rows.append(dict(seed=seed, model=name, fit_time_s=ft,
                                 **evaluate(yte, pred, proba, K, benign_idx)))
            f1c = f1_score(yte, pred, labels=list(range(K)), average=None, zero_division=0)
            cls_rows += [dict(seed=seed, model=name, cls=classes[k], f1=f1c[k]) for k in range(K)]
            if name == "LightGBM":
                lgb_model = m
                final.setdefault("model", {})[seed] = m

        # 2) ablation of the LightGBM design choices
        variants = {"A: default hyper-parameters, no class weights": lgbm({}, seed, a.n_jobs, class_weight=None),
                    "B: tuned, no class weights": lgbm(best["LightGBM"], seed, a.n_jobs, class_weight=None),
                    "C: tuned + class weights (proposed)": lgbm(best["LightGBM"], seed, a.n_jobs)}
        for vn, vm in variants.items():
            vm.fit(Xtr, ytr)
            pr = vm.predict(Xte)
            f1c = f1_score(yte, pr, labels=list(range(K)), average=None, zero_division=0)
            abl_rows.append(dict(seed=seed, variant=vn, macro_f1=f1_score(yte, pr, average="macro"),
                                 mcc=matthews_corrcoef(yte, pr),
                                 minority_f1=float(np.mean(f1c[minority])) if minority else np.nan))
        if a.dataset == "cicids2017" and DST_PORT in X_all.columns and not use_port:
            Xp_tr, Xp_te = X_all.loc[Xtr.index, feats + [DST_PORT]], X_all.loc[Xte.index, feats + [DST_PORT]]
            vm = lgbm(best["LightGBM"], seed, a.n_jobs).fit(Xp_tr, ytr)
            pr = vm.predict(Xp_te)
            f1c = f1_score(yte, pr, labels=list(range(K)), average=None, zero_division=0)
            abl_rows.append(dict(seed=seed, variant="D: C + Destination Port feature",
                                 macro_f1=f1_score(yte, pr, average="macro"), mcc=matthews_corrcoef(yte, pr),
                                 minority_f1=float(np.mean(f1c[minority])) if minority else np.nan))

        # 3) SHAP-guided feature reduction: ranking + k* chosen on VALIDATION data only
        Xf, Xv, yf, yv = train_test_split(Xtr, ytr, test_size=0.2, stratify=ytr, random_state=seed)
        mv = lgbm(best["LightGBM"], seed, a.n_jobs).fit(Xf, yf)
        Xvs, yvs = stratified_sample(Xv.reset_index(drop=True), yv, a.shap_n, seed, floor=100)
        _, sv_val = shap_values(mv, Xvs)
        imp_val = global_importance(sv_val)
        rank_val.append(imp_val)
        order = [feats[j] for j in np.argsort(-imp_val)]
        val_full = f1_score(yv, mv.predict(Xv), average="macro")
        val_f1, rows_k = {}, []
        for k in ks_grid + ["all"]:
            cols = feats if k == "all" else order[:k]
            vk = lgbm(best["LightGBM"], seed, a.n_jobs).fit(Xf[cols], yf)
            val_f1[k] = f1_score(yv, vk.predict(Xv[cols]), average="macro")
            mk = lgbm(best["LightGBM"], seed, a.n_jobs).fit(Xtr[cols], ytr)
            prk = mk.predict(Xte[cols])
            rows_k.append(dict(seed=seed, k=k, n_features=len(cols), macro_f1=f1_score(yte, prk, average="macro"),
                               mcc=matthews_corrcoef(yte, prk), val_macro_f1=val_f1[k]))
            if seed == 0:
                final.setdefault("reduced", {})[k] = (mk, cols)
        kstar = next((k for k in ks_grid if val_f1[k] >= val_full - a.eps), "all")
        kstars.append(kstar)
        for r in rows_k:
            r["is_k_star"] = (r["k"] == kstar)
        fs_rows += rows_k

        # 4) Leakage-free multi-split deletion faithfulness (validation SHAP ranking on proportional test slice)
        if a.faith_n < len(Xte):
            Xf_te, _, yf_te, _ = train_test_split(Xte, yte, train_size=a.faith_n, stratify=yte, random_state=seed)
        else:
            Xf_te, yf_te = Xte, yte
        Xf_vals = Xf_te.values
        base_f1 = f1_score(yf_te, lgb_model.predict(Xf_vals), average="macro")
        gain_seed = lgb_model.booster_.feature_importance(importance_type="gain")
        shap_idx = np.argsort(-imp_val)
        gain_idx = np.argsort(-gain_seed)
        rng_f = np.random.RandomState(seed)

        def perm_f1(col_indices, n_rep=3):
            vals = []
            for _ in range(n_rep):
                Z = Xf_vals.copy()
                for j in col_indices:
                    Z[:, j] = rng_f.permutation(Z[:, j])
                vals.append(f1_score(yf_te, lgb_model.predict(Z), average="macro"))
            return float(np.mean(vals))

        for k in ks_f:
            rnd_f1 = float(np.mean([perm_f1(rng_f.choice(len(feats), k, replace=False), n_rep=1) for _ in range(5)]))
            faith_rows.append(dict(seed=seed, k=k, baseline=base_f1,
                                   shap_top_k=perm_f1(shap_idx[:k], n_rep=3),
                                   gain_top_k=perm_f1(gain_idx[:k], n_rep=3),
                                   random_k=rnd_f1))

        if seed == 0:
            final.update(Xtr=Xtr, Xte=Xte, ytr=ytr, yte=yte)

    # ---- aggregate tables --------------------------------------------------- #
    tick("writing tables")
    det = pd.DataFrame(det_rows)
    det.to_csv(out / "raw_detection.csv", index=False)
    metrics = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "mcc",
               "detection_rate", "false_alarm_rate", "roc_auc_ovr", "fit_time_s"]
    g = det.groupby("model")[metrics].agg(["mean", "std"])
    tab = pd.DataFrame({m: [pct(g.loc[i, (m, "mean")], g.loc[i, (m, "std")]) if m != "fit_time_s"
                            else f"{g.loc[i, (m, 'mean')]:.1f}" for i in g.index] for m in metrics}, index=g.index)
    tab.to_csv(out / "table_detection.csv")
    pc = pd.DataFrame(cls_rows).groupby(["cls", "model"])["f1"].agg(["mean", "std"]).reset_index()
    pc["F1 (%)"] = [pct(m, s) for m, s in zip(pc["mean"], pc["std"])]
    pc.pivot(index="cls", columns="model", values="F1 (%)").to_csv(out / "table_per_class_f1.csv")
    ab = pd.DataFrame(abl_rows).groupby("variant")[["macro_f1", "mcc", "minority_f1"]].agg(["mean", "std"])
    pd.DataFrame({m: [pct(ab.loc[v, (m, "mean")], ab.loc[v, (m, "std")]) for v in ab.index]
                  for m in ["macro_f1", "mcc", "minority_f1"]}, index=ab.index).to_csv(out / "table_ablation.csv")
    fs = pd.DataFrame(fs_rows)
    fs["k"] = fs["k"].astype(str)
    fs.to_csv(out / "raw_feature_reduction.csv", index=False)
    fsg = fs.groupby("k")[["macro_f1", "mcc", "val_macro_f1"]].agg(["mean", "std"])
    fsg.to_csv(out / "table_feature_reduction_acc.csv")
    pd.Series(kstars, name="k_star").to_csv(out / "k_star_per_seed.csv", index=False)

    # Save multi-split faithfulness table and plot ---------------------------- #
    faith_df = pd.DataFrame(faith_rows)
    faith_df.to_csv(out / "raw_faithfulness.csv", index=False)
    fg = faith_df.groupby("k")[["baseline", "shap_top_k", "gain_top_k", "random_k"]].agg(["mean", "std"]).reset_index()
    fg.columns = ["k", "baseline_mean", "baseline_std", "shap_mean", "shap_std",
                  "gain_mean", "gain_std", "random_mean", "random_std"]
    fg.to_csv(out / "table_faithfulness.csv", index=False)

    plt.figure(figsize=(3.4, 2.6))
    for mcol, scol, lab in [("shap_mean", "shap_std", "SHAP top-k"),
                            ("gain_mean", "gain_std", "Gain top-k"),
                            ("random_mean", "random_std", "Random k")]:
        plt.plot(fg["k"], fg[mcol], marker="o", ms=3, label=lab)
        plt.fill_between(fg["k"], fg[mcol] - fg[scol], fg[mcol] + fg[scol], alpha=0.15)
    plt.xlabel("number of permuted features k")
    plt.ylabel("macro-F1")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out / "fig_faithfulness.png", dpi=300)
    plt.close()

    # ---- explanation analyses on the seed-0 final model --------------------- #
    tick("explanation analyses (seed 0)")
    model, Xte, yte, Xtr, ytr = final["model"][0], final["Xte"], final["yte"], final["Xtr"], final["ytr"]
    Xs, ys = stratified_sample(Xte.reset_index(drop=True), yte, a.shap_n, 0, floor=100)
    expl, sv = shap_values(model, Xs)
    imp = global_importance(sv)
    pred_s = model.predict(Xs)
    save_bar(imp, feats, out / "fig_shap_global.png", "Global feature importance (TreeSHAP)")
    save_class_heatmap(sv, feats, classes, out / "fig_shap_class_heatmap.png")
    per_class_top = {classes[k]: [feats[j] for j in np.argsort(-np.abs(sv[k]).mean(0))[:5]] for k in range(K)}
    pd.DataFrame(per_class_top).T.to_csv(out / "table_top5_features_per_class.csv", header=False)
    atk = [k for k in range(K) if k != benign_idx]
    focus = max(atk, key=lambda k: (ys == k).sum())
    try:
        plt.figure()
        shap.summary_plot(sv[focus], Xs, max_display=12, show=False)
        plt.savefig(out / "fig_shap_beeswarm.png", dpi=300, bbox_inches="tight")
        plt.close("all")
        tp = np.where((ys == focus) & (pred_s == focus))[0]
        if len(tp):
            save_waterfall(expl, sv, Xs, tp[0], focus, feats, out / "fig_waterfall_true_positive.png")
        err = np.where(ys != pred_s)[0]
        if len(err):
            save_waterfall(expl, sv, Xs, err[0], pred_s[err[0]], feats, out / "fig_waterfall_misclassified.png")
            json.dump({"true": classes[ys[err[0]]], "predicted": classes[pred_s[err[0]]]},
                      open(out / "misclassified_case.json", "w"))
        json.dump({"focus_class": classes[focus]}, open(out / "focus_class.json", "w"))
    except Exception as e:
        tick(f"  SHAP plot warning: {e}")

    # stability across seeds (rankings computed on validation data) ----------- #
    taus, jac = [], []
    for i in range(len(rank_val)):
        for j in range(i + 1, len(rank_val)):
            taus.append(kendalltau(rank_val[i], rank_val[j]).correlation)
            ti, tj = set(np.argsort(-rank_val[i])[:10]), set(np.argsort(-rank_val[j])[:10])
            jac.append(len(ti & tj) / len(ti | tj))
    gain = model.booster_.feature_importance(importance_type="gain")
    perm = permutation_importance(model, Xs, ys, scoring="f1_macro", n_repeats=3, random_state=0,
                                  n_jobs=1).importances_mean
    stab = {"kendall_tau_shap_across_splits": (np.mean(taus) if taus else np.nan, np.std(taus) if taus else np.nan),
            "top10_jaccard_across_splits": (np.mean(jac) if jac else np.nan, np.std(jac) if jac else np.nan),
            "kendall_tau_shap_vs_gain": (kendalltau(imp, gain).correlation, np.nan),
            "kendall_tau_shap_vs_permutation": (kendalltau(imp, perm).correlation, np.nan)}
    pd.DataFrame(stab, index=["mean", "std"]).T.to_csv(out / "table_stability_agreement.csv")

    # confusion matrix ---------------------------------------------------------- #
    cm = confusion_matrix(yte, model.predict(Xte), labels=list(range(K)), normalize="true")
    plt.figure(figsize=(3.6, 3.4))
    plt.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    plt.xticks(range(K), classes, rotation=60, ha="right", fontsize=6)
    plt.yticks(range(K), classes, fontsize=6)
    plt.xlabel("predicted", fontsize=7)
    plt.ylabel("true", fontsize=7)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out / "fig_confusion_matrix.png", dpi=300)
    plt.close()

    # latency / throughput / size / explanation cost ---------------------------- #
    tick("latency measurements (single thread)")
    lat_rows = []
    kstar_mode = pd.Series(kstars).mode().iloc[0]
    configs = [("All features", model, feats)]
    if kstar_mode != "all" and kstar_mode in final["reduced"]:
        configs.append((f"SHAP top-{kstar_mode}", *final["reduced"][kstar_mode]))
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except Exception:
        LimeTabularExplainer = None
    for label, m, cols in configs:
        m.set_params(n_jobs=1)
        Xn = Xte[cols].values.astype(np.float32)
        pick = np.random.RandomState(0).choice(len(Xn), min(1000, len(Xn)), replace=False)
        ts = []
        for i in pick:
            t = time.perf_counter()
            m.booster_.predict(Xn[i:i + 1], num_threads=1)
            ts.append((time.perf_counter() - t) * 1e3)
        b = Xn[:1024]
        t = time.perf_counter()
        for _ in range(20):
            m.booster_.predict(b, num_threads=1)
        thr = 20 * len(b) / (time.perf_counter() - t)
        ex = shap.TreeExplainer(m)
        sub = Xte[cols].iloc[pick[:100]]
        t = time.perf_counter()
        ex.shap_values(sub, check_additivity=False)
        shap_ms = (time.perf_counter() - t) / len(sub) * 1e3
        row = dict(config=label, n_features=len(cols), latency_median_ms=np.median(ts),
                   latency_p95_ms=np.percentile(ts, 95), throughput_flows_per_s=thr,
                   model_size_mb=len(pickle.dumps(m)) / 1e6, treeshap_ms_per_alert=shap_ms,
                   lime_ms_per_alert=np.nan)
        if LimeTabularExplainer is not None:
            le_ = LimeTabularExplainer(Xtr[cols].values[:5000], feature_names=cols,
                                       class_names=classes, discretize_continuous=True)
            t = time.perf_counter()
            for i in pick[:20]:
                le_.explain_instance(Xn[i], m.predict_proba, num_features=10, num_samples=1000)
            row["lime_ms_per_alert"] = (time.perf_counter() - t) / 20 * 1e3
        lat_rows.append(row)
    pd.DataFrame(lat_rows).to_csv(out / "table_latency.csv", index=False)

    # per-k cost of the SHAP-reduced models (seed 0) --------------------------- #
    cost = []
    for k, (mk, cols) in final["reduced"].items():
        mk.set_params(n_jobs=1)
        Xn = Xte[cols].values.astype(np.float32)
        pick = np.random.RandomState(0).choice(len(Xn), min(1000, len(Xn)), replace=False)
        ts = []
        for i in pick:
            t = time.perf_counter()
            mk.booster_.predict(Xn[i:i + 1], num_threads=1)
            ts.append((time.perf_counter() - t) * 1e3)
        cost.append(dict(k=str(k), n_features=len(cols), latency_median_ms=np.median(ts),
                         model_size_mb=len(pickle.dumps(mk)) / 1e6))
    pd.DataFrame(cost).to_csv(out / "table_feature_reduction_cost.csv", index=False)

    json.dump({"python": sys.version, "platform": platform.platform(), "processor": platform.processor(),
               "lightgbm": lgb.__version__, "shap": shap.__version__, "xgboost": xgb.__version__,
               "numpy": np.__version__, "pandas": pd.__version__, "args": vars(a)},
              open(out / "environment.json", "w"), indent=2, default=str)
    tick(f"done. Results in {out.resolve()}")


if __name__ == "__main__":
    main()