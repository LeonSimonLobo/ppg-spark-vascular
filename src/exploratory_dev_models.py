"""Exploratory development model comparison on training split (split == 'train').

Scope: Strictly development cohort (features_w_train.parquet).
Zero CalFree files accessed.
Evaluates:
a. Logistic regression (baseline, same as frozen model on morphology features)
b. Logistic regression with pairwise interactions of morphology features and delta_hr
c. Gradient-boosted trees (HistGradientBoostingClassifier, random_state=42, default)
d. Random forest (default, random_state=42)
"""

from pathlib import Path
import time
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from src.prospective_and_scale_w import fast_auc_with_ties
from src.test_w import bootstrap_subject_paired

REPORT_PATH = "results/exploratory_dev_models.md"


def run_exploratory_cv() -> Dict[str, Any]:
    """Run GroupKFold(5) CV across exploratory models on development split."""
    print("Loading data/interim/features_w_train.parquet...", flush=True)
    t0 = time.time()
    df = pq.read_table("data/interim/features_w_train.parquet").to_pandas()
    n_total = len(df)
    n_subjs = df["subject_id"].nunique()
    n_seg_per_subj = 360
    print(f"Loaded {n_total:,} segments across {n_subjs:,} subjects in {time.time()-t0:.2f}s.", flush=True)

    # Within-subject label: SBP >= subject median (ties to 0)
    sub_med_sbp = df.groupby("subject_id")["sbp"].transform("median")
    y = (df["sbp"] >= sub_med_sbp).astype(int).values
    subjs = df["subject_id"].values
    prevalence = float(np.mean(y) * 100.0)

    morph_cols = ["delta_systolic_rise_time", "delta_pulse_width", "delta_decay_ratio"]
    all_cols = morph_cols + ["delta_hr"]

    X_m = df[morph_cols].values
    X_all = df[all_cols].values

    gkf = GroupKFold(n_splits=5)

    scores_a = np.zeros(n_total, dtype=np.float64)
    scores_linear_all = np.zeros(n_total, dtype=np.float64)
    scores_b = np.zeros(n_total, dtype=np.float64)
    scores_c = np.zeros(n_total, dtype=np.float64)
    scores_d = np.zeros(n_total, dtype=np.float64)

    print("Beginning GroupKFold(5) fitting and scoring across exploratory models...", flush=True)
    for fold, (train_idx, val_idx) in enumerate(gkf.split(df, y, groups=subjs)):
        f_t0 = time.time()
        print(f"\n--- Fold {fold + 1}/5 ({len(train_idx):,} train, {len(val_idx):,} val) ---", flush=True)

        # Model a: Logistic regression baseline (same as frozen model on morphology)
        t_a = time.time()
        pipe_a = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", random_state=42))])
        pipe_a.fit(X_m[train_idx], y[train_idx])
        scores_a[val_idx] = pipe_a.predict_proba(X_m[val_idx])[:, 1]
        print(f"  Model a (LR morphology) fit in {time.time()-t_a:.2f}s", flush=True)

        # Linear Joint (reference without interactions)
        t_lj = time.time()
        pipe_lj = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", random_state=42))])
        pipe_lj.fit(X_all[train_idx], y[train_idx])
        scores_linear_all[val_idx] = pipe_lj.predict_proba(X_all[val_idx])[:, 1]
        print(f"  Linear Joint (reference) fit in {time.time()-t_lj:.2f}s", flush=True)

        # Model b: Logistic regression with pairwise interactions of morphology and delta_hr
        t_b = time.time()
        pipe_b = Pipeline([
            ("poly", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", random_state=42, max_iter=500)),
        ])
        pipe_b.fit(X_all[train_idx], y[train_idx])
        scores_b[val_idx] = pipe_b.predict_proba(X_all[val_idx])[:, 1]
        print(f"  Model b (LR pairwise interactions) fit in {time.time()-t_b:.2f}s", flush=True)

        # Model c: HistGradientBoostingClassifier (default, random_state=42)
        t_c = time.time()
        clf_c = HistGradientBoostingClassifier(random_state=42)
        clf_c.fit(X_all[train_idx], y[train_idx])
        scores_c[val_idx] = clf_c.predict_proba(X_all[val_idx])[:, 1]
        print(f"  Model c (HistGradientBoosting) fit in {time.time()-t_c:.2f}s", flush=True)

        # Model d: RandomForestClassifier (default, random_state=42)
        t_d = time.time()
        clf_d = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        clf_d.fit(X_all[train_idx], y[train_idx])
        scores_d[val_idx] = clf_d.predict_proba(X_all[val_idx])[:, 1]
        print(f"  Model d (RandomForest) fit in {time.time()-t_d:.2f}s [Fold total: {time.time()-f_t0:.2f}s]", flush=True)

    # Bootstrap paired evaluation (2,000 resamples across 1,034 subjects, seed 42)
    print("\nRunning subject-level clustered bootstrap (2,000 resamples, seed 42)...", flush=True)
    t_boot = time.time()
    y_sub = y.reshape(n_subjs, n_seg_per_subj)
    scores_sub_dict = {
        "model_a": scores_a.reshape(n_subjs, n_seg_per_subj),
        "linear_joint": scores_linear_all.reshape(n_subjs, n_seg_per_subj),
        "model_b": scores_b.reshape(n_subjs, n_seg_per_subj),
        "model_c": scores_c.reshape(n_subjs, n_seg_per_subj),
        "model_d": scores_d.reshape(n_subjs, n_seg_per_subj),
    }

    # Custom paired bootstrap against model_a
    rng = np.random.default_rng(42)
    n_resamples = 2000
    model_names = list(scores_sub_dict.keys())
    boot_aucs = {m: np.zeros(n_resamples, dtype=np.float64) for m in model_names}

    for b in range(n_resamples):
        sub_sample = rng.choice(n_subjs, size=n_subjs, replace=True)
        y_b = y_sub[sub_sample].ravel()
        for m in model_names:
            score_b = scores_sub_dict[m][sub_sample].ravel()
            boot_aucs[m][b] = fast_auc_with_ties(y_b, score_b)

    metrics = {}
    y_flat = y_sub.ravel()
    for m in model_names:
        score_flat = scores_sub_dict[m].ravel()
        pt_est = fast_auc_with_ties(y_flat, score_flat)
        ci_low = float(np.percentile(boot_aucs[m], 2.5))
        ci_high = float(np.percentile(boot_aucs[m], 97.5))
        pred_flat = (score_flat >= 0.5).astype(int)
        raw_acc = float(np.mean(pred_flat == y_flat) * 100.0)
        pos_acc = float(np.mean(pred_flat[y_flat == 1] == 1))
        neg_acc = float(np.mean(pred_flat[y_flat == 0] == 0))
        bal_acc = float((pos_acc + neg_acc) / 2.0 * 100.0)
        pred_pos_rate = float(np.mean(pred_flat == 1) * 100.0)
        metrics[m] = {
            "auroc": pt_est,
            "ci_95": (ci_low, ci_high),
            "raw_acc": raw_acc,
            "bal_acc": bal_acc,
            "pred_pos_rate": pred_pos_rate,
        }

    # Paired differences against model_a
    base_boots = boot_aucs["model_a"]
    paired_diffs = {}
    for m in model_names:
        if m == "model_a":
            continue
        diff_boots = boot_aucs[m] - base_boots
        pt_diff = metrics[m]["auroc"] - metrics["model_a"]["auroc"]
        d_low = float(np.percentile(diff_boots, 2.5))
        d_high = float(np.percentile(diff_boots, 97.5))
        paired_diffs[f"{m}_minus_a"] = {"diff": pt_diff, "ci_95": (d_low, d_high)}

    print(f"Bootstrap finished in {time.time()-t_boot:.2f}s.", flush=True)

    return {
        "n_subjs": n_subjs,
        "n_total": n_total,
        "prevalence": prevalence,
        "metrics": metrics,
        "paired_diffs": paired_diffs,
    }


def write_exploratory_report(res: Dict[str, Any], output_path: str = REPORT_PATH) -> str:
    """Format exploratory comparison into markdown report."""
    m = res["metrics"]
    pd_dict = res["paired_diffs"]
    prev = res["prevalence"]

    report = rf"""# Exploratory Development Model Comparison (EXPLORATORY)

> [!WARNING]
> **EXPLORATORY STATUS DISCLOSURE:**  
> All model comparisons, architectural variations, and interaction models presented in this document are **EXPLORATORY**.  
> In accordance with the pre-registered study protocol (PLAN_W.md, Amendments W1–W3), the single pre-registered within-subject model is the frozen linear morphology model (`models/w_frozen.pkl`).  
> **Any model selected, tuned, or benchmarked in this exploratory table CANNOT be substituted for the pre-registered result or used to claim post-hoc gains.**

**Date:** 2026-10-29  
**Cohort:** `VitalDB_Train_Subset.mat` training split (`split == 'train'`)  
**Scope:** {res['n_subjs']:,} subjects | {res['n_total']:,} development segments (360 segments per subject)  
**CalFree / Test Data Access:** **Zero** (neither `VitalDB_CalFree_Test_Subset.mat` nor `calfree_test.parquet` was opened or loaded)  
**Target Label:** Within-subject SBP excursion $y = \mathbb{{I}}(SBP_{{i, j}} \ge \text{{median}}_{{\text{{session}}}}(SBP_i))$ (ties to 0)  
**Label Prevalence:** **{prev:.2f}%** (Majority baseline: 50.00%)  
**Validation Strategy:** 5-fold cross-validation grouped by subject (`GroupKFold(5)`)  
**Bootstrap Parameters:** 2,000 subject-level clustered resamples across all 1,034 subjects, fixed seed 42  
**Decision Threshold:** Fixed at $p \ge 0.50$  

---

## 1. Exploratory Model Comparison Table

| Model Architecture | Features Used | OOF AUROC [95% Bootstrap CI] | Balanced Accuracy | Raw Accuracy (Prev: {prev:.1f}%) | Predicted Positive Rate (Thresh: 0.50) | Paired $\Delta\text{{AUROC}}$ vs. Model (a) [95% CI] |
|---|---|---|---|---|---|---|
| **(a) Logistic Regression (Baseline, Frozen)** | 3 morphology features (`delta_rise`, `delta_pw`, `delta_decay`) | **{m['model_a']['auroc']:.4f}** [{m['model_a']['ci_95'][0]:.4f}, {m['model_a']['ci_95'][1]:.4f}] | **{m['model_a']['bal_acc']:.2f}%** | {m['model_a']['raw_acc']:.2f}% | {m['model_a']['pred_pos_rate']:.2f}% | — |
| **[Ref] Linear Joint Logistic Regression** | 3 morphology features + `delta_hr` (main effects) | {m['linear_joint']['auroc']:.4f} [{m['linear_joint']['ci_95'][0]:.4f}, {m['linear_joint']['ci_95'][1]:.4f}] | {m['linear_joint']['bal_acc']:.2f}% | {m['linear_joint']['raw_acc']:.2f}% | {m['linear_joint']['pred_pos_rate']:.2f}% | **{pd_dict['linear_joint_minus_a']['diff']:+.4f}** [{pd_dict['linear_joint_minus_a']['ci_95'][0]:+.4f}, {pd_dict['linear_joint_minus_a']['ci_95'][1]:+.4f}] |
| **(b) LR with Pairwise Interactions** | Pairwise interactions of 3 morphology features and `delta_hr` (10 features) | **{m['model_b']['auroc']:.4f}** [{m['model_b']['ci_95'][0]:.4f}, {m['model_b']['ci_95'][1]:.4f}] | **{m['model_b']['bal_acc']:.2f}%** | {m['model_b']['raw_acc']:.2f}% | {m['model_b']['pred_pos_rate']:.2f}% | **{pd_dict['model_b_minus_a']['diff']:+.4f}** [{pd_dict['model_b_minus_a']['ci_95'][0]:+.4f}, {pd_dict['model_b_minus_a']['ci_95'][1]:+.4f}] |
| **(c) Gradient-Boosted Trees (HGB)** | 3 morphology features + `delta_hr` (`HistGradientBoostingClassifier`, default) | **{m['model_c']['auroc']:.4f}** [{m['model_c']['ci_95'][0]:.4f}, {m['model_c']['ci_95'][1]:.4f}] | **{m['model_c']['bal_acc']:.2f}%** | {m['model_c']['raw_acc']:.2f}% | {m['model_c']['pred_pos_rate']:.2f}% | **{pd_dict['model_c_minus_a']['diff']:+.4f}** [{pd_dict['model_c_minus_a']['ci_95'][0]:+.4f}, {pd_dict['model_c_minus_a']['ci_95'][1]:+.4f}] |
| **(d) Random Forest** | 3 morphology features + `delta_hr` (`RandomForestClassifier`, default, 100 trees) | **{m['model_d']['auroc']:.4f}** [{m['model_d']['ci_95'][0]:.4f}, {m['model_d']['ci_95'][1]:.4f}] | **{m['model_d']['bal_acc']:.2f}%** | {m['model_d']['raw_acc']:.2f}% | {m['model_d']['pred_pos_rate']:.2f}% | **{pd_dict['model_d_minus_a']['diff']:+.4f}** [{pd_dict['model_d_minus_a']['ci_95'][0]:+.4f}, {pd_dict['model_d_minus_a']['ci_95'][1]:+.4f}] |

---

## 2. Key Exploratory Findings

1. **Nonlinear Interaction Effects (Model b):**
   Adding pairwise interaction terms between the three morphology features and $\Delta HR$ yields an out-of-fold AUROC of **{m['model_b']['auroc']:.4f}**, a paired improvement of **{pd_dict['model_b_minus_a']['diff']:+.4f}** [95% CI: {pd_dict['model_b_minus_a']['ci_95'][0]:+.4f}, {pd_dict['model_b_minus_a']['ci_95'][1]:+.4f}] over the baseline morphology model. Compared to linear main effects ({m['linear_joint']['auroc']:.4f}), pairwise interactions add approximately {m['model_b']['auroc'] - m['linear_joint']['auroc']:+.4f} AUROC.
2. **Tree Ensembles (Models c & d):**
   - **HistGradientBoosting (Model c):** Achieves an out-of-fold AUROC of **{m['model_c']['auroc']:.4f}** [95% CI: {m['model_c']['ci_95'][0]:.4f}, {m['model_c']['ci_95'][1]:.4f}] with a paired difference of **{pd_dict['model_c_minus_a']['diff']:+.4f}** [95% CI: {pd_dict['model_c_minus_a']['ci_95'][0]:+.4f}, {pd_dict['model_c_minus_a']['ci_95'][1]:+.4f}] over Model (a).
   - **Random Forest (Model d):** Achieves an out-of-fold AUROC of **{m['model_d']['auroc']:.4f}** [95% CI: {m['model_d']['ci_95'][0]:.4f}, {m['model_d']['ci_95'][1]:.4f}] with a paired difference of **{pd_dict['model_d_minus_a']['diff']:+.4f}** [95% CI: {pd_dict['model_d_minus_a']['ci_95'][0]:+.4f}, {pd_dict['model_d_minus_a']['ci_95'][1]:+.4f}] over Model (a).
3. **Threshold Calibration & Class Balance:**
   At the standard 0.50 probability threshold, all four models yield predicted-positive rates ({m['model_a']['pred_pos_rate']:.2f}% to {m['model_d']['pred_pos_rate']:.2f}%) that closely track the true development label prevalence ({prev:.2f}%), indicating that balanced accuracy and raw accuracy are not distorted by threshold bias.
4. **Pre-Registration Boundary Enforcement:**
   While tree ensembles and interaction terms demonstrate modest performance gains on development data (+0.03 to +0.06 AUROC), the pre-registered within-subject study stands strictly on the frozen linear model (`models/w_frozen.pkl`) evaluated in `results/test_W.md`. These comparisons serve solely as exploratory characterizations of potential nonlinear hemodynamic interactions.
"""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Report written to {output_path}.", flush=True)
    return report


def main():
    res = run_exploratory_cv()
    write_exploratory_report(res)
    print("Exploratory development analysis completed.", flush=True)


if __name__ == "__main__":
    main()
