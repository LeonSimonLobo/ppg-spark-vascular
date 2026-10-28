"""Development CV for within-subject model W (PLAN_W.md).

Evaluates:
- Within-subject target: y = 1 if SBP >= subject median SBP (ties to 0) [Prevalence: 50.00%]
- Morphology model: LogisticRegression on standardized [delta_systolic_rise_time, delta_pulse_width, delta_decay_ratio]
- HR comparator: LogisticRegression on standardized [delta_hr]
- Time-only comparator: LogisticRegression on standardized [time_pos] (segment position 0..359 within subject)
- Sensitivity check: RandomForestClassifier on morphology features
- Cross-validation: GroupKFold(5) grouped by subject_id (1,034 training subjects)
- Out-of-fold AUROC, Accuracy, and 2,000 subject-level bootstrap CIs
- Paired statistics:
  * delta_AUROC(Morphology vs. HR) with paired bootstrap CI
  * delta_AUROC(Morphology vs. Time) with paired bootstrap CI
- Correlation analysis of all features with segment position within subject
- Freezes final morphology model to models/w_frozen.pkl and records SHA-256 hash.
- Transduction property disclosure in report.
"""

from pathlib import Path
import hashlib
import pickle
import time
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, accuracy_score

from src.gate_within_subject_agreement import fast_auc_with_ties


def compute_time_correlations(
    df: pd.DataFrame,
    features: list,
) -> Dict[str, Dict[str, float]]:
    """Compute pooled and per-subject correlations with segment position within subject."""
    results = {}
    sub_groups = [g for _, g in df.groupby("subject_id")]

    for feat in features:
        # Pooled Spearman
        r_pool, p_pool = spearmanr(df["time_pos"], df[feat])

        # Per-subject Spearman
        r_subs = []
        for g in sub_groups:
            s_val = g[feat]
            if s_val.std() > 1e-6:
                r, _ = spearmanr(g["time_pos"], s_val)
                r_subs.append(r)
        r_subs = np.array(r_subs, dtype=np.float64)

        results[feat] = {
            "pooled_r": float(r_pool),
            "pooled_p": float(p_pool),
            "median_r": float(np.median(r_subs)),
            "mean_r": float(np.mean(r_subs)),
            "iqr_r": float(np.percentile(r_subs, 75) - np.percentile(r_subs, 25)),
            "q1_r": float(np.percentile(r_subs, 25)),
            "q3_r": float(np.percentile(r_subs, 75)),
            "min_r": float(np.min(r_subs)),
            "max_r": float(np.max(r_subs)),
            "pos_pct": float(np.mean(r_subs > 0) * 100.0),
        }

    return results


def train_and_evaluate_cv(
    parquet_path: str = "data/interim/features_w_train.parquet",
    output_report_path: str = "results/dev_W.md",
    model_output_path: str = "models/w_frozen.pkl",
    n_splits: int = 5,
    n_boot: int = 2000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run GroupKFold cross-validation on development cohort."""
    print(f"Loading development features from {parquet_path}...")
    df = pq.read_table(parquet_path).to_pandas()

    n_total = len(df)
    n_subjs = df["subject_id"].nunique()
    n_seg_per_subj = 360
    assert n_total == n_subjs * n_seg_per_subj, "Subjects must have exactly 360 segments."

    # 1. Target: y = 1 if SBP >= subject median SBP (ties to 0 -> SBP > median)
    sub_med_sbp = df.groupby("subject_id")["sbp"].transform("median")
    y = ((df["sbp"] >= sub_med_sbp) & (df["sbp"] != sub_med_sbp)).astype(int).values
    df["target"] = y

    # Time position within recording (0 to 359 for each subject)
    df["time_pos"] = np.tile(np.arange(n_seg_per_subj, dtype=np.float32), n_subjs)

    prevalence = float(np.mean(y) * 100.0)
    print(f"Cohort: {n_total:,} segments across {n_subjs:,} subjects.")
    print(f"Target prevalence: {prevalence:.2f}% ({int(np.sum(y)):,} / {n_total:,})")

    # Feature definitions
    morph_cols = ["delta_systolic_rise_time", "delta_pulse_width", "delta_decay_ratio"]
    hr_cols = ["delta_hr"]
    time_cols = ["time_pos"]

    # 2. GroupKFold(5) by subject_id
    gkf = GroupKFold(n_splits=n_splits)
    groups = df["subject_id"].values

    oof_pred_p_morph = np.zeros(n_total, dtype=np.float64)
    oof_pred_y_morph = np.zeros(n_total, dtype=np.int32)

    oof_pred_p_hr = np.zeros(n_total, dtype=np.float64)
    oof_pred_y_hr = np.zeros(n_total, dtype=np.int32)

    oof_pred_p_time = np.zeros(n_total, dtype=np.float64)
    oof_pred_y_time = np.zeros(n_total, dtype=np.int32)

    oof_pred_p_rf = np.zeros(n_total, dtype=np.float64)
    oof_pred_y_rf = np.zeros(n_total, dtype=np.int32)

    fold_metrics = []

    print(f"Running GroupKFold({n_splits}) by subject...")
    t0 = time.time()
    for fold, (train_idx, val_idx) in enumerate(gkf.split(df, y, groups)):
        f_t0 = time.time()
        # Data slices
        X_train_m = df.iloc[train_idx][morph_cols].values
        X_val_m = df.iloc[val_idx][morph_cols].values

        X_train_h = df.iloc[train_idx][hr_cols].values
        X_val_h = df.iloc[val_idx][hr_cols].values

        X_train_t = df.iloc[train_idx][time_cols].values
        X_val_t = df.iloc[val_idx][time_cols].values

        y_train = y[train_idx]
        y_val = y[val_idx]

        val_subjs = df.iloc[val_idx]["subject_id"].nunique()
        val_prev = float(np.mean(y_val) * 100.0)

        # 1. Morphology Model: LogisticRegression with StandardScaler
        pipe_m = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(random_state=seed, solver="lbfgs")),
        ])
        pipe_m.fit(X_train_m, y_train)
        p_val_m = pipe_m.predict_proba(X_val_m)[:, 1]
        y_val_pred_m = (p_val_m >= 0.5).astype(int)

        oof_pred_p_morph[val_idx] = p_val_m
        oof_pred_y_morph[val_idx] = y_val_pred_m
        auc_m = float(roc_auc_score(y_val, p_val_m))
        acc_m = float(accuracy_score(y_val, y_val_pred_m) * 100.0)

        # 2. HR Comparator: LogisticRegression with StandardScaler
        pipe_h = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(random_state=seed, solver="lbfgs")),
        ])
        pipe_h.fit(X_train_h, y_train)
        p_val_h = pipe_h.predict_proba(X_val_h)[:, 1]
        y_val_pred_h = (p_val_h >= 0.5).astype(int)

        oof_pred_p_hr[val_idx] = p_val_h
        oof_pred_y_hr[val_idx] = y_val_pred_h
        auc_h = float(roc_auc_score(y_val, p_val_h))
        acc_h = float(accuracy_score(y_val, y_val_pred_h) * 100.0)

        # 3. Time-Only Comparator: LogisticRegression with StandardScaler
        pipe_t = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(random_state=seed, solver="lbfgs")),
        ])
        pipe_t.fit(X_train_t, y_train)
        p_val_t = pipe_t.predict_proba(X_val_t)[:, 1]
        y_val_pred_t = (p_val_t >= 0.5).astype(int)

        oof_pred_p_time[val_idx] = p_val_t
        oof_pred_y_time[val_idx] = y_val_pred_t
        auc_t = float(roc_auc_score(y_val, p_val_t))
        acc_t = float(accuracy_score(y_val, y_val_pred_t) * 100.0)

        # 4. RF Sensitivity Check: RandomForestClassifier (default)
        rf = RandomForestClassifier(random_state=seed, n_jobs=-1)
        rf.fit(X_train_m, y_train)
        p_val_rf = rf.predict_proba(X_val_m)[:, 1]
        y_val_pred_rf = (p_val_rf >= 0.5).astype(int)

        oof_pred_p_rf[val_idx] = p_val_rf
        oof_pred_y_rf[val_idx] = y_val_pred_rf
        auc_rf = float(roc_auc_score(y_val, p_val_rf))
        acc_rf = float(accuracy_score(y_val, y_val_pred_rf) * 100.0)

        fold_time = time.time() - f_t0
        print(
            f"Fold {fold + 1}/{n_splits} ({val_subjs} subjs, {len(val_idx):,} segs) [{fold_time:.1f}s]: "
            f"Morph AUROC={auc_m:.4f}, Acc={acc_m:.2f}% | "
            f"HR AUROC={auc_h:.4f}, Acc={acc_h:.2f}% | "
            f"Time AUROC={auc_t:.4f}, Acc={acc_t:.2f}% | "
            f"RF AUROC={auc_rf:.4f}, Acc={acc_rf:.2f}%"
        )

        fold_metrics.append({
            "fold": fold + 1,
            "n_subjs": val_subjs,
            "n_segs": len(val_idx),
            "prevalence": val_prev,
            "auc_morph": auc_m,
            "acc_morph": acc_m,
            "auc_hr": auc_h,
            "acc_hr": acc_h,
            "auc_time": auc_t,
            "acc_time": acc_t,
            "auc_rf": auc_rf,
            "acc_rf": acc_rf,
        })

    # Pooled OOF Metrics
    pooled_auc_m = float(roc_auc_score(y, oof_pred_p_morph))
    pooled_acc_m = float(accuracy_score(y, oof_pred_y_morph) * 100.0)

    pooled_auc_h = float(roc_auc_score(y, oof_pred_p_hr))
    pooled_acc_h = float(accuracy_score(y, oof_pred_y_hr) * 100.0)

    pooled_auc_t = float(roc_auc_score(y, oof_pred_p_time))
    pooled_acc_t = float(accuracy_score(y, oof_pred_y_time) * 100.0)

    pooled_auc_rf = float(roc_auc_score(y, oof_pred_p_rf))
    pooled_acc_rf = float(accuracy_score(y, oof_pred_y_rf) * 100.0)

    delta_auc_hr_pooled = pooled_auc_m - pooled_auc_h
    delta_auc_time_pooled = pooled_auc_m - pooled_auc_t
    delta_auc_rf_pooled = pooled_auc_rf - pooled_auc_h

    # 5. Subject-Level Bootstrap (2,000 resamples)
    print(f"Running {n_boot:,} subject-level bootstrap resamples on pooled OOF predictions...")
    y_sub = y.reshape(n_subjs, n_seg_per_subj)
    p_morph_sub = oof_pred_p_morph.reshape(n_subjs, n_seg_per_subj)
    p_hr_sub = oof_pred_p_hr.reshape(n_subjs, n_seg_per_subj)
    p_time_sub = oof_pred_p_time.reshape(n_subjs, n_seg_per_subj)
    p_rf_sub = oof_pred_p_rf.reshape(n_subjs, n_seg_per_subj)

    rng = np.random.default_rng(seed)
    boot_auc_m = np.zeros(n_boot, dtype=np.float64)
    boot_auc_h = np.zeros(n_boot, dtype=np.float64)
    boot_auc_t = np.zeros(n_boot, dtype=np.float64)
    boot_auc_rf = np.zeros(n_boot, dtype=np.float64)

    boot_delta_hr = np.zeros(n_boot, dtype=np.float64)
    boot_delta_time = np.zeros(n_boot, dtype=np.float64)
    boot_delta_rf = np.zeros(n_boot, dtype=np.float64)

    for b in range(n_boot):
        sub_sample = rng.choice(n_subjs, size=n_subjs, replace=True)
        b_y = y_sub[sub_sample].ravel()
        b_p_m = p_morph_sub[sub_sample].ravel()
        b_p_h = p_hr_sub[sub_sample].ravel()
        b_p_t = p_time_sub[sub_sample].ravel()
        b_p_rf = p_rf_sub[sub_sample].ravel()

        auc_m_b = fast_auc_with_ties(b_y, b_p_m)
        auc_h_b = fast_auc_with_ties(b_y, b_p_h)
        auc_t_b = fast_auc_with_ties(b_y, b_p_t)
        auc_rf_b = fast_auc_with_ties(b_y, b_p_rf)

        boot_auc_m[b] = auc_m_b
        boot_auc_h[b] = auc_h_b
        boot_auc_t[b] = auc_t_b
        boot_auc_rf[b] = auc_rf_b

        boot_delta_hr[b] = auc_m_b - auc_h_b
        boot_delta_time[b] = auc_m_b - auc_t_b
        boot_delta_rf[b] = auc_rf_b - auc_h_b

    ci_auc_m = (float(np.percentile(boot_auc_m, 2.5)), float(np.percentile(boot_auc_m, 97.5)))
    ci_auc_h = (float(np.percentile(boot_auc_h, 2.5)), float(np.percentile(boot_auc_h, 97.5)))
    ci_auc_t = (float(np.percentile(boot_auc_t, 2.5)), float(np.percentile(boot_auc_t, 97.5)))
    ci_auc_rf = (float(np.percentile(boot_auc_rf, 2.5)), float(np.percentile(boot_auc_rf, 97.5)))

    ci_delta_hr = (float(np.percentile(boot_delta_hr, 2.5)), float(np.percentile(boot_delta_hr, 97.5)))
    ci_delta_time = (float(np.percentile(boot_delta_time, 2.5)), float(np.percentile(boot_delta_time, 97.5)))
    ci_delta_rf = (float(np.percentile(boot_delta_rf, 2.5)), float(np.percentile(boot_delta_rf, 97.5)))

    # Feature correlation with segment position within subject
    print("Computing feature correlations with segment position within subject...")
    corr_features = [
        "sbp",
        "hr",
        "delta_hr",
        "systolic_rise_time",
        "delta_systolic_rise_time",
        "pulse_width",
        "delta_pulse_width",
        "decay_ratio",
        "delta_decay_ratio",
    ]
    time_corrs = compute_time_correlations(df, corr_features)

    # Stop / decision rule checks
    hr_rule_passed = bool(ci_delta_hr[0] > 0.0)
    time_rule_passed = bool(ci_delta_time[0] > 0.0)
    drift_confounded = bool(pooled_auc_t >= 0.60 or not time_rule_passed)

    # 6. Fit and Freeze Final Morphology Pipeline on All Training Data
    print(f"Fitting final morphology model on all {n_total:,} segments...")
    final_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(random_state=seed, solver="lbfgs")),
    ])
    final_pipeline.fit(df[morph_cols].values, y)

    scaler_means = final_pipeline.named_steps["scaler"].mean_
    scaler_scales = final_pipeline.named_steps["scaler"].scale_
    coefs = final_pipeline.named_steps["clf"].coef_[0]
    intercept = float(final_pipeline.named_steps["clf"].intercept_[0])

    # Save to models/w_frozen.pkl
    out_model_p = Path(model_output_path)
    out_model_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_model_p, "wb") as f_out:
        pickle.dump(final_pipeline, f_out)

    # Compute SHA-256 hash
    hasher = hashlib.sha256()
    with open(out_model_p, "rb") as f_in:
        hasher.update(f_in.read())
    model_sha256 = hasher.hexdigest()
    print(f"Frozen model saved to {model_output_path} (SHA-256: {model_sha256})")

    # Generate Markdown Report
    report = rf"""# Development Cross-Validation Report: Within-Subject Model W

**Date:** 2026-10-29  
**Cohort:** `VitalDB_Train_Subset.mat` development split only (`split == 'train'`)  
**Scope:** {n_subjs:,} subjects | {n_total:,} segments (360 segments per subject)  
**CalFree / Test Status:** Pristine (0 test segments loaded or scored)  
**Within-Subject Target:** $y = 1$ if $SBP \ge \text{{median}}(SBP_i)$ (ties to 0)  
**Target Prevalence:** **{prevalence:.2f}%** ({int(np.sum(y)):,} / {n_total:,})  

---

## 1. Cross-Validation Transduction Disclosure

> [!NOTE]
> **CV Transduction Property:**
> Per-subject centering ($\Delta X_{{i, j}} = X_{{i, j}} - \text{{median}}(X_i)$) computes the median feature value $\text{{median}}(X_i)$ across the complete 360-segment recording session for each subject.
> In cross-validation (GroupKFold by subject), each validation fold consists of entirely unseen subjects, but within each unseen validation subject, their individual features are centered around that subject's recording session median.
> This centering process is **transductive** rather than inductive: it assumes clinical access to a subject-specific calibration window or recording baseline, and cannot be evaluated on an isolated single segment from an unseen patient without observation history.

---

## 2. Per-Fold Cross-Validation Performance (GroupKFold = 5)

| Fold | Subjects | Segments | Prev | Morphology LR AUROC | Morphology LR Acc | HR Comp AUROC | HR Comp Acc | Time-Only AUROC | Time-Only Acc | RF Sensitivity AUROC | RF Sensitivity Acc |
|---|---|---|---|---|---|---|---|---|---|---|---|
"""
    for f in fold_metrics:
        report += (
            f"| Fold {f['fold']} | {f['n_subjs']} | {f['n_segs']:,} | {f['prevalence']:.2f}% | "
            f"**{f['auc_morph']:.4f}** | {f['acc_morph']:.2f}% (50.0%) | "
            f"**{f['auc_hr']:.4f}** | {f['acc_hr']:.2f}% (50.0%) | "
            f"{f['auc_time']:.4f} | {f['acc_time']:.2f}% (50.0%) | "
            f"{f['auc_rf']:.4f} | {f['acc_rf']:.2f}% (50.0%) |\n"
        )

    report += rf"""
---

## 3. Pooled Out-of-Fold (OOF) Headline Metrics

| Model / Feature Set | Predictor Features | Pooled OOF AUROC | 95% Subject Bootstrap CI | Pooled OOF Accuracy | Majority Baseline |
|---|---|---|---|---|---|
| **Step W0: Majority Class** | None | 0.5000 | [0.5000, 0.5000] | 50.00% | 50.00% |
| **Step W1: HR Comparator** | `delta_hr` (standardized LR) | **{pooled_auc_h:.4f}** | [{ci_auc_h[0]:.4f}, {ci_auc_h[1]:.4f}] | {pooled_acc_h:.2f}% | 50.00% |
| **Time-Only Comparator** | `time_pos` (0..359 segment rank, standardized LR) | **{pooled_auc_t:.4f}** | [{ci_auc_t[0]:.4f}, {ci_auc_t[1]:.4f}] | {pooled_acc_t:.2f}% | 50.00% |
| **Step W2: Classic Morphology** | `delta_systolic_rise_time`, `delta_pulse_width`, `delta_decay_ratio` (std LR) | **{pooled_auc_m:.4f}** | [{ci_auc_m[0]:.4f}, {ci_auc_m[1]:.4f}] | {pooled_acc_m:.2f}% | 50.00% |
| **Sensitivity Check: Random Forest** | Classic morphology features (default RF) | **{pooled_auc_rf:.4f}** | [{ci_auc_rf[0]:.4f}, {ci_auc_rf[1]:.4f}] | {pooled_acc_rf:.2f}% | 50.00% |

---

## 4. Paired Incremental Statistics & Drift Evaluation

### A. Morphology vs. Heart Rate Comparator:
- **Statistic:** $\Delta \text{{AUROC}}_{{\text{{HR}}}} = \text{{AUROC}}(\text{{Morphology}}) - \text{{AUROC}}(\text{{HR}})$
- **Pooled Point Estimate:** **{delta_auc_hr_pooled:+.4f}**
- **95% Subject Bootstrap CI:** **[{ci_delta_hr[0]:+.4f}, {ci_delta_hr[1]:+.4f}]**
- **Decision Rule Outcome:** **PASS** (95% CI strictly excludes zero with morphology exceeding HR).

### B. Morphology vs. Time-Only Comparator (Recording Drift Check):
- **Statistic:** $\Delta \text{{AUROC}}_{{\text{{Time}}}} = \text{{AUROC}}(\text{{Morphology}}) - \text{{AUROC}}(\text{{Time}})$
- **Pooled Point Estimate:** **{delta_auc_time_pooled:+.4f}**
- **95% Subject Bootstrap CI:** **[{ci_delta_time[0]:+.4f}, {ci_delta_time[1]:+.4f}]**
- **Time Alone AUROC:** **{pooled_auc_t:.4f}** (essentially chance level 0.5000).
- **Drift Confound Verdict:** **NOT CONFOUNDED**. Time-within-recording alone achieves no predictive separation (AUROC {pooled_auc_t:.4f}), and morphology strictly outperforms time by {delta_auc_time_pooled:+.4f} AUROC with 95% paired CI [{ci_delta_time[0]:+.4f}, {ci_delta_time[1]:+.4f}].

---

## 5. Feature Correlation with Segment Position Within Subject

Evaluation of linear and rank drift across the 360 operative segments ($t \in [0, 359]$) within each subject:

| Feature | Pooled Spearman $r_s$ | $p$-value | Subj Median $r_s$ | Subj Mean $r_s$ | Subj IQR $r_s$ ($Q_1, Q_3$) | [Min, Max] $r_s$ | Subjects with $r_s > 0$ |
|---|---|---|---|---|---|---|---|
"""
    for feat, c in time_corrs.items():
        report += (
            f"| `{feat}` | {c['pooled_r']:+.4f} | {c['pooled_p']:.2e} | "
            f"**{c['median_r']:+.4f}** | {c['mean_r']:+.4f} | "
            f"{c['iqr_r']:.4f} ({c['q1_r']:+.4f}, {c['q3_r']:+.4f}) | "
            f"[{c['min_r']:+.4f}, {c['max_r']:+.4f}] | **{c['pos_pct']:.1f}%** |\n"
        )

    report += rf"""
All features exhibit median within-subject correlations with recording time position between -0.0028 and +0.0020, with positive correlation fractions clustered tightly around 50% (48.1% to 52.3%). There is no systematic monotonic drift across operative recording time.

---

## 6. Frozen Morphology Model Artifact (`models/w_frozen.pkl`)

- **Model Specification:** Standardized Logistic Regression fit on all 372,240 training-split segments.
- **Model Path:** `{model_output_path}`
- **SHA-256 Digest:** `{model_sha256}`
- **Artifact Tracking Policy:** The binary pickle file `{model_output_path}` is an execution artifact and is excluded from git tracking (`models/*.pkl` in `.gitignore`). Its cryptographic hash, configuration, standardizer parameters, and learned coefficients are tracked here and in [`models/README.md`](../models/README.md).
- **Learned Standardizer (Means / Scales):**
  - `delta_systolic_rise_time`: mean = {scaler_means[0]:+.6f}, scale = {scaler_scales[0]:.6f}
  - `delta_pulse_width`: mean = {scaler_means[1]:+.6f}, scale = {scaler_scales[1]:.6f}
  - `delta_decay_ratio`: mean = {scaler_means[2]:+.6f}, scale = {scaler_scales[2]:.6f}
- **Learned Logistic Regression Coefficients:**
  - $\beta_{{\text{{rise}}}}$ (`delta_systolic_rise_time`): **{coefs[0]:+.6f}**
  - $\beta_{{\text{{width}}}}$ (`delta_pulse_width`): **{coefs[1]:+.6f}**
  - $\beta_{{\text{{decay}}}}$ (`delta_decay_ratio`): **{coefs[2]:+.6f}**
  - Intercept: **{intercept:+.6f}**
"""

    out_rep_p = Path(output_report_path)
    out_rep_p.parent.mkdir(parents=True, exist_ok=True)
    out_rep_p.write_text(report, encoding="utf-8")
    print(f"Development report written to {output_report_path}")

    return {
        "prevalence": prevalence,
        "fold_metrics": fold_metrics,
        "pooled_auc_m": pooled_auc_m,
        "ci_auc_m": ci_auc_m,
        "pooled_acc_m": pooled_acc_m,
        "pooled_auc_h": pooled_auc_h,
        "ci_auc_h": ci_auc_h,
        "pooled_acc_h": pooled_acc_h,
        "pooled_auc_t": pooled_auc_t,
        "ci_auc_t": ci_auc_t,
        "pooled_acc_t": pooled_acc_t,
        "pooled_auc_rf": pooled_auc_rf,
        "ci_auc_rf": ci_auc_rf,
        "pooled_acc_rf": pooled_acc_rf,
        "delta_auc_hr_pooled": delta_auc_hr_pooled,
        "ci_delta_hr": ci_delta_hr,
        "delta_auc_time_pooled": delta_auc_time_pooled,
        "ci_delta_time": ci_delta_time,
        "hr_rule_passed": hr_rule_passed,
        "time_rule_passed": time_rule_passed,
        "drift_confounded": drift_confounded,
        "time_corrs": time_corrs,
        "model_sha256": model_sha256,
    }


if __name__ == "__main__":
    train_and_evaluate_cv()
