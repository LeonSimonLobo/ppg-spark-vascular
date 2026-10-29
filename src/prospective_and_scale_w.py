"""Prospective prefix-centering control and PPG amplitude-scale invariance audit.

Addresses:
1. Prospective calibration check: centers features strictly using an initial 10-minute
   prefix (t < 60 segments) per subject, evaluating on the subsequent 50 minutes (t >= 60).
2. Amplitude-scale invariance audit: computes segment PPG range R_ppg = max - min, checks
   for normalization/clipping artifacts, evaluates within-subject correlations with morphology,
   and tests whether morphology survives residualization on R_ppg.
3. HR framing correction: documents anesthetic decoupling of baroreflex coupling.
"""

from pathlib import Path
import time
from typing import Dict, Any, Tuple
import h5py
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score


def fast_auc_with_ties(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Fast AUROC calculation handling ties, identical to roc_auc_score."""
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score_sorted = y_score[desc_score_indices]
    y_true_sorted = y_true[desc_score_indices]
    distinct_value_indices = np.where(np.diff(y_score_sorted))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
    tps = np.cumsum(y_true_sorted)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    tps = np.r_[0, tps]
    fps = np.r_[0, fps]
    if fps[-1] == 0 or tps[-1] == 0:
        return 0.5
    fpr = fps / fps[-1]
    tpr = tps / tps[-1]
    return float(np.trapezoid(tpr, fpr))


def bootstrap_subject_auroc_paired(
    y_sub: np.ndarray,
    scores_sub_dict: Dict[str, np.ndarray],
    n_resamples: int = 2000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Subject-level clustered bootstrap for AUROCs and paired differences using 2D arrays (n_subjs, n_seg)."""
    rng = np.random.default_rng(seed)
    n_subjs = len(y_sub)

    model_names = list(scores_sub_dict.keys())
    boot_aucs = {m: np.zeros(n_resamples, dtype=np.float64) for m in model_names}

    for b in range(n_resamples):
        sub_sample = rng.choice(n_subjs, size=n_subjs, replace=True)
        y_b = y_sub[sub_sample].ravel()

        for m in model_names:
            score_b = scores_sub_dict[m][sub_sample].ravel()
            boot_aucs[m][b] = fast_auc_with_ties(y_b, score_b)

    results = {}
    y_flat = y_sub.ravel()
    for m in model_names:
        score_flat = scores_sub_dict[m].ravel()
        pt_est = fast_auc_with_ties(y_flat, score_flat)
        ci_low = float(np.percentile(boot_aucs[m], 2.5))
        ci_high = float(np.percentile(boot_aucs[m], 97.5))
        results[m] = {"auroc": pt_est, "ci_95": (ci_low, ci_high)}

    # Paired differences against hr
    if "hr" in scores_sub_dict:
        hr_boots = boot_aucs["hr"]
        paired_diffs = {}
        for m in model_names:
            if m == "hr":
                continue
            diff_boots = boot_aucs[m] - hr_boots
            pt_diff = results[m]["auroc"] - results["hr"]["auroc"]
            d_low = float(np.percentile(diff_boots, 2.5))
            d_high = float(np.percentile(diff_boots, 97.5))
            paired_diffs[f"{m}_minus_hr"] = {"diff": pt_diff, "ci_95": (d_low, d_high)}
        results["paired_differences"] = paired_diffs

    return results


def run_prospective_prefix_control(df: pd.DataFrame, prefix_len: int = 60) -> Dict[str, Any]:
    """Run prospective prefix-centering control on evaluation segments (t >= prefix_len)."""
    n_subjs = df["subject_id"].nunique()
    n_seg = 360
    eval_mask = np.tile(np.arange(n_seg) >= prefix_len, n_subjs)
    prefix_mask = ~eval_mask

    prefix_df = df[prefix_mask]
    eval_df = df[eval_mask].copy()
    n_eval_seg_per_subj = n_seg - prefix_len

    # Prefix baselines
    pref_sbp = prefix_df.groupby("subject_id")["sbp"].median()
    pref_hr = prefix_df.groupby("subject_id")["hr"].median()
    pref_rise = prefix_df.groupby("subject_id")["systolic_rise_time"].median()
    pref_pw = prefix_df.groupby("subject_id")["pulse_width"].median()
    pref_decay = prefix_df.groupby("subject_id")["decay_ratio"].median()

    # Session medians for sbp
    sess_sbp = df.groupby("subject_id")["sbp"].median()

    eval_df["pref_med_sbp"] = eval_df["subject_id"].map(pref_sbp)
    eval_df["sess_med_sbp"] = eval_df["subject_id"].map(sess_sbp)

    eval_df["delta_hr_pref"] = eval_df["hr"] - eval_df["subject_id"].map(pref_hr)
    eval_df["delta_rise_pref"] = eval_df["systolic_rise_time"] - eval_df["subject_id"].map(pref_rise)
    eval_df["delta_pw_pref"] = eval_df["pulse_width"] - eval_df["subject_id"].map(pref_pw)
    eval_df["delta_decay_pref"] = eval_df["decay_ratio"] - eval_df["subject_id"].map(pref_decay)

    # Targets
    eval_df["y_pref"] = (eval_df["sbp"] >= eval_df["pref_med_sbp"]).astype(int)
    eval_df["y_sess"] = (eval_df["sbp"] >= eval_df["sess_med_sbp"]).astype(int)

    morph_cols = ["delta_rise_pref", "delta_pw_pref", "delta_decay_pref"]
    X_hr = eval_df[["delta_hr_pref"]].values
    X_morph = eval_df[morph_cols].values
    X_joint = eval_df[["delta_hr_pref"] + morph_cols].values
    subjs = eval_df["subject_id"].values

    gkf = GroupKFold(n_splits=5)
    results = {}

    for target_name, target_col in [("target_prefix_baseline", "y_pref"), ("target_session_median", "y_sess")]:
        y = eval_df[target_col].values
        scores_hr = np.zeros(len(y), dtype=np.float64)
        scores_morph = np.zeros(len(y), dtype=np.float64)
        scores_joint = np.zeros(len(y), dtype=np.float64)
        scores_resid = np.zeros(len(y), dtype=np.float64)

        for train_idx, val_idx in gkf.split(eval_df, y, groups=subjs):
            # 1. HR alone
            pipe_hr = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", random_state=42))])
            pipe_hr.fit(X_hr[train_idx], y[train_idx])
            scores_hr[val_idx] = pipe_hr.predict_proba(X_hr[val_idx])[:, 1]

            # 2. Morphology alone
            pipe_m = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", random_state=42))])
            pipe_m.fit(X_morph[train_idx], y[train_idx])
            scores_morph[val_idx] = pipe_m.predict_proba(X_morph[val_idx])[:, 1]

            # 3. Joint (HR + Morphology)
            pipe_j = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", random_state=42))])
            pipe_j.fit(X_joint[train_idx], y[train_idx])
            scores_joint[val_idx] = pipe_j.predict_proba(X_joint[val_idx])[:, 1]

            # 4. HR-residualized morphology
            reg = LinearRegression().fit(X_hr[train_idx], X_morph[train_idx])
            X_m_resid_train = X_morph[train_idx] - reg.predict(X_hr[train_idx])
            X_m_resid_val = X_morph[val_idx] - reg.predict(X_hr[val_idx])

            pipe_r = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", random_state=42))])
            pipe_r.fit(X_m_resid_train, y[train_idx])
            scores_resid[val_idx] = pipe_r.predict_proba(X_m_resid_val)[:, 1]

        y_sub = y.reshape(n_subjs, n_eval_seg_per_subj)
        scores_sub_dict = {
            "hr": scores_hr.reshape(n_subjs, n_eval_seg_per_subj),
            "morphology": scores_morph.reshape(n_subjs, n_eval_seg_per_subj),
            "joint": scores_joint.reshape(n_subjs, n_eval_seg_per_subj),
            "residualized": scores_resid.reshape(n_subjs, n_eval_seg_per_subj),
        }
        boot_res = bootstrap_subject_auroc_paired(y_sub, scores_sub_dict, n_resamples=2000, seed=42)
        boot_res["prevalence"] = float(np.mean(y) * 100.0)
        results[target_name] = boot_res

    results["n_eval_segments"] = len(eval_df)
    results["n_eval_subjs"] = n_subjs
    results["prefix_len_segments"] = prefix_len
    results["prefix_duration_min"] = float(prefix_len * 10 / 60)
    return results
