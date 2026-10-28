"""Baseline demographic models for hypertensive-range SBP classification.

Pre-registered Steps:
- Step 0a: Age only, logistic regression with standardized input.
- Step 0: Age, sex, and BMI, logistic regression with standardized inputs.

Per PLAN.md:
- Covariates are per-subject medians of segment values.
- Standardized inputs fit on training split only (split == 'train').
- Evaluated on validation split only (split == 'val').
- Effective sample size is 1,293 subjects across 465,480 segments (360 segments/subject).
"""

from typing import Dict, Tuple, Any, List
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score


def compute_subject_medians(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-subject medians for age, sex, and BMI and broadcast back to segments."""
    sub_med = df.groupby("subject_id")[["age", "sex", "bmi"]].median()
    df_out = df.copy()
    df_out["age_sub"] = df_out["subject_id"].map(sub_med["age"])
    df_out["sex_sub"] = df_out["subject_id"].map(sub_med["sex"])
    df_out["bmi_sub"] = df_out["subject_id"].map(sub_med["bmi"])
    return df_out


def fast_auc_and_balanced_acc(
    p_s: np.ndarray,
    pos_s: np.ndarray,
    neg_s: np.ndarray,
    threshold: float,
) -> Tuple[float, float]:
    """Compute exact AUROC and balanced accuracy from grouped subject-level counts."""
    total_pos = float(np.sum(pos_s))
    total_neg = float(np.sum(neg_s))
    if total_pos == 0 or total_neg == 0:
        return np.nan, np.nan

    distinct_p, inv = np.unique(p_s, return_inverse=True)
    pos_grp = np.bincount(inv, weights=pos_s)
    neg_grp = np.bincount(inv, weights=neg_s)

    cum_neg = np.cumsum(neg_grp)
    strictly_smaller_neg = np.empty_like(cum_neg)
    strictly_smaller_neg[0] = 0
    strictly_smaller_neg[1:] = cum_neg[:-1]

    auc = float(np.sum(pos_grp * (strictly_smaller_neg + 0.5 * neg_grp)) / (total_pos * total_neg))

    pred_pos = distinct_p >= threshold
    tp = float(np.sum(pos_grp[pred_pos]))
    fn = total_pos - tp
    fp = float(np.sum(neg_grp[pred_pos]))
    tn = total_neg - fp

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    bacc = float(0.5 * (sens + spec))
    return auc, bacc


def run_subject_bootstrap(
    p_subj: np.ndarray,
    pos_subj: np.ndarray,
    neg_subj: np.ndarray,
    threshold: float,
    n_boot: int = 2000,
    seed: int = 42,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Compute 95% subject-level bootstrap CIs for AUROC and balanced accuracy."""
    rng = np.random.default_rng(seed)
    n_sub = len(p_subj)
    aucs = np.empty(n_boot)
    baccs = np.empty(n_boot)

    for b in range(n_boot):
        idx = rng.choice(n_sub, size=n_sub, replace=True)
        aucs[b], baccs[b] = fast_auc_and_balanced_acc(
            p_subj[idx], pos_subj[idx], neg_subj[idx], threshold
        )

    ci_auc = (float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5)))
    ci_bacc = (float(np.percentile(baccs, 2.5)), float(np.percentile(baccs, 97.5)))
    return ci_auc, ci_bacc


def fit_and_evaluate_baselines(
    parquet_path: str = "data/parquet/train_file.parquet",
    sbp_threshold: float = 130.0,
    n_boot: int = 2000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Fit Step 0a and Step 0 on train split and evaluate on val split."""
    table = pq.read_table(
        parquet_path,
        columns=["segment_idx", "subject_id", "split", "age", "sex", "bmi", "sbp"],
    )
    df = table.to_pandas()
    df = compute_subject_medians(df)
    df["label"] = (df["sbp"] >= sbp_threshold).astype(int)

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()

    n_train_subjs = int(train_df["subject_id"].nunique())
    n_val_subjs = int(val_df["subject_id"].nunique())
    total_subjs = n_train_subjs + n_val_subjs

    # Step 0a: Age only
    scaler_0a = StandardScaler().fit(train_df[["age_sub"]])
    clf_0a = LogisticRegression(random_state=seed, solver="lbfgs").fit(
        scaler_0a.transform(train_df[["age_sub"]]), train_df["label"]
    )
    val_p_0a = clf_0a.predict_proba(scaler_0a.transform(val_df[["age_sub"]]))[:, 1]
    val_df["p_0a"] = val_p_0a

    # Step 0: Age, Sex, BMI
    scaler_0 = StandardScaler().fit(train_df[["age_sub", "sex_sub", "bmi_sub"]])
    clf_0 = LogisticRegression(random_state=seed, solver="lbfgs").fit(
        scaler_0.transform(train_df[["age_sub", "sex_sub", "bmi_sub"]]), train_df["label"]
    )
    val_p_0 = clf_0.predict_proba(scaler_0.transform(val_df[["age_sub", "sex_sub", "bmi_sub"]]))[:, 1]
    val_df["p_0"] = val_p_0

    # Optimal F1 threshold search on validation
    threshold_grid = np.linspace(0.01, 0.99, 197)

    f1s_0a = [
        f1_score(val_df["label"], (val_p_0a >= th).astype(int), zero_division=0)
        for th in threshold_grid
    ]
    best_th_0a = float(threshold_grid[np.argmax(f1s_0a)])
    best_f1_0a = float(np.max(f1s_0a))
    auc_0a = float(roc_auc_score(val_df["label"], val_p_0a))
    bacc_0a = float(balanced_accuracy_score(val_df["label"], (val_p_0a >= best_th_0a).astype(int)))

    f1s_0 = [
        f1_score(val_df["label"], (val_p_0 >= th).astype(int), zero_division=0)
        for th in threshold_grid
    ]
    best_th_0 = float(threshold_grid[np.argmax(f1s_0)])
    best_f1_0 = float(np.max(f1s_0))
    auc_0 = float(roc_auc_score(val_df["label"], val_p_0))
    bacc_0 = float(balanced_accuracy_score(val_df["label"], (val_p_0 >= best_th_0).astype(int)))

    # Subject aggregation for bootstrap
    val_subjs = np.array(list(val_df["subject_id"].unique()))
    subj_summary = val_df.groupby("subject_id").agg(
        p_0a=("p_0a", "first"),
        p_0=("p_0", "first"),
        n_pos=("label", "sum"),
        n_neg=("label", lambda s: len(s) - s.sum()),
    ).loc[val_subjs]

    ci_auc_0a, ci_bacc_0a = run_subject_bootstrap(
        subj_summary["p_0a"].values,
        subj_summary["n_pos"].values,
        subj_summary["n_neg"].values,
        threshold=best_th_0a,
        n_boot=n_boot,
        seed=seed,
    )

    ci_auc_0, ci_bacc_0 = run_subject_bootstrap(
        subj_summary["p_0"].values,
        subj_summary["n_pos"].values,
        subj_summary["n_neg"].values,
        threshold=best_th_0,
        n_boot=n_boot,
        seed=seed,
    )

    # Sensitivity analysis excluding p003938_1
    val_df_sens = val_df[val_df["subject_id"] != "p003938_1"]
    p_0a_sens = clf_0a.predict_proba(scaler_0a.transform(val_df_sens[["age_sub"]]))[:, 1]
    p_0_sens = clf_0.predict_proba(scaler_0.transform(val_df_sens[["age_sub", "sex_sub", "bmi_sub"]]))[:, 1]
    auc_0a_sens = float(roc_auc_score(val_df_sens["label"], p_0a_sens))
    auc_0_sens = float(roc_auc_score(val_df_sens["label"], p_0_sens))

    return {
        "n_train_subjs": n_train_subjs,
        "n_train_segments": len(train_df),
        "n_val_subjs": n_val_subjs,
        "n_val_segments": len(val_df),
        "total_subjs": total_subjs,
        "train_prevalence": float(train_df["label"].mean()),
        "val_prevalence": float(val_df["label"].mean()),
        "step_0a": {
            "name": "Step 0a (Age only)",
            "auroc": auc_0a,
            "auroc_ci": ci_auc_0a,
            "best_threshold": best_th_0a,
            "f1": best_f1_0a,
            "balanced_accuracy": bacc_0a,
            "bacc_ci": ci_bacc_0a,
            "coef": float(clf_0a.coef_[0, 0]),
            "intercept": float(clf_0a.intercept_[0]),
            "sens_auroc": auc_0a_sens,
            "sens_delta": auc_0a_sens - auc_0a,
        },
        "step_0": {
            "name": "Step 0 (Age, Sex, BMI)",
            "auroc": auc_0,
            "auroc_ci": ci_auc_0,
            "best_threshold": best_th_0,
            "f1": best_f1_0,
            "balanced_accuracy": bacc_0,
            "bacc_ci": ci_bacc_0,
            "coefs": {
                "age": float(clf_0.coef_[0, 0]),
                "sex": float(clf_0.coef_[0, 1]),
                "bmi": float(clf_0.coef_[0, 2]),
            },
            "intercept": float(clf_0.intercept_[0]),
            "sens_auroc": auc_0_sens,
            "sens_delta": auc_0_sens - auc_0,
        },
        "models": {
            "scaler_0a": scaler_0a,
            "clf_0a": clf_0a,
            "scaler_0": scaler_0,
            "clf_0": clf_0,
        },
    }
