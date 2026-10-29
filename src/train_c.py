"""Cross-subject model training, 5-fold grouped CV, and validation pipeline (PLAN_C.md).

Models (Amendment C1 Ladder):
- C0: Demographics baseline (age, sex, bmi)
- C1: Demographics + Heart Rate from IBI (age, sex, bmi, hr_ibi)
- C2m: C1 + morphology only (systolic_rise_time, pulse_width, decay_ratio)
- C2: C1 + morphology + auxiliary segment fields (systolic_rise_time, pulse_width, decay_ratio,
      beat_count, snr_ratio, ibi_sd)

Model Architectures (fixed settings, zero tuning):
- Logistic Regression: regParam 0.01 (C = 100.0, l2 penalty), StandardScaler fit on train fold
- GBT: HistGradientBoostingClassifier(max_iter=100, max_depth=5, random_state=42)

Primary target: SBP >= 120 mmHg (prevalence 37.7% in Train)
Sensitivity target: SBP >= 130 mmHg
Threshold: Fixed standard threshold 0.5 (p >= 0.5 -> 1)
Evaluated with 2,000 clustered subject-level bootstrap resamples (seed 42).
"""

import hashlib
from pathlib import Path
import pickle
import time
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.features_c import (
    PlanCImputer,
    get_feature_matrix,
    C0_COLS,
    C1_COLS,
    C2m_COLS,
    C2_COLS,
    ALL_EXTRACTED_COLS,
)
from src.prospective_and_scale_w import fast_auc_with_ties

LADDER_STEPS = ["C0", "C1", "C2m", "C2"]


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute standard classification metrics at a fixed threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    pos_mask = (y_true == 1)
    neg_mask = (y_true == 0)

    n_pos = np.sum(pos_mask)
    n_neg = np.sum(neg_mask)

    accuracy = float(np.mean(y_pred == y_true))
    majority_accuracy = float(max(n_pos, n_neg) / len(y_true))

    tpr = float(np.sum(y_pred[pos_mask] == 1) / n_pos) if n_pos > 0 else 0.0
    tnr = float(np.sum(y_pred[neg_mask] == 0) / n_neg) if n_neg > 0 else 0.0
    balanced_accuracy = 0.5 * (tpr + tnr)

    auroc = fast_auc_with_ties(y_true, y_prob)

    return {
        "accuracy": accuracy,
        "majority_accuracy": majority_accuracy,
        "balanced_accuracy": balanced_accuracy,
        "auroc": auroc,
        "prevalence": float(np.mean(y_true)),
        "predicted_positive_rate": float(np.mean(y_pred)),
        "tpr": tpr,
        "tnr": tnr,
    }


def clustered_subject_bootstrap_metrics(
    subject_ids: np.ndarray,
    y_true: np.ndarray,
    prob_dict: Dict[str, np.ndarray],
    threshold: float = 0.5,
    n_boot: int = 2000,
    seed: int = 42,
) -> Tuple[Dict[str, Dict[str, Tuple[float, float]]], Dict[str, Tuple[float, float]]]:
    """Compute clustered subject-level bootstrap percentile CIs for each model and paired differences.

    Returns:
    - per_model_cis: dict of {model_name: {metric_name: (ci_low, ci_high)}}
    - paired_cis: dict of {comparison_name: (ci_low, ci_high)}
    """
    unique_subjs, inv_idx = np.unique(subject_ids, return_inverse=True)
    n_subjs = len(unique_subjs)

    subj_indices = [np.where(inv_idx == s)[0] for s in range(n_subjs)]
    rng = np.random.default_rng(seed)

    model_names = list(prob_dict.keys())
    boot_accs = {m: np.zeros(n_boot, dtype=np.float64) for m in model_names}
    boot_bal_accs = {m: np.zeros(n_boot, dtype=np.float64) for m in model_names}
    boot_aurocs = {m: np.zeros(n_boot, dtype=np.float64) for m in model_names}

    for b in range(n_boot):
        sub_sample = rng.choice(n_subjs, size=n_subjs, replace=True)
        b_idx = np.concatenate([subj_indices[s] for s in sub_sample])

        b_y = y_true[b_idx]
        b_pos = (b_y == 1)
        b_neg = (b_y == 0)
        n_pos = np.sum(b_pos)
        n_neg = np.sum(b_neg)

        for m in model_names:
            b_p = prob_dict[m][b_idx]
            b_pred = (b_p >= threshold).astype(int)

            acc = np.mean(b_pred == b_y)
            tpr = np.sum(b_pred[b_pos] == 1) / n_pos if n_pos > 0 else 0.0
            tnr = np.sum(b_pred[b_neg] == 0) / n_neg if n_neg > 0 else 0.0
            bal_acc = 0.5 * (tpr + tnr)
            auc = fast_auc_with_ties(b_y, b_p)

            boot_accs[m][b] = acc
            boot_bal_accs[m][b] = bal_acc
            boot_aurocs[m][b] = auc

    # Extract 95% CIs [2.5, 97.5]
    per_model_cis = {}
    for m in model_names:
        per_model_cis[m] = {
            "accuracy": (float(np.percentile(boot_accs[m], 2.5)), float(np.percentile(boot_accs[m], 97.5))),
            "balanced_accuracy": (float(np.percentile(boot_bal_accs[m], 2.5)), float(np.percentile(boot_bal_accs[m], 97.5))),
            "auroc": (float(np.percentile(boot_aurocs[m], 2.5)), float(np.percentile(boot_aurocs[m], 97.5))),
        }

    # Stepwise paired differences (C1 - C0, C2m - C1, C2 - C2m, C2 - C0)
    paired_cis = {}
    for model_family in ["lr", "gbt"]:
        m0 = f"C0_{model_family}"
        m1 = f"C1_{model_family}"
        m2m = f"C2m_{model_family}"
        m2 = f"C2_{model_family}"

        # 1. Gain from Heart Rate (C1 - C0)
        if m0 in model_names and m1 in model_names:
            diff_acc_1_0 = boot_accs[m1] - boot_accs[m0]
            diff_auc_1_0 = boot_aurocs[m1] - boot_aurocs[m0]
            paired_cis[f"delta_acc_{m1}_minus_{m0}"] = (float(np.percentile(diff_acc_1_0, 2.5)), float(np.percentile(diff_acc_1_0, 97.5)))
            paired_cis[f"delta_auc_{m1}_minus_{m0}"] = (float(np.percentile(diff_auc_1_0, 2.5)), float(np.percentile(diff_auc_1_0, 97.5)))

        # 2. Gain from Pulse Waveform Morphology Shape (C2m - C1)
        if m1 in model_names and m2m in model_names:
            diff_acc_2m_1 = boot_accs[m2m] - boot_accs[m1]
            diff_auc_2m_1 = boot_aurocs[m2m] - boot_aurocs[m1]
            paired_cis[f"delta_acc_{m2m}_minus_{m1}"] = (float(np.percentile(diff_acc_2m_1, 2.5)), float(np.percentile(diff_acc_2m_1, 97.5)))
            paired_cis[f"delta_auc_{m2m}_minus_{m1}"] = (float(np.percentile(diff_auc_2m_1, 2.5)), float(np.percentile(diff_auc_2m_1, 97.5)))

        # 3. Gain from Signal Quality Fields (C2 - C2m)
        if m2m in model_names and m2 in model_names:
            diff_acc_2_2m = boot_accs[m2] - boot_accs[m2m]
            diff_auc_2_2m = boot_aurocs[m2] - boot_aurocs[m2m]
            paired_cis[f"delta_acc_{m2}_minus_{m2m}"] = (float(np.percentile(diff_acc_2_2m, 2.5)), float(np.percentile(diff_acc_2_2m, 97.5)))
            paired_cis[f"delta_auc_{m2}_minus_{m2m}"] = (float(np.percentile(diff_auc_2_2m, 2.5)), float(np.percentile(diff_auc_2_2m, 97.5)))

        # 4. Total Pipeline Gain (C2 - C0)
        if m0 in model_names and m2 in model_names:
            diff_acc_2_0 = boot_accs[m2] - boot_accs[m0]
            diff_auc_2_0 = boot_aurocs[m2] - boot_aurocs[m0]
            paired_cis[f"delta_acc_{m2}_minus_{m0}"] = (float(np.percentile(diff_acc_2_0, 2.5)), float(np.percentile(diff_acc_2_0, 97.5)))
            paired_cis[f"delta_auc_{m2}_minus_{m0}"] = (float(np.percentile(diff_auc_2_0, 2.5)), float(np.percentile(diff_auc_2_0, 97.5)))

    return per_model_cis, paired_cis


def train_and_eval_cv(
    train_df: pd.DataFrame,
    target_col: str = "y_120",
    n_splits: int = 5,
    seed: int = 42,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, float]], List[Dict[str, Any]]]:
    """Execute 5-fold cross-validation with seeded permutation of subjects and disjoint assertion.

    Fits C0, C1, C2m, C2 for Logistic Regression (C=100.0) and GBT.
    Returns:
    - oof_probs: dict of {model_name: oof_prob_array}
    - pooled_metrics: dict of {model_name: metric_dict}
    - fold_metrics: list of per-fold metric dictionaries
    """
    unique_subjs = np.array(sorted(train_df["subject_id"].unique()))
    n_subjs = len(unique_subjs)
    n_rows = len(train_df)

    print(f"Starting {n_splits}-fold CV across {n_subjs:,} subjects ({n_rows:,} segments) with seed={seed}...")

    # Seeded permutation of unique subjects
    rng = np.random.default_rng(seed)
    permuted_subjs = unique_subjs.copy()
    rng.shuffle(permuted_subjs)

    subj_folds = np.array_split(permuted_subjs, n_splits)
    subj_to_fold = {}
    for f_idx, f_subjs in enumerate(subj_folds):
        for s in f_subjs:
            subj_to_fold[s] = f_idx

    fold_assignments = train_df["subject_id"].map(subj_to_fold).to_numpy()

    model_keys = []
    for step in LADDER_STEPS:
        model_keys.extend([f"{step}_lr", f"{step}_gbt"])

    oof_probs = {k: np.zeros(n_rows, dtype=np.float32) for k in model_keys}
    fold_metrics = []
    y_all = train_df[target_col].to_numpy(dtype=int)

    for fold_idx in range(n_splits):
        t_fold = time.time()
        train_mask = (fold_assignments != fold_idx)
        val_mask = (fold_assignments == fold_idx)

        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]

        fold_train_df = train_df.iloc[train_idx].copy()
        fold_val_df = train_df.iloc[val_idx].copy()

        # Assert subject disjointness
        train_subjs = set(fold_train_df["subject_id"])
        val_subjs = set(fold_val_df["subject_id"])
        leakage = train_subjs & val_subjs
        assert len(leakage) == 0, f"Subject leakage detected in fold {fold_idx + 1}: {len(leakage)} overlapping subjects!"

        # Fit Imputer strictly on training fold
        imputer = PlanCImputer()
        imputer.fit(fold_train_df)
        df_tr_imp = imputer.transform(fold_train_df)
        df_val_imp = imputer.transform(fold_val_df)

        y_tr = fold_train_df[target_col].to_numpy(dtype=int)
        y_val = fold_val_df[target_col].to_numpy(dtype=int)

        fold_record = {
            "fold": fold_idx + 1,
            "n_train_subjs": len(train_subjs),
            "n_val_subjs": len(val_subjs),
            "n_train": len(y_tr),
            "n_val": len(y_val),
        }

        for step in LADDER_STEPS:
            X_tr, feat_names = get_feature_matrix(df_tr_imp, step)
            X_val, _ = get_feature_matrix(df_val_imp, step)

            # Fit Logistic Regression (regParam 0.01 -> C=100.0) with StandardScaler
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_val_s = scaler.transform(X_val)

            lr = LogisticRegression(
                C=100.0,
                penalty="l2",
                solver="lbfgs",
                max_iter=1000,
                random_state=seed,
            )
            lr.fit(X_tr_s, y_tr)
            prob_lr = lr.predict_proba(X_val_s)[:, 1]
            oof_probs[f"{step}_lr"][val_idx] = prob_lr.astype(np.float32)

            # Fit GBT (HistGradientBoostingClassifier, max_iter=100, max_depth=5)
            gbt = HistGradientBoostingClassifier(
                max_iter=100,
                max_depth=5,
                random_state=seed,
            )
            gbt.fit(X_tr, y_tr)
            prob_gbt = gbt.predict_proba(X_val)[:, 1]
            oof_probs[f"{step}_gbt"][val_idx] = prob_gbt.astype(np.float32)

            fold_record[f"{step}_lr_acc"] = float(np.mean((prob_lr >= 0.5) == y_val))
            fold_record[f"{step}_lr_auc"] = fast_auc_with_ties(y_val, prob_lr)
            fold_record[f"{step}_gbt_acc"] = float(np.mean((prob_gbt >= 0.5) == y_val))
            fold_record[f"{step}_gbt_auc"] = fast_auc_with_ties(y_val, prob_gbt)

        elapsed = time.time() - t_fold
        print(f"Fold {fold_idx + 1}/{n_splits} complete in {elapsed:.1f}s ({len(val_subjs)} validation subjects).")
        fold_metrics.append(fold_record)

    # Compute pooled metrics across all out-of-fold predictions
    pooled_metrics = {}
    for k in model_keys:
        pooled_metrics[k] = compute_metrics(y_all, oof_probs[k], threshold=0.5)

    return oof_probs, pooled_metrics, fold_metrics


def fit_and_freeze_full_models(
    train_df: pd.DataFrame,
    target_col: str = "y_120",
    models_dir: str = "models",
    seed: int = 42,
) -> Dict[str, Tuple[str, str]]:
    """Fit final C0, C1, C2m, C2 models on entire development set and freeze to disk.

    Returns:
    - dict of {model_key: (file_path, sha256_hash)}
    """
    out_dir = Path(models_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Fitting full imputer and freezing models on all development data...")
    imputer = PlanCImputer()
    imputer.fit(train_df)
    df_imp = imputer.transform(train_df)
    y_full = train_df[target_col].to_numpy(dtype=int)

    # Save fitted imputer
    imputer_path = out_dir / "c_imputer_frozen.pkl"
    with open(imputer_path, "wb") as f:
        pickle.dump(imputer, f)

    frozen_dict = {}

    for step in LADDER_STEPS:
        X_full, feat_names = get_feature_matrix(df_imp, step)

        # 1. Logistic Regression
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X_full)
        lr = LogisticRegression(
            C=100.0,
            penalty="l2",
            solver="lbfgs",
            max_iter=1000,
            random_state=seed,
        )
        lr.fit(X_s, y_full)

        lr_artifact = {
            "model": lr,
            "scaler": scaler,
            "step": step,
            "feature_names": feat_names,
            "target": target_col,
            "imputer_medians": imputer.medians_,
        }
        lr_path = out_dir / f"{step.lower()}_lr_frozen.pkl"
        with open(lr_path, "wb") as f:
            pickle.dump(lr_artifact, f)

        sha_lr = hashlib.sha256(lr_path.read_bytes()).hexdigest()
        frozen_dict[f"{step}_lr"] = (str(lr_path), sha_lr)
        print(f"Frozen {step}_lr: {lr_path} (SHA-256: {sha_lr})")

        # 2. GBT
        gbt = HistGradientBoostingClassifier(
            max_iter=100,
            max_depth=5,
            random_state=seed,
        )
        gbt.fit(X_full, y_full)

        gbt_artifact = {
            "model": gbt,
            "step": step,
            "feature_names": feat_names,
            "target": target_col,
            "imputer_medians": imputer.medians_,
        }
        gbt_path = out_dir / f"{step.lower()}_gbt_frozen.pkl"
        with open(gbt_path, "wb") as f:
            pickle.dump(gbt_artifact, f)

        sha_gbt = hashlib.sha256(gbt_path.read_bytes()).hexdigest()
        frozen_dict[f"{step}_gbt"] = (str(gbt_path), sha_gbt)
        print(f"Frozen {step}_gbt: {gbt_path} (SHA-256: {sha_gbt})")

    return frozen_dict


def evaluate_on_calfree(
    calfree_df: pd.DataFrame,
    frozen_models_dict: Dict[str, Tuple[str, str]],
    target_col: str = "y_120",
    seed: int = 42,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, float]], Dict[str, Any], Dict[str, Any]]:
    """Evaluate frozen models on CalFree extra validation cohort without refitting."""
    sample_key = list(frozen_models_dict.keys())[0]
    sample_path = frozen_models_dict[sample_key][0]
    with open(sample_path, "rb") as f:
        art = pickle.load(f)
    medians = art["imputer_medians"]

    imputer = PlanCImputer()
    imputer.medians_ = medians
    df_cf_imp = imputer.transform(calfree_df)

    y_cf = calfree_df[target_col].to_numpy(dtype=int)
    probs = {}

    for step in LADDER_STEPS:
        # Evaluate LR
        lr_path = frozen_models_dict[f"{step}_lr"][0]
        with open(lr_path, "rb") as f:
            art_lr = pickle.load(f)
        X_lr, _ = get_feature_matrix(df_cf_imp, step)
        X_lr_s = art_lr["scaler"].transform(X_lr)
        probs[f"{step}_lr"] = art_lr["model"].predict_proba(X_lr_s)[:, 1]

        # Evaluate GBT
        gbt_path = frozen_models_dict[f"{step}_gbt"][0]
        with open(gbt_path, "rb") as f:
            art_gbt = pickle.load(f)
        X_gbt, _ = get_feature_matrix(df_cf_imp, step)
        probs[f"{step}_gbt"] = art_gbt["model"].predict_proba(X_gbt)[:, 1]

    # Compute metrics
    cf_metrics = {k: compute_metrics(y_cf, probs[k], threshold=0.5) for k in probs}

    # Bootstrap CIs
    subjs = calfree_df["subject_id"].to_numpy()
    cf_model_cis, cf_paired_cis = clustered_subject_bootstrap_metrics(
        subjs, y_cf, probs, threshold=0.5, n_boot=2000, seed=seed
    )

    return probs, cf_metrics, cf_model_cis, cf_paired_cis
