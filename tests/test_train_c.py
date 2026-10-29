"""Unit tests for Study C training, 5-fold grouped CV, metrics, and bootstrap (PLAN_C.md)."""

import numpy as np
import pandas as pd
import pytest

from src.train_c import (
    compute_metrics,
    clustered_subject_bootstrap_metrics,
    train_and_eval_cv,
    fit_and_freeze_full_models,
)


def create_synthetic_cohort(
    n_subjs: int = 20,
    segs_per_subj: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic development cohort for fast unit testing."""
    rng = np.random.default_rng(seed)
    n_rows = n_subjs * segs_per_subj

    subjs = [f"subj_{i:03d}" for i in range(n_subjs) for _ in range(segs_per_subj)]
    ages = [40.0 + (i % 30) for i in range(n_subjs) for _ in range(segs_per_subj)]
    sexes = [float(i % 2) for i in range(n_subjs) for _ in range(segs_per_subj)]
    bmis = [22.0 + (i % 10) for i in range(n_subjs) for _ in range(segs_per_subj)]

    hr_ibis = rng.normal(74.0, 9.0, n_rows).astype(np.float32)
    hrs = rng.normal(75.0, 10.0, n_rows).astype(np.float32)
    ibi_sds = rng.uniform(0.01, 0.05, n_rows).astype(np.float32)
    rises = rng.normal(0.22, 0.04, n_rows).astype(np.float32)
    pws = rng.normal(0.24, 0.04, n_rows).astype(np.float32)
    decays = rng.normal(2.7, 0.5, n_rows).astype(np.float32)
    beats = (hrs / 6.0).astype(np.float32)
    clipped = rng.choice([0.0, 1.0], p=[0.95, 0.05], size=n_rows).astype(np.float32)
    snrs = rng.uniform(0.7, 0.99, n_rows).astype(np.float32)

    # Some NaNs in morphology to test imputation
    nan_mask = rng.choice([True, False], p=[0.05, 0.95], size=n_rows)
    rises[nan_mask] = np.nan
    pws[nan_mask] = np.nan

    # Target SBP
    sbps = 100.0 + 0.3 * np.array(ages) + 0.2 * hrs + rng.normal(0, 5, n_rows)

    df = pd.DataFrame({
        "subject_id": subjs,
        "segment_idx": np.arange(n_rows, dtype=np.int32),
        "age": np.array(ages, dtype=np.float32),
        "sex": np.array(sexes, dtype=np.float32),
        "bmi": np.array(bmis, dtype=np.float32),
        "hr_ibi": hr_ibis,
        "hr": hrs,
        "ibi_sd": ibi_sds,
        "systolic_rise_time": rises,
        "pulse_width": pws,
        "decay_ratio": decays,
        "beat_count": beats,
        "is_clipped": clipped,
        "snr_ratio": snrs,
        "sbp": sbps.astype(np.float32),
    })
    df["y_120"] = (df["sbp"] >= 120.0).astype(int)
    df["y_130"] = (df["sbp"] >= 130.0).astype(int)
    return df


def test_compute_metrics():
    """Test classification metrics computation."""
    y_true = np.array([0, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.4, 0.6, 0.8, 0.9])

    metrics = compute_metrics(y_true, y_prob, threshold=0.5)
    assert metrics["accuracy"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["auroc"] == 1.0
    assert metrics["majority_accuracy"] == 0.6
    assert metrics["prevalence"] == 0.6


def test_clustered_subject_bootstrap_metrics():
    """Test clustered subject bootstrap with paired differences."""
    df = create_synthetic_cohort(n_subjs=15, segs_per_subj=10)
    subjs = df["subject_id"].to_numpy()
    y_true = df["y_120"].to_numpy()

    # Create dummy probabilities
    prob_dict = {
        "C0_lr": np.full(len(df), 0.4),
        "C1_lr": np.where(y_true == 1, 0.7, 0.3),
        "C2_lr": np.where(y_true == 1, 0.8, 0.2),
    }

    model_cis, paired_cis = clustered_subject_bootstrap_metrics(
        subjs, y_true, prob_dict, threshold=0.5, n_boot=100, seed=42
    )

    assert "C0_lr" in model_cis
    assert "C1_lr" in model_cis
    assert "C2_lr" in model_cis
    assert "delta_acc_C2_lr_minus_C0_lr" in paired_cis
    assert "delta_auc_C2_lr_minus_C0_lr" in paired_cis

    # Since C2_lr has higher accuracy than C0_lr, delta should be positive
    low, high = paired_cis["delta_acc_C2_lr_minus_C0_lr"]
    assert low > 0.0


def test_cv_and_freezing(tmp_path):
    """Test 5-fold CV with seeded permutation and full model freezing on synthetic cohort."""
    df = create_synthetic_cohort(n_subjs=15, segs_per_subj=10)

    oof_probs, pooled_metrics, fold_metrics = train_and_eval_cv(
        df, target_col="y_120", n_splits=3, seed=42
    )

    assert len(fold_metrics) == 3
    assert "C0_lr" in pooled_metrics
    assert "C1_lr" in pooled_metrics
    assert "C2_gbt" in pooled_metrics
    assert 0.0 <= pooled_metrics["C2_gbt"]["accuracy"] <= 1.0

    # Test freezing
    frozen_dict = fit_and_freeze_full_models(
        df, target_col="y_120", models_dir=str(tmp_path), seed=42
    )

    assert "C0_lr" in frozen_dict
    assert "C1_lr" in frozen_dict
    assert "C2_gbt" in frozen_dict
    path, sha = frozen_dict["C2_gbt"]
    assert len(sha) == 64
