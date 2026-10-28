"""Unit tests for within-subject gate metric computation."""

import numpy as np
import pandas as pd
import pytest

from src.gate_within_subject import compute_gate_metrics


def test_gate_metrics_synthetic():
    np.random.seed(42)
    n_subjs = 10
    n_per_subj = 30
    n_total = n_subjs * n_per_subj

    subject_ids = [f"sub_{i}" for i in range(n_subjs) for _ in range(n_per_subj)]
    # True underlying subject mean SBP and HR
    subj_sbp_mean = {f"sub_{i}": 110 + 5 * i for i in range(n_subjs)}
    subj_hr_mean = {f"sub_{i}": 65 + 2 * i for i in range(n_subjs)}

    # Strong positive within-subject correlation: delta_sbp = 1.5 * delta_hr + noise
    delta_hrs = np.random.normal(0, 5, n_total)
    delta_sbps = 1.5 * delta_hrs + np.random.normal(0, 2, n_total)

    hrs = np.array([subj_hr_mean[s] for s in subject_ids]) + delta_hrs
    sbps = np.array([subj_sbp_mean[s] for s in subject_ids]) + delta_sbps

    df = pd.DataFrame({
        "subject_id": subject_ids,
        "sbp": sbps,
    })

    metrics = compute_gate_metrics(df, hrs)

    assert metrics["plausible_pass"] is True
    assert metrics["spearman_corr"] > 0
    assert metrics["median_subj_corr"] > 0
    assert metrics["coef_hr"] > 0
    assert metrics["signal_pass"] is True
    assert metrics["gate_passed"] is True


def test_gate_metrics_negative_fails():
    np.random.seed(42)
    n_subjs = 10
    n_per_subj = 30
    n_total = n_subjs * n_per_subj

    subject_ids = [f"sub_{i}" for i in range(n_subjs) for _ in range(n_per_subj)]
    # Strong negative within-subject correlation
    delta_hrs = np.random.normal(0, 5, n_total)
    delta_sbps = -2.0 * delta_hrs + np.random.normal(0, 1, n_total)

    hrs = 70 + delta_hrs
    sbps = 120 + delta_sbps

    df = pd.DataFrame({
        "subject_id": subject_ids,
        "sbp": sbps,
    })

    metrics = compute_gate_metrics(df, hrs)

    assert metrics["spearman_corr"] < 0
    assert metrics["signal_pass"] is False
    assert metrics["gate_passed"] is False
