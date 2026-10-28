"""Unit tests for training-split beat-agreement gate computation."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src.gate_within_subject_agreement import fast_auc_with_ties, compute_agreement_metrics


def test_fast_auc_with_ties_matches_sklearn():
    np.random.seed(42)
    y = np.random.choice([0, 1], size=1000, p=[0.75, 0.25])
    # Scores with many ties
    x = np.random.choice([-6, 0, 6, 12, 18], size=1000)

    sk_auc = roc_auc_score(y, x)
    my_auc = fast_auc_with_ties(y, x)

    assert np.isclose(sk_auc, my_auc, atol=1e-5)


def test_compute_agreement_metrics_synthetic():
    np.random.seed(42)
    n_subjs = 10
    n_seg_per_subj = 360
    n_train = n_subjs * n_seg_per_subj

    subject_ids = [f"sub_{i:04d}" for i in range(n_subjs) for _ in range(n_seg_per_subj)]
    sbp = np.random.normal(125, 15, size=n_train)

    train_df = pd.DataFrame({
        "subject_id": subject_ids,
        "sbp": sbp,
    })

    # Synthetic beats with high agreement (85% diff in {-1, 0, 1})
    ppg_beats = np.random.randint(10, 15, size=n_train)
    noise = np.random.choice([-2, -1, 0, 1, 2], size=n_train, p=[0.05, 0.35, 0.25, 0.25, 0.10])
    abp_beats = ppg_beats + noise

    metrics = compute_agreement_metrics(
        train_df,
        ppg_beats,
        abp_beats,
        n_boot=100,
        seed=42,
    )

    assert metrics["n_train"] == n_train
    assert metrics["n_subjs"] == n_subjs
    assert 80.0 <= metrics["pass_pct"] <= 90.0
    assert metrics["pass_rule_80"] is True
    assert "ci_agree_overall" in metrics
    assert len(metrics["ci_agree_overall"]) == 2
    assert metrics["ci_agree_overall"][0] <= metrics["pass_pct"] <= metrics["ci_agree_overall"][1]
    assert 0.0 <= metrics["hr_auc"] <= 1.0
