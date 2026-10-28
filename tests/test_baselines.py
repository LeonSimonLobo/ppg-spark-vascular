"""Unit tests for baseline demographic models and bootstrap procedures."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from src.baselines import (
    compute_subject_medians,
    fast_auc_and_balanced_acc,
    run_subject_bootstrap,
)


def test_compute_subject_medians():
    """Verify that per-subject medians are correctly broadcast to all segments."""
    df = pd.DataFrame({
        "subject_id": ["s1", "s1", "s1", "s2", "s2"],
        "age": [50.0, 52.0, 54.0, 30.0, 32.0],
        "sex": [1.0, 1.0, 1.0, 0.0, 0.0],
        "bmi": [24.0, 26.0, 25.0, 22.0, 22.0],
    })
    res = compute_subject_medians(df)
    assert np.allclose(res["age_sub"], [52.0, 52.0, 52.0, 31.0, 31.0])
    assert np.allclose(res["sex_sub"], [1.0, 1.0, 1.0, 0.0, 0.0])
    assert np.allclose(res["bmi_sub"], [25.0, 25.0, 25.0, 22.0, 22.0])


def test_fast_auc_and_balanced_acc_exact_match():
    """Verify that grouped fast_auc_and_balanced_acc matches scikit-learn exactly."""
    p_subj = np.array([0.85, 0.60, 0.60, 0.30, 0.10])
    pos_subj = np.array([12, 8, 15, 3, 1])
    neg_subj = np.array([2, 10, 5, 20, 30])
    threshold = 0.50

    # Expand to segment level
    y_full = np.concatenate([np.ones(p) for p in pos_subj] + [np.zeros(n) for n in neg_subj])
    p_full = np.concatenate([np.full(p, pr) for p, pr in zip(pos_subj, p_subj)] +
                            [np.full(n, pr) for n, pr in zip(neg_subj, p_subj)])

    fast_auc, fast_bacc = fast_auc_and_balanced_acc(p_subj, pos_subj, neg_subj, threshold)
    sk_auc = roc_auc_score(y_full, p_full)
    sk_bacc = balanced_accuracy_score(y_full, (p_full >= threshold).astype(int))

    assert np.isclose(fast_auc, sk_auc, atol=1e-7)
    assert np.isclose(fast_bacc, sk_bacc, atol=1e-7)


def test_run_subject_bootstrap():
    """Verify bootstrap CI generation and boundary consistency."""
    rng = np.random.default_rng(42)
    n_sub = 50
    p_s = rng.uniform(0.1, 0.9, size=n_sub)
    pos_s = rng.binomial(300, 0.2, size=n_sub)
    neg_s = 300 - pos_s

    pt_auc, pt_bacc = fast_auc_and_balanced_acc(p_s, pos_s, neg_s, threshold=0.5)
    ci_auc, ci_bacc = run_subject_bootstrap(p_s, pos_s, neg_s, threshold=0.5, n_boot=200, seed=42)

    assert ci_auc[0] <= pt_auc <= ci_auc[1]
    assert ci_bacc[0] <= pt_bacc <= ci_bacc[1]
    assert 0.0 <= ci_auc[0] and ci_auc[1] <= 1.0
    assert 0.0 <= ci_bacc[0] and ci_bacc[1] <= 1.0
