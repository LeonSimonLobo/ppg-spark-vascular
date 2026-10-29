"""Unit tests for exploratory model comparison utilities."""

import numpy as np
import pytest
from sklearn.preprocessing import PolynomialFeatures


def test_interaction_terms_dimension():
    """Verify that pairwise interaction of 4 features produces 10 features."""
    X = np.ones((10, 4))
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_poly = poly.fit_transform(X)
    # 4 main effects + 6 pairwise interactions = 10 features
    assert X_poly.shape == (10, 10)


def test_bootstrap_paired_dimensions():
    """Verify 2D subject matrix bootstrap logic."""
    from src.test_w import bootstrap_subject_paired

    rng = np.random.default_rng(42)
    n_subjs = 10
    n_seg = 20
    y_sub = rng.integers(0, 2, size=(n_subjs, n_seg))
    scores_dict = {
        "model_a": rng.random(size=(n_subjs, n_seg)),
        "model_b": rng.random(size=(n_subjs, n_seg)),
    }
    # Test paired diff
    res = bootstrap_subject_paired(y_sub, scores_dict, n_resamples=50, seed=42)
    assert "model_a" in res
    assert "model_b" in res
