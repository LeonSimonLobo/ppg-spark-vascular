"""Unit tests for prefix-centering prospective control and amplitude invariance audit."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression


def test_prefix_centering():
    """Verify that prefix centering uses only prefix segments."""
    # Create 2 subjects with 10 segments each
    # Subject 1: prefix (t < 4) has mean 10, suffix (t >= 4) has mean 20
    # Subject 2: prefix (t < 4) has mean 100, suffix (t >= 4) has mean 150
    data = []
    for s_id, (pref_val, suff_val) in enumerate([(10.0, 20.0), (100.0, 150.0)]):
        for t in range(10):
            val = pref_val + np.random.RandomState(t).normal(0, 0.1) if t < 4 else suff_val + np.random.RandomState(t).normal(0, 0.1)
            data.append({
                "subject_id": f"s_{s_id}",
                "segment_idx": t,
                "feat": val,
                "sbp": val * 2.0,
            })
    df = pd.DataFrame(data)

    prefix_len = 4
    prefix_df = df[df["segment_idx"] < prefix_len]
    eval_df = df[df["segment_idx"] >= prefix_len].copy()

    # Prefix medians
    med_feat = prefix_df.groupby("subject_id")["feat"].median()
    eval_df["pref_med"] = eval_df["subject_id"].map(med_feat)
    eval_df["delta_pref"] = eval_df["feat"] - eval_df["pref_med"]

    # In eval segments, delta_pref should be approximately suff_val - pref_val
    s0_deltas = eval_df[eval_df["subject_id"] == "s_0"]["delta_pref"]
    s1_deltas = eval_df[eval_df["subject_id"] == "s_1"]["delta_pref"]

    assert np.allclose(s0_deltas.median(), 10.0, atol=0.5)
    assert np.allclose(s1_deltas.median(), 50.0, atol=0.5)


def test_amplitude_range_calculation():
    """Verify max - min calculation on synthetic array."""
    sig = np.sin(np.linspace(0, 2 * np.pi, 1250)) * 2.5 + 1.0  # min -1.5, max 3.5 -> range 5.0
    rng = np.ptp(sig)
    assert np.isclose(rng, 5.0, atol=1e-5)


def test_linear_residualization_orthogonality():
    """Verify that OLS residuals are orthogonal to the regressor."""
    rng = np.random.RandomState(42)
    n = 1000
    x = rng.normal(0, 1, size=(n, 1))
    y = 2.5 * x.ravel() + rng.normal(0, 0.5, size=n)

    reg = LinearRegression().fit(x, y)
    y_pred = reg.predict(x)
    resids = y - y_pred

    # Covariance/dot product between residuals and regressor should be ~0
    dot_prod = np.dot(x.ravel(), resids) / n
    assert np.isclose(dot_prod, 0.0, atol=1e-5)


def test_format_prospective_and_scale_report(tmp_path):
    """Verify markdown report generation from results dict."""
    from src.generate_prospective_report import format_prospective_and_scale_report

    mock_prefix = {
        "n_eval_subjs": 100,
        "target_prefix_baseline": {
            "prevalence": 49.5,
            "hr": {"auroc": 0.55, "ci_95": (0.53, 0.57)},
            "morphology": {"auroc": 0.67, "ci_95": (0.65, 0.69)},
            "joint": {"auroc": 0.70, "ci_95": (0.68, 0.72)},
            "residualized": {"auroc": 0.69, "ci_95": (0.67, 0.71)},
            "paired_differences": {
                "morphology_minus_hr": {"diff": 0.12, "ci_95": (0.10, 0.14)},
                "joint_minus_hr": {"diff": 0.15, "ci_95": (0.13, 0.17)},
                "residualized_minus_hr": {"diff": 0.14, "ci_95": (0.12, 0.16)},
            },
        },
        "target_session_median": {
            "prevalence": 50.0,
            "hr": {"auroc": 0.55, "ci_95": (0.53, 0.57)},
            "morphology": {"auroc": 0.67, "ci_95": (0.65, 0.69)},
            "joint": {"auroc": 0.70, "ci_95": (0.68, 0.72)},
            "residualized": {"auroc": 0.69, "ci_95": (0.67, 0.71)},
            "paired_differences": {
                "morphology_minus_hr": {"diff": 0.12, "ci_95": (0.10, 0.14)},
                "joint_minus_hr": {"diff": 0.15, "ci_95": (0.13, 0.17)},
                "residualized_minus_hr": {"diff": 0.14, "ci_95": (0.12, 0.16)},
            },
        },
    }
    out_p = tmp_path / "test_report.md"
    rep = format_prospective_and_scale_report(mock_prefix, {}, output_path=str(out_p))
    assert out_p.exists()
    assert "Prospective Calibration Control" in rep
    assert "PASS" in rep
