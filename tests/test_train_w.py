"""Unit tests for development CV and nested models in train_w.py."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.train_w import train_and_evaluate_cv, train_and_evaluate_nested_cv


def create_synthetic_features(n_subjs=10, n_seg=360):
    np.random.seed(42)
    n_total = n_subjs * n_seg
    sub_ids = [f"sub_{i:04d}" for i in range(n_subjs) for _ in range(n_seg)]
    sbp = np.random.normal(120, 15, size=n_total).astype(np.float32)

    delta_hr = np.random.normal(0, 5, size=n_total).astype(np.float32)
    delta_rise = np.random.normal(0, 0.02, size=n_total).astype(np.float32)
    delta_pw = np.random.normal(0, 0.03, size=n_total).astype(np.float32)
    delta_decay = np.random.normal(0, 0.4, size=n_total).astype(np.float32)

    return pd.DataFrame({
        "segment_idx": np.arange(n_total),
        "subject_id": sub_ids,
        "split": ["train"] * n_total,
        "sbp": sbp,
        "delta_hr": delta_hr,
        "delta_systolic_rise_time": delta_rise,
        "delta_pulse_width": delta_pw,
        "delta_decay_ratio": delta_decay,
    })


def test_train_and_evaluate_nested_cv_synthetic(tmp_path):
    df = create_synthetic_features(n_subjs=10, n_seg=360)
    parquet_file = tmp_path / "test_nested_features.parquet"
    df.to_parquet(parquet_file, index=False)

    report_file = tmp_path / "test_nested_report.md"

    results = train_and_evaluate_nested_cv(
        parquet_path=str(parquet_file),
        output_report_path=str(report_file),
        n_splits=3,
        n_boot=50,
        seed=42,
    )

    assert results["prevalence"] == 50.0
    assert len(results["fold_metrics"]) == 3
    assert 0.0 <= results["pooled_auc_hr"] <= 1.0
    assert 0.0 <= results["pooled_auc_joint"] <= 1.0
    assert 0.0 <= results["pooled_auc_resid"] <= 1.0
    assert "ci_delta_joint" in results
    assert "ci_delta_resid" in results
    assert report_file.exists()
