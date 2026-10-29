"""Unit test for sanity and robustness controls in controls_w.py."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.controls_w import run_controls


def test_run_controls_synthetic(tmp_path):
    np.random.seed(42)
    n_subjs = 10
    n_seg = 360
    n_total = n_subjs * n_seg

    sub_ids = [f"sub_{i:04d}" for i in range(n_subjs) for _ in range(n_seg)]
    sbp = np.random.normal(120, 15, size=n_total).astype(np.float32)

    delta_hr = np.random.normal(0, 5, size=n_total).astype(np.float32)
    delta_rise = np.random.normal(0, 0.02, size=n_total).astype(np.float32)
    delta_pw = np.random.normal(0, 0.03, size=n_total).astype(np.float32)
    delta_decay = np.random.normal(0, 0.4, size=n_total).astype(np.float32)

    df = pd.DataFrame({
        "segment_idx": np.arange(n_total),
        "subject_id": sub_ids,
        "split": ["train"] * n_total,
        "sbp": sbp,
        "delta_hr": delta_hr,
        "delta_systolic_rise_time": delta_rise,
        "delta_pulse_width": delta_pw,
        "delta_decay_ratio": delta_decay,
    })

    parquet_file = tmp_path / "test_ctrl_features.parquet"
    df.to_parquet(parquet_file, index=False)

    report_file = tmp_path / "test_ctrl_report.md"

    results = run_controls(
        parquet_path=str(parquet_file),
        output_report_path=str(report_file),
        n_splits=3,
        n_boot=50,
        seed=42,
    )

    assert 0.0 <= results["pooled_auc_gbt_hr"] <= 1.0
    assert abs(results["pooled_auc_null_hr"] - 0.5) < 0.15
    assert abs(results["pooled_auc_null_joint"] - 0.5) < 0.15
    assert abs(results["pooled_auc_null_resid"] - 0.5) < 0.15
    assert report_file.exists()
