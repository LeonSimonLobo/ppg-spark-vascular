"""Unit tests for CalFree test runner logic and verification checks."""

from pathlib import Path
import pytest
import numpy as np
from src.test_w import (
    verify_frozen_model_hash,
    verify_test_lock,
    bootstrap_subject_paired,
    EXPECTED_FROZEN_MODEL_HASH,
    FROZEN_MODEL_PATH,
)


def test_verify_frozen_model_hash():
    """Verify that models/w_frozen.pkl matches the pre-registered SHA-256."""
    actual_hash = verify_frozen_model_hash(FROZEN_MODEL_PATH)
    assert actual_hash == EXPECTED_FROZEN_MODEL_HASH


def test_verify_test_lock(tmp_path, monkeypatch):
    """Verify lock guard prevents re-execution."""
    # When lock file does not exist: passes
    fake_lock = tmp_path / ".test_locked_W"
    monkeypatch.setattr("src.test_w.LOCK_FILE_PATH", str(fake_lock))
    verify_test_lock()

    # When lock file exists: raises RuntimeError
    fake_lock.write_text("LOCKED")
    with pytest.raises(RuntimeError, match="Test execution is locked"):
        verify_test_lock()


def test_bootstrap_subject_paired_synthetic():
    """Verify subject-level paired bootstrap logic."""
    rng = np.random.default_rng(42)
    n_subjs = 20
    n_seg = 50
    # Synthetic target and scores
    y_sub = rng.integers(0, 2, size=(n_subjs, n_seg))
    # Model 1 has slight signal
    s_m1 = y_sub * 0.3 + rng.normal(0, 1, size=(n_subjs, n_seg))
    # Model 2 has strong signal
    s_m2 = y_sub * 0.8 + rng.normal(0, 1, size=(n_subjs, n_seg))

    scores_dict = {"hr": s_m1, "morphology": s_m2}
    res = bootstrap_subject_paired(y_sub, scores_dict, n_resamples=100, seed=42)

    assert "hr" in res
    assert "morphology" in res
    assert "paired_differences" in res
    assert "morphology_minus_hr" in res["paired_differences"]
    diff = res["paired_differences"]["morphology_minus_hr"]["diff"]
    assert diff > 0.0  # Model 2 should beat Model 1
