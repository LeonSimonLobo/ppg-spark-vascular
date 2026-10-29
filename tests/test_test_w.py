"""Unit tests for CalFree test runner logic and verification checks."""

from pathlib import Path
import pytest
import numpy as np
from src.test_w import (
    verify_frozen_model_hashes,
    check_and_acquire_lock_pre_scoring,
    bootstrap_subject_paired,
    EXPECTED_W_HASH,
    FROZEN_W_PATH,
)


def test_verify_frozen_model_hashes():
    """Verify that frozen models match pre-registered SHA-256 digests."""
    hashes = verify_frozen_model_hashes()
    assert hashes["Primary Morphology (W)"] == EXPECTED_W_HASH


def test_frozen_model_hash_mismatch_raises(monkeypatch):
    """Verify that hash mismatch raises ValueError."""
    monkeypatch.setattr("src.test_w.EXPECTED_W_HASH", "wrong_hash_123456789")
    with pytest.raises(ValueError, match="Frozen model hash mismatch"):
        verify_frozen_model_hashes()


def test_check_and_acquire_lock_pre_scoring(tmp_path, monkeypatch):
    """Verify pre-scoring lock creation and re-execution blocking."""
    fake_lock = tmp_path / ".test_locked_W"
    monkeypatch.setattr("src.test_w.LOCK_FILE_PATH", str(fake_lock))

    # Initial acquisition succeeds and writes file
    assert not fake_lock.exists()
    check_and_acquire_lock_pre_scoring()
    assert fake_lock.exists()
    assert "STATUS: LOCKED_PRE_SCORING" in fake_lock.read_text()

    # Second acquisition raises RuntimeError
    with pytest.raises(RuntimeError, match="Test execution is LOCKED"):
        check_and_acquire_lock_pre_scoring()


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
