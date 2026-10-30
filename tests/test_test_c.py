"""Unit tests for Study C test runner pre-flight security gates and invariant checks."""

import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from src.test_c import (
    verify_preflight_gates,
    check_and_acquire_lock_pre_scoring,
    score_aami_cal_test,
    LOCK_FILE_PATH,
    REPORT_PATH,
    DEV_REPORT_PATH,
    PLAN_FILE_PATH,
)


def create_mock_environment(tmp_path: Path):
    """Helper to set up mock files for gating tests."""
    plan_file = tmp_path / "PLAN_C.md"
    plan_content = b"# Study C Specification Mock\n"
    plan_file.write_bytes(plan_content)
    plan_hash = hashlib.sha256(plan_content).hexdigest()

    # Create dummy frozen model files
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    model_paths = {}
    model_hashes = {}
    for key in ["c0_lr", "c1_lr", "c2m_lr", "c2_lr", "c0_gbt", "c1_gbt", "c2m_gbt", "c2_gbt"]:
        mf = models_dir / f"{key}.pkl"
        content = f"dummy model content for {key}".encode("utf-8")
        mf.write_bytes(content)
        model_paths[key] = str(mf)
        model_hashes[key] = hashlib.sha256(content).hexdigest()

    # Create mock dev report
    dev_report = tmp_path / "dev_C.md"
    lines = [
        "# Study C Development Cross-Validation Report",
        f"**Pre-Registered Plan Hash:** `{plan_hash}`",
        "",
        "### Frozen Model SHA-256 Checksums",
        "| Model Identifier | File Path | SHA-256 Digest |",
        "| :--- | :--- | :--- |",
    ]
    for key in sorted(model_paths.keys()):
        lines.append(f"| **`{key}`** | `{model_paths[key]}` | `{model_hashes[key]}` |")

    dev_report.write_text("\n".join(lines), encoding="utf-8")

    test_report = tmp_path / "test_C.md"
    lock_file = tmp_path / ".test_locked_C"

    return {
        "plan_file": plan_file,
        "plan_hash": plan_hash,
        "dev_report": dev_report,
        "models_dir": models_dir,
        "model_paths": model_paths,
        "model_hashes": model_hashes,
        "test_report": test_report,
        "lock_file": lock_file,
    }


def test_gate_1_missing_git_tag_fails(tmp_path):
    """Gate 1: Must fail if git tag 'plan-c-locked' does not exist in repository."""
    env = create_mock_environment(tmp_path)
    with patch("subprocess.check_output", return_value="v1.0\nv2.0\n"):
        with pytest.raises(RuntimeError, match="PRE-FLIGHT GATE 1 FAILED.*Required git tag 'plan-c-locked'"):
            verify_preflight_gates(
                dev_report_path=str(env["dev_report"]),
                plan_file_path=str(env["plan_file"]),
                test_report_path=str(env["test_report"]),
                lock_file_path=str(env["lock_file"]),
            )


def test_gate_2_missing_dev_report_fails(tmp_path):
    """Gate 2: Must fail if results/dev_C.md does not exist."""
    env = create_mock_environment(tmp_path)
    with patch("subprocess.check_output", return_value="plan-c-locked\n"):
        with pytest.raises(FileNotFoundError, match="PRE-FLIGHT GATE 2 FAILED"):
            verify_preflight_gates(
                dev_report_path=str(tmp_path / "non_existent_dev_report.md"),
                plan_file_path=str(env["plan_file"]),
                test_report_path=str(env["test_report"]),
                lock_file_path=str(env["lock_file"]),
            )


def test_gate_3_plan_hash_mismatch_fails(tmp_path):
    """Gate 3: Must fail if PLAN_C.md hash does not match hash recorded in dev report."""
    env = create_mock_environment(tmp_path)
    # Tamper with plan file
    env["plan_file"].write_bytes(b"# Tampered plan file content\n")

    with patch("subprocess.check_output", return_value="plan-c-locked\n"):
        with pytest.raises(ValueError, match="PRE-FLIGHT GATE 3 FAILED.*PLAN_C.md hash on disk does not match"):
            verify_preflight_gates(
                dev_report_path=str(env["dev_report"]),
                plan_file_path=str(env["plan_file"]),
                test_report_path=str(env["test_report"]),
                lock_file_path=str(env["lock_file"]),
            )


def test_gate_4_model_missing_fails(tmp_path):
    """Gate 4a: Must fail if any frozen model file is missing from disk."""
    env = create_mock_environment(tmp_path)
    # Remove one model file
    Path(env["model_paths"]["c0_lr"]).unlink()

    with patch("subprocess.check_output", return_value="plan-c-locked\n"):
        with pytest.raises(FileNotFoundError, match="PRE-FLIGHT GATE 4 FAILED: Model file.*does not exist"):
            verify_preflight_gates(
                dev_report_path=str(env["dev_report"]),
                plan_file_path=str(env["plan_file"]),
                test_report_path=str(env["test_report"]),
                lock_file_path=str(env["lock_file"]),
            )


def test_gate_4_model_hash_mismatch_fails(tmp_path):
    """Gate 4b: Must fail if any frozen model file hash does not match recorded digest."""
    env = create_mock_environment(tmp_path)
    # Corrupt one model file
    Path(env["model_paths"]["c1_gbt"]).write_bytes(b"corrupted model content")

    with patch("subprocess.check_output", return_value="plan-c-locked\n"):
        with pytest.raises(ValueError, match="PRE-FLIGHT GATE 4 FAILED: Hash mismatch for model"):
            verify_preflight_gates(
                dev_report_path=str(env["dev_report"]),
                plan_file_path=str(env["plan_file"]),
                test_report_path=str(env["test_report"]),
                lock_file_path=str(env["lock_file"]),
            )


def test_gate_5_test_report_already_exists_fails(tmp_path):
    """Gate 5: Must fail if test report already exists (prevent repeat evaluation)."""
    env = create_mock_environment(tmp_path)
    env["test_report"].write_text("prior test report", encoding="utf-8")

    with patch("subprocess.check_output", return_value="plan-c-locked\n"):
        with pytest.raises(RuntimeError, match="PRE-FLIGHT GATE 5 FAILED: Test report.*already exists"):
            verify_preflight_gates(
                dev_report_path=str(env["dev_report"]),
                plan_file_path=str(env["plan_file"]),
                test_report_path=str(env["test_report"]),
                lock_file_path=str(env["lock_file"]),
            )


def test_gate_6_lock_file_already_exists_fails(tmp_path):
    """Gate 6: Must fail if lock file already exists (prevent duplicate runs)."""
    env = create_mock_environment(tmp_path)
    env["lock_file"].write_text("prior lock", encoding="utf-8")

    with patch("subprocess.check_output", return_value="plan-c-locked\n"):
        with pytest.raises(RuntimeError, match="PRE-FLIGHT GATE 6 FAILED: Test lock.*already exists"):
            verify_preflight_gates(
                dev_report_path=str(env["dev_report"]),
                plan_file_path=str(env["plan_file"]),
                test_report_path=str(env["test_report"]),
                lock_file_path=str(env["lock_file"]),
            )


def test_all_gates_pass(tmp_path):
    """All 6 gates pass cleanly when invariants are satisfied."""
    env = create_mock_environment(tmp_path)

    with patch("subprocess.check_output", return_value="plan-c-locked\n"):
        hashes = verify_preflight_gates(
            dev_report_path=str(env["dev_report"]),
            plan_file_path=str(env["plan_file"]),
            test_report_path=str(env["test_report"]),
            lock_file_path=str(env["lock_file"]),
        )
        assert len(hashes) == 8
        for k in env["model_hashes"]:
            assert hashes[k] == env["model_hashes"][k]


def test_pre_scoring_lock_creation_and_blocking(tmp_path):
    """Test check_and_acquire_lock_pre_scoring writes lock and blocks second attempt."""
    lock_f = tmp_path / ".test_locked_C"
    assert not lock_f.exists()

    check_and_acquire_lock_pre_scoring(
        lock_path=str(lock_f),
        git_commit="test_commit_hash",
        plan_hash="test_plan_hash",
    )
    assert lock_f.exists()
    content = lock_f.read_text(encoding="utf-8")
    assert "Status: SCORING_IN_PROGRESS" in content
    assert "test_commit_hash" in content
    assert "test_plan_hash" in content

    # Second acquisition raises RuntimeError
    with pytest.raises(RuntimeError, match="Test execution is LOCKED"):
        check_and_acquire_lock_pre_scoring(lock_path=str(lock_f))


@pytest.mark.parametrize(
    "failing_gate",
    ["gate1", "gate2", "gate3", "gate4", "gate5", "gate6"],
)
def test_aami_file_never_touched_when_any_gate_fails(tmp_path, failing_gate):
    """CRITICAL SECURITY PROOF:

    Prove that if ANY pre-flight gate fails, no code path touches
    VitalDB_AAMI_Cal_Subset.mat or calls load_aami_cal_metadata.
    """
    env = create_mock_environment(tmp_path)

    # Configure which gate will fail
    tag_mock_output = "plan-c-locked\n"
    dev_path = str(env["dev_report"])
    plan_path = str(env["plan_file"])
    test_path = str(env["test_report"])
    lock_path = str(env["lock_file"])

    if failing_gate == "gate1":
        tag_mock_output = "unrelated-tag\n"
    elif failing_gate == "gate2":
        dev_path = str(tmp_path / "missing_dev.md")
    elif failing_gate == "gate3":
        env["plan_file"].write_bytes(b"tampered plan")
    elif failing_gate == "gate4":
        Path(env["model_paths"]["c0_lr"]).unlink()
    elif failing_gate == "gate5":
        env["test_report"].write_text("existing test report")
    elif failing_gate == "gate6":
        env["lock_file"].write_text("existing lock file")

    with patch("subprocess.check_output", return_value=tag_mock_output), \
         patch("h5py.File") as mock_h5py, \
         patch("src.test_c.load_aami_cal_metadata") as mock_load_meta:

        # Score runner should raise during pre-flight
        with pytest.raises((RuntimeError, FileNotFoundError, ValueError)):
            score_aami_cal_test(
                mat_path="data/raw/VitalDB_AAMI_Cal_Subset.mat",
                output_report=test_path,
                dev_report_path=dev_path,
                plan_path=plan_path,
                lock_path=lock_path,
            )

        # ASSERTION: Neither h5py.File nor load_aami_cal_metadata was EVER invoked!
        mock_h5py.assert_not_called()
        mock_load_meta.assert_not_called()


def test_save_per_segment_predictions_output(tmp_path):
    """Test output-only saving of per-segment predictions."""
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import HistGradientBoostingClassifier
    from src.test_c import score_aami_cal_models, LADDER_STEPS

    # Build tiny synthetic imputed dataframe (5 segments, 2 subjects)
    all_cols = [
        "age", "sex", "bmi", "hr_ibi", "hr", "ibi_sd",
        "systolic_rise_time", "pulse_width", "decay_ratio",
        "beat_count", "is_clipped", "snr_ratio",
    ]
    data = {c: np.random.rand(5).astype(np.float32) for c in all_cols}
    data["subject_id"] = ["p1", "p1", "p1", "p2", "p2"]
    data["segment_idx"] = np.arange(5, dtype=np.int32)
    data["sbp"] = np.array([115.0, 122.0, 135.0, 110.0, 128.0], dtype=np.float32)
    data["y_120"] = (data["sbp"] >= 120.0).astype(int)
    data["y_130"] = (data["sbp"] >= 130.0).astype(int)
    df_imp = pd.DataFrame(data)

    # Build tiny mock models dictionary matching each step's features
    from src.features_c import get_feature_matrix
    artifacts = {}
    for step in LADDER_STEPS:
        X_step, _ = get_feature_matrix(df_imp, step)
        y_step = df_imp["y_120"].values

        # Mock LR
        lr = LogisticRegression()
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X_step)
        lr.fit(X_s, y_step)
        artifacts[f"{step.lower()}_lr"] = {"model": lr, "scaler": scaler}

        # Mock GBT
        gbt = HistGradientBoostingClassifier(max_iter=5, random_state=42)
        gbt.fit(X_step, y_step)
        artifacts[f"{step.lower()}_gbt"] = {"model": gbt}

    probs, pred_df = score_aami_cal_models(df_imp, artifacts)

    assert len(pred_df) == 5
    expected_cols = ["subject_id", "segment_idx", "sbp", "y_120", "y_130"]
    for step in LADDER_STEPS:
        for fam in ["lr", "gbt"]:
            expected_cols.append(f"prob_{step}_{fam}")

    for c in expected_cols:
        assert c in pred_df.columns, f"Missing column {c} in pred_df"

    # Test parquet write and read back
    out_pq = tmp_path / "test_C_predictions.parquet"
    pred_df.to_parquet(str(out_pq), index=False)
    read_df = pd.read_parquet(str(out_pq))
    assert len(read_df) == 5
    assert list(read_df.columns) == list(pred_df.columns)


def test_stage2_report_generation_neutral_wording(tmp_path):
    """Test Stage 2 report generation with neutral wording (no MET/NOT MET)."""
    import numpy as np
    import pandas as pd
    from src.report_test_c import generate_test_report
    from src.train_c import LADDER_STEPS

    # Create synthetic predictions parquet
    n = 10
    cols = ["subject_id", "segment_idx", "sbp", "y_120", "y_130"]
    data = {
        "subject_id": [f"p{i % 3}" for i in range(n)],
        "segment_idx": np.arange(n, dtype=np.int32),
        "sbp": np.linspace(100, 150, n).astype(np.float32),
        "y_120": (np.linspace(100, 150, n) >= 120.0).astype(int),
        "y_130": (np.linspace(100, 150, n) >= 130.0).astype(int),
    }
    for step in LADDER_STEPS:
        for fam in ["lr", "gbt"]:
            data[f"prob_{step}_{fam}"] = np.linspace(0.1, 0.9, n)

    pred_pq = tmp_path / "test_C_predictions.parquet"
    pd.DataFrame(data).to_parquet(str(pred_pq), index=False)

    out_md = tmp_path / "test_C.md"
    diff_csv = tmp_path / "test_C_diffs.csv"

    generate_test_report(
        predictions_path=str(pred_pq),
        output_report=str(out_md),
        diffs_csv_path=str(diff_csv),
        n_boot=20, # small bootstrap for test
        seed=42,
        cohort_name="Synthetic_Cohort",
    )

    assert out_md.exists()
    assert diff_csv.exists()

    md_text = out_md.read_text(encoding="utf-8")
    assert "MET" not in md_text
    assert "BELOW" not in md_text
    assert "aspiration, not a pass/fail criterion" in md_text
    assert "Synthetic_Cohort" in md_text

    diff_df = pd.read_csv(str(diff_csv))
    assert len(diff_df) == 8
    expected_comps = {"C1_minus_C0", "C2m_minus_C1", "C2_minus_C2m", "C2_minus_C0"}
    assert set(diff_df["comparison"]) == expected_comps


