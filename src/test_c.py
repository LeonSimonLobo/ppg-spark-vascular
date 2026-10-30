"""Single-pass test evaluation script for Study C on VitalDB AAMI_Cal cohort (PLAN_C.md).

Pre-registered in PLAN_C.md (including Amendment C1).

Pre-Flight Security Gates (Order of Operations):
1. Gate 1: Git tag 'plan-c-locked' exists in the repository.
2. Gate 2: results/dev_C.md exists.
3. Gate 3: The SHA-256 hash of PLAN_C.md equals the hash recorded in results/dev_C.md.
4. Gate 4: Frozen model hashes in results/dev_C.md match the model files on disk.
5. Gate 5: results/test_C.md does NOT exist.
6. Gate 6: results/.test_locked_C does NOT exist.
7. Lock before scoring: results/.test_locked_C is created BEFORE any AAMI_Cal data is opened.

No code path opens VitalDB_AAMI_Cal_Subset.mat or any AAMI parquet until ALL gates pass.
"""

import hashlib
from pathlib import Path
import pickle
import re
import subprocess
import time
from typing import Dict, Any, Tuple, List, Optional
import h5py
import numpy as np
import pandas as pd

from src.features_c import (
    extract_features_block,
    PlanCImputer,
    get_feature_matrix,
    C0_COLS,
    C1_COLS,
    C2m_COLS,
    C2_COLS,
)
from src.prospective_and_scale_w import fast_auc_with_ties
from src.train_c import compute_metrics, clustered_subject_bootstrap_metrics, LADDER_STEPS

LOCK_FILE_PATH = "results/.test_locked_C"
REPORT_PATH = "results/test_C.md"
DEV_REPORT_PATH = "results/dev_C.md"
PLAN_FILE_PATH = "PLAN_C.md"
MAT_PATH = "data/raw/VitalDB_AAMI_Cal_Subset.mat"
PREDICTIONS_PATH = "results/test_C_predictions.parquet"


def verify_preflight_gates(
    dev_report_path: str = DEV_REPORT_PATH,
    plan_file_path: str = PLAN_FILE_PATH,
    test_report_path: str = REPORT_PATH,
    lock_file_path: str = LOCK_FILE_PATH,
) -> Dict[str, str]:
    """Enforce pre-flight security and protocol gates before any test execution.

    Order of operations:
    1. Gate 1: Check git tag 'plan-c-locked' exists in repository.
    2. Gate 2: Check dev report (results/dev_C.md) exists.
    3. Gate 3: Verify PLAN_C.md SHA-256 hash equals hash recorded in dev report.
    4. Gate 4: Parse frozen model hashes from dev report and verify models on disk match.
    5. Gate 5: Check results/test_C.md does NOT exist.
    6. Gate 6: Check results/.test_locked_C does NOT exist.

    Returns:
    - dict of verified frozen model hashes {model_key: expected_hash}
    """
    print("Executing pre-flight gates...")

    # Gate 1: Check git tag plan-c-locked exists
    try:
        tag_output = subprocess.check_output(
            ["git", "tag", "-l", "plan-c-locked"],
            stderr=subprocess.STDOUT,
            text=True,
        ).strip().splitlines()
        if "plan-c-locked" not in tag_output:
            raise RuntimeError(
                "PRE-FLIGHT GATE 1 FAILED: Required git tag 'plan-c-locked' does not exist in repository!\n"
                "The protocol requires tagging the repository prior to test scoring."
            )
        print("Gate 1 (Git Tag 'plan-c-locked'): PASS")
    except Exception as e:
        if "PRE-FLIGHT GATE 1 FAILED" in str(e):
            raise
        raise RuntimeError(f"Gate 1 Git tag check error: {e}")

    # Gate 2: Check results/dev_C.md exists
    dev_p = Path(dev_report_path)
    if not dev_p.exists():
        raise FileNotFoundError(
            f"PRE-FLIGHT GATE 2 FAILED: {dev_report_path} not found! "
            "Development CV and model freezing must be completed before test scoring."
        )
    print(f"Gate 2 (Development Report {dev_report_path} exists): PASS")
    dev_content = dev_p.read_text(encoding="utf-8")

    # Gate 3: Check PLAN_C.md SHA-256 hash equals the hash recorded in dev report
    plan_p = Path(plan_file_path)
    if not plan_p.exists():
        raise FileNotFoundError(f"PRE-FLIGHT GATE 3 FAILED: {plan_file_path} not found!")

    actual_plan_hash = hashlib.sha256(plan_p.read_bytes()).hexdigest()

    match_plan_hash = re.search(r"Pre-Registered Plan Hash:\*{0,2}\s*`([a-f0-9]{64})`", dev_content)
    if not match_plan_hash:
        raise ValueError(
            f"PRE-FLIGHT GATE 3 FAILED: No pre-registered plan hash found in {dev_report_path}!"
        )
    recorded_plan_hash = match_plan_hash.group(1)

    if actual_plan_hash != recorded_plan_hash:
        raise ValueError(
            f"PRE-FLIGHT GATE 3 FAILED: PLAN_C.md hash on disk does not match recorded plan hash in {dev_report_path}!\n"
            f"Recorded: {recorded_plan_hash}\n"
            f"On Disk:  {actual_plan_hash}\n"
            "Pre-registration plan text has been altered."
        )
    print(f"Gate 3 (PLAN_C.md Hash Verification): PASS ({actual_plan_hash[:16]}...)")

    # Gate 4: Parse frozen model hashes from dev report and verify model files on disk
    model_hash_matches = re.findall(r"\|\s*\*\*`([^`]+)`\*\*\s*\|\s*`([^`]+)`\s*\|\s*`([a-f0-9]{64})`\s*\|", dev_content)
    if not model_hash_matches:
        raise ValueError(
            f"PRE-FLIGHT GATE 4 FAILED: No frozen model hashes found in {dev_report_path}!"
        )

    expected_hashes = {}
    for key, path_str, sha in model_hash_matches:
        expected_hashes[key] = sha
        p = Path(path_str)
        if not p.exists():
            raise FileNotFoundError(f"PRE-FLIGHT GATE 4 FAILED: Model file {p} does not exist on disk!")
        actual_model_sha = hashlib.sha256(p.read_bytes()).hexdigest()
        if actual_model_sha != sha:
            raise ValueError(
                f"PRE-FLIGHT GATE 4 FAILED: Hash mismatch for model {p}!\n"
                f"Recorded in report: {sha}\n"
                f"Actual on disk:     {actual_model_sha}"
            )
    print(f"Gate 4 (Model Files Hash Verification on Disk): PASS ({len(expected_hashes)} models verified)")

    # Gate 5: Check results/test_C.md does NOT exist
    test_rep_p = Path(test_report_path)
    if test_rep_p.exists():
        raise RuntimeError(
            f"PRE-FLIGHT GATE 5 FAILED: Test report {test_report_path} already exists! "
            "Repeated evaluations on held-out test data are strictly prohibited."
        )
    print("Gate 5 (No prior test report): PASS")

    # Gate 6: Check results/.test_locked_C does NOT exist
    lock_p = Path(lock_file_path)
    if lock_p.exists():
        raise RuntimeError(
            f"PRE-FLIGHT GATE 6 FAILED: Test lock {lock_file_path} already exists! "
            "Scoring has already occurred or is currently in progress."
        )
    print("Gate 6 (No prior test lock file): PASS")

    return expected_hashes


def check_and_acquire_lock_pre_scoring(
    lock_path: str = LOCK_FILE_PATH,
    git_commit: str = "HEAD",
    plan_hash: str = "",
) -> None:
    """Acquire test lock BEFORE any test data is opened. Aborts if already locked."""
    lock_p = Path(lock_path)
    if lock_p.exists():
        raise RuntimeError(
            f"Test execution is LOCKED by existing {lock_path}! "
            "A test run has already occurred or is locked against repeated evaluations."
        )

    lock_content = (
        f"# TEST LOCK: Study C (AAMI_Cal single evaluation)\n"
        f"Acquired: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n"
        f"Git Commit: {git_commit}\n"
        f"Plan Hash: {plan_hash}\n"
        f"Status: SCORING_IN_PROGRESS\n"
    )
    lock_p.write_text(lock_content, encoding="utf-8")
    print(f"Acquired pre-scoring lock at {lock_path}")


def verify_frozen_model_hashes(expected_hashes: Dict[str, str]) -> Dict[str, Any]:
    """Verify and unpickle all frozen model artifacts."""
    loaded_artifacts = {}
    for key, expected_hash in expected_hashes.items():
        model_path = Path(f"models/{key}_frozen.pkl")
        if not model_path.exists():
            raise FileNotFoundError(f"Frozen model artifact missing: {model_path}")

        data = model_path.read_bytes()
        actual_hash = hashlib.sha256(data).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(
                f"HASH MISMATCH for {model_path}!\n"
                f"Expected: {expected_hash}\n"
                f"Actual:   {actual_hash}\n"
                "Halting test evaluation to preserve pre-registration integrity."
            )

        with open(model_path, "rb") as f:
            artifact = pickle.load(f)
        loaded_artifacts[key] = artifact
        print(f"Verified {key}: SHA-256 matches ({actual_hash[:16]}...)")

    return loaded_artifacts


def load_aami_cal_metadata(mat_path: str = MAT_PATH) -> pd.DataFrame:
    """Read demographics, SBP, subject IDs from AAMI_Cal MAT file."""
    print(f"Reading AAMI_Cal metadata from {mat_path}...")
    with h5py.File(mat_path, "r") as f:
        subset = f["Subset"]
        n_segs = subset["Signals"].shape[2]
        print(f"Reading AAMI_Cal metadata for {n_segs:,} segments...")

        subj_refs = subset["Subject"][0, :n_segs]
        gen_refs = subset["Gender"][0, :n_segs]

        subjs = []
        sex_arr = np.full(n_segs, np.nan, dtype=np.float32)
        prev_sub, prev_sex = None, np.nan
        for i in range(n_segs):
            s = "".join(chr(int(c)) for c in np.asarray(f[subj_refs[i]]).ravel())
            if s != prev_sub:
                c = "".join(chr(int(x)) for x in np.asarray(f[gen_refs[i]]).ravel()).strip().upper()
                prev_sex = 1.0 if c == "M" else (0.0 if c == "F" else np.nan)
                prev_sub = s
            subjs.append(s)
            sex_arr[i] = prev_sex

        age_arr = np.asarray(subset["Age"][0, :n_segs], dtype=np.float32)
        bmi_arr = np.asarray(subset["BMI"][0, :n_segs], dtype=np.float32)

        # SBP from Invasive Arterial Line
        if "SBP" in subset:
            sbp_arr = np.asarray(subset["SBP"][0, :n_segs], dtype=np.float32)
        else:
            abp_block = subset["Signals"][:, 0, :]
            sbp_arr = np.max(abp_block, axis=0).astype(np.float32)

        dbp_arr = np.asarray(subset["DBP"][0, :n_segs], dtype=np.float32) if "DBP" in subset else np.full(n_segs, np.nan, dtype=np.float32)

    df = pd.DataFrame({
        "subject_id": subjs,
        "segment_idx": np.arange(n_segs, dtype=np.int32),
        "age": age_arr,
        "sex": sex_arr,
        "bmi": bmi_arr,
        "sbp": sbp_arr,
        "dbp": dbp_arr,
    })
    return df


def extract_aami_cal_features(
    mat_path: str = MAT_PATH,
    meta_df: Optional[pd.DataFrame] = None,
    chunk_size: int = 15000,
    fs: int = 125,
) -> pd.DataFrame:
    """Extract PPG Channel 1 features in chunks from MAT file."""
    if meta_df is None:
        meta_df = load_aami_cal_metadata(mat_path=mat_path)
    n_segs = len(meta_df)

    print(f"\nExtracting PPG Channel 1 features from {mat_path}...")
    hr_ibi_arr = np.full(n_segs, np.nan, dtype=np.float32)
    hr_arr = np.full(n_segs, np.nan, dtype=np.float32)
    ibi_sd_arr = np.full(n_segs, np.nan, dtype=np.float32)
    rise_arr = np.full(n_segs, np.nan, dtype=np.float32)
    pw_arr = np.full(n_segs, np.nan, dtype=np.float32)
    dec_arr = np.full(n_segs, np.nan, dtype=np.float32)
    beat_arr = np.zeros(n_segs, dtype=np.float32)
    clipped_arr = np.zeros(n_segs, dtype=np.float32)
    snr_arr = np.zeros(n_segs, dtype=np.float32)

    t0 = time.time()
    with h5py.File(mat_path, "r") as f:
        sig_ds = f["Subset"]["Signals"]
        for c_start in range(0, n_segs, chunk_size):
            c_end = min(n_segs, c_start + chunk_size)
            ppg_block = sig_ds[:, 1, c_start:c_end]

            block_res = extract_features_block(ppg_block, fs=fs)
            hr_ibi_arr[c_start:c_end] = block_res["hr_ibi"]
            hr_arr[c_start:c_end] = block_res["hr"]
            ibi_sd_arr[c_start:c_end] = block_res["ibi_sd"]
            rise_arr[c_start:c_end] = block_res["systolic_rise_time"]
            pw_arr[c_start:c_end] = block_res["pulse_width"]
            dec_arr[c_start:c_end] = block_res["decay_ratio"]
            beat_arr[c_start:c_end] = block_res["beat_count"]
            clipped_arr[c_start:c_end] = block_res["is_clipped"]
            snr_arr[c_start:c_end] = block_res["snr_ratio"]

    elapsed_feat = time.time() - t0
    print(f"Feature extraction complete in {elapsed_feat:.1f}s ({n_segs / elapsed_feat:.1f} seg/s).")

    feat_df = meta_df.copy()
    feat_df["hr_ibi"] = hr_ibi_arr
    feat_df["hr"] = hr_arr
    feat_df["ibi_sd"] = ibi_sd_arr
    feat_df["systolic_rise_time"] = rise_arr
    feat_df["pulse_width"] = pw_arr
    feat_df["decay_ratio"] = dec_arr
    feat_df["beat_count"] = beat_arr
    feat_df["is_clipped"] = clipped_arr
    feat_df["snr_ratio"] = snr_arr
    return feat_df


def impute_aami_cal_features(
    feat_df: pd.DataFrame,
    imputer_medians: Dict[str, float],
) -> pd.DataFrame:
    """Apply frozen PlanCImputer medians to feature DataFrame."""
    imputer = PlanCImputer()
    imputer.medians_ = imputer_medians
    return imputer.transform(feat_df)


def score_aami_cal_models(
    df_imp: pd.DataFrame,
    artifacts: Dict[str, Any],
) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
    """Score imputed DataFrame with frozen models and construct predictions table."""
    print("\nScoring with frozen models...")
    probs = {}
    for step in LADDER_STEPS:
        # Check both lower and original casing in artifacts
        art_lr = artifacts.get(f"{step.lower()}_lr") or artifacts.get(f"{step}_lr")
        X_lr, _ = get_feature_matrix(df_imp, step)
        X_lr_s = art_lr["scaler"].transform(X_lr)
        prob_lr = art_lr["model"].predict_proba(X_lr_s)[:, 1]
        probs[f"{step}_lr"] = prob_lr

        art_gbt = artifacts.get(f"{step.lower()}_gbt") or artifacts.get(f"{step}_gbt")
        X_gbt, _ = get_feature_matrix(df_imp, step)
        prob_gbt = art_gbt["model"].predict_proba(X_gbt)[:, 1]
        probs[f"{step}_gbt"] = prob_gbt

    y_120_vals = df_imp["y_120"].values if "y_120" in df_imp else (df_imp["sbp"] >= 120.0).astype(int).values
    y_130_vals = df_imp["y_130"].values if "y_130" in df_imp else (df_imp["sbp"] >= 130.0).astype(int).values
    pred_df = pd.DataFrame({
        "subject_id": df_imp["subject_id"].values,
        "segment_idx": df_imp["segment_idx"].values,
        "sbp": df_imp["sbp"].values,
        "y_120": y_120_vals,
        "y_130": y_130_vals,
    })
    for step in LADDER_STEPS:
        for fam in ["lr", "gbt"]:
            k = f"{step}_{fam}"
            pred_df[f"prob_{k}"] = probs[k]

    return probs, pred_df


def score_aami_cal_test(
    mat_path: str = MAT_PATH,
    output_report: str = REPORT_PATH,
    dev_report_path: str = DEV_REPORT_PATH,
    plan_path: str = PLAN_FILE_PATH,
    lock_path: str = LOCK_FILE_PATH,
    predictions_path: str = PREDICTIONS_PATH,
    n_boot: int = 2000,
    seed: int = 42,
    git_commit: str = "HEAD",
):
    """Execute single-pass test evaluation on AAMI_Cal."""
    print("=" * 70)
    print("STUDY C: AAMI_Cal Single-Pass Test Evaluation")
    print("=" * 70)

    # 1. Execute Pre-Flight Gates
    expected_hashes = verify_preflight_gates(
        dev_report_path=dev_report_path,
        plan_file_path=plan_path,
        test_report_path=output_report,
        lock_file_path=lock_path,
    )

    plan_p = Path(plan_path)
    plan_hash = hashlib.sha256(plan_p.read_bytes()).hexdigest()

    # 2. Acquire Pre-Scoring Lock
    check_and_acquire_lock_pre_scoring(lock_path=lock_path, git_commit=git_commit, plan_hash=plan_hash)

    try:
        # 3. Verify Hashes
        print("\nVerifying frozen model hashes...")
        artifacts = verify_frozen_model_hashes(expected_hashes)

        # 4. Load Metadata (FIRST AND ONLY CODE PATH OPENING AAMI DATA)
        meta_df = load_aami_cal_metadata(mat_path=mat_path)
        n_segs = len(meta_df)
        n_subjs = meta_df["subject_id"].nunique()
        print(f"Loaded AAMI_Cal: {n_segs:,} segments across {n_subjs:,} subjects.")

        meta_df["y_120"] = (meta_df["sbp"] >= 120.0).astype(int)
        meta_df["y_130"] = (meta_df["sbp"] >= 130.0).astype(int)
        prev_120 = float(np.mean(meta_df["y_120"]))
        prev_130 = float(np.mean(meta_df["y_130"]))
        print(f"AAMI_Cal SBP >= 120 prevalence: {prev_120 * 100.0:.2f}%")
        print(f"AAMI_Cal SBP >= 130 prevalence: {prev_130 * 100.0:.2f}%")

        # 5. Extract PPG Features in Chunks
        feat_df = extract_aami_cal_features(mat_path=mat_path, meta_df=meta_df, chunk_size=15000, fs=125)

        # 6. Apply Frozen Imputer (Zero row drops)
        c0_art = artifacts.get("c0_lr") or artifacts.get("C0_lr")
        imputer_medians = c0_art["imputer_medians"] if c0_art and "imputer_medians" in c0_art else artifacts["c_imputer"].medians_
        df_imp = impute_aami_cal_features(feat_df=feat_df, imputer_medians=imputer_medians)

        # 7. Score with Frozen Models & Save Predictions
        probs, pred_df = score_aami_cal_models(df_imp=df_imp, artifacts=artifacts)

        pred_p = Path(predictions_path)
        pred_p.parent.mkdir(parents=True, exist_ok=True)
        pred_df.to_parquet(str(pred_p), index=False)
        print(f"Saved per-segment predictions to {predictions_path}")

        # Stage 2: Generate Report and Diffs CSV from predictions parquet
        from src.report_test_c import generate_test_report
        generate_test_report(
            predictions_path=predictions_path,
            output_report=output_report,
            diffs_csv_path="results/test_C_diffs.csv",
            dev_report_path=dev_report_path,
            plan_path=plan_path,
            n_boot=n_boot,
            seed=seed,
            git_commit=git_commit,
            cohort_name="AAMI_Cal",
        )

        # Finalize lock status
        lock_p = Path(LOCK_FILE_PATH)
        lock_p.write_text(
            f"# TEST LOCK: Study C (AAMI_Cal single evaluation)\n"
            f"Completed: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n"
            f"Git Commit: {git_commit}\n"
            f"Plan Hash: {plan_hash}\n"
            f"Status: COMPLETED_AND_PERMANENTLY_LOCKED\n",
            encoding="utf-8",
        )
        print("Test evaluation successfully completed and permanently locked.")

    except Exception as e:
        print(f"ERROR during test execution: {e}")
        raise


from src.report_test_c import write_aami_cal_report

