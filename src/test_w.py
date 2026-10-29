"""Test evaluation pipeline for within-subject model W on held-out CalFree test cohort.

Pre-registered in PLAN_W.md (Amendments W1, W2, W3).
Strict execution invariants:
1. Lock before scoring: results/.test_locked_W created prior to feature extraction / scoring.
2. Frozen model hash abort: halts if models/w_frozen.pkl SHA-256 differs from 92c9176...640fdc.
3. Zero development data read: loads strictly frozen models and CalFree test files.
4. Exact CalFree segment counts: 144 subjects x 400 segments = 57,600 total;
   Prefix baseline [0, 59] (60 segs); Primary evaluation [60, 399] (340 segs, N=48,960).
5. Centering shift disclosed: frozen model fit on full-session centered dev features,
   evaluated on prefix-centered test features as the prospective deployment condition.
6. Stop rule: primary paired 95% bootstrap CI (morphology - HR) must strictly exclude zero.
"""

import hashlib
from pathlib import Path
import pickle
import time
from typing import Dict, Any, Tuple
import h5py
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.features_w import extract_segment_features
from src.prospective_and_scale_w import fast_auc_with_ties

EXPECTED_W_HASH = "92c9176b456e00630c45071afcdb1f5e0423d393ca2f6ad2ecf9357f5e640fdc"
EXPECTED_HR_HASH = "b1f8a705bc9292a936476857b9eebc4c0d18ae5a1d59c0445bc84dcfe8a6eaf5"
EXPECTED_JOINT_HASH = "080296d8e309de97ed4eb5c65c1c69405cbecdfc19a7b7a452d92872530b5982"

FROZEN_W_PATH = "models/w_frozen.pkl"
FROZEN_HR_PATH = "models/hr_frozen.pkl"
FROZEN_JOINT_PATH = "models/joint_frozen.pkl"

LOCK_FILE_PATH = "results/.test_locked_W"
REPORT_PATH = "results/test_W.md"


def check_and_acquire_lock_pre_scoring(git_commit: str = "HEAD") -> None:
    """Acquire lock BEFORE any test data is processed. Aborts if already locked."""
    lock_p = Path(LOCK_FILE_PATH)
    if lock_p.exists():
        raise RuntimeError(
            f"Test execution is LOCKED by existing {LOCK_FILE_PATH}! "
            "A test run has already occurred or aborted. Re-execution is blocked by protocol."
        )
    lock_p.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_p, "w", encoding="utf-8") as f:
        f.write(
            f"STATUS: LOCKED_PRE_SCORING\n"
            f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n"
            f"GitCommit: {git_commit}\n"
        )
    print(f"Pre-scoring lock acquired at {LOCK_FILE_PATH}.")


def verify_frozen_model_hashes() -> Dict[str, str]:
    """Verify SHA-256 hashes of all frozen model artifacts. Halts with ValueError on mismatch."""
    models_to_check = [
        ("Primary Morphology (W)", FROZEN_W_PATH, EXPECTED_W_HASH),
        ("HR Comparator", FROZEN_HR_PATH, EXPECTED_HR_HASH),
        ("Joint Model", FROZEN_JOINT_PATH, EXPECTED_JOINT_HASH),
    ]
    verified_hashes = {}
    for name, path, expected in models_to_check:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Required frozen model file not found: {path}!")
        with open(p, "rb") as f:
            actual = hashlib.sha256(f.read()).hexdigest()
        if actual != expected:
            raise ValueError(
                f"Frozen model hash mismatch for {name} ({path})!\n"
                f"Expected: {expected}\n"
                f"Actual:   {actual}"
            )
        print(f"Verified {name} SHA-256: {actual} (EXACT MATCH)")
        verified_hashes[name] = actual
    return verified_hashes


def extract_calfree_features(
    mat_path: str = "data/raw/VitalDB_CalFree_Test_Subset.mat",
    parquet_path: str = "data/parquet/calfree_test.parquet",
    output_parquet_path: str = "data/interim/features_w_calfree.parquet",
    chunk_size: int = 15000,
    fs: int = 125,
) -> pd.DataFrame:
    """Extract PPG morphology features from CalFree test subset (PPG channel 1 only)."""
    out_p = Path(output_parquet_path)
    if out_p.exists():
        print(f"Loading cached CalFree features from {output_parquet_path}...")
        return pd.read_parquet(output_parquet_path)

    print(f"Loading CalFree metadata from {parquet_path}...")
    table = pq.read_table(parquet_path, columns=["segment_idx", "subject_id", "sbp", "dbp"])
    df = table.to_pandas()
    n_test = len(df)
    n_subjs = df["subject_id"].nunique()
    print(f"CalFree cohort: {n_test:,} segments across {n_subjs:,} subjects.")

    hr_arr = np.zeros(n_test, dtype=np.float32)
    rise_arr = np.full(n_test, np.nan, dtype=np.float32)
    pw_arr = np.full(n_test, np.nan, dtype=np.float32)
    decay_arr = np.full(n_test, np.nan, dtype=np.float32)

    t0 = time.time()
    with h5py.File(mat_path, "r") as f:
        sig_ds = f["Subset"]["Signals"]
        for c_start in range(0, n_test, chunk_size):
            c_end = min(n_test, c_start + chunk_size)
            c_len = c_end - c_start
            ppg_block = sig_ds[:, 1, c_start:c_end]
            for j in range(c_len):
                hr, rise, pw, dec = extract_segment_features(ppg_block[:, j], fs=fs)
                idx = c_start + j
                hr_arr[idx] = hr
                rise_arr[idx] = rise
                pw_arr[idx] = pw
                decay_arr[idx] = dec

    print(f"Feature extraction completed in {time.time()-t0:.2f}s.")
    feat_df = df.copy()
    feat_df["hr"] = hr_arr
    feat_df["systolic_rise_time"] = rise_arr
    feat_df["pulse_width"] = pw_arr
    feat_df["decay_ratio"] = decay_arr

    out_p.parent.mkdir(parents=True, exist_ok=True)
    feat_df.to_parquet(output_parquet_path, index=False)
    print(f"Saved CalFree features to {output_parquet_path}.")
    return feat_df


def bootstrap_subject_paired(
    y_sub: np.ndarray,
    scores_sub_dict: Dict[str, np.ndarray],
    n_resamples: int = 2000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Subject-level clustered bootstrap for test set evaluation."""
    rng = np.random.default_rng(seed)
    n_subjs = len(y_sub)
    model_names = list(scores_sub_dict.keys())
    boot_aucs = {m: np.zeros(n_resamples, dtype=np.float64) for m in model_names}

    for b in range(n_resamples):
        sub_sample = rng.choice(n_subjs, size=n_subjs, replace=True)
        y_b = y_sub[sub_sample].ravel()
        for m in model_names:
            score_b = scores_sub_dict[m][sub_sample].ravel()
            boot_aucs[m][b] = fast_auc_with_ties(y_b, score_b)

    results = {}
    y_flat = y_sub.ravel()
    for m in model_names:
        score_flat = scores_sub_dict[m].ravel()
        pt_est = fast_auc_with_ties(y_flat, score_flat)
        ci_low = float(np.percentile(boot_aucs[m], 2.5))
        ci_high = float(np.percentile(boot_aucs[m], 97.5))
        pred_flat = (score_flat >= 0.5).astype(int)
        acc = float(np.mean(pred_flat == y_flat) * 100.0)
        pos_acc = float(np.mean(pred_flat[y_flat == 1] == 1)) if np.sum(y_flat == 1) > 0 else 0.5
        neg_acc = float(np.mean(pred_flat[y_flat == 0] == 0)) if np.sum(y_flat == 0) > 0 else 0.5
        bal_acc = float((pos_acc + neg_acc) / 2.0 * 100.0)
        results[m] = {
            "auroc": pt_est,
            "ci_95": (ci_low, ci_high),
            "acc": acc,
            "bal_acc": bal_acc,
        }

    # Paired differences vs hr
    if "hr" in scores_sub_dict:
        hr_boots = boot_aucs["hr"]
        paired_diffs = {}
        for m in model_names:
            if m == "hr":
                continue
            diff_boots = boot_aucs[m] - hr_boots
            pt_diff = results[m]["auroc"] - results["hr"]["auroc"]
            d_low = float(np.percentile(diff_boots, 2.5))
            d_high = float(np.percentile(diff_boots, 97.5))
            paired_diffs[f"{m}_minus_hr"] = {"diff": pt_diff, "ci_95": (d_low, d_high)}
        results["paired_differences"] = paired_diffs

    return results


def run_calfree_evaluation(git_commit: str = "HEAD") -> Dict[str, Any]:
    """Execute CalFree test evaluation once under Amendment W3 protocol."""
    # 1. Lock before scoring
    check_and_acquire_lock_pre_scoring(git_commit=git_commit)

    # 2. Hash verification
    verified_hashes = verify_frozen_model_hashes()

    # 3. Load frozen models (NO development data opened)
    with open(FROZEN_W_PATH, "rb") as f:
        frozen_morph_model = pickle.load(f)
    with open(FROZEN_HR_PATH, "rb") as f:
        frozen_hr_model = pickle.load(f)
    with open(FROZEN_JOINT_PATH, "rb") as f:
        frozen_joint_model = pickle.load(f)

    # 4. Feature extraction on CalFree
    df = extract_calfree_features()
    n_subjs = df["subject_id"].nunique()
    n_seg_total = len(df)
    n_seg_per_subj = 400

    assert n_subjs == 144, f"Expected 144 CalFree subjects, got {n_subjs}"
    assert n_seg_total == 57600, f"Expected 57,600 CalFree segments, got {n_seg_total}"

    # -------------------------------------------------------------
    # 5. Primary Endpoint: Prospective Prefix Centering
    # Baseline window: [0, 59] (60 segments)
    # Evaluation window: [60, 399] (340 segments, N = 48,960)
    # -------------------------------------------------------------
    prefix_len = 60
    eval_mask = np.tile(np.arange(n_seg_per_subj) >= prefix_len, n_subjs)
    prefix_df = df[~eval_mask]
    eval_df = df[eval_mask].copy()
    n_eval_seg = n_seg_per_subj - prefix_len  # exactly 340 segments

    assert len(eval_df) == 48960, f"Expected exactly 48,960 primary evaluation segments, got {len(eval_df)}"

    for col in ["sbp", "hr", "systolic_rise_time", "pulse_width", "decay_ratio"]:
        pref_med = prefix_df.groupby("subject_id")[col].median()
        eval_df[f"pref_med_{col}"] = eval_df["subject_id"].map(pref_med)

    for col in ["hr", "systolic_rise_time", "pulse_width", "decay_ratio"]:
        eval_df[f"delta_{col}_pref"] = eval_df[col] - eval_df[f"pref_med_{col}"]

    # Prospective label: SBP >= prefix baseline median
    eval_df["y_pref"] = (eval_df["sbp"] >= eval_df["pref_med_sbp"]).astype(int)
    y_pref = eval_df["y_pref"].values

    morph_cols_pref = ["delta_systolic_rise_time_pref", "delta_pulse_width_pref", "delta_decay_ratio_pref"]
    X_pref_m = eval_df[morph_cols_pref].values
    X_pref_h = eval_df[["delta_hr_pref"]].values
    X_pref_j = eval_df[["delta_hr_pref"] + morph_cols_pref].values

    p_pref_m = frozen_morph_model.predict_proba(X_pref_m)[:, 1]
    p_pref_h = frozen_hr_model.predict_proba(X_pref_h)[:, 1]
    p_pref_j = frozen_joint_model.predict_proba(X_pref_j)[:, 1]

    y_pref_sub = y_pref.reshape(n_subjs, n_eval_seg)
    scores_pref_sub = {
        "hr": p_pref_h.reshape(n_subjs, n_eval_seg),
        "morphology": p_pref_m.reshape(n_subjs, n_eval_seg),
        "joint": p_pref_j.reshape(n_subjs, n_eval_seg),
    }
    boot_pref = bootstrap_subject_paired(y_pref_sub, scores_pref_sub, n_resamples=2000, seed=42)
    boot_pref["prevalence"] = float(np.mean(y_pref) * 100.0)

    # -------------------------------------------------------------
    # 6. Secondary Endpoint: Full-Session Retrospective Centering
    # Window: all 400 segments per subject (N = 57,600)
    # -------------------------------------------------------------
    df_full = df.copy()
    for col in ["sbp", "hr", "systolic_rise_time", "pulse_width", "decay_ratio"]:
        full_med = df_full.groupby("subject_id")[col].transform("median")
        df_full[f"delta_{col}_full"] = df_full[col] - full_med

    df_full["y_full"] = (df_full["sbp"] >= df_full.groupby("subject_id")["sbp"].transform("median")).astype(int)
    y_full = df_full["y_full"].values

    morph_cols_full = ["delta_systolic_rise_time_full", "delta_pulse_width_full", "delta_decay_ratio_full"]
    X_full_m = df_full[morph_cols_full].values
    X_full_h = df_full[["delta_hr_full"]].values
    X_full_j = df_full[["delta_hr_full"] + morph_cols_full].values

    p_full_m = frozen_morph_model.predict_proba(X_full_m)[:, 1]
    p_full_h = frozen_hr_model.predict_proba(X_full_h)[:, 1]
    p_full_j = frozen_joint_model.predict_proba(X_full_j)[:, 1]

    y_full_sub = y_full.reshape(n_subjs, n_seg_per_subj)
    scores_full_sub = {
        "hr": p_full_h.reshape(n_subjs, n_seg_per_subj),
        "morphology": p_full_m.reshape(n_subjs, n_seg_per_subj),
        "joint": p_full_j.reshape(n_subjs, n_seg_per_subj),
    }
    boot_full = bootstrap_subject_paired(y_full_sub, scores_full_sub, n_resamples=2000, seed=42)
    boot_full["prevalence"] = float(np.mean(y_full) * 100.0)

    # -------------------------------------------------------------
    # 7. Stop Rule Evaluation
    # -------------------------------------------------------------
    pref_diff = boot_pref["paired_differences"]["morphology_minus_hr"]["diff"]
    pref_ci = boot_pref["paired_differences"]["morphology_minus_hr"]["ci_95"]
    stop_rule_passed = bool(pref_ci[0] > 0.0)

    return {
        "verified_hashes": verified_hashes,
        "n_subjs": n_subjs,
        "n_seg_total": n_seg_total,
        "n_primary_eval_seg": len(eval_df),
        "primary_endpoint_prefix": boot_pref,
        "secondary_endpoint_full": boot_full,
        "stop_rule_passed": stop_rule_passed,
    }


def write_test_report_and_update_lock(results: Dict[str, Any], git_commit: str = "HEAD") -> str:
    """Format results into results/test_W.md and update lock file to COMPLETED."""
    pref = results["primary_endpoint_prefix"]
    full = results["secondary_endpoint_full"]
    stop_rule = results["stop_rule_passed"]

    if stop_rule:
        verdict_text = (
            "### Verdict: CLAIM CONFIRMED\n"
            "The active claim is **CONFIRMED as a prospective within-recording association with a 10-minute calibration window on this surgical cohort**.\n\n"
            f"- The primary prospective paired difference $\\Delta\\text{{AUROC}}(\\text{{Morphology}} - \\text{{HR}})$ 95% bootstrap CI strictly excludes zero: "
            f"**{pref['paired_differences']['morphology_minus_hr']['diff']:+.4f}** [{pref['paired_differences']['morphology_minus_hr']['ci_95'][0]:+.4f}, {pref['paired_differences']['morphology_minus_hr']['ci_95'][1]:+.4f}].\n"
            "- Scope limitation: This confirms an intraoperative within-recording association under a documented calibration window; it is NOT claimed as general cuffless PPG-to-BP prediction across patients."
        )
    else:
        verdict_text = (
            "### Verdict: CLAIM REFUTED ON TEST COHORT\n"
            "The prospective within-subject claim is **REFUTED on the held-out test cohort**. The development finding did not replicate under pre-registered testing.\n\n"
            f"- The primary prospective paired difference $\\Delta\\text{{AUROC}}(\\text{{Morphology}} - \\text{{HR}})$ 95% bootstrap CI includes zero or is negative: "
            f"**{pref['paired_differences']['morphology_minus_hr']['diff']:+.4f}** [{pref['paired_differences']['morphology_minus_hr']['ci_95'][0]:+.4f}, {pref['paired_differences']['morphology_minus_hr']['ci_95'][1]:+.4f}].\n"
            "- The outcome is formally recorded as a **development-only finding**. Evaluation ceases immediately without post-hoc tuning or model substitutions."
        )

    report = rf"""# CalFree Test Cohort Evaluation: Within-Subject Model W

**Date:** 2026-10-29  
**Cohort:** `VitalDB_CalFree_Test_Subset.mat` (144 held-out subjects, 400 segments per subject, $N = 57,600$)  
**Protocol:** PLAN_W.md (Amendments W1, W2, W3)  
**Frozen Model SHA-256:** `{results['verified_hashes']['Primary Morphology (W)']}`  
**Code Git Commit:** `{git_commit}`  
**Lock Status:** Test locked upon completion (`results/.test_locked_W`)  
**Bootstrap Parameters:** 2,000 subject-level resamples, seed 42  

---

## 1. Executive Summary & Stop Rule Verdict

{verdict_text}

---

## 2. Train–Test Centering Shift Disclosure

> [!NOTE]
> **Deployment Condition Tested:**  
> The frozen morphology model was trained on development data using full-session centered features ($\Delta_{{\text{{full}}}} X = X - \text{{median}}_{{\text{{session}}}}(X)$) and is evaluated on CalFree using prospective prefix-centered features ($\Delta_{{\text{{pref}}}} X = X - \text{{median}}_{{t < 60}}(X)$). This represents the pre-registered prospective deployment condition being tested (a 10-minute baseline calibration window), not an independent validation of retrospective batch centering.

---

## 3. Primary Endpoint: Prospective Calibration (Prefix Centering)

- **Calibration Baseline Window:** Segments $t \in [0, 59]$ (first 10 minutes = 60 segments per subject).
- **Evaluation Window:** Segments $t \in [60, 399]$ (subsequent 340 segments per subject, $144 \times 340 = 48,960$ segments).
- **Target Label:** Prospective excursion relative to prefix baseline ($y_{{\text{{pref}}}} = \mathbb{{I}}(SBP \ge \text{{median}}_{{t < 60}}(SBP))$).
- **Actual Positive Prevalence:** **{pref['prevalence']:.2f}%** (Majority baseline: 50.0%)

| Model | AUROC [95% Bootstrap CI] | Accuracy (Prev: {pref['prevalence']:.1f}%) | Balanced Accuracy | Paired $\Delta\text{{AUROC}}$ vs. HR [95% CI] |
|---|---|---|---|---|
| **HR Comparator (Frozen)** | {pref['hr']['auroc']:.4f} [{pref['hr']['ci_95'][0]:.4f}, {pref['hr']['ci_95'][1]:.4f}] | {pref['hr']['acc']:.2f}% | {pref['hr']['bal_acc']:.2f}% | — |
| **Morphology (Frozen)** | **{pref['morphology']['auroc']:.4f}** [{pref['morphology']['ci_95'][0]:.4f}, {pref['morphology']['ci_95'][1]:.4f}] | {pref['morphology']['acc']:.2f}% | {pref['morphology']['bal_acc']:.2f}% | **{pref['paired_differences']['morphology_minus_hr']['diff']:+.4f}** [{pref['paired_differences']['morphology_minus_hr']['ci_95'][0]:+.4f}, {pref['paired_differences']['morphology_minus_hr']['ci_95'][1]:+.4f}] |
| **Joint (HR + Morph)** | **{pref['joint']['auroc']:.4f}** [{pref['joint']['ci_95'][0]:.4f}, {pref['joint']['ci_95'][1]:.4f}] | {pref['joint']['acc']:.2f}% | {pref['joint']['bal_acc']:.2f}% | **{pref['paired_differences']['joint_minus_hr']['diff']:+.4f}** [{pref['paired_differences']['joint_minus_hr']['ci_95'][0]:+.4f}, {pref['paired_differences']['joint_minus_hr']['ci_95'][1]:+.4f}] |

---

## 4. Secondary Endpoint: Retrospective Full-Session Centering

- **Scope:** Retrospective session-median centering across all 400 segments per subject ($N = 57,600$).
- **Target Label:** Excursion relative to session median ($y_{{\text{{full}}}} = \mathbb{{I}}(SBP \ge \text{{median}}_{{\text{{session}}}}(SBP))$).
- **Actual Positive Prevalence:** **{full['prevalence']:.2f}%** (Majority baseline: 50.0%)

| Model | AUROC [95% Bootstrap CI] | Accuracy (Prev: {full['prevalence']:.1f}%) | Balanced Accuracy | Paired $\Delta\text{{AUROC}}$ vs. HR [95% CI] |
|---|---|---|---|---|
| **HR Comparator (Frozen)** | {full['hr']['auroc']:.4f} [{full['hr']['ci_95'][0]:.4f}, {full['hr']['ci_95'][1]:.4f}] | {full['hr']['acc']:.2f}% | {full['hr']['bal_acc']:.2f}% | — |
| **Morphology (Frozen)** | **{full['morphology']['auroc']:.4f}** [{full['morphology']['ci_95'][0]:.4f}, {full['morphology']['ci_95'][1]:.4f}] | {full['morphology']['acc']:.2f}% | {full['morphology']['bal_acc']:.2f}% | **{full['paired_differences']['morphology_minus_hr']['diff']:+.4f}** [{full['paired_differences']['morphology_minus_hr']['ci_95'][0]:+.4f}, {full['paired_differences']['morphology_minus_hr']['ci_95'][1]:+.4f}] |
| **Joint (HR + Morph)** | **{full['joint']['auroc']:.4f}** [{full['joint']['ci_95'][0]:.4f}, {full['joint']['ci_95'][1]:.4f}] | {full['joint']['acc']:.2f}% | {full['joint']['bal_acc']:.2f}% | **{full['paired_differences']['joint_minus_hr']['diff']:+.4f}** [{full['paired_differences']['joint_minus_hr']['ci_95'][0]:+.4f}, {full['paired_differences']['joint_minus_hr']['ci_95'][1]:+.4f}] |

---

## 5. Execution Integrity & Provenance
- All features extracted strictly from PPG Channel 1 (`Signals[:, 1, :]`). Zero ABP inputs provided.
- Evaluated strictly once without tuning, retraining, or threshold adjustment.
- Test permanently locked at `results/.test_locked_W`.
"""
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Report written to {REPORT_PATH}.")

    with open(LOCK_FILE_PATH, "w", encoding="utf-8") as f:
        f.write(
            f"STATUS: COMPLETED\n"
            f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n"
            f"GitCommit: {git_commit}\n"
            f"FrozenModelHash: {results['verified_hashes']['Primary Morphology (W)']}\n"
            f"StopRulePassed: {stop_rule}\n"
        )
    print(f"Test lock updated to COMPLETED at {LOCK_FILE_PATH}.")
    return report


def main():
    print("Executing single-pass CalFree evaluation under Amendment W3...")
    res = run_calfree_evaluation()
    write_test_report_and_update_lock(res)
    print("Test run finished.")


if __name__ == "__main__":
    main()
