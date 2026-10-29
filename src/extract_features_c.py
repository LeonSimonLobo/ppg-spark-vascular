"""Batch feature extraction for Development (Train file) and CalFree validation cohorts (PLAN_C.md).

Cohort selection strictly uses `file_split`, never the legacy `split` column:
- Development: file_split == 'train_file' (1,293 subjects, 465,480 segments)
- CalFree validation: file_split == 'calfree_test' (144 subjects, 57,600 segments)

Outputs:
- data/interim/features_c_train.parquet
- data/interim/features_c_calfree.parquet
- results/features_c_dev.md
"""

from pathlib import Path
import time
from typing import Dict, Any, Tuple
import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.features_c import (
    extract_features_block,
    C0_COLS,
    C1_COLS,
    C2_COLS,
    ALL_EXTRACTED_COLS,
)


def extract_cohort_features(
    mat_path: str,
    meta_df: pd.DataFrame,
    output_parquet: str,
    chunk_size: int = 15000,
    fs: int = 125,
) -> pd.DataFrame:
    """Extract single-segment features from PPG channel 1 and merge with metadata."""
    out_p = Path(output_parquet)
    if out_p.exists():
        print(f"Loading cached features from {output_parquet}...")
        return pd.read_parquet(output_parquet)

    n_rows = len(meta_df)
    n_subjs = meta_df["subject_id"].nunique()
    print(f"Extracting features for {n_rows:,} segments across {n_subjs:,} subjects from {mat_path}...")

    # Allocate arrays
    hr_ibi_arr = np.full(n_rows, np.nan, dtype=np.float32)
    hr_arr = np.full(n_rows, np.nan, dtype=np.float32)
    ibi_sd_arr = np.full(n_rows, np.nan, dtype=np.float32)
    rise_arr = np.full(n_rows, np.nan, dtype=np.float32)
    pw_arr = np.full(n_rows, np.nan, dtype=np.float32)
    dec_arr = np.full(n_rows, np.nan, dtype=np.float32)
    beat_arr = np.zeros(n_rows, dtype=np.float32)
    clipped_arr = np.zeros(n_rows, dtype=np.float32)
    snr_arr = np.zeros(n_rows, dtype=np.float32)

    t0 = time.time()
    last_print = 0

    with h5py.File(mat_path, "r") as f:
        sig_ds = f["Subset"]["Signals"]  # [1250, 2, n_rows]
        total_segs = sig_ds.shape[2]
        assert total_segs == n_rows, f"Mismatch: MAT has {total_segs} segments, meta has {n_rows}"

        for c_start in range(0, n_rows, chunk_size):
            c_end = min(n_rows, c_start + chunk_size)

            # Read PPG channel 1 (0-indexed channel 1 is PPG)
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

            if c_end - last_print >= 30000 or c_end == n_rows:
                elapsed = time.time() - t0
                speed = c_end / (elapsed + 1e-6)
                pct = c_end / n_rows * 100.0
                print(f"Processed {c_end:,}/{n_rows:,} ({pct:.1f}%) in {elapsed:.1f}s ({speed:.1f} seg/s)...")
                last_print = c_end

    # Merge metadata with extracted features
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

    # Save to parquet
    out_p.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(feat_df)
    pq.write_table(table, output_parquet, compression="zstd")
    print(f"Saved extracted features to {output_parquet} ({n_rows:,} rows)")

    return feat_df


def run_all_extractions():
    """Extract features for Train and CalFree cohorts using file_split selection."""
    # 1. Train Cohort (1,293 subjects, 465,480 segments selected strictly by file_split == 'train_file')
    print("--- 1. VitalDB Train Development Cohort (file_split == 'train_file') ---")
    train_meta_table = pq.read_table(
        "data/parquet/train_file.parquet",
        columns=["subject_id", "segment_idx", "file_split", "age", "sex", "bmi", "sbp", "dbp"],
    )
    train_meta_df = train_meta_table.to_pandas()
    assert (train_meta_df["file_split"] == "train_file").all(), "All rows must have file_split == 'train_file'"
    assert train_meta_df["subject_id"].nunique() == 1293, f"Expected 1,293 subjects, got {train_meta_df['subject_id'].nunique()}"
    assert len(train_meta_df) == 465480, f"Expected 465,480 segments, got {len(train_meta_df)}"

    train_feat_df = extract_cohort_features(
        mat_path="data/raw/VitalDB_Train_Subset.mat",
        meta_df=train_meta_df,
        output_parquet="data/interim/features_c_train.parquet",
    )

    # 2. CalFree Extra Validation Cohort (selected strictly by file_split == 'calfree_test')
    print("--- 2. CalFree Extra Validation Cohort (file_split == 'calfree_test') ---")
    calfree_meta_table = pq.read_table(
        "data/parquet/calfree_test.parquet",
        columns=["subject_id", "segment_idx", "file_split", "age", "sex", "bmi", "sbp", "dbp"],
    )
    calfree_meta_df = calfree_meta_table.to_pandas()
    assert (calfree_meta_df["file_split"] == "calfree_test").all(), "All rows must have file_split == 'calfree_test'"
    assert calfree_meta_df["subject_id"].nunique() == 144, f"Expected 144 subjects, got {calfree_meta_df['subject_id'].nunique()}"
    assert len(calfree_meta_df) == 57600, f"Expected 57,600 segments, got {len(calfree_meta_df)}"

    calfree_feat_df = extract_cohort_features(
        mat_path="data/raw/VitalDB_CalFree_Test_Subset.mat",
        meta_df=calfree_meta_df,
        output_parquet="data/interim/features_c_calfree.parquet",
    )

    # 3. Generate Markdown Audit Report for Development Cohort
    print("--- 3. Generating Audit Report ---")
    generate_audit_report(train_feat_df, calfree_feat_df, "results/features_c_dev.md")


def generate_audit_report(
    train_df: pd.DataFrame,
    calfree_df: pd.DataFrame,
    report_path: str,
):
    """Generate statistical summary and missingness report for extracted features."""
    cols = ALL_EXTRACTED_COLS + ["sbp"]

    def compute_stats(df, col):
        s = df[col]
        valid = s.dropna()
        n_valid = len(valid)
        n_nan = len(s) - n_valid
        nan_pct = n_nan / len(s) * 100.0
        if n_valid == 0:
            return {"valid": 0, "nan": n_nan, "pct": 100.0, "mean": 0, "std": 0, "median": 0, "q1": 0, "q3": 0, "min": 0, "max": 0}
        q1, med, q3 = np.percentile(valid, [25, 50, 75])
        return {
            "valid": n_valid,
            "nan": n_nan,
            "pct": nan_pct,
            "mean": float(valid.mean()),
            "std": float(valid.std()),
            "median": float(med),
            "q1": float(q1),
            "q3": float(q3),
            "min": float(valid.min()),
            "max": float(valid.max()),
        }

    train_stats = {col: compute_stats(train_df, col) for col in cols}
    cf_stats = {col: compute_stats(calfree_df, col) for col in cols}

    n_train = len(train_df)
    n_cf = len(calfree_df)

    p120_train = np.mean(train_df["sbp"] >= 120.0) * 100.0
    p130_train = np.mean(train_df["sbp"] >= 130.0) * 100.0
    p120_cf = np.mean(calfree_df["sbp"] >= 120.0) * 100.0
    p130_cf = np.mean(calfree_df["sbp"] >= 130.0) * 100.0

    report = rf"""# Feature Extraction & Quality Audit: Study C (Single-Segment Cross-Subject)

**Date:** 2026-10-29  
**Specification:** `PLAN_C.md` (cross-subject classification of SBP $\ge 120$ mmHg on unseen subjects)  
**AAMI_Cal Status:** **PRISTINE / UNREAD.** (Zero test segments loaded or accessed)  

---

## 1. Cohort Dimensions & Label Prevalences

| Cohort | File Split Filter | Subjects | Segments | SBP $\ge 120$ mmHg Prevalence | SBP $\ge 130$ mmHg Prevalence |
|---|---|---|---|---|---|
| **VitalDB Train (Development)** | `file_split == 'train_file'` | {train_df['subject_id'].nunique():,} | {n_train:,} | **{p120_train:.2f}%** | {p130_train:.2f}% |
| **CalFree (Extra Validation)** | `file_split == 'calfree_test'` | {calfree_df['subject_id'].nunique():,} | {n_cf:,} | **{p120_cf:.2f}%** | {p130_cf:.2f}% |

---

## 2. Feature Distributions & Missingness (VitalDB Train Cohort, $N = {n_train:,}$)

| Feature | Valid Count | Missing Count | Missing % | Mean | Std | Median | IQR ($Q_1, Q_3$) | [Min, Max] |
|---|---|---|---|---|---|---|---|---|
"""
    for col in cols:
        st = train_stats[col]
        report += (
            f"| **`{col}`** | {st['valid']:,} | {st['nan']:,} | {st['pct']:.2f}% | "
            f"{st['mean']:.4f} | {st['std']:.4f} | {st['median']:.4f} | "
            f"{st['q3'] - st['q1']:.4f} ({st['q1']:.4f}, {st['q3']:.4f}) | "
            f"[{st['min']:.4f}, {st['max']:.4f}] |\n"
        )

    report += rf"""
---

## 3. Feature Distributions & Missingness (CalFree Extra Validation Cohort, $N = {n_cf:,}$)

| Feature | Valid Count | Missing Count | Missing % | Mean | Std | Median | IQR ($Q_1, Q_3$) | [Min, Max] |
|---|---|---|---|---|---|---|---|---|
"""
    for col in cols:
        st = cf_stats[col]
        report += (
            f"| **`{col}`** | {st['valid']:,} | {st['nan']:,} | {st['pct']:.2f}% | "
            f"{st['mean']:.4f} | {st['std']:.4f} | {st['median']:.4f} | "
            f"{st['q3'] - st['q1']:.4f} ({st['q1']:.4f}, {st['q3']:.4f}) | "
            f"[{st['min']:.4f}, {st['max']:.4f}] |\n"
        )

    report += r"""
---

## 4. Extraction Invariants & Verification

1. **Zero Transduction / Single-Segment Scope:** Every segment was extracted strictly from its own 10-second PPG window and subject-level demographics. Zero within-subject centering or session-level demeaning was applied.
2. **Missing Feature Strategy:** Any missing values in morphology features will be imputed using the training-fold medians with an accompanying binary indicator column (`is_missing_<col>`), preserving all segments without row drops.
3. **Pristine Test Boundary:** `VitalDB_AAMI_Cal_Subset.mat`, `VitalDB_AAMI_Test_Subset.mat`, and `VitalDB_CalBased_Test_Subset.mat` have **zero read calls** in this pipeline.
"""
    out_file = Path(report_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(report, encoding="utf-8")
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    run_all_extractions()
