# Study C PySpark vs NumPy Feature Extraction Parity Report

**Date:** 2026-10-29  
**Cohort:** `VitalDB_Train_Subset` (`train_file.parquet`)  
**Sample Size:** 2,000 randomly sampled segments (seed=42)  
**Tolerance:** $\max(|x_{\text{NumPy}} - x_{\text{Spark}}|) < 10^{-6}$  
**Status:** **ALL PARITY CHECKS PASSED (100% BIT-EXACT / PARITY VERIFIED)**  

---

## Parity Verification Results

| Column Name | Type | NaN Count | NaN Rate | Max Abs Diff | Mean Abs Diff | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`subject_id`** | Metadata | 0 | 0.00% | `0.00e+00` | `0.00e+00` | **PASS** |
| **`segment_idx`** | Metadata | 0 | 0.00% | `0.00e+00` | `0.00e+00` | **PASS** |
| **`file_split`** | Metadata | 0 | 0.00% | `0.00e+00` | `0.00e+00` | **PASS** |
| **`split`** | Metadata | 0 | 0.00% | `0.00e+00` | `0.00e+00` | **PASS** |
| **`age`** | Metadata | 0 | 0.00% | `0.00e+00` | `0.00e+00` | **PASS** |
| **`sex`** | Metadata | 0 | 0.00% | `0.00e+00` | `0.00e+00` | **PASS** |
| **`height`** | Metadata | 0 | 0.00% | `0.00e+00` | `0.00e+00` | **PASS** |
| **`weight`** | Metadata | 0 | 0.00% | `0.00e+00` | `0.00e+00` | **PASS** |
| **`bmi`** | Metadata | 0 | 0.00% | `0.00e+00` | `0.00e+00` | **PASS** |
| **`sbp`** | Metadata | 0 | 0.00% | `0.00e+00` | `0.00e+00` | **PASS** |
| **`dbp`** | Metadata | 0 | 0.00% | `0.00e+00` | `0.00e+00` | **PASS** |
| **`hr_ibi`** | Feature | 0 | 0.00% | `0.00e+00` | `0.00e+00` | **PASS** |
| **`hr`** | Feature | 0 | 0.00% | `0.00e+00` | `0.00e+00` | **PASS** |
| **`ibi_sd`** | Feature | 0 | 0.00% | `0.00e+00` | `0.00e+00` | **PASS** |
| **`systolic_rise_time`** | Feature | 2 | 0.10% | `0.00e+00` | `0.00e+00` | **PASS** |
| **`pulse_width`** | Feature | 2 | 0.10% | `0.00e+00` | `0.00e+00` | **PASS** |
| **`decay_ratio`** | Feature | 2 | 0.10% | `0.00e+00` | `0.00e+00` | **PASS** |
| **`beat_count`** | Feature | 0 | 0.00% | `0.00e+00` | `0.00e+00` | **PASS** |
| **`is_clipped`** | Feature | 0 | 0.00% | `0.00e+00` | `0.00e+00` | **PASS** |
| **`snr_ratio`** | Feature | 0 | 0.00% | `0.00e+00` | `0.00e+00` | **PASS** |

---

## Summary of Invariants Verified
1. **Identical Output Columns:** Exactly 20 columns (11 metadata, 9 extracted features).
2. **Identical NaN Patterns:** Zero discrepancy in missingness or feature extraction failures.
3. **Numerical Equivalence:** All non-NaN float values match within machine precision ($< 10^{-6}$).
4. **Single-Segment Inductive Invariant:** Extraction is strictly per-segment with zero cross-segment leakage.
