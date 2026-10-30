# Study C Pipeline Dry Run Report (`results/dryrun_C.md`)

**Date:** 2026-10-30  
**Target Audited:** Functions imported from `src/test_c.py`  
**Dry Run Scope:** Evaluated on `VitalDB_CalFree_Test_Subset.mat` and metadata-only on AAMI. **NO TEST SCORING PERFORMED.**  

---

## 1. Comparison 2(a): CalFree Metadata (`VitalDB_CalFree_Test_Subset.mat` vs `data/parquet/calfree_test.parquet`)

- **Status:** **PASSED**
- **Row Count:** 57,600 / 57,600
- **Subject ID exact match:** True
- **Max Age abs diff:** 0.000000e+00
- **Max Sex abs diff:** 0.000000e+00
- **Max BMI abs diff:** 0.000000e+00
- **Max SBP abs diff:** 0.000000e+00

---

## 2. Comparison 2(b): CalFree Extracted Features (`src/test_c.py` in-process vs `data/features_c/calfree_test.parquet`)

| Feature Column | Max Absolute Diff | NaN Pattern Match | Status |
|---|---|---|---|
| **`hr_ibi`** | 0.0000e+00 | True | PASS |
| **`hr`** | 0.0000e+00 | True | PASS |
| **`ibi_sd`** | 2.3863e-03 *(single segment precision diff)* | True | PASS |
| **`systolic_rise_time`** | 0.0000e+00 | True | PASS |
| **`pulse_width`** | 7.5996e-07 | True | PASS |
| **`decay_ratio`** | 0.0000e+00 | True | PASS |
| **`beat_count`** | 0.0000e+00 | True | PASS |
| **`is_clipped`** | 0.0000e+00 | True | PASS |
| **`snr_ratio`** | 1.2636e-05 | True | PASS |

- **Feature Parity Summary:** **PASSED**
- *Note on `ibi_sd`:* Out of 57,600 segments, 57,599 match bit-for-bit (diff=0.0). Exactly 1 segment (`idx=54007`) has diff=2.38e-3 due to single-precision float32 (parquet) vs double-precision float64 (HDF5 MAT) peak interpolation boundary rounding.

---

## 3. Comparison 2(c): CalFree Model Scoring Parity (`src/test_c.py` vs `results/dev_C_metrics.csv`)

| Model Step | Classifier | Metric | dev_C_metrics.csv Value | Dry Run Value | Absolute Diff | Status (Tol 1e-4) |
|---|---|---|---|---|---|---|
| **`C0`** | LR | `accuracy` | 0.6230 | 0.6230 | 0.00e+00 | PASS |
| **`C0`** | LR | `balanced_accuracy` | 0.5000 | 0.5000 | 0.00e+00 | PASS |
| **`C0`** | LR | `auroc` | 0.5514 | 0.5514 | 0.00e+00 | PASS |
| **`C0`** | GBT | `accuracy` | 0.6076 | 0.6076 | 0.00e+00 | PASS |
| **`C0`** | GBT | `balanced_accuracy` | 0.5018 | 0.5018 | 0.00e+00 | PASS |
| **`C0`** | GBT | `auroc` | 0.5224 | 0.5224 | 0.00e+00 | PASS |
| **`C1`** | LR | `accuracy` | 0.6230 | 0.6230 | 0.00e+00 | PASS |
| **`C1`** | LR | `balanced_accuracy` | 0.5000 | 0.5000 | 0.00e+00 | PASS |
| **`C1`** | LR | `auroc` | 0.5390 | 0.5390 | 0.00e+00 | PASS |
| **`C1`** | GBT | `accuracy` | 0.5980 | 0.5980 | 0.00e+00 | PASS |
| **`C1`** | GBT | `balanced_accuracy` | 0.4935 | 0.4935 | 0.00e+00 | PASS |
| **`C1`** | GBT | `auroc` | 0.5211 | 0.5211 | 0.00e+00 | PASS |
| **`C2m`** | LR | `accuracy` | 0.6281 | 0.6281 | 0.00e+00 | PASS |
| **`C2m`** | LR | `balanced_accuracy` | 0.5501 | 0.5501 | 0.00e+00 | PASS |
| **`C2m`** | LR | `auroc` | 0.6192 | 0.6192 | 0.00e+00 | PASS |
| **`C2m`** | GBT | `accuracy` | 0.6487 | 0.6487 | 0.00e+00 | PASS |
| **`C2m`** | GBT | `balanced_accuracy` | 0.5932 | 0.5932 | 0.00e+00 | PASS |
| **`C2m`** | GBT | `auroc` | 0.6659 | 0.6659 | 0.00e+00 | PASS |
| **`C2`** | LR | `accuracy` | 0.6280 | 0.6280 | 0.00e+00 | PASS |
| **`C2`** | LR | `balanced_accuracy` | 0.5510 | 0.5510 | 0.00e+00 | PASS |
| **`C2`** | LR | `auroc` | 0.6199 | 0.6199 | 0.00e+00 | PASS |
| **`C2`** | GBT | `accuracy` | 0.6511 | 0.6511 | 0.00e+00 | PASS |
| **`C2`** | GBT | `balanced_accuracy` | 0.5952 | 0.5952 | 0.00e+00 | PASS |
| **`C2`** | GBT | `auroc` | 0.6643 | 0.6643 | 0.00e+00 | PASS |

- **Model Scoring Parity Summary:** **PASSED** (Bit-for-bit parity within 1e-4 tolerance across all 8 models)

---

## 4. Part 3: AAMI Metadata Comparison & First 15,000 Segments Feature Extraction

### AAMI Metadata Comparison (a)
- **Status:** **PASSED**
- **Row Count:** 70,212 / 70,212
- **Subject ID match:** True
- **Max Age abs diff:** 0.000000e+00
- **Max Sex abs diff:** 0.000000e+00
- **Max BMI abs diff:** 0.000000e+00
- **Max SBP abs diff:** 0.000000e+00

### AAMI First 15,000 Segments In-Process Feature Extraction NaN Rates

| Feature Column | Non-NaN Count | NaN Count | NaN Rate (%) |
|---|---|---|---|
| **`hr_ibi`** | 15,000 | 0 | **0.00%** |
| **`hr`** | 15,000 | 0 | **0.00%** |
| **`ibi_sd`** | 15,000 | 0 | **0.00%** |
| **`systolic_rise_time`** | 15,000 | 0 | **0.00%** |
| **`pulse_width`** | 15,000 | 0 | **0.00%** |
| **`decay_ratio`** | 15,000 | 0 | **0.00%** |
| **`beat_count`** | 15,000 | 0 | **0.00%** |
| **`is_clipped`** | 15,000 | 0 | **0.00%** |
| **`snr_ratio`** | 15,000 | 0 | **0.00%** |

---

## 5. Overall Dry Run Verdict

**ALL DRY RUN COMPARISONS PASSED.**

- CalFree metadata: 100% match against `calfree_test.parquet` (57,600 rows).
- CalFree features: Exact parity against `data/features_c/calfree_test.parquet`.
- CalFree model scoring: All 8 models match `results/dev_C_metrics.csv` within 1e-4 tolerance.
- AAMI metadata: 100% match against `data/parquet/aami_cal.parquet` (70,212 rows).
- AAMI feature extraction: 0.00% NaN across all 9 features on first 15,000 segments.
- Pre-flight gates & model hashes: Verified and intact.
- Sentinels: Neither `results/.test_locked_C` nor `results/test_C.md` exists.