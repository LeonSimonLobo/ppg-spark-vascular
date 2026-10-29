# Study C Distributed Feature Extraction Report (PySpark)

**Date:** 2026-10-29  
**Pipeline:** PySpark `mapInPandas` + Arrow with `src/features_c.py:extract_features_block`  
**Executor Cores:** `8`  
**Partitions:** `64`  
**Status:** **SUCCESSFUL EXTRACTION & VERIFICATION**  

---

## 1. Cohort Extraction & Throughput Summary

| Cohort Identifier | Input Parquet | Output Parquet | Total Rows | Unique Subjects | Elapsed Time (s) | Throughput (rows/s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`train_file`** | `data/parquet/train_file.parquet` | `data\features_c\train_file.parquet` | **465,480** | **1,293** | **92.64s** | **5024.8** |
| **`calfree_test`** | `data/parquet/calfree_test.parquet` | `data\features_c\calfree_test.parquet` | **57,600** | **144** | **52.05s** | **1106.7** |

---

## 2. Feature Missingness & Extraction Failure Rates

| Feature Name | Train File (`train_file`) NaN % | CalFree Test (`calfree_test`) NaN % | Failure Handling Mode |
| :--- | :--- | :--- | :--- |
| **`hr_ibi`** | 0.00% | 0.00% | Kept as NaN (PlanCImputer training-median imputation) |
| **`hr`** | 0.00% | 0.00% | Kept as NaN (PlanCImputer training-median imputation) |
| **`ibi_sd`** | 0.00% | 0.00% | Kept as NaN (PlanCImputer training-median imputation) |
| **`systolic_rise_time`** | 0.00% | 0.00% | Kept as NaN (PlanCImputer training-median imputation) |
| **`pulse_width`** | 0.00% | 0.00% | Kept as NaN (PlanCImputer training-median imputation) |
| **`decay_ratio`** | 0.00% | 0.00% | Kept as NaN (PlanCImputer training-median imputation) |
| **`beat_count`** | 0.00% | 0.00% | Kept as NaN (PlanCImputer training-median imputation) |
| **`is_clipped`** | 0.00% | 0.00% | Kept as NaN (PlanCImputer training-median imputation) |
| **`snr_ratio`** | 0.00% | 0.00% | Kept as NaN (PlanCImputer training-median imputation) |

---

## 3. Protocol Invariants Verified
1. **Single-Segment Feature Purity:** Features are extracted strictly from individual 10-s waveforms with zero per-subject centering or transduction.
2. **Zero Rows Dropped:** Both cohorts retain 100% of rows (Train: 465,480 rows; CalFree: 57,600 rows). Extraction failures are preserved as NaN for downstream indicator-flagged imputation.
3. **Schema Conformance:** Canonical 20-column schema matches `OUTPUT_SCHEMA` exactly across both datasets.
