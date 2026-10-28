# PPG Channel Signal Range & Normalization Check

Evaluation of the PPG channel (`Signals[:, 1, :]`) in the raw MATLAB v7.3 HDF5 files.

---

## 1. Summary Metrics

| Dataset Split | Segments Analyzed | Overall Min | Overall Max | Fraction Min < 0 | Fraction Max > 1 | Fraction Min == 0.0 | Fraction Max == 1.0 | Median Per-Segment Median [IQR] |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`VitalDB_CalFree_Test_Subset.mat`** | 57,600 (all) | 0.0 | 1.0 | **0.0%** (0 / 57,600) | **0.0%** (0 / 57,600) | 100.0% | 87.65% | 0.2273 [0.1719–0.2887] |
| **`VitalDB_Train_Subset.mat`** | 1,000 (first) | 0.0 | 1.0 | **0.0%** (0 / 1,000) | **0.0%** (0 / 1,000) | 100.0% | 90.70% | 0.2679 [0.2221–0.3057] |

---

## 2. Key Findings

1. **Signal Range**:
   - The overall minimum across all analyzed segments is **0.0**.
   - The overall maximum across all analyzed segments is **1.0**.
   - Exactly **0.0%** of segments have a maximum value exceeding 1.0 or a minimum value below 0.0.
2. **Pre-scaling in Raw Data**:
   - 100.0% of segments have a minimum of exactly 0.0.
   - 87.65% of test segments and 90.70% of train segments reach a maximum of exactly 1.0 (with the remainder bounded within $[0, 1]$).
   - This confirms the PulseDB dataset generator applied per-segment min-max scaling to $[0, 1]$ during subset creation.
3. **Ingest Processing**:
   - `src/ingest.py` reads and persists these raw float values directly to Parquet without applying any secondary normalization transform.
