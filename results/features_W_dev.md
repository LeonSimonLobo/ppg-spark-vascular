# Features W Extraction Report: Development / Training Split

**Date:** 2026-10-29  
**Cohort:** `VitalDB_Train_Subset.mat` training split (`split == 'train'`)  
**Scope:** 1,034 subjects | 372,240 segments (exactly 360 segments per subject)  
**CalFree / Test Status:** Pristine (0 test segments loaded or scored)  
**Inputs:** PPG channel 1 only (no ABP inputs used in feature extraction or centering)  

---

## 1. Summary Statistics and Missingness

| Feature | Valid Count | NaN Count | NaN Share | Mean | Std | Median | IQR ($Q_1, Q_3$) | [Min, Max] |
|---|---|---|---|---|---|---|---|---|
| **`hr`** (bpm) | 372,240 | 0 | 0.00% | 75.69 | 13.53 | 72.0 | 18.0 (66.0, 84.0) | [42.0, 156.0] |
| **`systolic_rise_time`** (s) | 372,240 | 0 | 0.00% | 0.2209 | 0.0478 | 0.2040 | 0.0720 (0.1840, 0.2560) | [0.1280, 0.3040] |
| **`pulse_width`** (s) | 372,240 | 0 | 0.00% | 0.2341 | 0.0522 | 0.2330 | 0.0756 (0.1934, 0.2689) | [0.1110, 0.5543] |
| **`decay_ratio`** | 372,240 | 0 | 0.00% | 2.7147 | 0.7484 | 2.6842 | 0.9634 (2.2105, 3.1739) | [0.4737, 7.1818] |

### Centered Features ($\Delta X_{i, j} = X_{i, j} - \text{median}(X_i)$):
| Feature | Valid Count | NaN Count | Mean | Std | Median | IQR ($Q_1, Q_3$) |
|---|---|---|---|---|---|---|
| **`delta_hr`** (bpm) | 372,240 | 0 | +0.3982 | 7.8197 | +0.0000 | 12.0000 (-6.0000, +6.0000) |
| **`delta_systolic_rise_time`** (s) | 372,240 | 0 | +0.0042 | 0.0364 | +0.0000 | 0.0160 (-0.0080, +0.0080) |
| **`delta_pulse_width`** (s) | 372,240 | 0 | +0.0021 | 0.0306 | +0.0000 | 0.0282 (-0.0130, +0.0152) |
| **`delta_decay_ratio`** | 372,240 | 0 | -0.0068 | 0.5428 | +0.0000 | 0.4907 (-0.2440, +0.2468) |

---

## 2. Segment-Level Missingness & Coverage

- **Total Segments:** 372,240
- **Segments with ANY NaN across Morphology (`systolic_rise_time`, `pulse_width`, `decay_ratio`):** **0.00%** (0 / 372,240)
- **Complete Morphology Coverage:** **100.00%** (372,240 / 372,240)
- **Segments with ANY NaN across all 4 features (including `hr`):** **0.00%** (0 / 372,240)

---

## 3. Centering Sanity Verification

Per-subject centering subtracts $\text{median}(X_i)$ for each subject $i$. Across all 1,034 training subjects:
- Maximum absolute subject-median of `delta_hr`: **0.000000** (expected 0.0)
- Maximum absolute subject-median of `delta_systolic_rise_time`: **0.000000** (expected 0.0)
- Maximum absolute subject-median of `delta_pulse_width`: **0.000000** (expected 0.0)
- Maximum absolute subject-median of `delta_decay_ratio`: **0.000000** (expected 0.0)

Centering verification **PASSED**.
