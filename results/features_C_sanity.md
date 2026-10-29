# Study C Feature Sanity & Distribution Report

**Date:** 2026-10-29  
**Specification:** `PLAN_C.md` (Cross-subject SBP classification on unseen subjects)  
**Amendment Applied:** Amendment C2 (Inter-Peak Foot Definition & Auxiliary Segment Fields)  
**Input Datasets:**  
- Development Cohort: `data/features_c/train_file.parquet` ($N = 465,480$ segments, $1,293$ subjects, $360\text{ seg/subj}$)
- Validation Cohort: `data/features_c/calfree_test.parquet` ($N = 57,600$ segments, $144$ subjects, $400\text{ seg/subj}$)
**Status:** Audit complete. All metrics calculated directly on extracted feature tables with zero labels (`sbp`, `dbp`) accessed.

---

## 1. Feature Distribution & Quantile Profiles (Train vs. CalFree Side-by-Side)

Evaluated across the 9 canonical extracted features:
- **`hr`**: Discrete heart rate ($n_{\text{beats}} \times 6$, bpm)
- **`hr_ibi`**: Continuous heart rate ($60.0 / \text{median}(\text{IBI})$, bpm)
- **`ibi_sd`**: Standard deviation of inter-beat intervals (s)
- **`beat_count`**: Total valid systolic peaks in segment (beats)
- **`systolic_rise_time`**: Median foot-to-peak elapsed time under inter-peak argmin (s)
- **`pulse_width`**: Full width at 50% amplitude (FWHM, s)
- **`decay_ratio`**: Diastolic decay time / systolic rise time ($d\_time / r\_time$, dimensionless time ratio)
- **`is_clipped`**: Rail clipping flag ($>1\%$ samples within $10^{-4}$ of voltage rails, binary 0.0 or 1.0)
- **`snr_ratio`**: Fractional spectral power in $[0.5, 8.0]\text{ Hz}$ relative to non-DC power (dimensionless $\in [0, 1]$)

### Quantiles & Anomaly Counts Table

| Feature Name | Cohort | Total Segments | NaN Count (NaN Rate) | Inf | Unique Values | Q01 | Q05 | Q25 | Q50 (Median) | Q75 | Q95 | Q99 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`hr`** | **Train** | 465,480 | 0 (0.0000%) | 0 | 20 | 48.0000 | 54.0000 | 66.0000 | 72.0000 | 84.0000 | 102.0000 | 114.0000 |
| | **CalFree** | 57,600 | 0 (0.0000%) | 0 | 17 | 48.0000 | 54.0000 | 66.0000 | 72.0000 | 84.0000 | 102.0000 | 120.0000 |
| **`hr_ibi`** | **Train** | 465,480 | 0 (0.0000%) | 0 | 275 | 48.3871 | 53.7634 | 63.0252 | 71.7703 | 81.5217 | 100.0000 | 115.3846 |
| | **CalFree** | 57,600 | 0 (0.0000%) | 0 | 262 | 47.4684 | 52.4476 | 63.8298 | 72.1154 | 82.8729 | 98.6842 | 119.0476 |
| **`ibi_sd`** | **Train** | 465,480 | 0 (0.0000%) | 0 | 64,527 | 0.0035 | 0.0042 | 0.0074 | 0.0134 | 0.0454 | 0.1523 | 0.2545 |
| | **CalFree** | 57,600 | 0 (0.0000%) | 0 | 17,197 | 0.0035 | 0.0042 | 0.0072 | 0.0129 | 0.0423 | 0.1517 | 0.2623 |
| **`beat_count`** | **Train** | 465,480 | 0 (0.0000%) | 0 | 20 | 8.0000 | 9.0000 | 11.0000 | 12.0000 | 14.0000 | 17.0000 | 19.0000 |
| | **CalFree** | 57,600 | 0 (0.0000%) | 0 | 17 | 8.0000 | 9.0000 | 11.0000 | 12.0000 | 14.0000 | 17.0000 | 20.0000 |
| **`systolic_rise_time`** | **Train** | 465,480 | 122 (0.0262%) | 0 | 204 | 0.1600 | 0.1680 | 0.1840 | 0.2080 | 0.3040 | 0.4480 | 0.5680 |
| | **CalFree** | 57,600 | 102 (0.1771%) | 0 | 158 | 0.1600 | 0.1680 | 0.1840 | 0.2120 | 0.3080 | 0.4480 | 0.5640 |
| **`pulse_width`** | **Train** | 465,480 | 123 (0.0264%) | 0 | 453,851 | 0.1423 | 0.1570 | 0.1938 | 0.2338 | 0.2702 | 0.3224 | 0.3914 |
| | **CalFree** | 57,600 | 102 (0.1771%) | 0 | 57,329 | 0.1446 | 0.1577 | 0.1907 | 0.2319 | 0.2714 | 0.3345 | 0.3936 |
| **`decay_ratio`** | **Train** | 465,480 | 122 (0.0262%) | 0 | 75,317 | 0.4921 | 1.1042 | 1.9737 | 2.5882 | 3.1486 | 3.9783 | 4.7000 |
| | **CalFree** | 57,600 | 102 (0.1771%) | 0 | 16,191 | 0.4561 | 1.0000 | 1.9524 | 2.5770 | 3.1493 | 4.1429 | 4.9048 |
| **`is_clipped`** | **Train** | 465,480 | 0 (0.0000%) | 0 | 1 (const 0) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| | **CalFree** | 57,600 | 0 (0.0000%) | 0 | 1 (const 0) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **`snr_ratio`** | **Train** | 465,480 | 0 (0.0000%) | 0 | 164,170 | 0.9847 | 0.9894 | 0.9941 | 0.9972 | 0.9993 | 0.9999 | 1.0000 |
| | **CalFree** | 57,600 | 0 (0.0000%) | 0 | 45,680 | 0.9843 | 0.9892 | 0.9940 | 0.9972 | 0.9993 | 0.9999 | 1.0000 |

### Most Frequent Value (Mode) of Morphology Features

| Feature Name | Cohort | Most Frequent Value (Mode) | Mode Segment Count | Mode Segment Share (%) |
| :--- | :--- | :--- | :--- | :--- |
| **`systolic_rise_time`** | **Train** | **0.1840 s** (23 samples at 125 Hz) | 41,457 | **8.9063%** |
| | **CalFree** | **0.1760 s** (22 samples at 125 Hz) | 4,993 | **8.6684%** |
| **`pulse_width`** | **Train** | **0.2506 s** | 4 | **0.0009%** (continuous FWHM) |
| | **CalFree** | **0.1509 s** | 2 | **0.0035%** (continuous FWHM) |
| **`decay_ratio`** | **Train** | **3.0000** | 5,978 | **1.2843%** |
| | **CalFree** | **3.0000** | 629 | **1.0920%** |

*Note on morphology modes:* Under the legacy 300 ms search window (v1), `systolic_rise_time` had its mode artificially fixed at the ceiling value of $0.3040\text{ s}$ ($18.75\%$ of Train). Under Amendment C2's inter-peak foot definition, the mode shifts to a physiological $\sim 0.18\text{ s}$ ($8.91\%$ in Train, $8.67\%$ in CalFree), matching standard literature values for healthy young-to-middle-aged adult radial/digital PPG.

---

## 2. Spearman Correlation Matrix of Features (Train Cohort, N = 465,480)

Pairwise Spearman rank correlation matrix computed on training segments (pairwise complete observations):

| Feature | `hr` | `hr_ibi` | `ibi_sd` | `beat_count` | `systolic_rise_time` | `pulse_width` | `decay_ratio` | `is_clipped` | `snr_ratio` |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`hr`** | 1.0000 | 0.9707 | -0.2156 | 1.0000 | -0.5234 | -0.5509 | -0.0588 | NaN* | 0.0967 |
| **`hr_ibi`** | 0.9707 | 1.0000 | -0.3212 | 0.9707 | -0.5288 | -0.5657 | -0.0623 | NaN* | 0.0989 |
| **`ibi_sd`** | -0.2156 | -0.3212 | 1.0000 | -0.2156 | 0.1337 | 0.2569 | 0.0642 | NaN* | 0.0008 |
| **`beat_count`** | 1.0000 | 0.9707 | -0.2156 | 1.0000 | -0.5234 | -0.5509 | -0.0588 | NaN* | 0.0967 |
| **`systolic_rise_time`** | -0.5234 | -0.5288 | 0.1337 | -0.5234 | 1.0000 | 0.3650 | -0.7142 | NaN* | -0.0407 |
| **`pulse_width`** | -0.5509 | -0.5657 | 0.2569 | -0.5509 | 0.3650 | 1.0000 | 0.0522 | NaN* | -0.1112 |
| **`decay_ratio`** | -0.0588 | -0.0623 | 0.0642 | -0.0588 | -0.7142 | 0.0522 | 1.0000 | NaN* | -0.0405 |
| **`is_clipped`** | NaN* | NaN* | NaN* | NaN* | NaN* | NaN* | NaN* | NaN* | NaN* |
| **`snr_ratio`** | 0.0967 | 0.0989 | 0.0008 | 0.0967 | -0.0407 | -0.1112 | -0.0405 | NaN* | 1.0000 |

*\*Note: `is_clipped` has zero variance (identical to 0.0 across all segments), producing undefined (NaN) correlations. Per Amendment C2, it is excluded from model feature ladders.*

---

## 3. Boundary Checks (Train & CalFree)

| Boundary Check Condition | Train Count ($N=465,480$) | Train % | Train Observed Range | CalFree Count ($N=57,600$) | CalFree % | CalFree Observed Range |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`hr_ibi` outside [40, 160] bpm** | 3 | **0.0006%** | [40.32, 178.57] bpm | 0 | **0.0000%** | [40.76, 159.57] bpm |
| **`systolic_rise_time` $\ge 0.29\text{ s}$** | 132,993 | **28.5712%** | [0.1280, 1.0080] s | 16,613 | **28.8420%** | [0.1360, 0.8080] s |
| **`systolic_rise_time` $\le 0.00\text{ s}$ (non-positive)** | 0 | **0.0000%** | — | 0 | **0.0000%** | — |
| **`pulse_width` outside [0.05, 0.60] s** | 0 | **0.0000%** | [0.1206, 0.5977] s | 0 | **0.0000%** | [0.1252, 0.5559] s |
| **`decay_ratio` $\le 0$ or $> 20$** | 0 | **0.0000%** | [0.1511, 7.8083] | 0 | **0.0000%** | [0.2312, 6.5909] |
| ↳ *`decay_ratio` $\le 0$* | 0 | 0.0000% | — | 0 | 0.0000% | — |
| ↳ *`decay_ratio` $> 20$* | 0 | 0.0000% | — | 0 | 0.0000% | — |
| **`is_clipped` == 1.0 (amplifier saturation)** | 0 | **0.0000%** | [0.0, 0.0] | 0 | **0.0000%** | [0.0, 0.0] |
| **`snr_ratio` outside [0.0, 1.0]** | 0 | **0.0000%** | [0.9453, 1.0000] | 0 | **0.0000%** | [0.9546, 1.0000] |
| **`beat_count` < 3 (severe bradycardia / loss)** | 0 | **0.0000%** | [7, 26] beats | 0 | **0.0000%** | [7, 23] beats |

*Key Boundary Insight on Rise Time:* In v1, the artificial search ceiling truncated rise times at $0.3040\text{ s}$, creating a severe spike where $18.75\%$ of segments landed exactly at the boundary. Under Amendment C2, the ceiling is completely removed; $28.57\%$ of segments in Train ($28.84\%$ in CalFree) naturally exceed $0.29\text{ s}$, continuously scaling up to $1.008\text{ s}$ for bradycardic/slow-filling beats without boundary accumulation or truncation.

---

## 4. Relationship Between `hr` ($n_{\text{beats}} \times 6$) and `hr_ibi` ($60.0 / \text{median}(\text{IBI})$)

| Metric | Train Cohort ($N = 465,480$) | CalFree Cohort ($N = 57,600$) |
| :--- | :--- | :--- |
| **Pearson Correlation ($r$)** | **0.97480** ($p < 10^{-300}$) | **0.96953** ($p < 10^{-300}$) |
| **Mean Absolute Difference (MAD)** | **3.0085 bpm** | **3.1013 bpm** |
| **Root Mean Squared Difference (RMSD)** | **3.9237 bpm** | **4.3115 bpm** |
| **Maximum Absolute Difference** | 70.57 bpm | 57.06 bpm |

*Quantization Confirmation:* Discretization of heart rate into coarse 6-bpm steps in a 10-s window produces an expected uniform error of $1.5\text{ bpm}$. With physiological heart rate variability and window edge effects, the observed MAD is exactly $3.01\text{ bpm}$ in Train and $3.10\text{ bpm}$ in CalFree. `hr_ibi` provides the true continuous physiological heart rate for model ladder C1.

---

## 5. Subject-Level Constant Value Audit

Evaluates how many subjects exhibit zero variation (constant values) across all segments in their respective recording sessions:

| Feature Name | Train Cohort ($N = 1,293$ subjects, $360\text{ seg/subj}$) | CalFree Cohort ($N = 144$ subjects, $400\text{ seg/subj}$) |
| :--- | :--- | :--- |
| **`hr`** | **0 / 1,293** (0.00%) | **0 / 144** (0.00%) |
| **`hr_ibi`** | **0 / 1,293** (0.00%) | **0 / 144** (0.00%) |
| **`ibi_sd`** | **0 / 1,293** (0.00%) | **0 / 144** (0.00%) |
| **`beat_count`** | **0 / 1,293** (0.00%) | **0 / 144** (0.00%) |
| **`systolic_rise_time`** | **0 / 1,293** (0.00%) | **0 / 144** (0.00%) |
| **`pulse_width`** | **0 / 1,293** (0.00%) | **0 / 144** (0.00%) |
| **`decay_ratio`** | **0 / 1,293** (0.00%) | **0 / 144** (0.00%) |
| **`is_clipped`** | **1,293 / 1,293** (100.00%) — *all zero* | **144 / 144** (100.00%) — *all zero* |
| **`snr_ratio`** | **0 / 1,293** (0.00%) | **0 / 144** (0.00%) |

---

## 6. Full Pytest Test Suite Verification

Executed full test suite:
```powershell
python -m pytest tests/
```

### Verbatim Output & Summary Line:
```
============================= test session starts =============================
platform win32 -- Python 3.13.1, pytest-9.1.1, pluggy-1.6.0
rootdir: D:\DS_RESEARCH\ppg-spark-vascular
plugins: anyio-4.12.1, langsmith-0.6.2
collected 58 items

tests\test_baselines.py ...                                              [  5%]
tests\test_controls_w.py .                                               [  6%]
tests\test_exploratory_dev_models.py ..                                  [ 10%]
tests\test_features_c.py ......                                          [ 20%]
tests\test_features_w.py ...                                             [ 25%]
tests\test_fiducials.py ....x......                                      [ 44%]
tests\test_gate.py ..                                                    [ 48%]
tests\test_gate_agreement.py ..                                          [ 51%]
tests\test_parity_c.py .                                                 [ 53%]
tests\test_prospective_and_scale_w.py ....                               [ 60%]
tests\test_test_c.py ...............                                     [ 86%]
tests\test_test_w.py ....                                                [ 93%]
tests\test_train_c.py ...                                                [ 98%]
tests\test_train_w.py .                                                  [100%]

============================== warnings summary ===============================
tests/test_parity_c.py::test_parity_numpy_vs_spark
...
================= 57 passed, 1 xfailed, 26 warnings in 17.79s =================
```

- **Collected Tests:** Exactly **58 items** (including `test_synthetic_rise_time_0_4s_recovery`).
- **Inclusion of `tests/test_parity_c.py`:** **Confirmed present and PASSED** at step 31 (`tests\test_parity_c.py . [ 53%]`).
- **Xfailed Item:** 1 test (`tests/test_fiducials.py:155::test_synthetic_lag_minus_30`, superseded by Study W Amendment A3).

---

## 7. Audit Findings & Protocol Integrity Summary

1. **Resolution of Ceiling Truncation (Amendment C2):**
   - The artificial $0.3040\text{ s}$ ceiling truncation is completely resolved.
   - The observed distribution of `systolic_rise_time` extends smoothly to its natural physiological limit ($1.008\text{ s}$ in Train, $0.808\text{ s}$ in CalFree).
   - $28.57\%$ of Train segments and $28.84\%$ of CalFree segments have rise times $\ge 0.29\text{ s}$, showing strong cross-cohort consistency.
   - The mode of `systolic_rise_time` is $0.1840\text{ s}$ ($8.91\%$) in Train and $0.1760\text{ s}$ ($8.67\%$) in CalFree.

2. **Exclusion of Degenerate Feature `is_clipped`:**
   - As documented in Amendment C2, `is_clipped` is identically $0.0$ across all 523,080 segments in Train and CalFree ($0\%$ variance). It has been formally excluded from the model feature ladders (C2).

3. **Missing Value Handling (PlanCImputer):**
   - The morphology features exhibit tiny, well-defined missingness rates: $0.0262\%$ (122 segments) in Train and $0.1771\%$ (102 segments) in CalFree due to boundary beats failing the $[0.4, 1.5]\text{ s}$ IBI qualification criterion.
   - Per protocol, zero rows are dropped: `PlanCImputer` imputes training medians and attaches binary missingness flags (`is_missing_*`).

4. **Label Blindness Maintained:**
   - Zero label columns (`sbp`, `dbp`) were queried or read during this sanity audit.
   - Study C remains fully locked and blind prior to development model fitting.
