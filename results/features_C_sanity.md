# Study C Feature Sanity & Distribution Report

**Date:** 2026-10-29  
**Specification:** `PLAN_C.md` (Cross-subject SBP classification on unseen subjects)  
**Input Datasets:**  
- Development Cohort: `data/features_c/train_file.parquet` ($N = 465,480$ segments, $1,293$ subjects, $360\text{ seg/subj}$)
- Validation Cohort: `data/features_c/calfree_test.parquet` ($N = 57,600$ segments, $144$ subjects, $400\text{ seg/subj}$)
**Status:** Invariant and sanity audit complete. All numbers reported as-is without feature code alterations.

---

## 1. Feature Distribution & Quantile Profiles (Train vs. CalFree Side-by-Side)

Evaluated across the 9 canonical extracted features:
- **`hr`**: Discrete heart rate ($n_{\text{beats}} \times 6$, bpm)
- **`hr_ibi`**: Continuous heart rate ($60.0 / \text{median}(\text{IBI})$, bpm)
- **`ibi_sd`**: Standard deviation of inter-beat intervals (s)
- **`beat_count`**: Total valid systolic peaks in segment (beats)
- **`systolic_rise_time`**: Median foot-to-peak elapsed time (s)
- **`pulse_width`**: Full width at 50% amplitude (FWHM, s)
- **`decay_ratio`**: Diastolic decay time / systolic rise time ($d\_time / r\_time$, dimensionless time ratio)
- **`is_clipped`**: Rail clipping flag ($>1\%$ samples within $10^{-4}$ of voltage rails, binary 0.0 or 1.0)
- **`snr_ratio`**: Fractional spectral power in $[0.5, 8.0]\text{ Hz}$ relative to non-DC power (dimensionless $\in [0, 1]$)

### Quantiles & Anomaly Counts Table

| Feature Name | Cohort | Total Segments | NaN | Inf | Unique Values | Q01 | Q05 | Q25 | Q50 (Median) | Q75 | Q95 | Q99 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`hr`** | **Train** | 465,480 | 0 | 0 | 20 | 48.0000 | 54.0000 | 66.0000 | 72.0000 | 84.0000 | 102.0000 | 114.0000 |
| | **CalFree** | 57,600 | 0 | 0 | 17 | 48.0000 | 54.0000 | 66.0000 | 72.0000 | 84.0000 | 102.0000 | 120.0000 |
| **`hr_ibi`** | **Train** | 465,480 | 0 | 0 | 275 | 48.3871 | 53.7634 | 63.0252 | 71.7703 | 81.5217 | 100.0000 | 115.3846 |
| | **CalFree** | 57,600 | 0 | 0 | 262 | 47.4684 | 52.4476 | 63.8298 | 72.1154 | 82.8729 | 98.6842 | 119.0476 |
| **`ibi_sd`** | **Train** | 465,480 | 0 | 0 | 64,527 | 0.0035 | 0.0042 | 0.0074 | 0.0134 | 0.0454 | 0.1523 | 0.2545 |
| | **CalFree** | 57,600 | 0 | 0 | 17,197 | 0.0035 | 0.0042 | 0.0072 | 0.0129 | 0.0423 | 0.1517 | 0.2623 |
| **`beat_count`** | **Train** | 465,480 | 0 | 0 | 20 | 8.0000 | 9.0000 | 11.0000 | 12.0000 | 14.0000 | 17.0000 | 19.0000 |
| | **CalFree** | 57,600 | 0 | 0 | 17 | 8.0000 | 9.0000 | 11.0000 | 12.0000 | 14.0000 | 17.0000 | 20.0000 |
| **`systolic_rise_time`** | **Train** | 465,480 | 0 | 0 | 44 | 0.1520 | 0.1680 | 0.1840 | 0.2040 | 0.2560 | 0.3040 | 0.3040 |
| | **CalFree** | 57,600 | 0 | 0 | 45 | 0.1520 | 0.1640 | 0.1840 | 0.2040 | 0.2560 | 0.3040 | 0.3040 |
| **`pulse_width`** | **Train** | 465,480 | 0 | 0 | 454,010 | 0.1404 | 0.1552 | 0.1927 | 0.2326 | 0.2687 | 0.3200 | 0.3838 |
| | **CalFree** | 57,600 | 0 | 0 | 57,429 | 0.1408 | 0.1566 | 0.1891 | 0.2307 | 0.2696 | 0.3302 | 0.3869 |
| **`decay_ratio`** | **Train** | 465,480 | 0 | 0 | 43,202 | 0.9429 | 1.5294 | 2.2096 | 2.6818 | 3.1739 | 3.9615 | 4.6800 |
| | **CalFree** | 57,600 | 0 | 0 | 10,430 | 0.8947 | 1.4545 | 2.1923 | 2.6500 | 3.1739 | 4.1364 | 4.8676 |
| **`is_clipped`** | **Train** | 465,480 | 0 | 0 | 1 (const 0) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| | **CalFree** | 57,600 | 0 | 0 | 1 (const 0) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **`snr_ratio`** | **Train** | 465,480 | 0 | 0 | 164,170 | 0.9847 | 0.9894 | 0.9941 | 0.9972 | 0.9993 | 0.9999 | 1.0000 |
| | **CalFree** | 57,600 | 0 | 0 | 45,680 | 0.9843 | 0.9892 | 0.9940 | 0.9972 | 0.9993 | 0.9999 | 1.0000 |

---

## 2. Spearman Correlation Matrix of Features (Train Cohort, N = 465,480)

Pairwise Spearman rank correlation matrix computed on the 465,480 training segments:

| Feature | `hr` | `hr_ibi` | `ibi_sd` | `beat_count` | `systolic_rise_time` | `pulse_width` | `decay_ratio` | `is_clipped` | `snr_ratio` |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`hr`** | 1.0000 | 0.9707 | -0.2156 | 1.0000 | -0.5376 | -0.5437 | -0.2248 | NaN* | 0.0967 |
| **`hr_ibi`** | 0.9707 | 1.0000 | -0.3212 | 0.9707 | -0.5397 | -0.5565 | -0.2292 | NaN* | 0.0989 |
| **`ibi_sd`** | -0.2156 | -0.3212 | 1.0000 | -0.2156 | 0.1472 | 0.2497 | 0.0951 | NaN* | 0.0008 |
| **`beat_count`** | 1.0000 | 0.9707 | -0.2156 | 1.0000 | -0.5376 | -0.5437 | -0.2248 | NaN* | 0.0967 |
| **`systolic_rise_time`** | -0.5376 | -0.5397 | 0.1472 | -0.5376 | 1.0000 | 0.4658 | -0.5176 | NaN* | -0.0580 |
| **`pulse_width`** | -0.5437 | -0.5565 | 0.2497 | -0.5437 | 0.4658 | 1.0000 | 0.1173 | NaN* | -0.1118 |
| **`decay_ratio`** | -0.2248 | -0.2292 | 0.0951 | -0.2248 | -0.5176 | 0.1173 | 1.0000 | NaN* | -0.0539 |
| **`is_clipped`** | NaN* | NaN* | NaN* | NaN* | NaN* | NaN* | NaN* | NaN* | NaN* |
| **`snr_ratio`** | 0.0967 | 0.0989 | 0.0008 | 0.0967 | -0.0580 | -0.1118 | -0.0539 | NaN* | 1.0000 |

*\*Note: `is_clipped` has zero variance (identical to 0.0 across all segments), producing undefined (NaN) correlations.*

---

## 3. Boundary Checks (Train & CalFree)

| Boundary Check Condition | Train Count ($N=465,480$) | Train % | Train Observed Range | CalFree Count ($N=57,600$) | CalFree % | CalFree Observed Range |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`hr_ibi` outside [40, 160] bpm** | 3 | **0.0006%** | [40.32, 178.57] bpm | 0 | **0.0000%** | [40.76, 159.57] bpm |
| **`systolic_rise_time` at ceiling ($\ge 0.29\text{ s}$) or zero ($\le 0\text{ s}$)** | 87,267 | **18.7477%** | [0.1200, 0.3040] s | 11,148 | **19.3542%** | [0.1280, 0.3040] s |
| ↳ *`systolic_rise_time` $\ge 0.29\text{ s}$ (ceiling hit)* | 87,267 | 18.7477% | — | 11,148 | 19.3542% | — |
| ↳ *`systolic_rise_time` $\le 0.00\text{ s}$ (non-positive)* | 0 | 0.0000% | — | 0 | 0.0000% | — |
| **`pulse_width` outside [0.05, 0.60] s** | 0 | **0.0000%** | [0.1110, 0.5543] s | 0 | **0.0000%** | [0.1169, 0.5514] s |
| **`decay_ratio` $\le 0$ or $> 20$** | 0 | **0.0000%** | [0.3548, 12.0000] | 0 | **0.0000%** | [0.3667, 10.3333] |
| ↳ *`decay_ratio` $\le 0$* | 0 | 0.0000% | — | 0 | 0.0000% | — |
| ↳ *`decay_ratio` $> 20$* | 0 | 0.0000% | — | 0 | 0.0000% | — |
| **`is_clipped` == 1.0 (amplifier saturation)** | 0 | **0.0000%** | [0.0, 0.0] | 0 | **0.0000%** | [0.0, 0.0] |
| **`snr_ratio` outside [0.0, 1.0]** | 0 | **0.0000%** | [0.9453, 1.0000] | 0 | **0.0000%** | [0.9546, 1.0000] |
| **`beat_count` < 3 (severe bradycardia / loss)** | 0 | **0.0000%** | [7, 26] beats | 0 | **0.0000%** | [7, 23] beats |

---

## 4. Relationship Between `hr` ($n_{\text{beats}} \times 6$) and `hr_ibi` ($60.0 / \text{median}(\text{IBI})$)

| Metric | Train Cohort ($N = 465,480$) | CalFree Cohort ($N = 57,600$) |
| :--- | :--- | :--- |
| **Pearson Correlation ($r$)** | **0.97480** ($p < 10^{-300}$) | **0.96953** ($p < 10^{-300}$) |
| **Mean Absolute Difference (MAD)** | **3.0085 bpm** | **3.1013 bpm** |
| **Root Mean Squared Difference (RMSD)** | **3.9237 bpm** | **4.3115 bpm** |
| **Maximum Absolute Difference** | 46.50 bpm | 40.95 bpm |

*Empirical Confirmation of Amendment C1:* In a 10-second segment, $n_{\text{beats}} \times 6$ is discretized in coarse 6-bpm increments. The expected quantization error of a uniform 6-bpm step is $\mathbb{E}[|\epsilon|] = 6 / 4 = 1.5\text{ bpm}$ under steady rate, but with physiological heart rate variability and boundary beat truncation, the observed MAD is almost exactly $3.01\text{ bpm}$ ($\approx \text{step} / 2$). `hr_ibi` recovers continuous physiological rate without quantization jumps.

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
collected 57 items

tests\test_baselines.py ...                                              [  5%]
tests\test_controls_w.py .                                               [  7%]
tests\test_exploratory_dev_models.py ..                                  [ 10%]
tests\test_features_c.py .....                                           [ 19%]
tests\test_features_w.py ...                                             [ 24%]
tests\test_fiducials.py ....x......                                      [ 43%]
tests\test_gate.py ..                                                    [ 47%]
tests\test_gate_agreement.py ..                                          [ 50%]
tests\test_parity_c.py .                                                 [ 52%]
tests\test_prospective_and_scale_w.py ....                               [ 59%]
tests\test_test_c.py ...............                                     [ 85%]
tests\test_test_w.py ....                                                [ 92%]
tests\test_train_c.py ...                                                [ 98%]
tests\test_train_w.py .                                                  [100%]

================= 56 passed, 1 xfailed, 26 warnings in 17.20s =================
```

- **Collected Tests:** Exactly **57 items**.
- **Inclusion of `tests/test_parity_c.py`:** **Confirmed present and PASSED** at step 30 (`tests\test_parity_c.py . [ 52%]`).
- **Xfailed Item:** 1 test (`tests/test_fiducials.py:155::test_synthetic_lag_minus_30`, superseded by Amendment A3).

---

## 7. Audit Findings (Reported As-Is)

Per instructions, the following findings are cataloged directly from the data without changing any feature extraction code:

1. **Finding 1 — Search-Window Ceiling Truncation in `systolic_rise_time`:**
   - In `src/features_c.py` (lines 130–143), the foot search window preceding a systolic peak is defined as $w_{\text{foot}} = \text{round}(0.3 \times f_s) = 38\text{ samples}$ at $125\text{ Hz}$ ($38/125 = 0.3040\text{ s}$).
   - In **18.75% of Train segments (87,267 rows)** and **19.35% of CalFree segments (11,148 rows)**, the foot detection hits this window boundary, clipping `systolic_rise_time` at exactly $0.3040\text{ s}$ ($\ge 0.29\text{ s}$).
   - Both Q95 and Q99 are saturated at $0.3040\text{ s}$.

2. **Finding 2 — Degenerate Variance in `is_clipped`:**
   - Across all 523,080 segments ($465,480$ Train + $57,600$ CalFree), `is_clipped` is strictly $0.0$ ($0$ saturated segments under the $1\%$ threshold).
   - In all 1,293 Train subjects and 144 CalFree subjects, `is_clipped` has zero variance. Its correlation with other features is undefined (NaN).
   - Downstream models (logistic regression and GBT) will handle zero-variance features safely via zero coefficient weights or unused tree splits, but it provides no discriminative information in this cohort.

3. **Finding 3 — Pulse Width and Decay Ratio Physiological Stability:**
   - `pulse_width` is strictly bounded between $0.1110\text{ s}$ and $0.5543\text{ s}$ (0% outside $[0.05, 0.60]\text{ s}$).
   - `decay_ratio` is strictly positive and finite, ranging between $0.3548$ and $12.0000$ (0% non-positive or $> 20$).

4. **Finding 4 — High Spectral Signal Quality Across All Segments:**
   - `snr_ratio` exceeds $0.945$ for 100% of segments (median $0.9972$), confirming high physiological band energy across VitalDB recordings.
   - `beat_count` is at least 7 beats in every segment ($0\%$ segments with $< 3$ beats).
