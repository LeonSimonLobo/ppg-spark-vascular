# Beat-Level Verification of Study C Morphology Definition

**Date:** 2026-10-29  
**Specification:** `PLAN_C.md` Amendment C2  
**Dataset:** `data/parquet/train_file.parquet` (Development Cohort)  
**Sample Size:** 20,000 segments sampled stratified by subject (seed 42, all 1293 subjects represented)  
**Analyzed Units:** 211,141 valid qualifying beats across 19,997 segments (3 segments had no qualifying beats)  
**Protocol Constraints:** No labels accessed (`sbp`, `dbp`, `age`, `sex`, `bmi` excluded). Pure measurement and reporting.  

---

## 1. Primary Metrics (Over Beats and Over Segments)

### Summary Table

| Evaluation Metric | Over Beats ($N = 211,141$) | Over Segments ($N = 19,997$) | Definition / Interpretation |
| :--- | :--- | :--- | :--- |
| **(a) Peak equals highest PPG sample** ($|p_k - \operatorname{argmax}| \le 2$) | **99.9380%** (211,010 / 211,141) | **99.9400%** (mean share per seg)<br>*99.55%* segs with 100% pass | Fraction of beats where detected peak $p_k$ is the global maximum in $[\mathrm{foot}_k, \mathrm{foot}_{k+1}]$ |
| **(b) Rise time / Beat period** (Median [Q05, Q95]) | **0.2772** [0.1963, 0.5652] | **0.2747** [0.1991, 0.4836] | Upstroke duration relative to total beat interval $(p_k - p_{k-1})$ |
| **(c) Rise time > 0.40 s share** | **8.6918%** (18,352 / 211,141) | **8.0962%** (median RT > 0.4s)<br>*25.95%* (any beat RT > 0.4s) | Prevalence of long systolic upstrokes (> 400 ms) |
| ↳ *Share where (a) fails among RT > 0.40 s* | **0.0763%** | **0.6794%** (segs with med RT > 0.4s)<br>*0.27%* (segs with any beat RT > 0.4s) | Failure of peak supremacy among long upstrokes |

---

## 2. Distribution of Rise Time / Beat Period (Metric b)

| Quantile | Over Beats ($N = 211,141$) | Over Segments (Segment Median, $N = 19,997$) |
| :--- | :--- | :--- |
| **Q05** | 0.1963 | 0.1991 |
| **Q25** | 0.2376 | 0.2377 |
| **Q50 (Median)** | **0.2772** | **0.2747** |
| **Q75** | 0.3496 | 0.3362 |
| **Q95** | 0.5652 | 0.4836 |
| **Q99** | 0.7089 | 0.6842 |

---

## 3. Median Rise Time by Continuous Heart Rate (hr_ibi) Bins (Metric d)

| `hr_ibi` Bin | Beat Count | Median Rise Time (Over Beats) | Segment Count | Median Rise Time (Over Segments) |
| :--- | :--- | :--- | :--- | :--- |
| **`< 60 bpm`** | 24,611 | **0.3280 s** | 3,209 | **0.3280 s** |
| **`60 - 80 bpm`** | 112,644 | **0.2160 s** | 11,252 | **0.2160 s** |
| **`80 - 100 bpm`** | 57,364 | **0.1920 s** | 4,495 | **0.1880 s** |
| **`> 100 bpm`** | 16,522 | **0.1760 s** | 1,041 | **0.1760 s** |

---

## 4. Inspection of Figure `results/morphology_check_C.png`

The figure `results/morphology_check_C.png` visualizes 18 random segments (6 rows x 3 columns):
- **Column 1 (Median RT < 0.25 s):** Crisp arterial pulses with typical steep systolic upstroke and dicrotic notch.
- **Column 2 (Median RT 0.25 - 0.40 s):** Broader systolic pulses, often associated with lower heart rates or smoother upstrokes.
- **Column 3 (Median RT > 0.40 s):** Long rise times where the inter-peak argmin foot search selects the trough preceding the systolic wave.

---

## 5. Result of 0.4 s Synthetic Rise Time Unit Test

The unit test `test_synthetic_rise_time_0_4s_recovery` in `tests/test_features_c.py` generates a synthetic asymmetric waveform with true rise time of 0.40 s (50 samples at 125 Hz) and decay time of 0.60 s (75 samples).
- **Assertion 1 (Legacy Ceiling Exceeded):** $\text{systolic\_rise\_time} > 0.304\text{ s}$ (**PASS**).
- **Assertion 2 (True Recovery):** $\text{systolic\_rise\_time} = 0.4000\text{ s}$ ($|\text{recovered} - 0.40| < 0.01$, **PASS**).
- **Assertion 3 (Pulse Width):** $\text{pulse\_width} = 0.5000\text{ s}$ ($|\text{recovered} - 0.50| < 0.01$, **PASS**).
- **Assertion 4 (Decay Ratio):** $\text{decay\_ratio} = 1.5000$ ($|\text{recovered} - 1.50| < 0.05$, **PASS**).
