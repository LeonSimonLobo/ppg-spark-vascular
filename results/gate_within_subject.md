# Within-Subject Gate Evaluation (PROMPT 2)

**Date:** 2026-10-29  
**Cohort:** `VitalDB_Train_Subset.mat` training split only (`split == 'train'`)  
**Scope:** 1,034 subjects | 372,240 segments (10 s at 125 Hz; exactly 360 segments per subject)  
**CalFree / Test Status:** Pristine (0 test segments loaded or scored)  

---

## 1. Criterion 1: Physiological Beat Count Plausibility

- **Rule:** $\ge 95\%$ of segments must yield heart rate in [30, 150] bpm (5 to 25 detected beats per 10 s).
- **Segments within bounds:** **99.99%** (372,210 / 372,240)
- **Heart Rate Distribution (bpm):**
  - Median: 72.0 bpm
  - IQR: 18.0 bpm (Q1: 66.0, Q3: 84.0)
  - Min / Max: 42 bpm / 156 bpm
- **Criterion 1 Status:** **PASS** (99.99% vs. 95.0% threshold)

---

## 2. Criterion 2: Within-Subject Heart Rate Signal

- **Centered Feature:** $\Delta HR = HR - \text{median}(HR_i)$ (bpm)
- **Centered Target:** $\Delta SBP = SBP - \text{median}(SBP_i)$ (mmHg)

### Association Metrics:
| Metric | Value | Threshold / Expected | Pass |
|---|---|---|---|
| **Pooled Spearman Correlation ($r_s$)** | **+0.1376** ($p = 0.00e+00$) | $r_s > 0$ | Yes |
| **Per-Subject Median Correlation** | **+0.1792** (Mean: +0.1457) | Positive | Yes |
| **Subjects with Positive Correlation** | **68.86%** (712/1034) | $> 50\%$ | Yes |
| **Logistic Regression Slope ($\beta_{HR}$)** | **+0.2144** (std coef) | $\beta_{HR} > 0$ | Yes |
| **Classification AUROC ($SBP \ge 130$)** | **0.5569** | AUROC $> 0.50$ | Yes |
| **Classification AUROC (Within-Subject $\Delta SBP \ge 0$)** | **0.5583** ($\beta = +0.2044$) | AUROC $> 0.50$ | Yes |

---

## 3. Gate Decision Summary

- **Criterion 1 (Plausibility $\ge 95\%$):** **PASS** (99.99%)
- **Criterion 2 (Within-Subject Signal):** **PASS** (AUROC = 0.5569, $r_s = +0.1376$, $\beta = +0.2144$)
- **Overall Gate Outcome:** **PASS**: The within-subject heart-rate comparator and PPG beat detection meet all pre-registered criteria on the training split. Progression to within-subject morphology modeling is authorized.
