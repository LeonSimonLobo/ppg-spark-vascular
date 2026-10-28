# Training-Split Beat-Agreement Gate (PLAN_W.md Section 7)

**Date:** 2026-10-29  
**Cohort:** `VitalDB_Train_Subset.mat` training split (`split == 'train'`)  
**Scope:** 1,034 subjects | 372,240 segments (10 s at 125 Hz; 360 segments per subject)  
**CalFree / Test Status:** Pristine (0 test segments loaded or evaluated)  
**Agreement Criterion:** $|n_{\text{PPG}} - n_{\text{ABP}}| \le 1$ beat per 10 s segment  
**Threshold:** $\ge 80.0\%$ passing segments  

---

## 1. Overall Beat Agreement

- **Passing Segments ($|n_{\text{PPG}} - n_{\text{ABP}}| \le 1$):** **96.92%** (360,757 / 372,240)
- **95% Subject-Level Bootstrap CI (2,000 resamples):** **[96.44%, 97.35%]**
- **Threshold Rule ($\ge 80.0\%$):** **PASS** (96.92% vs. 80.0%)

---

## 2. Agreement by SBP Class

### Primary Split ($SBP \ge 130$ mmHg):
| Subgroup | Total Segments | Passing Segments ($|d| \le 1$) | Pass Rate | 95% Bootstrap CI | Median Diff ($n_{\text{PPG}} - n_{\text{ABP}}$) | IQR of Diff ($Q_1, Q_3$) |
|---|---|---|---|---|---|---|
| **$SBP \ge 130$ mmHg** | 80,507 | 79,030 | **98.17%** | [97.71%, 98.54%] | +0.0 | 1.0 (+0.0, +1.0) |
| **$SBP < 130$ mmHg** | 291,733 | 281,727 | **96.57%** | [96.04%, 97.07%] | +0.0 | 2.0 (-1.0, +1.0) |

### Sensitivity Cut-Points:
| Cut-Point | Class | Total Segments | Passing Segments | Pass Rate | Median Diff | IQR of Diff |
|---|---|---|---|---|---|---|
| **120 mmHg** | $SBP \ge 120$ | 142,293 | 139,495 | 98.03% | +0.0 | 1.0 (+0.0, +1.0) |
| | $SBP < 120$ | 229,947 | 221,262 | 96.22% | +0.0 | 2.0 (-1.0, +1.0) |
| **140 mmHg** | $SBP \ge 140$ | 39,406 | 38,759 | 98.36% | +0.0 | 1.0 (+0.0, +1.0) |
| | $SBP < 140$ | 332,834 | 321,998 | 96.74% | +0.0 | 2.0 (-1.0, +1.0) |

---

## 3. Heart Rate Baseline AUROC on the Same Subjects

- **Feature:** Within-subject centered heart rate ($\Delta HR_{i, j} = HR_{i, j} - \text{median}(HR_i)$)
- **Target:** Acute hypertensive state ($SBP \ge 130$ mmHg)
- **Point Estimate AUROC:** **0.5569**
- **95% Subject-Level Bootstrap CI (2,000 resamples):** **[0.5449, 0.5686]**

---

## 4. Decision & Gate Summary

- **Beat Agreement ($\ge 80\%$):** **PASS** (96.92%)
- **Conclusion:** **PASS**: The training-split beat agreement (|PPG beats − ABP beats| ≤ 1) exceeds the 80% pre-registered threshold.
