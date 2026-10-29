# Nested Model Comparison & HR Residualization Report (dev_W_nested.md)

**Date:** 2026-10-29  
**Cohort:** `VitalDB_Train_Subset.mat` development split only (`split == 'train'`)  
**Scope:** 1,034 subjects | 372,240 segments (360 segments per subject)  
**CalFree / Test Status:** Pristine (0 test segments loaded or scored)  
**Within-Subject Target:** $y = 1$ if $SBP \ge \text{median}(SBP_i)$ (ties to 0)  
**Target Prevalence:** **50.00%** (186,120 / 372,240)  

---

## 1. Cross-Validation Transduction Disclosure

> [!NOTE]
> **CV Transduction Property:**
> Per-subject centering ($\Delta X_{i, j} = X_{i, j} - \text{median}(X_i)$) utilizes the median across all 360 operative segments for each subject.
> In cross-validation (GroupKFold by subject), each validation fold consists of unseen subjects, but features within each unseen subject are centered relative to that subject's recording session median.
> This centering process is **transductive** rather than purely inductive: it assumes clinical access to a patient-specific calibration window or baseline monitoring session.

---

## 2. Within-Subject Correlation: Morphology Features vs. $\Delta HR$

Spearman rank correlation of centered PPG morphology features with centered heart rate ($\Delta HR_{i, j}$) across 1,034 subjects:

| Morphology Feature | Pooled Spearman $r_s$ | $p$-value | Subj Median $r_s$ | Subj Mean $r_s$ | Subj IQR $r_s$ ($Q_1, Q_3$) | [Min, Max] $r_s$ | Subjects with $r_s > 0$ |
|---|---|---|---|---|---|---|---|
| `delta_systolic_rise_time` | -0.3194 | 0.00e+00 | **-0.3551** | -0.3392 | 0.3818 (-0.5386, -0.1567) | [-0.9093, +0.6122] | **10.9%** |
| `delta_pulse_width` | -0.3304 | 0.00e+00 | **-0.3448** | -0.3242 | 0.4043 (-0.5314, -0.1271) | [-0.9573, +0.5507] | **14.4%** |
| `delta_decay_ratio` | -0.2361 | 0.00e+00 | **-0.2729** | -0.2649 | 0.4306 (-0.4916, -0.0610) | [-0.9349, +0.6105] | **19.4%** |

All three morphology features are moderately negatively correlated with $\Delta HR$ (pooled $r_s \in [-0.33, -0.24]$, subject median $r_s \in [-0.36, -0.27]$), reflecting mechanical cycle shortening at higher heart rates.

---

## 3. Per-Fold Cross-Validation Performance (GroupKFold = 5)

| Fold | Subjects | Segments | Prev | HR Alone AUROC | HR Alone Acc | HR + Morph AUROC | HR + Morph Acc | Resid Morph AUROC | Resid Morph Acc |
|---|---|---|---|---|---|---|---|---|---|
| Fold 1 | 207 | 74,520 | 50.00% | **0.5441** | 53.08% (50.0%) | **0.7214** | 66.44% (50.0%) | **0.7161** | 66.05% (50.0%) |
| Fold 2 | 207 | 74,520 | 50.00% | **0.5713** | 54.80% (50.0%) | **0.7075** | 65.18% (50.0%) | **0.6956** | 64.39% (50.0%) |
| Fold 3 | 207 | 74,520 | 50.00% | **0.5752** | 55.14% (50.0%) | **0.6879** | 63.55% (50.0%) | **0.6735** | 62.48% (50.0%) |
| Fold 4 | 207 | 74,520 | 50.00% | **0.5490** | 53.34% (50.0%) | **0.7045** | 65.24% (50.0%) | **0.6984** | 64.66% (50.0%) |
| Fold 5 | 206 | 74,160 | 50.00% | **0.5511** | 53.87% (50.0%) | **0.6979** | 64.18% (50.0%) | **0.6915** | 63.89% (50.0%) |

---

## 4. Pooled Out-of-Fold (OOF) Headline Metrics

| Model Specification | Predictor Features | Pooled OOF AUROC | 95% Subject Bootstrap CI | Pooled OOF Accuracy | Majority Baseline |
|---|---|---|---|---|---|
| **Majority Class Baseline** | None | 0.5000 | [0.5000, 0.5000] | 50.00% | 50.00% |
| **Model 1: HR Alone** | `delta_hr` (standardized LR) | **0.5574** | [0.5481, 0.5663] | 54.05% | 50.00% |
| **Model 2: HR + Morphology** | `delta_hr`, `delta_rise`, `delta_pw`, `delta_decay` (std LR) | **0.7038** | [0.6946, 0.7134] | 64.92% | 50.00% |
| **Model 3: HR-Residualized Morphology** | OLS-residualized morphology ($\tilde{M}_k \perp \Delta HR$, std LR) | **0.6950** | [0.6852, 0.7050] | 64.30% | 50.00% |

---

## 5. Paired Incremental Statistics vs. HR Alone

### A. Model 2 vs. Model 1: HR + Morphology vs. HR Alone
- **Statistic:** $\Delta \text{AUROC}_{\text{joint}} = \text{AUROC}(\text{HR + Morphology}) - \text{AUROC}(\text{HR Alone})$
- **Pooled Point Estimate:** **+0.1465**
- **95% Subject-Level Bootstrap CI (2,000 draws):** **[+0.1350, +0.1590]**
- **Hypothesis Test Outcome:** **PASS: Strictly excludes zero ($\Delta \text{AUROC}_{\text{lower}} = +0.1350 > 0$)**. Combining morphology with heart rate provides an incremental gain of +0.1465 AUROC.

### B. Model 3 vs. Model 1: HR-Residualized Morphology vs. HR Alone
- **Statistic:** $\Delta \text{AUROC}_{\text{resid}} = \text{AUROC}(\text{HR-Residualized Morphology}) - \text{AUROC}(\text{HR Alone})$
- **Pooled Point Estimate:** **+0.1376**
- **95% Subject-Level Bootstrap CI (2,000 draws):** **[+0.1243, +0.1521]**
- **Hypothesis Test Outcome:** **PASS: Strictly excludes zero ($\Delta \text{AUROC}_{\text{lower}} = +0.1243 > 0$)**. Even after stripping all linear heart rate dependence out of morphology on the training folds, the residual morphology features alone outperform HR alone by +0.1376 AUROC.

---

## 6. Summary of Findings

1. **Orthogonal Information:** PPG morphology is moderately correlated with heart rate due to cardiac cycle duration ($r_s \approx -0.3$), but this dependency accounts for only a minor fraction of morphology's predictive power.
2. **Residual Power:** Stripping out all linear correlation with $\Delta HR$ via training-fold OLS yields residualized morphology features that achieve AUROC **0.6950** on their own, dramatically exceeding HR alone (**0.5574**).
3. **Synergy:** Incorporating both HR and morphology into a joint model reaches AUROC **0.7038**, demonstrating that PPG morphology provides substantial orthogonal information beyond heart rate for tracking within-subject blood pressure excursions.
