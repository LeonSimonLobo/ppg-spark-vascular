# Development Cross-Validation Report: Within-Subject Model W

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
> Per-subject centering ($\Delta X_{i, j} = X_{i, j} - \text{median}(X_i)$) computes the median feature value $\text{median}(X_i)$ across the complete 360-segment recording session for each subject.
> In cross-validation (GroupKFold by subject), each validation fold consists of entirely unseen subjects, but within each unseen validation subject, their individual features are centered around that subject's recording session median.
> This centering process is **transductive** rather than inductive: it assumes clinical access to a subject-specific calibration window or recording baseline, and cannot be evaluated on an isolated single segment from an unseen patient without observation history.

---

## 2. Per-Fold Cross-Validation Performance (GroupKFold = 5)

| Fold | Subjects | Segments | Prev | Morphology LR AUROC | Morphology LR Acc | HR Comp AUROC | HR Comp Acc | Time-Only AUROC | Time-Only Acc | RF Sensitivity AUROC | RF Sensitivity Acc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Fold 1 | 207 | 74,520 | 50.00% | **0.7017** | 65.13% (50.0%) | **0.5441** | 53.08% (50.0%) | 0.4988 | 49.81% (50.0%) | 0.6652 | 61.94% (50.0%) |
| Fold 2 | 207 | 74,520 | 50.00% | **0.6794** | 63.46% (50.0%) | **0.5713** | 54.80% (50.0%) | 0.4992 | 49.83% (50.0%) | 0.6512 | 60.88% (50.0%) |
| Fold 3 | 207 | 74,520 | 50.00% | **0.6523** | 60.90% (50.0%) | **0.5752** | 55.14% (50.0%) | 0.5041 | 50.38% (50.0%) | 0.6388 | 59.84% (50.0%) |
| Fold 4 | 207 | 74,520 | 50.00% | **0.6793** | 63.51% (50.0%) | **0.5490** | 53.34% (50.0%) | 0.5019 | 50.27% (50.0%) | 0.6476 | 60.50% (50.0%) |
| Fold 5 | 206 | 74,160 | 50.00% | **0.6720** | 62.68% (50.0%) | **0.5511** | 53.87% (50.0%) | 0.5017 | 50.02% (50.0%) | 0.6436 | 60.27% (50.0%) |

---

## 3. Pooled Out-of-Fold (OOF) Headline Metrics

| Model / Feature Set | Predictor Features | Pooled OOF AUROC | 95% Subject Bootstrap CI | Pooled OOF Accuracy | Majority Baseline |
|---|---|---|---|---|---|
| **Step W0: Majority Class** | None | 0.5000 | [0.5000, 0.5000] | 50.00% | 50.00% |
| **Step W1: HR Comparator** | `delta_hr` (standardized LR) | **0.5574** | [0.5481, 0.5663] | 54.05% | 50.00% |
| **Time-Only Comparator** | `time_pos` (0..359 segment rank, standardized LR) | **0.5003** | [0.4985, 0.5022] | 50.06% | 50.00% |
| **Step W2: Classic Morphology** | `delta_systolic_rise_time`, `delta_pulse_width`, `delta_decay_ratio` (std LR) | **0.6767** | [0.6671, 0.6867] | 63.13% | 50.00% |
| **Sensitivity Check: Random Forest** | Classic morphology features (default RF) | **0.6493** | [0.6426, 0.6559] | 60.69% | 50.00% |

---

## 4. Paired Incremental Statistics & Drift Evaluation

### A. Morphology vs. Heart Rate Comparator:
- **Statistic:** $\Delta \text{AUROC}_{\text{HR}} = \text{AUROC}(\text{Morphology}) - \text{AUROC}(\text{HR})$
- **Pooled Point Estimate:** **+0.1194**
- **95% Subject Bootstrap CI:** **[+0.1053, +0.1345]**
- **Decision Rule Outcome:** **PASS** (95% CI strictly excludes zero with morphology exceeding HR).

### B. Morphology vs. Time-Only Comparator (Recording Drift Check):
- **Statistic:** $\Delta \text{AUROC}_{\text{Time}} = \text{AUROC}(\text{Morphology}) - \text{AUROC}(\text{Time})$
- **Pooled Point Estimate:** **+0.1764**
- **95% Subject Bootstrap CI:** **[+0.1669, +0.1866]**
- **Time Alone AUROC:** **0.5003** (essentially chance level 0.5000).
- **Drift Confound Verdict:** **NOT CONFOUNDED**. Time-within-recording alone achieves no predictive separation (AUROC 0.5003), and morphology strictly outperforms time by +0.1764 AUROC with 95% paired CI [+0.1669, +0.1866].

---

## 5. Feature Correlation with Segment Position Within Subject

Evaluation of linear and rank drift across the 360 operative segments ($t \in [0, 359]$) within each subject:

| Feature | Pooled Spearman $r_s$ | $p$-value | Subj Median $r_s$ | Subj Mean $r_s$ | Subj IQR $r_s$ ($Q_1, Q_3$) | [Min, Max] $r_s$ | Subjects with $r_s > 0$ |
|---|---|---|---|---|---|---|---|
| `sbp` | -0.0028 | 8.36e-02 | **-0.0028** | -0.0025 | 0.0782 (-0.0421, +0.0361) | [-0.1454, +0.1504] | **48.1%** |
| `hr` | -0.0015 | 3.68e-01 | **-0.0023** | -0.0023 | 0.0777 (-0.0417, +0.0359) | [-0.1421, +0.2042] | **48.6%** |
| `delta_hr` | -0.0026 | 1.13e-01 | **-0.0023** | -0.0023 | 0.0777 (-0.0417, +0.0359) | [-0.1421, +0.2042] | **48.6%** |
| `systolic_rise_time` | -0.0001 | 9.48e-01 | **-0.0009** | -0.0006 | 0.0690 (-0.0342, +0.0347) | [-0.1591, +0.1812] | **49.2%** |
| `delta_systolic_rise_time` | -0.0004 | 7.87e-01 | **-0.0009** | -0.0006 | 0.0690 (-0.0342, +0.0347) | [-0.1591, +0.1812] | **49.2%** |
| `pulse_width` | +0.0005 | 7.83e-01 | **-0.0022** | +0.0006 | 0.0767 (-0.0371, +0.0396) | [-0.1753, +0.1535] | **48.4%** |
| `delta_pulse_width` | +0.0007 | 6.69e-01 | **-0.0022** | +0.0006 | 0.0767 (-0.0371, +0.0396) | [-0.1753, +0.1535] | **48.4%** |
| `decay_ratio` | +0.0006 | 7.29e-01 | **+0.0020** | +0.0012 | 0.0672 (-0.0314, +0.0358) | [-0.1914, +0.1599] | **52.3%** |
| `delta_decay_ratio` | +0.0001 | 9.44e-01 | **+0.0020** | +0.0012 | 0.0672 (-0.0314, +0.0358) | [-0.1914, +0.1599] | **52.3%** |

All features exhibit median within-subject correlations with recording time position between -0.0028 and +0.0020, with positive correlation fractions clustered tightly around 50% (48.1% to 52.3%). There is no systematic monotonic drift across operative recording time.

---

## 6. Frozen Morphology Model Artifact (`models/w_frozen.pkl`)

- **Model Specification:** Standardized Logistic Regression fit on all 372,240 training-split segments.
- **Model Path:** `models/w_frozen.pkl`
- **SHA-256 Digest:** `92c9176b456e00630c45071afcdb1f5e0423d393ca2f6ad2ecf9357f5e640fdc`
- **Artifact Tracking Policy:** The binary pickle file `models/w_frozen.pkl` is an execution artifact and is excluded from git tracking (`models/*.pkl` in `.gitignore`). Its cryptographic hash, configuration, standardizer parameters, and learned coefficients are tracked here and in [`models/README.md`](../models/README.md).
- **Learned Standardizer (Means / Scales):**
  - `delta_systolic_rise_time`: mean = +0.004236, scale = 0.036408
  - `delta_pulse_width`: mean = +0.002124, scale = 0.030572
  - `delta_decay_ratio`: mean = -0.006777, scale = 0.542768
- **Learned Logistic Regression Coefficients:**
  - $\beta_{\text{rise}}$ (`delta_systolic_rise_time`): **-0.327440**
  - $\beta_{\text{width}}$ (`delta_pulse_width`): **+0.717397**
  - $\beta_{\text{decay}}$ (`delta_decay_ratio`): **-0.349621**
  - Intercept: **+0.008174**
