# Development Cross-Validation Report: Within-Subject Model W

**Date:** 2026-10-29  
**Cohort:** `VitalDB_Train_Subset.mat` development split only (`split == 'train'`)  
**Scope:** 1,034 subjects | 372,240 segments (360 segments per subject)  
**CalFree / Test Status:** Pristine (0 test segments loaded or scored)  
**Within-Subject Target:** $y = 1$ if $SBP \ge \text{median}(SBP_i)$ (ties to 0)  
**Target Prevalence:** **50.00%** (186,120 / 372,240)  

---

## 1. Per-Fold Cross-Validation Performance (GroupKFold = 5)

| Fold | Subjects | Segments | Prevalence | Morphology LR AUROC | Morphology LR Acc | HR Comparator AUROC | HR Comparator Acc | RF Sensitivity AUROC | RF Sensitivity Acc |
|---|---|---|---|---|---|---|---|---|---|
| Fold 1 | 207 | 74,520 | 50.00% | **0.7017** | 65.13% (50.0%) | **0.5441** | 53.08% (50.0%) | 0.6652 | 61.94% (50.0%) |
| Fold 2 | 207 | 74,520 | 50.00% | **0.6794** | 63.46% (50.0%) | **0.5713** | 54.80% (50.0%) | 0.6512 | 60.88% (50.0%) |
| Fold 3 | 207 | 74,520 | 50.00% | **0.6523** | 60.90% (50.0%) | **0.5752** | 55.14% (50.0%) | 0.6388 | 59.84% (50.0%) |
| Fold 4 | 207 | 74,520 | 50.00% | **0.6793** | 63.51% (50.0%) | **0.5490** | 53.34% (50.0%) | 0.6476 | 60.50% (50.0%) |
| Fold 5 | 206 | 74,160 | 50.00% | **0.6720** | 62.68% (50.0%) | **0.5511** | 53.87% (50.0%) | 0.6436 | 60.27% (50.0%) |

---

## 2. Pooled Out-of-Fold (OOF) Headline Metrics

| Model / Feature Set | Predictor Features | Pooled OOF AUROC | 95% Subject Bootstrap CI | Pooled OOF Accuracy | Majority Baseline |
|---|---|---|---|---|---|
| **Step W0: Majority Class** | None | 0.5000 | [0.5000, 0.5000] | 50.00% | 50.00% |
| **Step W1: HR Comparator** | `delta_hr` (standardized LR) | **0.5574** | [0.5481, 0.5663] | 54.05% | 50.00% |
| **Step W2: Classic Morphology** | `delta_systolic_rise_time`, `delta_pulse_width`, `delta_decay_ratio` (std LR) | **0.6767** | [0.6671, 0.6867] | 63.13% | 50.00% |
| **Sensitivity Check: Random Forest** | Classic morphology features (default RF) | **0.6493** | [0.6426, 0.6559] | 60.69% | 50.00% |

---

## 3. Paired Incremental Statistic: $\Delta \text{AUROC}$ (Morphology vs. HR)

- **Definition:** $\Delta \text{AUROC} = \text{AUROC}(\text{Morphology}) - \text{AUROC}(\text{HR})$ evaluated on the exact same out-of-fold bootstrap subject resamples (2,000 draws).
- **Pooled Point Estimate ($\Delta \text{AUROC}$):** **+0.1194**
- **95% Subject-Level Bootstrap CI:** **[+0.1053, +0.1345]**
- **Sensitivity RF Point Estimate ($\Delta \text{AUROC}_{\text{RF}}$):** **+0.0919** (95% CI: [+0.0808, +0.1037])

### Stop / Decision Rule Evaluation (PLAN_W.md Section 6):
- **Criterion:** The 95% CI of $\Delta \text{AUROC}$ must strictly exclude zero with morphology exceeding HR ($\Delta \text{AUROC}_{\text{lower}} > 0$).
- **Measured Outcome:** Lower bound of 95% CI is **+0.1053**.
- **Formal Status:** **PASS: Morphology AUROC strictly exceeds HR AUROC with 95% paired bootstrap CI excluding zero.**

---

## 4. Frozen Morphology Model Artifact (`models/w_frozen.pkl`)

- **Model Specification:** Standardized Logistic Regression fit on all 372,240 training-split segments.
- **Model Path:** `models/w_frozen.pkl`
- **SHA-256 Digest:** `92c9176b456e00630c45071afcdb1f5e0423d393ca2f6ad2ecf9357f5e640fdc`
- **Learned Standardizer (Means / Scales):**
  - `delta_systolic_rise_time`: mean = +0.004236, scale = 0.036408
  - `delta_pulse_width`: mean = +0.002124, scale = 0.030572
  - `delta_decay_ratio`: mean = -0.006777, scale = 0.542768
- **Learned Logistic Regression Coefficients:**
  - $\beta_{\text{rise}}$ (`delta_systolic_rise_time`): **-0.327440**
  - $\beta_{\text{width}}$ (`delta_pulse_width`): **+0.717397**
  - $\beta_{\text{decay}}$ (`delta_decay_ratio`): **-0.349621**
  - Intercept: **+0.008174**
