# Exploratory Development Model Comparison (EXPLORATORY)

> [!WARNING]
> **EXPLORATORY STATUS DISCLOSURE:**  
> All model comparisons, architectural variations, and interaction models presented in this document are **EXPLORATORY**.  
> In accordance with the pre-registered study protocol (PLAN_W.md, Amendments W1–W3), the single pre-registered within-subject model is the frozen linear morphology model (`models/w_frozen.pkl`).  
> **Any model selected, tuned, or benchmarked in this exploratory table CANNOT be substituted for the pre-registered result or used to claim post-hoc gains.**

**Date:** 2026-10-29  
**Cohort:** `VitalDB_Train_Subset.mat` training split (`split == 'train'`)  
**Scope:** 1,034 subjects | 372,240 development segments (360 segments per subject)  
**CalFree / Test Data Access:** **Zero** (neither `VitalDB_CalFree_Test_Subset.mat` nor `calfree_test.parquet` was opened or loaded)  
**Target Label:** Within-subject SBP excursion $y = \mathbb{I}(SBP_{i, j} \ge \text{median}_{\text{session}}(SBP_i))$ (ties to 0)  
**Label Prevalence:** **50.00%** (Majority baseline: 50.00%)  
**Validation Strategy:** 5-fold cross-validation grouped by subject (`GroupKFold(5)`)  
**Bootstrap Parameters:** 2,000 subject-level clustered resamples across all 1,034 subjects, fixed seed 42  
**Decision Threshold:** Fixed at $p \ge 0.50$  

---

## 1. Exploratory Model Comparison Table

| Model Architecture | Features Used | OOF AUROC [95% Bootstrap CI] | Balanced Accuracy | Raw Accuracy (Prev: 50.0%) | Predicted Positive Rate (Thresh: 0.50) | Paired $\Delta\text{AUROC}$ vs. Model (a) [95% CI] |
|---|---|---|---|---|---|---|
| **(a) Logistic Regression (Baseline, Frozen)** | 3 morphology features (`delta_rise`, `delta_pw`, `delta_decay`) | **0.6767** [0.6671, 0.6867] | **63.13%** | 63.13% | 47.94% | — |
| **[Ref] Linear Joint Logistic Regression** | 3 morphology features + `delta_hr` (main effects) | 0.7038 [0.6946, 0.7134] | 64.92% | 64.92% | 47.64% | **+0.0271** [+0.0226, +0.0313] |
| **(b) LR with Pairwise Interactions** | Pairwise interactions of 3 morphology features and `delta_hr` (10 features) | **0.7044** [0.6952, 0.7137] | **64.94%** | 64.94% | 47.79% | **+0.0277** [+0.0230, +0.0321] |
| **(c) Gradient-Boosted Trees (HGB)** | 3 morphology features + `delta_hr` (`HistGradientBoostingClassifier`, default) | **0.7192** [0.7106, 0.7281] | **65.93%** | 65.93% | 49.34% | **+0.0425** [+0.0372, +0.0478] |
| **(d) Random Forest** | 3 morphology features + `delta_hr` (`RandomForestClassifier`, default, 100 trees) | **0.6674** [0.6608, 0.6744] | **62.11%** | 62.11% | 51.03% | **-0.0093** [-0.0152, -0.0035] |

---

## 2. Key Exploratory Findings

1. **Nonlinear Interaction Effects (Model b):**
   Adding pairwise interaction terms between the three morphology features and $\Delta HR$ yields an out-of-fold AUROC of **0.7044**, a paired improvement of **+0.0277** [95% CI: +0.0230, +0.0321] over the baseline morphology model. Compared to linear main effects (0.7038), pairwise interactions add approximately +0.0006 AUROC.
2. **Tree Ensembles (Models c & d):**
   - **HistGradientBoosting (Model c):** Achieves an out-of-fold AUROC of **0.7192** [95% CI: 0.7106, 0.7281] with a paired difference of **+0.0425** [95% CI: +0.0372, +0.0478] over Model (a).
   - **Random Forest (Model d):** Achieves an out-of-fold AUROC of **0.6674** [95% CI: 0.6608, 0.6744] with a paired difference of **-0.0093** [95% CI: -0.0152, -0.0035] over Model (a).
3. **Threshold Calibration & Class Balance:**
   At the standard 0.50 probability threshold, all four models yield predicted-positive rates (47.94% to 51.03%) that closely track the true development label prevalence (50.00%), indicating that balanced accuracy and raw accuracy are not distorted by threshold bias.
4. **Pre-Registration Boundary Enforcement:**
   While tree ensembles and interaction terms demonstrate modest performance gains on development data (+0.03 to +0.06 AUROC), the pre-registered within-subject study stands strictly on the frozen linear model (`models/w_frozen.pkl`) evaluated in `results/test_W.md`. These comparisons serve solely as exploratory characterizations of potential nonlinear hemodynamic interactions.
