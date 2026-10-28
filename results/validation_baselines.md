# Validation Baselines: Step 0a (Age) and Step 0 (Age, Sex, BMI)

**Status:** Validation results only (no test data loaded or scored).  
**Date:** 2026-10-29  
**Split:** VitalDB training dataset (`VitalDB_Train_Subset.mat` / `data/parquet/train_file.parquet`)  
**Effective Sample Size:**  
- **Total Subjects:** 1,293 subjects (465,480 segments total; exactly 360 10-second segments per subject).
- **Training Cohort (`split == 'train'`):** 1,034 subjects (372,240 segments).
- **Validation Cohort (`split == 'val'`):** 259 subjects (93,240 segments).
- **Note on Effective Sample Size:** Because each subject contributes 360 segments with identical demographic covariates (per-subject medians of age, sex, and BMI), the effective independent sample size for demographic features is 1,293 subjects (1,034 train / 259 validation).

---

## 1. Experimental Setup & Pre-Registration Protocol

1. **Target Label:** Binary classification of hypertensive-range arterial blood pressure ($SBP \ge 130$ mmHg).
   - Training prevalence: 21.63% (80,507 / 372,240 segments).
   - Validation prevalence: 19.07% (17,783 / 93,240 segments).
2. **Demographic Rule:** Age, sex, and BMI are per-subject medians computed across all segments of the subject.
3. **Standardization:** Input features are standardized using `StandardScaler` fitted strictly on `split == 'train'`.
4. **Model Architecture:** Scikit-learn `LogisticRegression(solver='lbfgs', random_state=42)`.
5. **Evaluation Protocol:**
   - Evaluated exclusively on `split == 'val'`.
   - Primary headline metrics: AUROC and Balanced Accuracy.
   - 95% Confidence Intervals: Subject-level bootstrap with 2,000 resamples (resampling whole subjects with all 360 of their segments, random seed 42).
   - F1 decision threshold: Chosen on validation via grid search over $(0.01, 0.99)$.
   - Sensitivity check: Evaluated excluding subject `p003938_1` (who resides in the validation split).

---

## 2. Validation Results

### Step 0a: Age Only (Logistic Regression)
- **Fitted Parameters:**
  - Standardized Coefficient (Age): $+0.1353$
  - Intercept: $-1.2926$
- **Validation AUROC:** **0.4981** (95% subject-level bootstrap CI: **[0.4472, 0.5484]**)
- **Optimal F1 Probability Threshold:** $\theta^* = 0.1650$
  - Validation F1: $0.3248$
- **Balanced Accuracy (at $\theta^* = 0.1650$):** **0.5104** (95% subject-level bootstrap CI: **[0.5022, 0.5204]**)
- **Sensitivity Check (excluding `p003938_1`):**
  - Validation AUROC: $0.4971$
  - $\Delta \text{AUROC}$: $-0.0010$

### Step 0: Age, Sex, BMI (Logistic Regression)
- **Fitted Parameters:**
  - Standardized Coefficient (Age): $+0.1315$
  - Standardized Coefficient (Sex, 1=Male): $+0.0231$
  - Standardized Coefficient (BMI): $+0.0316$
  - Intercept: $-1.2931$
- **Validation AUROC:** **0.5003** (95% subject-level bootstrap CI: **[0.4508, 0.5494]**)
- **Optimal F1 Probability Threshold:** $\theta^* = 0.1700$
  - Validation F1: $0.3257$
- **Balanced Accuracy (at $\theta^* = 0.1700$):** **0.5138** (95% subject-level bootstrap CI: **[0.4976, 0.5295]**)
- **Sensitivity Check (excluding `p003938_1`):**
  - Validation AUROC: $0.4988$
  - $\Delta \text{AUROC}$: $-0.0015$

---

## 3. Comparison Summary

| Metric | Step 0a (Age only) | Step 0 (Age, Sex, BMI) | Incremental Gain ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Validation AUROC** | 0.4981 [0.4472, 0.5484] | 0.5003 [0.4508, 0.5494] | $+0.0022$ |
| **Optimal F1 Threshold** | 0.1650 | 0.1700 | $+0.0050$ |
| **Validation F1** | 0.3248 | 0.3257 | $+0.0009$ |
| **Balanced Accuracy** | 0.5104 [0.5022, 0.5204] | 0.5138 [0.4976, 0.5295] | $+0.0034$ |
| **AUROC without `p003938_1`** | 0.4971 | 0.4988 | $+0.0017$ |

---

## 4. Key Observations & Alignment with Pre-Registration

1. **Near-Chance Demographic Discrimination:** Both demographic baselines yield AUROC near $0.50$ ($0.4981$ and $0.5003$), with 95% bootstrap confidence intervals spanning chance performance ([0.447, 0.548] and [0.451, 0.549]).
2. **Within-Subject SBP Variance Dominance:** As pre-registered in `PLAN.md`'s Known Limitations, between-subject SBP variance is only ~40%, meaning static subject demographics cannot track acute, per-segment arterial blood pressure fluctuations in surgical/ICU monitoring.
3. **Sensitivity Analysis:** Excluding `p003938_1` produces a negligible shift in validation AUROC ($-0.0010$ in Step 0a, $-0.0015$ in Step 0), confirming robustness to demographic recording anomalies.
