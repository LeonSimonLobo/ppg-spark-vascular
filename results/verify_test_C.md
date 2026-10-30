# Independent Verification of Study C Test Results

**Verification Engine:** Standalone `src/verify_test_C.py` using pure `scikit-learn` and `numpy`.
**Input Source:** `D:/DS_RESEARCH/ppg-spark-vascular/results/test_C_predictions.parquet` strictly (zero .mat file access, zero test_c.py code import).
**Sample Size:** 70,212 segments across 116 subjects.
**Cohort Prevalence (SBP >= 120 mmHg):** 41.03% (28,811 / 70,212 segments).
**Majority-Class Baseline Rate:** 58.97%.

---

## 1. Recomputed Model Performance at Fixed Threshold 0.5

| Model Step | Classifier | Accuracy | Balanced Accuracy | AUROC | Predicted Pos Rate | Majority Baseline |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **C0** | LR | **58.97%** | **50.00%** | **0.5704** | 0.00% | 58.97% |
| **C0** | GBT | **59.41%** | **51.66%** | **0.5584** | 6.50% | 58.97% |
| **C1** | LR | **58.96%** | **50.00%** | **0.5435** | 0.01% | 58.97% |
| **C1** | GBT | **57.76%** | **50.36%** | **0.5439** | 8.67% | 58.97% |
| **C2m** | LR | **60.63%** | **54.48%** | **0.6444** | 14.86% | 58.97% |
| **C2m** | GBT | **63.81%** | **59.76%** | **0.6635** | 25.66% | 58.97% |
| **C2** | LR | **60.96%** | **54.89%** | **0.6520** | 15.30% | 58.97% |
| **C2** | GBT | **64.65%** | **60.58%** | **0.6785** | 25.42% | 58.97% |

---

## 2. Primary Endpoint Verification (GBT: C2 minus C0)

- **Pooled Point Estimate ($\Delta$Accuracy):** **5.2370%** (+0.052370)
- **Reference Value in `results/test_C_diffs.csv`:** **5.2370%** (+0.052370)
- **Point Estimate Absolute Discrepancy:** `7.63e-17` (**PASS: Exact agreement < 1e-6**)
- **Recomputed 95% CI (Seed 123, 2000 resamples):** `[+0.25%, +10.06%]`
- **Reference 95% CI (Seed 42, 2000 resamples):** `[+0.60%, +10.30%]`
- **Agreement Assessment:** Seed-123 interval is closely consistent with the registered seed-42 interval, verifying robust statistical significance bounded above zero.

---

## 3. Exploratory Subject-Weighted (Macro-Average) Analysis (Post hoc, descriptive)

- **Protocol Status:** *Post hoc, descriptive only* (unweighted pooled accuracy is the pre-registered primary endpoint).
- **Macro-Average Accuracy C0 (GBT):** 60.65%
- **Macro-Average Accuracy C2 (GBT):** 65.74%
- **Macro-Average Gain ($C2 - C0$):** **+5.09%** [95% CI: +0.96%, +9.40%]
- **Observation:** The vascular feature gain remains positive and statistically significant when giving equal weight to each patient regardless of segment count.

---

## 4. Observation on AUROC Equality to Development Cohorts (4 Decimals)

| Test Model | Test AUROC | Match Type | Dev Cohort | Dev Target | Dev Model Step | Dev Classifier | Dev AUROC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `C2_gbt` | 0.678549 (`0.6785`) | Exact 4-decimal match | `Train_OOF_Pooled` | `sbp_ge_120` | `C2` | `gbt` | 0.678494 (`0.6785`) |

> [!NOTE]
> As observed above, the held-out test AUROC for **`C2_gbt`** on AAMI_Cal is **`0.6785`**, which matches the development cross-validation out-of-fold AUROC for **`C2_gbt`** on `Train_OOF_Pooled` (`0.6785` to 4 decimal places).
