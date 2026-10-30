# AAMI Cal Subset Ingestion Report (`results/aami_ingest_report.md`)

**Cohort:** VitalDB AAMI Calibration Subset (`VitalDB_AAMI_Cal_Subset.mat`)  
**Destination:** `data/parquet/aami_cal.parquet`  
**Protocol Specification:** `PLAN_C.md` / Study C  
**Evaluation State:** Ingestion and descriptive audit ONLY. **NO SCORING PERFORMED.**  

---

## 1. Segments and Subject Counts

- **Total Segments Written:** 70,212 of 70,212 in source file (dropped: 0) (Expectation: **70,212**)
- **Unique Subjects:** 116 (Expectation: **116**)
- **Verification Check:** PASS
- **Segments per Subject:** min = 3, 25% = 258, median = 546, 75% = 843, max = 1935

---

## 2. Demographic Completeness (Non-Null %)

- **Age:** 100.00% non-null
- **Sex / Gender:** 100.00% non-null (female share: 41.4%)
- **BMI:** 100.00% non-null (median [IQR]: 23.4 [21.1-25.8])
- **Height:** 100.00% non-null
- **Weight:** 100.00% non-null

---

## 3. Cross-Cohort Subject Overlap (Data Leakage Audit)

- **Overlap with `train_file` (1,293 subjects):** **0** subjects (Expectation: **0** - PASS)
- **Overlap with `calfree_test` (144 subjects):** **0** subjects (Expectation: **0** - PASS)
- **Independence Status:** Strictly disjoint subject cohort. Zero patient overlap across splits.

---

## 4. Signal Channel & PPG Correlation Check

- **Selected PPG Channel Index:** Channel 1
- **Channel 0 Correlation with ABP:** 0.3380
- **Channel 1 Correlation with ABP:** 0.9608
- **Decision Rule:** Selected channel 1 has superior correlation with invasive arterial pressure.

---

## 5. Descriptive Target Distributions (SBP & DBP)

> [!NOTE]
> These statistics are purely descriptive of cohort distribution. No classification models have been run.

### SBP Quantiles (mmHg)
- **1%:** 79.2 mmHg
- **5%:** 88.0 mmHg
- **25%:** 103.1 mmHg
- **50% (Median):** 115.5 mmHg
- **75%:** 128.7 mmHg
- **95%:** 149.2 mmHg
- **99%:** 166.1 mmHg

### DBP Quantiles (mmHg)
- **5%:** 45.6 mmHg | **50%:** 62.9 mmHg | **95%:** 84.2 mmHg

### Target Class Prevalence
- **Primary Target ($SBP \ge 120$ mmHg):** **28,811** segments (**41.03%**) | Subjects (median SBP): 38/116 (32.76%)
- **Sensitivity Target ($SBP \ge 130$ mmHg):** **16,012** segments (**22.81%**) | Subjects (median SBP): 11/116 (9.48%)

---

## 6. Study C Feature Extraction NaN Rates

Computed across all 70,212 segments using canonical single-segment `extract_features_block`:

| Feature | Description | AAMI_Cal NaN Rate (%) | Missingness Handling |
|---|---|---|---|
| **`hr_ibi`** | Study C Feature | **0.00%** | Preserved as NaN; imputed by frozen PlanCImputer training medians |
| **`hr`** | Study C Feature | **0.00%** | Preserved as NaN; imputed by frozen PlanCImputer training medians |
| **`ibi_sd`** | Study C Feature | **0.00%** | Preserved as NaN; imputed by frozen PlanCImputer training medians |
| **`systolic_rise_time`** | Study C Feature | **0.00%** | Preserved as NaN; imputed by frozen PlanCImputer training medians |
| **`pulse_width`** | Study C Feature | **0.00%** | Preserved as NaN; imputed by frozen PlanCImputer training medians |
| **`decay_ratio`** | Study C Feature | **0.00%** | Preserved as NaN; imputed by frozen PlanCImputer training medians |
| **`beat_count`** | Study C Feature | **0.00%** | Preserved as NaN; imputed by frozen PlanCImputer training medians |
| **`is_clipped`** | Study C Feature | **0.00%** | Preserved as NaN; imputed by frozen PlanCImputer training medians |
| **`snr_ratio`** | Study C Feature | **0.00%** | Preserved as NaN; imputed by frozen PlanCImputer training medians |
