# Study C Development & Cross-Cohort Validation Report

**Date:** 2026-10-30  
**Specification:** `PLAN_C.md` (Cross-subject SBP classification on unseen subjects)  
**Pre-Registered Plan Hash:** `e851b885f144022920476b9de238304a52fade4fd46157404a8b3378c8bbb0f3`  
**Git HEAD Commit:** `d74dd46d506d55e2a252f3a124f4face860316d4`  
**Train Features Parquet Hash (64 parts):** `1fdf00698c1e3b292d276bb012032db07add33cfd81f46bc299a84296298702b`  
**CalFree Features Parquet Hash (64 parts):** `161786696e5ea3c06e5e4931c87019558391098ce9e7ead6332128a7e37be11e`  
**Final Test Boundary:** `VitalDB_AAMI_Cal_Subset.mat` is **PRISTINE and UNREAD**. Scored strictly once after models are frozen.  

---

## 1. Study Setup & Data Splits

- **Development Cohort:** VitalDB Train file (`file_split == 'train_file'`, 1,293 subjects, 465,480 segments, exactly 360 seg/subj).
- **Development Protocol:** 5-fold cross-validation with seeded subject permutation and disjoint fold assertion.
- **Extra Validation Cohort:** CalFree (`file_split == 'calfree_test'`, 144 subjects, 57,600 segments, 400 seg/subj).
- **Primary Target Label:** $SBP \ge 120\text{ mmHg}$ (Development: **175,338** pos / **290,142** neg, **37.67%**; CalFree: **21,713** pos / **35,887** neg, **37.70%**).
- **Sensitivity Target Label:** $SBP \ge 130\text{ mmHg}$ (Development: **98,290** pos / **367,190** neg, **21.12%**).
- **Decision Threshold:** Fixed standard threshold $0.5$ ($P(\hat{y}=1) \ge 0.5$, no recalibration).
- **Bootstrap Protocol:** Clustered subject-level bootstrap with 2,000 resamples (seed 42).
- **Primary Endpoint:** Paired accuracy difference $C2 - C0$ for the primary classifier (HistGradientBoostingClassifier).

---

## 2. Primary Endpoint (SBP $\ge 120$ mmHg): 5-Fold Grouped CV Pooled OOF Results

| Model Step | Classifier | Predictors | Accuracy [95% CI] | Balanced Acc [95% CI] | AUROC [95% CI] | F1 Score | Majority Baseline |
|---|---|---|---|---|---|---|---|
| **C0** | **GBT (Primary)** | Age, Sex, BMI | **59.25%** [57.66%, 60.87%] | **49.93%** [48.84%, 50.97%] | **0.5120** [0.4928, 0.5322] | 0.1831 | 62.33% |
| **C0** | Logistic Regression | Age, Sex, BMI | **62.33%** [60.85%, 63.85%] | **50.00%** [50.00%, 50.00%] | **0.5063** [0.4869, 0.5253] | 0.0000 | 62.33% |
| **C1** | **GBT (Primary)** | C0 + PPG Heart Rate (hr_ibi) | **60.22%** [58.70%, 61.81%] | **51.09%** [50.05%, 52.16%] | **0.5323** [0.5141, 0.5512] | 0.2104 | 62.33% |
| **C1** | Logistic Regression | C0 + PPG Heart Rate (hr_ibi) | **62.32%** [60.83%, 63.84%] | **50.00%** [49.97%, 50.02%] | **0.5252** [0.5067, 0.5437] | 0.0010 | 62.33% |
| **C2m** | **GBT (Primary)** | C1 + Morphology Only (rise_time, pulse_width, decay_ratio) | **65.86%** [64.70%, 67.13%] | **60.32%** [59.22%, 61.44%] | **0.6744** [0.6597, 0.6896] | 0.4552 | 62.33% |
| **C2m** | Logistic Regression | C1 + Morphology Only (rise_time, pulse_width, decay_ratio) | **64.69%** [63.34%, 66.05%] | **56.30%** [55.35%, 57.26%] | **0.6469** [0.6303, 0.6625] | 0.3224 | 62.33% |
| **C2** | **GBT (Primary)** | C1 + Morphology + Auxiliary Fields (beat_count, snr_ratio, ibi_sd) | **65.94%** [64.80%, 67.21%] | **60.46%** [59.34%, 61.58%] | **0.6785** [0.6640, 0.6934] | 0.4583 | 62.33% |
| **C2** | Logistic Regression | C1 + Morphology + Auxiliary Fields (beat_count, snr_ratio, ibi_sd) | **64.77%** [63.46%, 66.13%] | **56.54%** [55.56%, 57.50%] | **0.6490** [0.6326, 0.6647] | 0.3310 | 62.33% |

### Stepwise Paired Gain Decomposition (5-Fold CV Pooled OOF, SBP $\ge 120$ mmHg)

| Comparison | Classifier | Paired $\Delta\text{Accuracy}$ [95% CI] | Paired $\Delta\text{Balanced Acc}$ [95% CI] | Paired $\Delta\text{AUROC}$ [95% CI] |
|---|---|---|---|---|
| C1 vs C0 (Gain from Heart Rate IBI) | **GBT (Primary)** | **+0.97%** [+0.02%, +1.92%] | **+1.16%** [+0.19%, +2.15%] | **+0.0203** [+0.0068, +0.0335] |
| C2m vs C1 (Gain from Pulse Morphology Shape) | **GBT (Primary)** | **+5.64%** [+4.34%, +6.91%] | **+9.23%** [+7.91%, +10.55%] | **+0.1421** [+0.1251, +0.1595] |
| C2 vs C2m (Gain from Auxiliary Segment Fields) | **GBT (Primary)** | **+0.08%** [-0.20%, +0.36%] | **+0.14%** [-0.18%, +0.46%] | **+0.0041** [+0.0009, +0.0074] |
| **C2 vs C0 (Total Pipeline Gain - Primary Endpoint)** | **GBT (Primary)** | **+6.69%** [+5.29%, +8.10%] | **+10.53%** [+9.13%, +11.93%] | **+0.1665** [+0.1464, +0.1851] |
| C1 vs C0 (Gain from Heart Rate IBI) | Logistic Regression | **-0.01%** [-0.04%, +0.00%] | **-0.00%** [-0.03%, +0.02%] | **+0.0189** [+0.0081, +0.0296] |
| C2m vs C1 (Gain from Pulse Morphology Shape) | Logistic Regression | **+2.37%** [+1.35%, +3.34%] | **+6.30%** [+5.35%, +7.25%] | **+0.1217** [+0.1012, +0.1407] |
| C2 vs C2m (Gain from Auxiliary Segment Fields) | Logistic Regression | **+0.09%** [-0.07%, +0.24%] | **+0.23%** [+0.06%, +0.40%] | **+0.0021** [-0.0001, +0.0044] |
| **C2 vs C0 (Total Pipeline Gain - Primary Endpoint)** | Logistic Regression | **+2.44%** [+1.42%, +3.42%] | **+6.54%** [+5.56%, +7.50%] | **+0.1428** [+0.1223, +0.1625] |

---

## 3. Per-Fold Cross-Validation Metrics (Primary Target: SBP $\ge 120$ mmHg)

| Fold | Model Step | Classifier | Accuracy | Balanced Acc | AUROC | F1 Score | Majority Baseline |
|---|---|---|---|---|---|---|---|
| Fold 1 | **C0** | GBT | 60.17% | 48.96% | 0.5073 | 0.1568 | 64.56% |
| Fold 1 | **C0** | LR | 64.56% | 50.00% | 0.4881 | 0.0000 | 64.56% |
| Fold 1 | **C1** | GBT | 61.85% | 52.04% | 0.5271 | 0.2543 | 64.56% |
| Fold 1 | **C1** | LR | 64.56% | 50.01% | 0.5111 | 0.0004 | 64.56% |
| Fold 1 | **C2m** | GBT | 66.39% | 60.42% | 0.6738 | 0.4569 | 64.56% |
| Fold 1 | **C2m** | LR | 65.88% | 56.51% | 0.6514 | 0.3359 | 64.56% |
| Fold 1 | **C2** | GBT | 65.83% | 59.89% | 0.6787 | 0.4501 | 64.56% |
| Fold 1 | **C2** | LR | 66.00% | 56.94% | 0.6544 | 0.3498 | 64.56% |
| Fold 2 | **C0** | GBT | 62.02% | 51.25% | 0.5087 | 0.1925 | 63.92% |
| Fold 2 | **C0** | LR | 63.92% | 50.00% | 0.5592 | 0.0000 | 63.92% |
| Fold 2 | **C1** | GBT | 60.59% | 50.01% | 0.5207 | 0.1799 | 63.92% |
| Fold 2 | **C1** | LR | 63.92% | 50.00% | 0.5744 | 0.0000 | 63.92% |
| Fold 2 | **C2m** | GBT | 67.21% | 60.10% | 0.6775 | 0.4321 | 63.92% |
| Fold 2 | **C2m** | LR | 65.83% | 56.16% | 0.6587 | 0.3117 | 63.92% |
| Fold 2 | **C2** | GBT | 67.65% | 60.60% | 0.6818 | 0.4406 | 63.92% |
| Fold 2 | **C2** | LR | 65.99% | 56.60% | 0.6598 | 0.3268 | 63.92% |
| Fold 3 | **C0** | GBT | 58.55% | 50.80% | 0.5218 | 0.1853 | 59.93% |
| Fold 3 | **C0** | LR | 59.93% | 50.00% | 0.5078 | 0.0000 | 59.93% |
| Fold 3 | **C1** | GBT | 60.99% | 53.51% | 0.5616 | 0.2456 | 59.93% |
| Fold 3 | **C1** | LR | 59.86% | 49.98% | 0.5213 | 0.0041 | 59.93% |
| Fold 3 | **C2m** | GBT | 64.80% | 60.32% | 0.6654 | 0.4623 | 59.93% |
| Fold 3 | **C2m** | LR | 63.49% | 56.51% | 0.6343 | 0.3195 | 59.93% |
| Fold 3 | **C2** | GBT | 64.60% | 60.28% | 0.6649 | 0.4658 | 59.93% |
| Fold 3 | **C2** | LR | 63.30% | 56.35% | 0.6347 | 0.3179 | 59.93% |
| Fold 4 | **C0** | GBT | 58.58% | 49.98% | 0.5348 | 0.1999 | 61.79% |
| Fold 4 | **C0** | LR | 61.79% | 50.00% | 0.5105 | 0.0000 | 61.79% |
| Fold 4 | **C1** | GBT | 60.66% | 51.91% | 0.5533 | 0.2237 | 61.79% |
| Fold 4 | **C1** | LR | 61.79% | 50.00% | 0.5293 | 0.0000 | 61.79% |
| Fold 4 | **C2m** | GBT | 66.17% | 61.41% | 0.6816 | 0.4822 | 61.79% |
| Fold 4 | **C2m** | LR | 64.51% | 57.20% | 0.6610 | 0.3606 | 61.79% |
| Fold 4 | **C2** | GBT | 66.46% | 61.80% | 0.6907 | 0.4892 | 61.79% |
| Fold 4 | **C2** | LR | 64.65% | 57.45% | 0.6654 | 0.3678 | 61.79% |
| Fold 5 | **C0** | GBT | 56.93% | 48.61% | 0.4858 | 0.1798 | 61.44% |
| Fold 5 | **C0** | LR | 61.44% | 50.00% | 0.4961 | 0.0000 | 61.44% |
| Fold 5 | **C1** | GBT | 57.01% | 48.15% | 0.5006 | 0.1449 | 61.44% |
| Fold 5 | **C1** | LR | 61.44% | 50.00% | 0.5149 | 0.0001 | 61.44% |
| Fold 5 | **C2m** | GBT | 64.72% | 59.34% | 0.6754 | 0.4394 | 61.44% |
| Fold 5 | **C2m** | LR | 63.73% | 55.27% | 0.6395 | 0.2801 | 61.44% |
| Fold 5 | **C2** | GBT | 65.17% | 59.71% | 0.6801 | 0.4424 | 61.44% |
| Fold 5 | **C2** | LR | 63.92% | 55.56% | 0.6428 | 0.2891 | 61.44% |

---

## 4. Sensitivity Analysis (SBP $\ge 130$ mmHg): 5-Fold Grouped CV Pooled OOF Results

| Model Step | Classifier | Accuracy [95% CI] | Balanced Acc [95% CI] | AUROC [95% CI] | F1 Score | Majority Baseline |
|---|---|---|---|---|---|---|
| **C0** | GBT | **78.39%** [77.19%, 79.65%] | **50.33%** [49.83%, 50.93%] | **0.5246** [0.5021, 0.5466] | 0.0333 | 78.88% |
| **C0** | Logistic Regression | **78.88%** [77.72%, 80.09%] | **50.00%** [50.00%, 50.00%] | **0.5098** [0.4887, 0.5310] | 0.0000 | 78.88% |
| **C1** | GBT | **78.20%** [76.99%, 79.38%] | **50.34%** [49.81%, 50.93%] | **0.5340** [0.5139, 0.5550] | 0.0396 | 78.88% |
| **C1** | Logistic Regression | **78.88%** [77.72%, 80.09%] | **50.00%** [50.00%, 50.00%] | **0.5335** [0.5128, 0.5550] | 0.0000 | 78.88% |
| **C2m** | GBT | **78.64%** [77.50%, 79.83%] | **54.42%** [53.57%, 55.31%] | **0.6936** [0.6778, 0.7100] | 0.1983 | 78.88% |
| **C2m** | Logistic Regression | **78.64%** [77.47%, 79.85%] | **51.47%** [50.89%, 52.15%] | **0.6728** [0.6542, 0.6905] | 0.0809 | 78.88% |
| **C2** | GBT | **78.73%** [77.61%, 79.89%] | **54.52%** [53.65%, 55.46%] | **0.6956** [0.6796, 0.7121] | 0.2004 | 78.88% |
| **C2** | Logistic Regression | **78.64%** [77.47%, 79.85%] | **51.50%** [50.92%, 52.18%] | **0.6738** [0.6558, 0.6916] | 0.0822 | 78.88% |

---

## 5. Extra Validation: CalFree Cohort ($N = 57,600$ segments, 144 subjects)

Evaluated strictly using frozen models fit on the development set (zero refitting on CalFree).

| Model Step | Classifier | Accuracy [95% CI] | Balanced Acc [95% CI] | AUROC [95% CI] | F1 Score | Majority Baseline |
|---|---|---|---|---|---|---|
| **C0** | GBT | **60.76%** [56.08%, 65.23%] | **50.18%** [48.16%, 52.14%] | **0.5224** [0.4698, 0.5770] | 0.1210 | 62.30% |
| **C0** | Logistic Regression | **62.30%** [57.46%, 66.76%] | **50.00%** [50.00%, 50.00%] | **0.5514** [0.4943, 0.6072] | 0.0000 | 62.30% |
| **C1** | GBT | **59.80%** [55.37%, 64.34%] | **49.35%** [47.32%, 51.30%] | **0.5211** [0.4686, 0.5721] | 0.1146 | 62.30% |
| **C1** | Logistic Regression | **62.30%** [57.46%, 66.76%] | **50.00%** [50.00%, 50.00%] | **0.5390** [0.4863, 0.5924] | 0.0000 | 62.30% |
| **C2m** | GBT | **64.87%** [61.26%, 68.52%] | **59.32%** [55.97%, 62.61%] | **0.6659** [0.6220, 0.7095] | 0.4411 | 62.30% |
| **C2m** | Logistic Regression | **62.81%** [58.93%, 66.59%] | **55.01%** [51.75%, 58.11%] | **0.6192** [0.5665, 0.6723] | 0.3209 | 62.30% |
| **C2** | GBT | **65.11%** [61.48%, 68.65%] | **59.52%** [56.17%, 62.81%] | **0.6643** [0.6203, 0.7092] | 0.4432 | 62.30% |
| **C2** | Logistic Regression | **62.80%** [58.80%, 66.70%] | **55.10%** [51.89%, 58.24%] | **0.6199** [0.5680, 0.6714] | 0.3256 | 62.30% |

### CalFree Paired Differences ($C2 - C0$ Endpoint Preview)

| Comparison | Classifier | Paired $\Delta\text{Accuracy}$ [95% CI] | Paired $\Delta\text{Balanced Acc}$ [95% CI] | Paired $\Delta\text{AUROC}$ [95% CI] |
|---|---|---|---|---|
| C1 vs C0 (Gain from Heart Rate IBI) | GBT | **-0.96%** [-3.58%, +1.84%] | **-0.83%** [-3.26%, +1.70%] | **-0.0013** [-0.0524, +0.0477] |
| C2m vs C1 (Gain from Pulse Morphology Shape) | GBT | **+5.07%** [+1.03%, +9.22%] | **+9.97%** [+6.29%, +13.76%] | **+0.1448** [+0.0844, +0.2078] |
| C2 vs C2m (Gain from Auxiliary Segment Fields) | GBT | **+0.23%** [-0.38%, +0.92%] | **+0.20%** [-0.45%, +0.98%] | **-0.0016** [-0.0108, +0.0066] |
| **C2 vs C0 (Total Pipeline Gain - Primary Endpoint)** | GBT | **+4.34%** [+0.35%, +8.63%] | **+9.34%** [+5.90%, +12.91%] | **+0.1419** [+0.0864, +0.2005] |
| C1 vs C0 (Gain from Heart Rate IBI) | Logistic Regression | **+0.00%** [+0.00%, +0.00%] | **+0.00%** [+0.00%, +0.00%] | **-0.0124** [-0.0527, +0.0262] |
| C2m vs C1 (Gain from Pulse Morphology Shape) | Logistic Regression | **+0.51%** [-3.07%, +4.03%] | **+5.01%** [+1.75%, +8.11%] | **+0.0801** [+0.0244, +0.1377] |
| C2 vs C2m (Gain from Auxiliary Segment Fields) | Logistic Regression | **-0.01%** [-0.42%, +0.42%] | **+0.09%** [-0.33%, +0.52%] | **+0.0007** [-0.0056, +0.0075] |
| **C2 vs C0 (Total Pipeline Gain - Primary Endpoint)** | Logistic Regression | **+0.50%** [-3.11%, +4.12%] | **+5.10%** [+1.89%, +8.24%] | **+0.0685** [+0.0176, +0.1213] |

---

## 6. Frozen Model Artifacts & SHA-256 Checksums

All models below are frozen and ready for the single-pass evaluation on `AAMI_Cal`:

| Model Key | Path | SHA-256 Digest |
|---|---|---|
| **`c_imputer`** | `models\c_imputer_frozen.pkl` | `d0a8e3d67cad417d1115f6b957ccd0ba901e1ac378e78eeec06b7df8d16bd641` |
| **`C0_lr`** | `models\c0_lr_frozen.pkl` | `02490af84b823d77cfb05e416ea74982ad577cbb425cf1ebf38b532f5e7dd4b3` |
| **`C0_gbt`** | `models\c0_gbt_frozen.pkl` | `fd5d7d29b0b18a592370d9d9885a5ed824efb5bfbdca043dc5e64cf907068af3` |
| **`C1_lr`** | `models\c1_lr_frozen.pkl` | `74012839bdd119f60619aee8c8ea07997125bb78eb2cf89e8556a22da948b0c5` |
| **`C1_gbt`** | `models\c1_gbt_frozen.pkl` | `0ddc345f04e8aba38da90bbac8afaf26ce02c48bdeea78082a888931f840feb9` |
| **`C2m_lr`** | `models\c2m_lr_frozen.pkl` | `c7032cc9204ad4225af592588d2334df1d94dd9bd9ba5f351876530d52c755d9` |
| **`C2m_gbt`** | `models\c2m_gbt_frozen.pkl` | `2869f2b83975e5335573904bb8c1a2fb5b412bbf3896537656c25da04265b54a` |
| **`C2_lr`** | `models\c2_lr_frozen.pkl` | `4def87939d3e73fdbfb28a2f0f94e669423c25a1f7d0fd2e7e1e0d5ddeb8fa8a` |
| **`C2_gbt`** | `models\c2_gbt_frozen.pkl` | `8a6e9e1aea7c552455e4a9d1fa1b0883ceaab71a9ace1be018f8bc36af4f649f` |

---

## 7. Generated Visualizations

- **Accuracy Ladder Plot:** `results/dev_C_accuracy_ladder.png`
- **Paired Gain Decomposition Plot:** `results/dev_C_paired_diffs.png`

---

## 8. Pre-Registration Invariant Status

- `PLAN_C.md` pre-registration and Amendments C1, C2, C3 committed to git.
- `VitalDB_AAMI_Cal_Subset.mat` has **NOT** been opened or read.
- Tag `plan-c-locked` has **NOT** been created.
- **Execution Note:** The first development pipeline attempt completed Steps 1–4 (cross-validation, bootstrap, and full model freezing) and crashed in Step 5 due to an artifact dictionary loading bug in `evaluate_on_calfree` (attempting to subscript a pickled `PlanCImputer` instance). The bug was fixed with zero changes to any feature definition, model, hyperparameter, fold assignment, seed, threshold, or bootstrap setting.
