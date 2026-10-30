# Study C Headline Numbers & Source Traceability Matrix

This table aggregates every headline number reported for Study C across development, validation, and held-out test evaluation, mapped to its exact source file and column.

---

## 1. Cohort Sizes, Prevalences, and Class Counts

| Cohort | Role | Subjects | Segments | Target | Positives (%) | Negatives (%) | Source File & Field |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train Development** | 5-Fold Grouped CV | 1,293 | 465,480 | $SBP \ge 120$ | 175,338 (37.67%) | 290,142 (62.33%) | `results/dev_C.md`: Section 1 |
| **Train Development** | Sensitivity | 1,293 | 465,480 | $SBP \ge 130$ | 98,290 (21.12%) | 367,190 (78.88%) | `results/dev_C.md`: Section 1 |
| **CalFree Validation** | Prospective Validation | 144 | 57,600 | $SBP \ge 120$ | 21,713 (37.70%) | 35,887 (62.30%) | `results/dev_C.md`: Section 1 |
| **AAMI_Cal Test** | Final Held-Out Evaluation | 116 | 70,212 | $SBP \ge 120$ | 28,811 (41.03%) | 41,401 (58.97%) | `results/test_C.md`: Section 1 |

---

## 2. Three-Cohort Primary Endpoint Gain ($C2 - C0$ Total Pipeline Gain)

| Cohort | Target | Classifier | Metric | Point Estimate | 95% Confidence Interval | Source File & Column |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Train_OOF_Pooled` | `sbp_ge_120` | **LR** | Accuracy Gain | **+2.44%** | [+1.42%, +3.42%] | `results/dev_C_diffs.csv`: `delta_accuracy` |
| `Train_OOF_Pooled` | `sbp_ge_120` | **LR** | Balanced Acc Gain | **+6.54%** | [+5.56%, +7.50%] | `results/dev_C_diffs.csv`: `delta_balanced_accuracy` |
| `Train_OOF_Pooled` | `sbp_ge_120` | **LR** | AUROC Gain | **+0.1428** | [+0.1223, +0.1625] | `results/dev_C_diffs.csv`: `delta_auroc` |
| `Train_OOF_Pooled` | `sbp_ge_130` | **LR** | Accuracy Gain | **-0.24%** | [-0.64%, +0.13%] | `results/dev_C_diffs.csv`: `delta_accuracy` |
| `Train_OOF_Pooled` | `sbp_ge_130` | **LR** | Balanced Acc Gain | **+1.50%** | [+0.92%, +2.18%] | `results/dev_C_diffs.csv`: `delta_balanced_accuracy` |
| `Train_OOF_Pooled` | `sbp_ge_130` | **LR** | AUROC Gain | **+0.1640** | [+0.1431, +0.1849] | `results/dev_C_diffs.csv`: `delta_auroc` |
| `CalFree_Validation` | `sbp_ge_120` | **LR** | Accuracy Gain | **+0.50%** | [-3.11%, +4.12%] | `results/dev_C_diffs.csv`: `delta_accuracy` |
| `CalFree_Validation` | `sbp_ge_120` | **LR** | Balanced Acc Gain | **+5.10%** | [+1.89%, +8.24%] | `results/dev_C_diffs.csv`: `delta_balanced_accuracy` |
| `CalFree_Validation` | `sbp_ge_120` | **LR** | AUROC Gain | **+0.0685** | [+0.0176, +0.1213] | `results/dev_C_diffs.csv`: `delta_auroc` |
| `Train_OOF_Pooled` | `sbp_ge_120` | **GBT** | Accuracy Gain | **+6.69%** | [+5.29%, +8.10%] | `results/dev_C_diffs.csv`: `delta_accuracy` |
| `Train_OOF_Pooled` | `sbp_ge_120` | **GBT** | Balanced Acc Gain | **+10.53%** | [+9.13%, +11.93%] | `results/dev_C_diffs.csv`: `delta_balanced_accuracy` |
| `Train_OOF_Pooled` | `sbp_ge_120` | **GBT** | AUROC Gain | **+0.1665** | [+0.1464, +0.1851] | `results/dev_C_diffs.csv`: `delta_auroc` |
| `Train_OOF_Pooled` | `sbp_ge_130` | **GBT** | Accuracy Gain | **+0.34%** | [-0.32%, +1.00%] | `results/dev_C_diffs.csv`: `delta_accuracy` |
| `Train_OOF_Pooled` | `sbp_ge_130` | **GBT** | Balanced Acc Gain | **+4.19%** | [+3.27%, +5.18%] | `results/dev_C_diffs.csv`: `delta_balanced_accuracy` |
| `Train_OOF_Pooled` | `sbp_ge_130` | **GBT** | AUROC Gain | **+0.1710** | [+0.1497, +0.1920] | `results/dev_C_diffs.csv`: `delta_auroc` |
| `CalFree_Validation` | `sbp_ge_120` | **GBT** | Accuracy Gain | **+4.34%** | [+0.35%, +8.63%] | `results/dev_C_diffs.csv`: `delta_accuracy` |
| `CalFree_Validation` | `sbp_ge_120` | **GBT** | Balanced Acc Gain | **+9.34%** | [+5.90%, +12.91%] | `results/dev_C_diffs.csv`: `delta_balanced_accuracy` |
| `CalFree_Validation` | `sbp_ge_120` | **GBT** | AUROC Gain | **+0.1419** | [+0.0864, +0.2005] | `results/dev_C_diffs.csv`: `delta_auroc` |
| `AAMI_Cal` | `sbp_ge_120` | **LR** | Accuracy Gain | **+1.99%** | [-1.31%, +5.51%] | `results/test_C_diffs.csv`: `delta_accuracy` |
| `AAMI_Cal` | `sbp_ge_120` | **LR** | Balanced Acc Gain | **+4.89%** | [+1.72%, +8.32%] | `results/test_C_diffs.csv`: `delta_balanced_accuracy` |
| `AAMI_Cal` | `sbp_ge_120` | **LR** | AUROC Gain | **+0.0817** | [+0.0140, +0.1500] | `results/test_C_diffs.csv`: `delta_auroc` |
| `AAMI_Cal` | `sbp_ge_120` | **GBT** | Accuracy Gain | **+5.24%** | [+0.60%, +10.30%] | `results/test_C_diffs.csv`: `delta_accuracy` |
| `AAMI_Cal` | `sbp_ge_120` | **GBT** | Balanced Acc Gain | **+8.92%** | [+4.28%, +14.04%] | `results/test_C_diffs.csv`: `delta_balanced_accuracy` |
| `AAMI_Cal` | `sbp_ge_120` | **GBT** | AUROC Gain | **+0.1202** | [+0.0493, +0.1952] | `results/test_C_diffs.csv`: `delta_auroc` |

---

## 3. Stepwise Gain Decomposition Across Cohorts (Primary Classifier: GBT)

| Cohort | Comparison | Physiological Component | $\Delta\text{Accuracy}$ [95% CI] | $\Delta\text{AUROC}$ [95% CI] | Source File |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `Train_OOF_Pooled` | `C1_minus_C0` | Heart Rate Contribution ($C1 - C0$) | **+0.97%** [+0.02%, +1.92%] | **+0.0203** [+0.0068, +0.0335] | `results/dev_C_diffs.csv` |
| `Train_OOF_Pooled` | `C2m_minus_C1` | Pulse Morphology Contribution ($C2m - C1$) | **+5.64%** [+4.34%, +6.91%] | **+0.1421** [+0.1251, +0.1595] | `results/dev_C_diffs.csv` |
| `Train_OOF_Pooled` | `C2_minus_C2m` | Auxiliary Signal Quality Contribution ($C2 - C2m$) | **+0.08%** [-0.20%, +0.36%] | **+0.0041** [+0.0009, +0.0074] | `results/dev_C_diffs.csv` |
| `Train_OOF_Pooled` | `C2_minus_C0` | Total Vascular Pipeline Gain ($C2 - C0$) | **+6.69%** [+5.29%, +8.10%] | **+0.1665** [+0.1464, +0.1851] | `results/dev_C_diffs.csv` |
| `CalFree_Validation` | `C1_minus_C0` | Heart Rate Contribution ($C1 - C0$) | **-0.96%** [-3.58%, +1.84%] | **-0.0013** [-0.0524, +0.0477] | `results/dev_C_diffs.csv` |
| `CalFree_Validation` | `C2m_minus_C1` | Pulse Morphology Contribution ($C2m - C1$) | **+5.07%** [+1.03%, +9.22%] | **+0.1448** [+0.0844, +0.2078] | `results/dev_C_diffs.csv` |
| `CalFree_Validation` | `C2_minus_C2m` | Auxiliary Signal Quality Contribution ($C2 - C2m$) | **+0.23%** [-0.38%, +0.92%] | **-0.0016** [-0.0108, +0.0066] | `results/dev_C_diffs.csv` |
| `CalFree_Validation` | `C2_minus_C0` | Total Vascular Pipeline Gain ($C2 - C0$) | **+4.34%** [+0.35%, +8.63%] | **+0.1419** [+0.0864, +0.2005] | `results/dev_C_diffs.csv` |
| `AAMI_Cal` | `C1_minus_C0` | Heart Rate Contribution ($C1 - C0$) | **-1.65%** [-7.07%, +1.92%] | **-0.0145** [-0.0706, +0.0388] | `results/test_C_diffs.csv` |
| `AAMI_Cal` | `C2m_minus_C1` | Pulse Morphology Contribution ($C2m - C1$) | **+6.05%** [+0.37%, +12.35%] | **+0.1196** [+0.0499, +0.1963] | `results/test_C_diffs.csv` |
| `AAMI_Cal` | `C2_minus_C2m` | Auxiliary Signal Quality Contribution ($C2 - C2m$) | **+0.83%** [-0.03%, +1.83%] | **+0.0151** [+0.0038, +0.0267] | `results/test_C_diffs.csv` |
| `AAMI_Cal` | `C2_minus_C0` | Total Vascular Pipeline Gain ($C2 - C0$) | **+5.24%** [+0.60%, +10.30%] | **+0.1202** [+0.0493, +0.1952] | `results/test_C_diffs.csv` |

---

## 4. Absolute Model Performance Metrics ($SBP \ge 120\text{ mmHg}$, Threshold 0.5)

| Cohort | Step | Classifier | Accuracy [95% CI] | Balanced Acc [95% CI] | AUROC [95% CI] | Majority Baseline | Source File |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Train_OOF_Pooled` | **C0** | LR | **62.33%** [60.85%, 63.85%] | **50.00%** [50.00%, 50.00%] | **0.5063** [0.4869, 0.5253] | 62.33% | `results/dev_C_metrics.csv` |
| `Train_OOF_Pooled` | **C0** | GBT | **59.25%** [57.66%, 60.87%] | **49.93%** [48.84%, 50.97%] | **0.5120** [0.4928, 0.5322] | 62.33% | `results/dev_C_metrics.csv` |
| `Train_OOF_Pooled` | **C1** | LR | **62.32%** [60.83%, 63.84%] | **50.00%** [49.97%, 50.02%] | **0.5252** [0.5067, 0.5437] | 62.33% | `results/dev_C_metrics.csv` |
| `Train_OOF_Pooled` | **C1** | GBT | **60.22%** [58.70%, 61.81%] | **51.09%** [50.05%, 52.16%] | **0.5323** [0.5141, 0.5512] | 62.33% | `results/dev_C_metrics.csv` |
| `Train_OOF_Pooled` | **C2m** | LR | **64.69%** [63.34%, 66.05%] | **56.30%** [55.35%, 57.26%] | **0.6469** [0.6303, 0.6625] | 62.33% | `results/dev_C_metrics.csv` |
| `Train_OOF_Pooled` | **C2m** | GBT | **65.86%** [64.70%, 67.13%] | **60.32%** [59.22%, 61.44%] | **0.6744** [0.6597, 0.6896] | 62.33% | `results/dev_C_metrics.csv` |
| `Train_OOF_Pooled` | **C2** | LR | **64.77%** [63.46%, 66.13%] | **56.54%** [55.56%, 57.50%] | **0.6490** [0.6326, 0.6647] | 62.33% | `results/dev_C_metrics.csv` |
| `Train_OOF_Pooled` | **C2** | GBT | **65.94%** [64.80%, 67.21%] | **60.46%** [59.34%, 61.58%] | **0.6785** [0.6640, 0.6934] | 62.33% | `results/dev_C_metrics.csv` |
| `CalFree_Validation` | **C0** | LR | **62.30%** [57.46%, 66.76%] | **50.00%** [50.00%, 50.00%] | **0.5514** [0.4943, 0.6072] | 62.30% | `results/dev_C_metrics.csv` |
| `CalFree_Validation` | **C0** | GBT | **60.76%** [56.08%, 65.23%] | **50.18%** [48.16%, 52.14%] | **0.5224** [0.4698, 0.5770] | 62.30% | `results/dev_C_metrics.csv` |
| `CalFree_Validation` | **C1** | LR | **62.30%** [57.46%, 66.76%] | **50.00%** [50.00%, 50.00%] | **0.5390** [0.4863, 0.5924] | 62.30% | `results/dev_C_metrics.csv` |
| `CalFree_Validation` | **C1** | GBT | **59.80%** [55.37%, 64.34%] | **49.35%** [47.32%, 51.30%] | **0.5211** [0.4686, 0.5721] | 62.30% | `results/dev_C_metrics.csv` |
| `CalFree_Validation` | **C2m** | LR | **62.81%** [58.93%, 66.59%] | **55.01%** [51.75%, 58.11%] | **0.6192** [0.5665, 0.6723] | 62.30% | `results/dev_C_metrics.csv` |
| `CalFree_Validation` | **C2m** | GBT | **64.87%** [61.26%, 68.52%] | **59.32%** [55.97%, 62.61%] | **0.6659** [0.6220, 0.7095] | 62.30% | `results/dev_C_metrics.csv` |
| `CalFree_Validation` | **C2** | LR | **62.80%** [58.80%, 66.70%] | **55.10%** [51.89%, 58.24%] | **0.6199** [0.5680, 0.6714] | 62.30% | `results/dev_C_metrics.csv` |
| `CalFree_Validation` | **C2** | GBT | **65.11%** [61.48%, 68.65%] | **59.52%** [56.17%, 62.81%] | **0.6643** [0.6203, 0.7092] | 62.30% | `results/dev_C_metrics.csv` |
| `AAMI_Cal` | **C0** | LR | **58.97%** [53.09%, 65.22%] | **50.00%** [50.00%, 50.00%] | **0.5704** [0.5025, 0.6341] | 58.97% | `results/test_C.md` |
| `AAMI_Cal` | **C0** | GBT | **59.41%** [53.42%, 65.66%] | **51.66%** [49.30%, 54.55%] | **0.5584** [0.4871, 0.6307] | 58.97% | `results/test_C.md` |
| `AAMI_Cal` | **C1** | LR | **58.96%** [53.09%, 65.20%] | **50.00%** [49.98%, 50.00%] | **0.5435** [0.4736, 0.6130] | 58.97% | `results/test_C.md` |
| `AAMI_Cal` | **C1** | GBT | **57.76%** [51.58%, 64.06%] | **50.36%** [45.89%, 54.12%] | **0.5439** [0.4715, 0.6130] | 58.97% | `results/test_C.md` |
| `AAMI_Cal` | **C2m** | LR | **60.63%** [55.29%, 66.31%] | **54.48%** [51.24%, 58.00%] | **0.6444** [0.5833, 0.7102] | 58.97% | `results/test_C.md` |
| `AAMI_Cal` | **C2m** | GBT | **63.81%** [58.57%, 68.96%] | **59.76%** [55.41%, 64.42%] | **0.6635** [0.5982, 0.7348] | 58.97% | `results/test_C.md` |
| `AAMI_Cal` | **C2** | LR | **60.96%** [55.62%, 66.42%] | **54.89%** [51.72%, 58.32%] | **0.6520** [0.5912, 0.7169] | 58.97% | `results/test_C.md` |
| `AAMI_Cal` | **C2** | GBT | **64.65%** [59.31%, 69.70%] | **60.58%** [56.15%, 65.35%] | **0.6785** [0.6160, 0.7440] | 58.97% | `results/test_C.md` |

---

## 5. Benchmark Results

No benchmark.csv file present (external hardware benchmark not conducted; PySpark extraction throughput documented in `results/spark_features_C.md`).
