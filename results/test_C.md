# Study C Held-Out Test Evaluation: VitalDB AAMI_Cal Cohort

**Date:** 2026-10-30  
**Pre-Registration Specification:** `PLAN_C.md` (including Amendments 1-3)  
**Pre-Registered Plan Hash:** `e851b885f144022920476b9de238304a52fade4fd46157404a8b3378c8bbb0f3`  
**Code Git Commit:** `HEAD`  
**Lock Status:** Permanently locked at `results/.test_locked_C` (Single evaluation).  
**Test Dataset:** VitalDB AAMI_Cal (116 subjects, 70,212 segments).  
**Excluded Datasets:** Untouched test subsets strictly preserved.  

---

## 1. Study Setup & AAMI_Cal Cohort Characteristics

- **Subjects:** 116 unseen surgical/ICU patients
- **Segments Evaluated:** 70,212 (10-second non-overlapping PPG windows)
- **Primary Target:** $SBP \ge 120\text{ mmHg}$
- **Cohort Prevalence:** **41.03%** (28,811 positive / 70,212 total segments)
- **Majority-Class Baseline Rate:** **58.97%**
- **Decision Threshold:** Fixed standard threshold $0.5$ (pre-registered, zero tuning on test set)
- **Bootstrap Protocol:** Clustered subject-level bootstrap with 2,000 resamples (seed 42)

---

## 2. Primary Endpoint: $C2 - C0$ Accuracy Gain on AAMI_Cal

| Model Step | Classifier | Predictors | Accuracy [95% CI] | Balanced Acc [95% CI] | AUROC [95% CI] | Predicted Pos Rate | Majority Baseline |
|---|---|---|---|---|---|---|---|
| **C0** | Logistic Reg (regParam 0.01) | Age, Sex, BMI | **58.97%** [53.09%, 65.22%] | **50.00%** [50.00%, 50.00%] | **0.5704** [0.5025, 0.6341] | 0.00% | 58.97% |
| **C0** | GBT (depth 5, iter 100) | Age, Sex, BMI | **59.41%** [53.42%, 65.66%] | **51.66%** [49.30%, 54.55%] | **0.5584** [0.4871, 0.6307] | 6.50% | 58.97% |
| **C1** | Logistic Reg (regParam 0.01) | C0 + PPG Heart Rate (hr_ibi) | **58.96%** [53.09%, 65.20%] | **50.00%** [49.98%, 50.00%] | **0.5435** [0.4736, 0.6130] | 0.01% | 58.97% |
| **C1** | GBT (depth 5, iter 100) | C0 + PPG Heart Rate (hr_ibi) | **57.76%** [51.58%, 64.06%] | **50.36%** [45.89%, 54.12%] | **0.5439** [0.4715, 0.6130] | 8.67% | 58.97% |
| **C2m** | Logistic Reg (regParam 0.01) | C1 + Morphology Only (rise_time, pulse_width, decay_ratio) | **60.63%** [55.29%, 66.31%] | **54.48%** [51.24%, 58.00%] | **0.6444** [0.5833, 0.7102] | 14.86% | 58.97% |
| **C2m** | GBT (depth 5, iter 100) | C1 + Morphology Only (rise_time, pulse_width, decay_ratio) | **63.81%** [58.57%, 68.96%] | **59.76%** [55.41%, 64.42%] | **0.6635** [0.5982, 0.7348] | 25.66% | 58.97% |
| **C2** | Logistic Reg (regParam 0.01) | C1 + Morphology + Auxiliary Fields (beat_count, snr_ratio, ibi_sd) | **60.96%** [55.62%, 66.42%] | **54.89%** [51.72%, 58.32%] | **0.6520** [0.5912, 0.7169] | 15.30% | 58.97% |
| **C2** | GBT (depth 5, iter 100) | C1 + Morphology + Auxiliary Fields (beat_count, snr_ratio, ibi_sd) | **64.65%** [59.31%, 69.70%] | **60.58%** [56.15%, 65.35%] | **0.6785** [0.6160, 0.7440] | 25.42% | 58.97% |

### Paired Hypothesis Testing & Stepwise Gain Decomposition

| Comparison | Model Family | Paired $\Delta\text{Accuracy}$ [95% CI] | Paired $\Delta\text{AUROC}$ [95% CI] |
|---|---|---|---|
| **C1 vs C0 (Gain from Heart Rate)** | Logistic Regression | **-0.01%** [-0.02%, +0.00%] | **-0.0268** [-0.0621, +0.0092] |
| **C2m vs C1 (Gain from Pulse Morphology)** | Logistic Regression | **+1.67%** [-1.56%, +5.21%] | **+0.1009** [+0.0383, +0.1647] |
| **C2 vs C2m (Gain from Signal Quality Fields)** | Logistic Regression | **+0.32%** [-0.37%, +1.07%] | **+0.0076** [-0.0012, +0.0189] |
| **C2 vs C0 (Primary Endpoint: Total Pipeline Gain)** | Logistic Regression | **+1.99%** [-1.31%, +5.51%] | **+0.0817** [+0.0140, +0.1500] |
| **C1 vs C0 (Gain from Heart Rate)** | GBT | **-1.65%** [-7.07%, +1.92%] | **-0.0145** [-0.0706, +0.0388] |
| **C2m vs C1 (Gain from Pulse Morphology)** | GBT | **+6.05%** [+0.37%, +12.35%] | **+0.1196** [+0.0499, +0.1963] |
| **C2 vs C2m (Gain from Signal Quality Fields)** | GBT | **+0.83%** [-0.03%, +1.83%] | **+0.0151** [+0.0038, +0.0267] |
| **C2 vs C0 (Primary Endpoint: Total Pipeline Gain)** | GBT | **+5.24%** [+0.60%, +10.30%] | **+0.1202** [+0.0493, +0.1952] |

> [!NOTE]
> The exploratory benchmark goal of +4–5 percentage points accuracy gain is an aspiration, not a pass/fail criterion.

---

## 3. Verified Frozen Model Checksums

| Model Key | Expected & Verified SHA-256 Digest |
|---|---|
| **`c_imputer`** | `d0a8e3d67cad417d1115f6b957ccd0ba901e1ac378e78eeec06b7df8d16bd641` |
| **`C0_lr`** | `02490af84b823d77cfb05e416ea74982ad577cbb425cf1ebf38b532f5e7dd4b3` |
| **`C0_gbt`** | `fd5d7d29b0b18a592370d9d9885a5ed824efb5bfbdca043dc5e64cf907068af3` |
| **`C1_lr`** | `74012839bdd119f60619aee8c8ea07997125bb78eb2cf89e8556a22da948b0c5` |
| **`C1_gbt`** | `0ddc345f04e8aba38da90bbac8afaf26ce02c48bdeea78082a888931f840feb9` |
| **`C2m_lr`** | `c7032cc9204ad4225af592588d2334df1d94dd9bd9ba5f351876530d52c755d9` |
| **`C2m_gbt`** | `2869f2b83975e5335573904bb8c1a2fb5b412bbf3896537656c25da04265b54a` |
| **`C2_lr`** | `4def87939d3e73fdbfb28a2f0f94e669423c25a1f7d0fd2e7e1e0d5ddeb8fa8a` |
| **`C2_gbt`** | `8a6e9e1aea7c552455e4a9d1fa1b0883ceaab71a9ace1be018f8bc36af4f649f` |

---

## 4. Pre-Registration Protocol Adherence

- The single-pass test was evaluated strictly once without repeated attempts.
- No post-hoc tuning, threshold adjustments, or feature modifications were performed.
- Results reflect exact observed cross-subject test performance on unseen patients.
