# CalFree Test Cohort Evaluation: Within-Subject Model W

**Date:** 2026-10-29  
**Cohort:** `VitalDB_CalFree_Test_Subset.mat` (144 held-out subjects, 400 segments per subject, $N = 57,600$)  
**Protocol:** PLAN_W.md (Amendments W1, W2, W3)  
**Frozen Model SHA-256:** `92c9176b456e00630c45071afcdb1f5e0423d393ca2f6ad2ecf9357f5e640fdc`  
**Code Git Commit:** `HEAD`  
**Lock Status:** Test locked upon completion (`results/.test_locked_W`)  
**Bootstrap Parameters:** 2,000 subject-level resamples, seed 42  

---

## 1. Executive Summary & Stop Rule Verdict

### Verdict: CLAIM CONFIRMED
The active claim is **CONFIRMED as a prospective within-recording association with a 10-minute calibration window on this surgical cohort**.

- The primary prospective paired difference $\Delta\text{AUROC}(\text{Morphology} - \text{HR})$ 95% bootstrap CI strictly excludes zero: **+0.0973** [+0.0591, +0.1368].
- Scope limitation: This confirms an intraoperative within-recording association under a documented calibration window; it is NOT claimed as general cuffless PPG-to-BP prediction across patients.

---

## 2. Train–Test Centering Shift Disclosure

> [!NOTE]
> **Deployment Condition Tested:**  
> The frozen morphology model was trained on development data using full-session centered features ($\Delta_{\text{full}} X = X - \text{median}_{\text{session}}(X)$) and is evaluated on CalFree using prospective prefix-centered features ($\Delta_{\text{pref}} X = X - \text{median}_{t < 60}(X)$). This represents the pre-registered prospective deployment condition being tested (a 10-minute baseline calibration window), not an independent validation of retrospective batch centering.

---

## 3. Primary Endpoint: Prospective Calibration (Prefix Centering)

- **Calibration Baseline Window:** Segments $t \in [0, 59]$ (first 10 minutes = 60 segments per subject).
- **Evaluation Window:** Segments $t \in [60, 399]$ (subsequent 340 segments per subject, $144 \times 340 = 48,960$ segments).
- **Target Label:** Prospective excursion relative to prefix baseline ($y_{\text{pref}} = \mathbb{I}(SBP \ge \text{median}_{t < 60}(SBP))$).
- **Actual Positive Prevalence:** **49.67%** (Majority baseline: 50.0%)

| Model | AUROC [95% Bootstrap CI] | Accuracy (Prev: 49.7%) | Balanced Accuracy | Paired $\Delta\text{AUROC}$ vs. HR [95% CI] |
|---|---|---|---|---|
| **HR Comparator (Frozen)** | 0.5482 [0.5198, 0.5746] | 53.93% | 53.79% | — |
| **Morphology (Frozen)** | **0.6455** [0.6161, 0.6747] | 60.86% | 60.85% | **+0.0973** [+0.0591, +0.1368] |
| **Joint (HR + Morph)** | **0.6734** [0.6430, 0.7039] | 63.44% | 63.42% | **+0.1252** [+0.0927, +0.1593] |

---

## 4. Secondary Endpoint: Retrospective Full-Session Centering

- **Scope:** Retrospective session-median centering across all 400 segments per subject ($N = 57,600$).
- **Target Label:** Excursion relative to session median ($y_{\text{full}} = \mathbb{I}(SBP \ge \text{median}_{\text{session}}(SBP))$).
- **Actual Positive Prevalence:** **50.00%** (Majority baseline: 50.0%)

| Model | AUROC [95% Bootstrap CI] | Accuracy (Prev: 50.0%) | Balanced Accuracy | Paired $\Delta\text{AUROC}$ vs. HR [95% CI] |
|---|---|---|---|---|
| **HR Comparator (Frozen)** | 0.5460 [0.5174, 0.5734] | 53.44% | 53.44% | — |
| **Morphology (Frozen)** | **0.6460** [0.6175, 0.6741] | 60.85% | 60.85% | **+0.0999** [+0.0631, +0.1375] |
| **Joint (HR + Morph)** | **0.6719** [0.6423, 0.7009] | 62.82% | 62.82% | **+0.1259** [+0.0945, +0.1582] |

---

## 5. Execution Integrity & Provenance
- All features extracted strictly from PPG Channel 1 (`Signals[:, 1, :]`). Zero ABP inputs provided.
- Evaluated strictly once without tuning, retraining, or threshold adjustment.
- Test permanently locked at `results/.test_locked_W`.
