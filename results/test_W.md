# CalFree Test Cohort Evaluation: Within-Subject Model W

**Date:** 2026-10-29  
**Cohort:** `VitalDB_CalFree_Test_Subset.mat` (144 held-out subjects, 400 segments per subject, $N = 57,600$)  
**Protocol:** PLAN_W.md (Amendments W1, W2, W3)  
**Frozen Model SHA-256:** `92c9176b456e00630c45071afcdb1f5e0423d393ca2f6ad2ecf9357f5e640fdc`  
**Execution Runner Commit:** `07c3d8b`  
**Pre-Registration Amendment W3 Commit:** `a424a3b`  
**Lock Status:** Test permanently locked (`results/.test_locked_W`)  
**Bootstrap Parameters:** 2,000 subject-level clustered resamples across all 144 test subjects, fixed seed 42  
**Classification Threshold:** Fixed default probability threshold of $0.50$ (frozen from model training; zero post-hoc threshold tuning)  

---

## 1. Executive Summary & Stop Rule Verdict

### Verdict
**The pre-registered stop rule (W3, primary endpoint) was met on the single CalFree evaluation.**

- **Primary Endpoint Result:** The paired difference $\Delta\text{AUROC}(\text{Morphology} - \text{HR})$ 95% subject-level bootstrap confidence interval strictly excludes zero:
  $$\Delta\text{AUROC} = \mathbf{+0.0973} \quad [95\%\text{ CI: } +0.0591, +0.1368]$$
- **Scope Limitation:** This outcome indicates that pulse contour morphology timing features track acute arterial blood pressure excursions within each recording session when calibrated against a 10-minute subject baseline window in this surgical cohort. It is **NOT** a general confirmation of PPG-to-BP prediction across unseen individuals without baseline calibration.

---

## 2. Primary Endpoint: Prospective Calibration (Prefix Centering)

- **Calibration Baseline Window:** Segments $t \in [0, 59]$ (initial 10 minutes = 60 segments per subject).
- **Evaluation Window:** Segments $t \in [60, 399]$ (subsequent 340 segments per subject, $144 \times 340 = 48,960$ segments across all 144 test subjects).
- **Subject Baselines:** Set strictly from prefix: $\mu_{\text{pref}}(X_i) = \text{median}_{t < 60}(X_{i, t})$.
- **Features:** $\Delta_{\text{pref}} X_{i, t} = X_{i, t} - \mu_{\text{pref}}(X_i)$ for PPG timing features (`systolic_rise_time`, `pulse_width`, `decay_ratio`) and `hr`.
- **Target Label:** Prospective excursion relative to prefix baseline ($y_{\text{pref}} = \mathbb{I}(SBP_{i, t} \ge \mu_{\text{pref}}(SBP_i))$, ties to 0).
- **Actual Positive Prevalence:** **49.67%** (24,316 positive / 48,960 total segments; baseline majority rate: 50.33%).
- **Decision Threshold:** Fixed at $p \ge 0.50$.

### Primary Endpoint Results Table

| Model | AUROC [95% Bootstrap CI] | Balanced Accuracy | Raw Accuracy (Prev: 49.7%) | Predicted Positive Rate (Thresh: 0.50) | Paired $\Delta\text{AUROC}$ vs. HR [95% CI] |
|---|---|---|---|---|---|
| **HR Comparator (Frozen)** | 0.5482 [0.5198, 0.5746] | 53.79% | 53.93% | 30.01% | — |
| **Morphology (Frozen `w_frozen.pkl`)** | **0.6455** [0.6161, 0.6747] | **60.85%** | 60.86% | 48.56% | **+0.0973** [+0.0591, +0.1368] |

### Metric and Accuracy Framing
The morphology model achieves 60.85% balanced accuracy and 60.86% raw accuracy against an empirical ~50% median-split baseline (approximately 11 percentage points above chance). This is an excursion-classification performance on a balanced task, not an absolute hypertension detection rate against a skewed majority baseline. At threshold 0.50, the predicted positive rate for morphology (48.56%) is closely aligned with the true label prevalence (49.67%), whereas the HR comparator substantially under-predicts positive excursions (30.01%), driving down its balanced accuracy (53.79%).

---

## 3. Secondary Endpoint: Retrospective Full-Session Centering

- **Protocol:** Demeaned by full session median across all 400 segments per subject ($N = 57,600$).
- **Target Label:** Excursion relative to session median ($y_{\text{full}} = \mathbb{I}(SBP_{i, t} \ge \text{median}_{\text{session}}(SBP_i))$, ties to 0).
- **Actual Positive Prevalence:** **50.00%** (28,800 positive / 57,600 total segments; baseline majority rate: 50.00%).
- **Decision Threshold:** Fixed at $p \ge 0.50$.

### Secondary Endpoint Results Table

| Model | AUROC [95% Bootstrap CI] | Balanced Accuracy | Raw Accuracy (Prev: 50.0%) | Predicted Positive Rate (Thresh: 0.50) | Paired $\Delta\text{AUROC}$ vs. HR [95% CI] |
|---|---|---|---|---|---|
| **HR Comparator (Frozen)** | 0.5460 [0.5174, 0.5734] | 53.44% | 53.44% | 29.73% | — |
| **Morphology (Frozen `w_frozen.pkl`)** | **0.6460** [0.6175, 0.6741] | **60.85%** | 60.85% | 47.47% | **+0.0999** [+0.0631, +0.1375] |

### Interpretation of Centering Comparison
The prospective prefix-centered AUROC (0.6455) and retrospective full-session centered AUROC (0.6460) are virtually identical ($\Delta = -0.0005$). This similarity demonstrates that the two centering choices produce nearly identical relative rankings on this cohort. However, this does **not** indicate an unconstrained prospective capability: prefix centering remains a within-recording design that requires an uninterrupted 10-minute baseline prefix from the same physical monitoring session.

---

## 4. Secondary & Exploratory Analysis: Joint Model (HR + Morphology)

The joint model (`joint_frozen.pkl`, combining `delta_hr` with the three centered morphology features) was evaluated as a secondary exploratory sensitivity check:

### Joint Model Performance Table

| Evaluation Setup | Joint AUROC [95% CI] | Balanced Accuracy | Raw Accuracy | Predicted Positive Rate | Joint vs. HR $\Delta\text{AUROC}$ [95% CI] | Joint vs. Morphology Alone $\Delta\text{AUROC}$ [95% CI] |
|---|---|---|---|---|---|---|
| **Primary (10-min Prefix)** | 0.6734 [0.6430, 0.7039] | 63.42% | 63.44% | 46.99% | +0.1252 [+0.0927, +0.1593] | **+0.0279** [+0.0159, +0.0401] |
| **Secondary (Full Session)** | 0.6719 [0.6423, 0.7009] | 62.82% | 62.82% | 45.98% | +0.1259 [+0.0945, +0.1582] | **+0.0259** [+0.0147, +0.0374] |

### Signal Attribution & Incremental Significance
- The joint model's comparison against HR alone (+0.1252) is **not** the pre-registered test. Because the HR comparator is weak (AUROC 0.5482), morphology alone accounts for $+0.0973$ (~78%) of the total joint gain over HR.
- Adding HR to morphology provides a small incremental gain of **+0.0279** on the primary endpoint, whose paired 95% bootstrap CI ([+0.0159, +0.0401]) strictly excludes zero. While this increment is statistically measurable, the predictive signal resides predominantly in the pulse contour timing intervals rather than heart rate frequency.

---

## 5. Complete Protocol History & Provenance

To maintain full transparency, the sequence of scientific iterations leading to this evaluation is documented below:

1. **Initial Pre-Registration (`PLAN.md` - Between-Subject Claim):**
   - Originally sought to predict absolute arterial blood pressure and classify hypertension across unseen subjects using pulse transit time (PTT) and pulse arrival time (PAT).
   - **Detector Agreement Failures (Amendments A2 & A3):**
     - Under Amendment A2, validation beat-count agreement within $[0, 60)$ lag window was **68.93%** (falling below the $\ge 80.0\%$ pre-registered threshold; `results/superseded/detector_agreement.md`).
     - Under Amendment A3, a symmetric $[-37, 37)$ lag window yielded **44.09%** validation agreement (falling below the $\ge 80.0\%$ threshold due to systematic search boundary clipping; `results/superseded/detector_agreement_A3.md`).
   - **Baseline Sanity Failure:**
     - On the validation split (`results/superseded/validation_baselines.md`), demographic baselines produced chance-level discrimination: Age alone AUROC was **0.4981** [95% CI: 0.4472, 0.5484], and combined Demographics (Age + Sex + BMI) achieved AUROC **0.5003** [95% CI: 0.4508, 0.5494].
     - On the training split sanity check (`results/superseded/baseline_sanity.md`), subject-level Age–SBP correlation achieved AUROC **0.5240** for distinguishing subjects with median $SBP \ge 130$ mmHg.
   - **Test Set Cancellation (Amendment A4):** Between-subject evaluation on CalFree was formally **cancelled** to prevent evaluating a broken cross-subject model.
2. **Re-Framing to Within-Subject Pulse Contour Tracking (`PLAN_W.md`):**
   - Reframed the core hypothesis: Does PPG morphology track acute *within-subject* arterial pressure variations relative to a patient's own baseline, beyond heart rate alone?
   - **Amendment W1:** Pre-registered the within-subject heart rate comparator ($\Delta HR$) and the paired subject-level bootstrap stop rule ($\Delta\text{AUROC} > 0$).
   - **Amendment W2:** Clarified the beat agreement gate as a cohort-level quality check (all segments retained for modeling). The gate passed at **96.92%** on the training split (`results/gate_within_subject_agreement.md`).
3. **Development-Only Controls & Sensitivity Checks:**
   - **Label Permutation:** Within-subject label shuffling collapsed all model AUROCs to 0.499, ruling out preprocessing or cross-fold data leakage.
   - **Recording Drift:** Evaluating segment recording position alone yielded AUROC 0.500 (chance), ruling out temporal drift.
   - **Nonlinear HR:** Gradient-boosted trees on HR alone reached AUROC 0.555, confirming linear HR was not failing simply due to model capacity.
   - **PPG Amplitude Invariance:** Confirmed that segment amplitude range $R_{\text{PPG}} \equiv 1.0000 \pm 10^{-16}$ across all segments (zero scale variance), resolving the earlier "12.35% peak below 1.0" observation as an IEEE 754 precision artifact from MATLAB normalization.
   - **Prospective Prefix Control:** Evaluated 10-minute prefix calibration on development data, yielding AUROC 0.675 and confirming the signal survives prospective baseline subtraction.
4. **Pre-Registration Amendment W3:**
   - Pre-registered the primary prefix-centered endpoint, secondary full-session endpoint, paired bootstrap difference test, and single-pass lock policy prior to touching CalFree.

---

## 6. Documented Limitations

1. **Single Clinical Cohort:** Evaluated entirely within the VitalDB surgical and ICU dataset (derived from PulseDB). The findings have not been evaluated on an independent external dataset or ambulatory cohort.
2. **Train–Test Centering Shift:** The frozen morphology model was trained on development features centered using full-session medians ($\Delta_{\text{full}} X$) and tested on CalFree features centered using prospective prefix medians ($\Delta_{\text{pref}} X$). While pre-registered, this represents a shifted deployment condition rather than an independent validation of retrospective centering.
3. **Invasive Catheter Target:** The target is invasive radial/femoral arterial-line SBP under general anesthesia and operative monitoring, not non-invasive auscultatory cuff BP. Labels reflect excursions relative to each patient's baseline, not absolute blood pressure values.
4. **Prevalence and Metric Baseline:** The within-subject classification task operates against an empirical ~50% prevalence baseline (median-split excursion), which is an easier baseline than the original 87% majority rate for absolute hypertension ($SBP \ge 130$). Consequently, balanced accuracy and AUROC are the primary metrics; raw accuracy must always be interpreted alongside the ~50% baseline rate.
5. **Sample Size & Precision:** While the CalFree cohort contains 57,600 segments, they are clustered across 144 subjects. Subject-level bootstrap confidence intervals are reasonably wide ([+0.0591, +0.1368]), reflecting subject-to-subject heterogeneity in pulse contour coupling.

---

## 7. Execution Integrity & Numerical Audit Verification

- **Code Provenance:** All CalFree features were computed strictly from PPG Channel 1 (`Signals[:, 1, :]`) via `src/features_w.py`. Zero ABP inputs were provided to the feature extractor or models.
- **Single-Pass Execution:** Executed via `src/test_w.py` with pre-scoring lock acquired at `results/.test_locked_W`.
- **Runner Code Diff:** `git diff 07c3d8b 33895a3 -- src/test_w.py` produces an **empty diff**, verifying that the test runner script is bit-for-bit identical between the pre-run commit (`07c3d8b`) and the committed state.
- **Numerical Audit Verification:** A pairwise diff of all numerical results between the initial post-test commit (`31cd178`) and this finalized report confirms that **every AUROC, confidence interval bound, raw accuracy, balanced accuracy, predicted positive rate, and prevalence value is identical**. Editorial revisions refined the framing and disclosed context without altering a single metric.
- **Permanent Lock Enforced:** The test is permanently locked against re-execution at `results/.test_locked_W`. Zero further test runs or exploratory parameter adjustments are permitted on CalFree under protocol PLAN_W.md.
