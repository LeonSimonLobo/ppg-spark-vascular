# Pre-Registration Plan: Within-Subject PPG Morphology Study (PLAN_W.md)

## 1. Research Question
On held-out subjects (VitalDB CalFree test cohort), does quality-aware PPG morphology track acute within-subject variations in arterial blood pressure (classifying hypertensive-range SBP $\ge 130$ mmHg) beyond what is explained by heart rate alone?

---

## 2. Target and Label
- **Per-Segment Label:** Binary classification of $SBP \ge 130$ mmHg (primary). Sensitivity checks at 120 and 140 mmHg.
- **Label Source:** Invasive arterial-line catheter BP from surgical and ICU patients in VitalDB.
- **Scope:** Proxy for acute vascular and hemodynamic state under operative monitoring, not a chronic ambulatory hypertension diagnosis.

---

## 3. Data and Provenance
- **Training Cohort:** `VitalDB_Train_Subset.mat` / `data/parquet/train_file.parquet` (465,480 segments, 1,293 subjects; 1,034 train / 259 validation, seed 42 split).
- **Test Cohort:** `VitalDB_CalFree_Test_Subset.mat` / `data/parquet/calfree_test.parquet` (57,600 segments, 144 subjects; exactly 400 segments per subject).
- **Subject Overlap:** 0 subjects between train and test.
- **Signals:** PPG channel 1 (125 Hz, 10-second non-overlapping segments, 1,250 samples per segment).

---

## 4. Within-Subject Baselines & Comparators

1. **Baseline W0 (Majority Class Baseline):**
   - Constant prediction of the majority negative class ($SBP < 130$ mmHg).
   - **CalFree Majority Rate:** Computed directly from the CalFree labels ($SBP < 130$ mmHg) as **78.9375%** (45,468 / 57,600 segments; positive prevalence of $SBP \ge 130$ is 21.0625% [12,132 / 57,600 segments]).
   - If evaluated on within-subject demeaned labels ($SBP \ge \text{median}(SBP_i)$), the majority rate is exactly 50.0000% (28,800 / 57,600 segments).
   - Expected Balanced Accuracy: 50.0%; AUROC: 0.500.

2. **Baseline W1 (Heart Rate Within-Subject Comparator):**
   - Heart rate ($HR$ in beats per minute) derived from PPG beat count: $HR = n_{\text{beats}} \times 6$ (for 10-second windows).
   - Centered per subject:
     $$\Delta HR_{i, j} = HR_{i, j} - \text{median}(HR_i)$$
     where $\text{median}(HR_i)$ is the subject's median heart rate across all segments.
   - Evaluated as the non-morphology within-subject comparator via standardized logistic regression.

3. **Transductive Centering Disclosure & Implications:**
   - **Transductive Nature:** Computing per-subject medians $\text{median}(x_i)$ utilizes the full set of 400 segments across the recording session for each test subject.
   - **Implications:** Per-subject centering is transductive rather than purely inductive. In practical deployment, it requires an established subject-specific baseline or calibration window (e.g., initial baseline monitoring or batch processing of the operative case), and cannot be computed on a single isolated segment from an unseen patient without prior observations.

---

## 5. Fiducial Detection and Documented Filter Deviation

- **Algorithm:** Elgendi et al. (2013) two-moving-average method (*PLoS ONE*, doi:10.1371/journal.pone.0076585):
  - Systolic peak window $W_1 = 111\text{ ms}$ (14 samples at 125 Hz).
  - Beat window $W_2 = 667\text{ ms}$ (83 samples at 125 Hz).
  - Offset parameter $\beta = 0.02$ ($\alpha = \beta \times \bar{z}$).
  - Threshold formula: $\text{THR}_1 = \text{MA}_{\text{beat}} + \alpha$.
- **Filtering Implementation and Documented Deviation:**
  - Applied directly to Signals channel 1 as stored, with no secondary filtering.
  - **Documented Deviation:** In the original Elgendi et al. (2013) paper, a zero-phase second-order Butterworth filter (bandpass 0.5–8 Hz) was specified. In this study, this filter is replaced by the upstream source filter applied in PulseDB (4th-order Chebyshev-II filter, bandpass [0.5, 8] Hz, as documented in Section 2.4.2 of Wang et al., 2023). No secondary Butterworth filter is applied to avoid cascaded phase distortion and double-filtering attenuation.
- **Foot & Turning Points:**
  - Foot: Minimum in the 300 ms window preceding each systolic peak.
  - Turning point: Minimum between consecutive systolic peaks.

---

## 6. Model Ladder & Stop Rules

- **Ladder Steps:**
  - **Step W0:** Majority class on CalFree (78.9375%).
  - **Step W1:** Heart rate comparator (per-subject centered HR, logistic regression).
  - **Step W2:** Classic PPG morphology (per-subject centered systolic rise time, pulse width, decay ratio, random forest).
  - **Step W3:** Step W2 + signal quality filter (rejected segment share reported).
  - **Step W4:** Step W3 + GBT tuned on validation.

- **Stop / Incremental Criterion:**
  - **Paired Incremental Criterion:** To claim that PPG morphology adds information beyond heart rate for within-subject blood pressure tracking, the 95% bootstrap confidence interval of $\Delta \text{AUROC} = \text{AUROC}(\text{morphology}) - \text{AUROC}(\text{HR})$, paired on the same bootstrap subject resamples (2,000 whole-subject resamples), must strictly exclude zero with morphology exceeding HR ($\Delta \text{AUROC}_{\text{lower}} > 0$).
  - The previous CI lower-bound rule is removed.
  - If the paired 95% CI of $\Delta \text{AUROC}$ includes zero or is negative, the outcome is formally reported as a **null result** (morphology provides no statistically significant within-subject gain beyond heart rate).

---

## 7. Gate on Training-Split Subjects Only (PROMPT 2)

Before any feature extraction or evaluation on held-out test subjects, the within-subject pipeline must pass an empirical gate evaluated strictly on training-split subjects (`split == 'train'` of `VitalDB_Train_Subset.mat`):
1. **Physiological Beat Count Plausibility:** $\ge 95\%$ of training-split segments must yield heart rates within physiological bounds (30 to 150 bpm, corresponding to 5 to 25 detected beats per 10 s segment).
2. **Within-Subject Heart Rate Signal:** Per-subject centered heart rate ($\Delta HR$) must exhibit a positive association with within-subject SBP variations:
   - Positive Spearman rank correlation: $r_s(\Delta HR, \Delta SBP) > 0$.
   - Positive logistic regression coefficient ($\beta_{HR} > 0$) for predicting $SBP \ge 130$ mmHg.
   - Classification AUROC $> 0.50$.
3. **Gate Decision Rule:** If either criterion fails on training-split subjects, execution stops immediately, and no morphology models or test evaluations are run.
4. **Segment Inclusion Policy:** All segments included; the gate is a cohort-level check, not a per-segment filter. All 372,240 training-split segments (and all 57,600 CalFree test segments) are retained in the modeling dataset for Steps W0 through W2. Segment quality filtering is tested strictly at Step W3 (signal quality filter) per the pre-registered model ladder.

---

## 8. Metrics and Statistical Protocol
- **Primary Headline Metrics:** AUROC and Balanced Accuracy with 95% subject-level bootstrap confidence intervals (2,000 whole-subject resamples on the test cohort).
- **Secondary Metric:** F1 score evaluated at the decision threshold optimized on the validation split.
- **Prevalence Context:** Plain accuracy must always be presented alongside the test positive prevalence (21.06% segments, 12.5% subjects at SBP $\ge 130$).

---

## 9. Known Limitations
- Per-subject centering is transductive and assumes access to the recording session.
- The CalFree test cohort has only 18 positive subjects at the 130 mmHg threshold (CIs will be wide).
- Operative interventions in surgical datasets introduce within-subject variance that may not reflect natural physiological pulse wave dynamics.

---

## 10. Claim Status
No result is claimed until produced by a script and recorded in `results/`. A null or negative result will be reported as such without alteration of the protocol.


---

## 11. Amendment W3: Pre-Registration of CalFree Test Evaluation Protocol

**Date:** 2026-10-29  
**Trigger:** Successful completion and passing of all development gates:
1. Empirical beat agreement gate: 96.92% (PASS, $\ge 80\%$, `eba57e3`).
2. Robustness controls: label permutation collapsed to 0.499, time drift chance at 0.500, nonlinear HR GBT at 0.555 (`8420ca1`).
3. Prospective prefix-centering control: morphology AUROC = 0.6754, joint = 0.7027 under 10-minute prefix calibration (PASS, `b62d99d`).
4. Amplitude-scale invariance audit: $R_{\text{PPG}} \equiv 1.0000 \pm 10^{-16}$, 0.0% peak below 0.9999; 12.35% resolved as IEEE 754 precision artifact (PASS, `b62d99d`).
5. Frozen model hash verified: `models/w_frozen.pkl` (`92c9176b456e00630c45071afcdb1f5e0423d393ca2f6ad2ecf9357f5e640fdc`).

### 11.1 Primary Endpoint (Prospective Calibration)
- **Deployment Protocol:** Prefix-centered within-subject evaluation on held-out test subjects (`VitalDB_CalFree_Test_Subset.mat`, 144 subjects, 400 segments per subject).
- **Calibration Prefix Window:** First 60 segments ($t \in [0, 59]$, 10 minutes) per subject used strictly to set subject baselines:
  $$\mu_{\text{pref}}(X_i) = \text{median}_{t < 60}(X_{i, t})$$
- **Evaluation Window:** Subsequent 340 segments ($t \in [60, 399]$, 56.7 minutes) per subject ($144 \times 340 = 48,960$ test segments).
- **Features:** $\Delta_{\text{pref}} X_{i, t} = X_{i, t} - \mu_{\text{pref}}(X_i)$ for PPG timing features (`systolic_rise_time`, `pulse_width`, `decay_ratio`) and `hr`.
- **Primary Target Label:** Prospective hypertensive-range excursion relative to prefix baseline:
  $$y_{\text{pref}, i, t} = \mathbb{I}(SBP_{i, t} \ge \mu_{\text{pref}}(SBP_i)) \quad (\text{ties to } 0)$$
- **Model and Refit/Frozen Status:** **FROZEN (zero refitting, zero tuning).** Evaluated strictly using `models/w_frozen.pkl` (SHA-256: `92c9176b456e00630c45071afcdb1f5e0423d393ca2f6ad2ecf9357f5e640fdc`), trained on the 372,240 development segments. No retraining, fine-tuning, or threshold modification is permitted.

### 11.2 Secondary Endpoint (Full-Session Centering)
- **Retrospective Setup:** Full-session median centering across all 400 segments per subject ($N = 57,600$ segments).
- **Target Label:** Excursion relative to session median $y_{\text{full}, i, t} = \mathbb{I}(SBP_{i, t} \ge \text{median}_{\text{session}}(SBP_i))$ (ties to 0).
- **Model:** Frozen model `models/w_frozen.pkl` evaluated on full-session centered features $\Delta_{\text{full}} X_{i, t} = X_{i, t} - \text{median}_{\text{session}}(X_i)$.

### 11.3 Paired-Difference Hypothesis Test Against HR
- **Primary Statistic:** Paired difference in AUROC:
  $$\Delta \text{AUROC} = \text{AUROC}(\text{Morphology}) - \text{AUROC}(\text{HR})$$
  and $\Delta \text{AUROC} = \text{AUROC}(\text{Joint}) - \text{AUROC}(\text{HR})$.
- **Bootstrap Protocol:** Clustered subject-level bootstrap with 2,000 whole-subject resamples across all 144 test subjects, evaluated with fixed seed `42`.
- Both morphology and HR models are evaluated on identical bootstrap resamples to form the paired difference empirical distribution and 95% percentile confidence intervals ($[2.5\%, 97.5\%]$).

### 11.4 Stop Rule
- **Existing PLAN_W Stop Rule:** The paired 95% bootstrap confidence interval of $\Delta \text{AUROC} = \text{AUROC}(\text{Morphology}) - \text{AUROC}(\text{HR})$ must strictly exclude zero with morphology exceeding HR ($\Delta \text{AUROC}_{\text{lower}} > 0$).
- **[Added Criterion - Amendment W3]:**
  - If the paired 95% bootstrap CI on the primary prospective endpoint (prefix-centered) strictly excludes zero on CalFree ($\Delta \text{AUROC}_{\text{lower}} > 0$), the active claim is **CONFIRMED** as a prospective within-recording association in this surgical cohort with a documented 10-minute calibration window.
  - If the paired 95% bootstrap CI includes zero or is negative ($\Delta \text{AUROC}_{\text{lower}} \le 0$), the prospective within-subject claim is **REFUTED on the test cohort**, the run stops immediately, and the outcome is formally reported as a **development-only finding** without exploring alternative models or post-hoc refits.

### 11.5 Feature Computation Code & Verification Hash
- **PPG Input:** Evaluated strictly from PPG channel 1 (`Signals[:, 1, :]`) of `VitalDB_CalFree_Test_Subset.mat`. Zero ABP inputs are read or provided to the feature extractor.
- **Code Provenance:** Feature extraction is executed via `src/features_w.py`, committed at `e5493d8` (and validated at `b62d99d`).
- **Single Execution Script:** All operations (CalFree feature extraction, SHA-256 verification of `models/w_frozen.pkl`, single-pass scoring, bootstrap CIs, markdown generation to `results/test_W.md`, and creation of `results/.test_locked_W`) are bundled in `src/test_w.py`.
