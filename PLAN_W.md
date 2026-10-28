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

1. **Baseline 0 (Majority Class Baseline):**
   - Constant prediction of the majority negative class ($SBP < 130$ mmHg).
   - **Actual CalFree Majority Rate:** **78.9% of segments** (45,432 / 57,600 segments; or 87.5% of subjects with median SBP $< 130$ mmHg).
   - Expected Balanced Accuracy: 50.0%; AUROC: 0.500.

2. **Baseline 1 (Heart Rate Within-Subject Comparator):**
   - Heart rate ($HR$ in beats per minute) derived from PPG beat count: $HR = n_{\text{beats}} \times 6$ (for 10-second windows).
   - Centered per subject:
     $$\Delta HR_{i, j} = HR_{i, j} - \text{median}(HR_i)$$
     where $\text{median}(HR_i)$ is the subject's median heart rate across all segments.
   - Evaluated as the non-morphology within-subject comparator via standardized logistic regression.

3. **Transductive Centering Disclosure & Implications:**
   - **Transductive Nature:** Computing per-subject medians $\text{median}(x_i)$ utilizes the full set of 400 segments across the recording session for each test subject.
   - **Implications:** Per-subject centering is transductive rather than purely inductive. In practical deployment, it requires an established subject-specific baseline or calibration window (e.g., initial baseline monitoring or batch processing of the operative case), and cannot be computed on a single isolated segment from an unseen patient without prior observations.

---

## 5. Fiducial Detection and Parameters

- **Algorithm:** Elgendi et al. (2013) two-moving-average method (*PLoS ONE*, doi:10.1371/journal.pone.0076585):
  - Systolic peak window $W_1 = 111\text{ ms}$ (14 samples at 125 Hz).
  - Beat window $W_2 = 667\text{ ms}$ (83 samples at 125 Hz).
  - Offset parameter $\beta = 0.02$ ($\alpha = \beta \times \bar{z}$).
  - Threshold formula: $\text{THR}_1 = \text{MA}_{\text{beat}} + \alpha$.
  - Filtering in source: bandpass 0.5–8 Hz; applied to channel 1 as stored without secondary filtering.
- **Foot & Turning Points:**
  - Foot: Minimum in the 300 ms window preceding each systolic peak.
  - Turning point: Minimum between consecutive systolic peaks.

---

## 6. Model Ladder & Stop Rules

- **Ladder Steps:**
  - **Step W0:** Majority class on CalFree (78.9%).
  - **Step W1:** Heart rate comparator (per-subject centered HR, logistic regression).
  - **Step W2:** Classic PPG morphology (per-subject centered systolic rise time, pulse width, decay ratio, random forest).
  - **Step W3:** Step W2 + signal quality filter (rejected segment share reported).
  - **Step W4:** Step W3 + GBT tuned on validation.

- **Stop / Claim Rules:**
  - **Primary Stop Rule:** To claim that PPG morphology adds information beyond heart rate for within-subject blood pressure tracking, the morphology model's validation AUROC and balanced accuracy must **strictly beat the heart-rate comparator's 95% bootstrap confidence interval lower bound** (and point estimate).
  - If morphology fails to beat the heart-rate comparator's CI lower bound, the outcome is formally reported as a **null result** (morphology provides no incremental within-subject information beyond heart rate).

---

## 7. Metrics and Statistical Protocol
- **Primary Headline Metrics:** AUROC and Balanced Accuracy with 95% subject-level bootstrap confidence intervals (2,000 whole-subject resamples on the test cohort).
- **Secondary Metric:** F1 score evaluated at the decision threshold optimized on the validation split.
- **Prevalence Context:** Plain accuracy must always be presented alongside the test positive prevalence (21.1% segments, 12.5% subjects at SBP $\ge 130$).

---

## 8. Known Limitations
- Per-subject centering is transductive and assumes access to the recording session.
- The CalFree test cohort has only 18 positive subjects at the 130 mmHg threshold (CIs will be wide).
- Operative interventions in surgical datasets introduce within-subject variance that may not reflect natural physiological pulse wave dynamics.

---

## 9. Claim Status
No result is claimed until produced by a script and recorded in `results/`. A null or negative result will be reported as such without alteration of the protocol.
