# PLAN_C

Written after the null analyses W and N (results in `results/`). Committed before `AAMI_Cal` is read. This claim was formulated after those analyses.

## 1. Claim
From a single 10-s PPG segment plus demographics, a quality-aware PPG pipeline classifies elevated SBP on unseen subjects more accurately than demographics alone. The gain is decomposed into heart rate vs waveform shape.

---

## 2. Cohorts and Splits

- **Development Cohort:** VitalDB Train file (`data/parquet/train_file.parquet` / `data/raw/VitalDB_Train_Subset.mat`, 1,293 subjects, 465,480 segments). Evaluated via 5-fold cross-validation grouped by subject (`GroupKFold(5)`).
- **Extra Validation Cohort:** CalFree (`data/parquet/calfree_test.parquet` / `data/raw/VitalDB_CalFree_Test_Subset.mat`, 144 subjects, 57,600 segments). Already seen earlier during Study W; used for secondary cross-cohort validation without tuning.
- **Final Held-Out Test Cohort:** `VitalDB_AAMI_Cal_Subset.mat` (116 subjects, 70,212 segments), scored strictly once after models are frozen.
- **Excluded Cohorts:** `VitalDB_AAMI_Test_Subset.mat` and `VitalDB_CalBased_Test_Subset.mat` are **not used** and must remain untouched.

---

## 3. Target Labels

- **Primary Label:** Elevated SBP defined as $SBP \ge 120\text{ mmHg}$.
  - Primary rationale: SBP $\ge 120\text{ mmHg}$ has ~37.7% prevalence in the VitalDB Train dataset, which is closest to 50% among standard clinical cutoffs $\{120, 130, 140\}\text{ mmHg}$.
- **Sensitivity Analysis Label:** SBP $\ge 130\text{ mmHg}$ (stage 1 hypertension threshold).

---

## 4. Single-Segment Inductive Features (Zero Transduction)

All features are extracted strictly from the single 10-second PPG segment and patient demographics.
- **Strict Invariant:** **No per-subject normalization, centering, demeaning, or baseline subtraction.** Every segment is evaluated independently as an unseen observation.

### Feature Ladder
- **C0 (Demographics Baseline):**
  - `age` (years)
  - `sex` (binary indicator)
  - `bmi` ($\text{kg}/\text{m}^2$)
- **C1 (Demographics + PPG Heart Rate):**
  - Features from **C0**
  - PPG heart rate: `hr` (beats per minute, derived from Elgendi peak detector: $n_{\text{beats}} \times 6$)
- **C2 (Demographics + Heart Rate + Morphology + Quality):**
  - Features from **C1**
  - Morphology features:
    - `systolic_rise_time`: Mean duration (seconds) from foot to systolic peak.
    - `pulse_width`: Mean duration (seconds) of the pulse at 50% amplitude.
    - `decay_ratio`: Ratio of dicrotic notch / diastolic level to systolic amplitude.
  - Quality fields:
    - `beat_count`: Number of detected pulses in the 10-second segment.
    - `is_clipped`: Boolean flag indicating sensor saturation or ADC rail clipping.
    - `snr_ratio`: Relative signal-to-noise / energy ratio in the physiological band [0.5, 8] Hz.
- **C3 (Exploratory / Optional):**
  - End-to-end 1D-CNN operating directly on the raw 1,250-sample PPG waveform. Optional, exploratory only.

### Missing Feature Handling (Zero Segment Dropping)
- If peak detection fails or morphology features cannot be extracted:
  - Impute missing numeric features using the **median of that feature computed strictly on the training fold**.
  - Append a binary missing-indicator column (`is_missing_<feature>`) for each imputed feature.
- **Zero Rows Dropped:** All 70,212 segments of `AAMI_Cal` (and all development segments) are retained and scored.

---

## 5. Model Architectures & Hyperparameters

Fixed settings across all comparisons; no hyperparameter tuning or search:
1. **Logistic Regression:**
   - Standardized features (using training-fold mean and standard deviation).
   - L2 regularization with regularization parameter `regParam = 0.01` (corresponding to inverse regularizer $C = 100.0$).
2. **Gradient-Boosted Trees (GBT):**
   - `HistGradientBoostingClassifier` (or equivalent gradient boosting).
   - Fixed hyperparameters: `max_iter = 100` (`maxIter 100`), `max_depth = 5` (`maxDepth 5`), `random_state = 42`.
   - Native missing value handling or standard imputation.

---

## 6. Endpoints & Success Metrics

- **Primary Endpoint:**
  - Accuracy difference between model **C2** and model **C0** on `AAMI_Cal`:
    $$\Delta \text{Accuracy} = \text{Accuracy}(C2) - \text{Accuracy}(C0)$$
  - Evaluated via paired clustered bootstrap over subjects with 2,000 resamples (seed 42).
- **Secondary & Descriptive Metrics on AAMI_Cal:**
  - Balanced Accuracy: $\frac{\text{TPR} + \text{TNR}}{2}$
  - Area Under the ROC Curve (AUROC)
  - Majority-class accuracy (baseline rate of predicting most prevalent class)
  - AAMI cohort true prevalence of $SBP \ge 120\text{ mmHg}$
  - Stepwise decomposition: $\Delta \text{Accuracy}(C1 - C0)$ (gain from HR) vs $\Delta \text{Accuracy}(C2 - C1)$ (gain from waveform morphology & quality).
- **Benchmark Goal:**
  - A gain of +4 to 5 percentage points of accuracy above demographics alone ($C2 - C0 \ge +4.0\%$).
  - *Note:* This goal is an aspiration and empirical benchmark, not a pass/fail prerequisite for publication. Whatever is measured will be faithfully reported.

---

## 7. Protocol Rules & Pre-Registration Safeguards

1. **Test Lock:** `PLAN_C.md` must be committed to git before `VitalDB_AAMI_Cal_Subset.mat` is opened or read.
2. **Frozen Model Hash:** Models for C0, C1, C2 must be trained on the development data, serialized to `models/`, and their SHA-256 hashes recorded before testing.
3. **Pre-Scoring Lock:** A lock file `results/.test_locked_C` must be created prior to scoring `AAMI_Cal`.
4. **Single-Pass Evaluation:** `AAMI_Cal` will be scored strictly once.
5. **No Post-Hoc Alterations:** No feature definitions, model hyperparameters, decision thresholds, or imputation rules may be altered after `AAMI_Cal` is scored.
