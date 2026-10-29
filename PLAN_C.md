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


---

## 8. Amendments

### Amendment C1 (Pre-Registered Before Any Study C Extraction or Results)
**Date:** 2026-10-29  
**Timing:** Decided and committed before any Study C feature extraction, data processing, model training, or results existed (no files matching `data/features_c*` or `results/dev_C*` exist).

1. **Heart Rate Metric (C1):**
   - Model **C1** is refined as **C0 + `hr_ibi`**, where:
     $$\text{hr\_ibi} = \frac{60.0}{\text{median}(\text{IBI})}$$
     evaluated from the median peak-to-peak inter-beat interval in seconds ($\text{IBI} = \Delta \text{peaks} / f_s$, in bpm).
   - *Rationale:* The original integer beat count metric ($HR = n_{\text{beats}} \times 6$) is quantized in coarse 6-bpm steps across 10-second segments. `hr_ibi` provides a continuous, physiological estimate of heart rate. $n_{\text{beats}}$ remains strictly as the signal quality field `beat_count`.

2. **Decay Ratio Definition Correction:**
   - `decay_ratio` is defined strictly as implemented in the code (`src/features_w.py` and `src/features_c.py`):
     $$\text{decay\_ratio} = \frac{d\_time}{r\_time} = \frac{t_{\text{turn}} - t_{\text{peak}}}{t_{\text{peak}} - t_{\text{foot}}}$$
     This is a **time ratio** (diastolic decay duration divided by systolic rise duration), exactly matching Study W, rather than an amplitude ratio.

3. **Refined Feature Ladder & Secondary Morphology Ablation (C2m):**
   - **C1:** $\text{C0} + \text{`hr\_ibi`}$
   - **C2m (Secondary Morphology Ablation):** $\text{C1} + \text{morphology only}$ (`systolic_rise_time`, `pulse_width`, `decay_ratio`), without any quality fields. This isolates the pure contribution of arterial pulse waveform geometry from signal quality artifacts.
   - **C2 (Full Pipeline):** $\text{C1} + \text{morphology} + \text{quality fields}$ (`systolic_rise_time`, `pulse_width`, `decay_ratio`, `beat_count`, `is_clipped`, `snr_ratio`, `ibi_sd`). Here, `ibi_sd` (standard deviation of inter-beat intervals in seconds) and `beat_count` are included as quality/regularity fields.
   - The stepwise gain is decomposed cleanly as:
     $$\Delta \text{Gain}(\text{Morphology Shape}) = C2m - C1$$
     $$\Delta \text{Gain}(\text{Signal Quality}) = C2 - C2m$$
     $$\Delta \text{Gain}(\text{Total Pipeline}) = C2 - C0$$


### Amendment C2 (Pre-Registered Before Development Model Training & Results)
**Date:** 2026-10-29  
**Timing:** Decided and committed before any model training, validation scoring, or development results were generated. Decided strictly on the basis of the label-free feature distribution sanity audit (`results/superseded/features_C_sanity_v1.md`); no Study C label analysis (`sbp`, `dbp`) or model result existed when this was decided.

1. **Inter-Peak Foot Definition & Physiological Morphology Extraction (`src/features_c.py` ONLY):**
   - *Rationale:* The previous fixed 300 ms foot search window preceding a systolic peak caused an artificial search-window ceiling at $0.304\text{ s}$ ($38\text{ samples}$ at $125\text{ Hz}$), truncating $18.7\%$ of Train and $19.4\%$ of CalFree segments, which also contaminated `pulse_width` and `decay_ratio`.
   - *Scope:* Changed strictly in `src/features_c.py` only (do not modify `src/fiducials.py` or any Study W code).
   - *Foot Definition:* Replace the 300 ms window by:
     $$\text{foot}_k = \operatorname{argmin}_{t \in [p_{k-1}, p_k]} \text{PPG}(t)$$
     where $p_k$ is the sample index of systolic peak $k$.
   - *Beat Qualification Criterion:* Beat $k$ is used for morphology extraction if and only if it has both an immediately preceding peak $p_{k-1}$ and a following peak $p_{k+1}$, and both adjacent inter-peak intervals satisfy physiological bounds:
     $$0.4\text{ s} \le \frac{p_k - p_{k-1}}{f_s} \le 1.5\text{ s} \quad \text{and} \quad 0.4\text{ s} \le \frac{p_{k+1} - p_k}{f_s} \le 1.5\text{ s}$$
     (corresponding to physiological heart rates of 40–150 bpm; no free parameter was tuned).
   - *Per-Beat Metrics:*
     - $\text{systolic\_rise\_time}_k = \frac{p_k - \text{foot}_k}{f_s}$
     - $\text{pulse\_width}_k = \text{FWHM at the 50% amplitude level between } y(\text{foot}_k) \text{ and } y(p_k)$
     - $\text{decay\_ratio}_k = \frac{(\text{foot}_{k+1} - p_k) / f_s}{\text{systolic\_rise\_time}_k}$, where $\text{foot}_{k+1} = \operatorname{argmin}_{t \in [p_k, p_{k+1}]} \text{PPG}(t)$
   - *Segment Aggregation:* Segment-level value is the median over all qualified beats in the segment. If no beat qualifies, the segment value is set to $\text{NaN}$ (handled downstream via training-median imputation plus a binary missing-indicator column as pre-registered in the plan).
   - `hr_ibi`, `ibi_sd`, `beat_count` and the Elgendi peak detector remain completely unchanged.

2. **Exclusion of Degenerate Feature (`is_clipped`):**
   - The label-free audit demonstrated that `is_clipped` is identically $0.0$ across all 523,080 segments in both Train and CalFree (zero rail saturation).
   - `is_clipped` is excluded from all model input feature ladders (C2) and reported as degenerate with zero variance.

3. **Wording Refinement & Cohort Quality Limitation:**
   - The descriptor "quality-aware" is formally removed from the claim.
   - The feature group $\{\text{beat\_count}, \text{ibi\_sd}, \text{snr\_ratio}\}$ is designated as "auxiliary segment fields".
   - *Documented Limitation:* VitalDB/PulseDB pre-filtering leaves no low-quality segments in the dataset ($\text{snr\_ratio} \ge 0.945$, zero clipping, $\ge 7\text{ beats}$ everywhere). Consequently, the signal quality filtering capability of the pipeline cannot be empirically evaluated on this cohort; additionally, $\text{beat\_count}$ identically equals $\text{hr} / 6$.

4. **Change Freezing Protocol:**
   - No other feature definition or model architecture change is permitted prior to the development cross-validation run. Any subsequent modification requires another dated amendment committed to git before creating the `plan-c-locked` tag.
