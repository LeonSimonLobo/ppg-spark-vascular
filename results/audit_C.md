# Study C Code & Protocol Audit against PLAN_C.md

**Date:** 2026-10-29  
**Repository:** `D:\DS_RESEARCH\ppg-spark-vascular`  
**Execution Invariant:** No models trained. No feature extractions run. No AAMI rows or test results read. `src/test_c.py` not executed. `results/test_W.md` not opened.

---

## 1. Git State, Tags, and Pre-Registration Verification

### 1.1 Git Status
```
On branch main
Your branch is ahead of 'origin/main' by 37 commits.
  (use "git push" to publish your local commits)

Untracked files:
  (use "git add <file>..." to include in what will be committed)
	results/inventory.md
	src/extract_features_c.py
	src/features_c.py
	src/run_dev_c.py
	src/test_c.py
	src/train_c.py
	tests/test_features_c.py
	tests/test_train_c.py

nothing added to commit but untracked files present (use "git add" to track)
```

### 1.2 Git Log (Last 15 Commits)
```
584c5d4 Pre-registration: PLAN_C cross-subject SBP classification on AAMI_Cal
171a2b2 Exploratory: development model comparison (no CalFree access)
fe047ec Audit: reconcile protocol history figures, verify numeric identity to 31cd178, and add joint paired CI
33895a3 Lock file: record exact execution runner commit 07c3d8b
9da1ec5 Results: calibrate test_W.md with precise stop rule wording, full protocol history, and limitations
31cd178 CalFree evaluation: primary prospective endpoint confirmed (test locked)
07c3d8b Test runner hardening: lock before scoring, hash abort, and isolated CalFree scope
fc48d9d Test execution script: single-pass CalFree evaluation runner and unit tests (Amendment W3)
a424a3b Amendment W3: pre-register CalFree primary prospective and secondary retrospective endpoints
b62d99d Controls: prospective prefix-centering, amplitude-scale invariance, and HR reframing
8420ca1 Robustness controls: within-subject label permutation and nonlinear HR GBT
d4a51f4 Nested models: HR, HR+morphology, and HR-residualized morphology (dev data)
43ed07c Time-only comparator, drift audit, transduction disclosure, and models hash record
5648d52 Development CV for within-subject model W (no test data)
b20670f Amendment W2: segment inclusion policy (all segments included; gate is cohort check)
```

### 1.3 Git Tag List
```
(empty - no tags currently exist)
```
*Note: Git tag `plan-c-locked` must be created prior to test execution.*

### 1.4 Full Text of `PLAN_C.md` and Comparison Against Specification
- **SHA-256 Digest:** `9be7f444e02097dccba501fbe7fb39a02f2295b19095ec36f96cbedb304afc79`
- **Committed at:** `584c5d4` (prior to reading any `AAMI_Cal` data)
- **Specification Comparison:** **IDENTICAL.** Every requirement from the prompt specification (claim, 1,293 VitalDB Train subjects, 5-fold CV by subject, CalFree extra validation, AAMI_Cal final test, SBP $\ge 120$ primary, SBP $\ge 130$ sensitivity, C0/C1/C2 feature ladder, median imputation with indicator columns, Logistic Regression with regParam 0.01, GBT maxIter 100 maxDepth 5, paired bootstrap over subjects, +4-5 accuracy benchmark goal) is fully and faithfully incorporated.

```markdown
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
```

---

## 2. PASS/FAIL Audit Table

| Item | Requirement Summary | Status | Files and Line Numbers | Detailed Audit Findings |
|:---|:---|:---:|:---|:---|
| **a** | Features come from one segment only: no per-subject centering, no baseline window, no `delta_` columns, no use of other segments. | **PASS** | `src/features_c.py:53-162`, `191-216`<br>`src/extract_features_c.py:34-82` | All extracted features operate strictly on the 10-second segment ($N=1250$ points) and patient-level demographics. Zero `delta_` columns, zero subject demeaning, and zero cross-segment aggregation. |
| **b** | Heart rate calculation: add `hr_ibi = 60 / median(IBI)` and `ibi_sd`; define `C1 = C0 + hr_ibi`. | **FAIL** | `src/features_c.py:28-30`, `88`, `154`<br>`src/extract_features_c.py:43`, `85`<br>`src/train_c.py:10`<br>`src/test_c.py:162`, `175` | Currently, heart rate is computed solely as `hr = float(n_pks * 6.0)`. Neither `hr_ibi` nor `ibi_sd` is computed, and `C1` is currently defined as `C0 + hr`.<br>**Fix:** Compute inter-beat intervals $\text{IBI} = \Delta \text{peaks} / f_s$. When $\ge 2$ peaks: $\text{hr\_ibi} = 60.0 / \text{median}(\text{IBI})$ and $\text{ibi\_sd} = \text{std}(\text{IBI})$. Set `C1 = C0 + hr_ibi` (with indicator). Move `hr` and `ibi_sd` to `C2`. |
| **c** | Failed extraction is NaN plus missing indicator, no rows dropped, imputer fit on training folds only. | **PASS** | `src/features_c.py:76-86`, `148-150`, `243-274`<br>`src/train_c.py:176-180` | Failures in peak detection or morphology yield `np.nan`. `PlanCImputer` fits medians strictly on training fold data, imputes missing values, and appends `is_missing_<col>` (1.0 for missing, 0.0 for valid). No segments are dropped. |
| **d** | Label is per segment: SBP $\ge 120$ (sensitivity: $\ge 130$). | **PASS** | `src/run_dev_c.py:50-51`<br>`src/train_c.py:168`<br>`src/test_c.py:150-151` | Target labels are strictly defined per segment as binary threshold indicators: $y_{120} = \mathbb{I}(SBP \ge 120)$ and $y_{130} = \mathbb{I}(SBP \ge 130)$. |
| **e** | Development cohort uses `file_split == 'train_file'` (1,293 subjects); CalFree uses `file_split == 'calfree_test'`. Selection must use `file_split`, never `split`. | **FAIL** | `src/extract_features_c.py:108-110`, `120-124`<br>`src/run_dev_c.py:45` | `extract_features_c.py` read `split` instead of `file_split` from `train_file.parquet`. It did not explicitly filter or assert `file_split == 'train_file'` and `file_split == 'calfree_test'`. `run_dev_c.py` also lacked an assertion for `file_split == 'train_file'`.<br>**Fix:** Change column selection to `file_split`, assert `file_split == 'train_file'` on Development, assert `file_split == 'calfree_test'` on CalFree, and remove any dependency on the legacy `split` column. |
| **f** | 5 folds, seeded permutation of subjects, asserted subject-disjoint. | **FAIL** | `src/train_c.py:188-189` | Currently uses standard `GroupKFold(n_splits=5)`, which does not apply a seeded subject permutation, and lacks an assertion that train and validation folds are subject-disjoint.<br>**Fix:** Permute unique subjects using `np.random.default_rng(seed).shuffle(unique_subjs)`, partition into 5 folds, and explicitly assert `len(set(train_subjs) & set(val_subjs)) == 0` for each fold. |
| **g** | Bootstrap resamples subjects, not segments: 2,000 resamples, seed 42, paired across ladder steps. | **PASS** | `src/train_c.py:77-165` | `clustered_subject_bootstrap_metrics` resamples subjects with replacement (`size=n_subjs`), aggregates all segments belonging to the resampled subjects, and evaluates paired differences on identical bootstrap draws across ladder steps with `seed=42`. |
| **h** | `test_c.py` refuses to run unless git tag `plan-c-locked` exists, plan hash matches, `results/dev_C.md` exists with frozen model hashes, and no `results/test_C.md` or lock file exists. | **FAIL** | `src/test_c.py:39-56`, `58-70` | `src/test_c.py` currently checks only `results/.test_locked_C`. It does not check for the git tag `plan-c-locked`, does not verify the SHA-256 hash of `PLAN_C.md`, does not check for the presence of `results/dev_C.md`, and does not check for `results/test_C.md`.<br>**Fix:** Add strict pre-flight assertions verifying: (1) tag `plan-c-locked` via git CLI, (2) `PLAN_C.md` SHA-256 hash against constant, (3) existence and parsed hashes in `results/dev_C.md`, (4) non-existence of `results/test_C.md` and `results/.test_locked_C`. |
| **i** | Which stages use Spark (`get_spark`, `mapInPandas`)? Plain statement on Spark usage. | **PASS** *(Audit)* | `src/features.py:17`, `68`<br>`src/spark_smoke_test.py:12`, `22` | **Plain Statement:** Spark is **NOT** used in any feature extraction, training, or evaluation stage of Study C or Study W. `pyspark` is only imported in an unused prototype file (`src/features.py`) and a smoke test (`src/spark_smoke_test.py`). Feature extraction runs entirely on CPU using NumPy, SciPy, h5py, and PyArrow. `get_spark` does not exist in the codebase. |
| **j** | What do `src/benchmark.py` and `src/features.py` actually do? Is benchmark a stub? | **PASS** *(Audit)* | `src/benchmark.py:27-57`<br>`src/features.py:46-85` | **Plain Statement:** `src/benchmark.py` is a **stub**. It generates synthetic sine waves and computes `np.mean` and `np.std` in a loop; it does not process real PPG data. `src/features.py` is an unused early skeleton with un-implemented Spark TODOs (`# TODO: Load df = spark.read.parquet...`). |
| **k** | Diff `src/ingest.py` against `src/ingest_script.py`. If identical, delete `ingest_script.py`. | **PASS** *(Audit)* | `src/ingest.py`<br>`src/ingest_script.py` | `src/ingest.py` and `src/ingest_script.py` are bit-for-bit identical (SHA-256: `64a7c06eb1846b0267fb5a49f75bfeb5179426f04f056157f8976a20ffc7ec1b`). `src/ingest_script.py` is redundant and will be deleted. |

---

## 3. Minimal Proposed Fixes (Zero Model Runs, Zero Data Processing)

All fixes below are code-only improvements addressing audit failures without triggering model training or feature extraction:

1. **Fix 1 (Item k):** Delete redundant duplicate file `src/ingest_script.py`.
2. **Fix 2 (Item b):** In `src/features_c.py`:
   - Add `hr_ibi = 60.0 / median(IBI)` and `ibi_sd = std(IBI)`.
   - Update `C1_COLS = C0_COLS + ["hr_ibi"]`.
   - Update `C2_COLS = C1_COLS + ["hr", "ibi_sd", "systolic_rise_time", "pulse_width", "decay_ratio", "beat_count", "is_clipped", "snr_ratio"]`.
3. **Fix 3 (Item e):** In `src/extract_features_c.py`:
   - Read `file_split` from `data/parquet/train_file.parquet`.
   - Explicitly filter/assert `file_split == 'train_file'` for Development (asserting 1,293 subjects and 465,480 rows).
   - Filter/assert `file_split == 'calfree_test'` for CalFree (asserting 144 subjects and 57,600 rows).
   - Compute `hr_ibi` and `ibi_sd` alongside `hr`.
4. **Fix 4 (Item f):** In `src/train_c.py`:
   - Replace unseeded `GroupKFold` with a seeded subject permutation (`rng.shuffle(unique_subjs)` with `seed=42`).
   - Explicitly assert `len(set(train_subjs) & set(val_subjs)) == 0` for all 5 folds.
   - Align ladder matrices with `C1 = C0 + hr_ibi`.
5. **Fix 5 (Item e):** In `src/run_dev_c.py`:
   - Assert `(train_df["file_split"] == "train_file").all()`.
   - Assert `train_df["subject_id"].nunique() == 1293` and `len(train_df) == 465480`.
6. **Fix 6 (Item h):** In `src/test_c.py`:
   - Add strict pre-flight gate function:
     1. Refuse to run unless `git tag -l plan-c-locked` returns a tag.
     2. Refuse to run unless `hashlib.sha256(open("PLAN_C.md").read()).hexdigest()` equals `9be7f444e02097dccba501fbe7fb39a02f2295b19095ec36f96cbedb304afc79`.
     3. Refuse to run unless `results/dev_C.md` exists and contains frozen model hashes.
     4. Refuse to run if `results/test_C.md` or `results/.test_locked_C` already exists.
7. **Fix 7 (Tests):** Update `tests/test_features_c.py` and `tests/test_train_c.py` to cover `hr_ibi`, `ibi_sd`, C1 definition, and seeded CV assertions.
