# Project Inventory: Source Code, Results, Feature Tables, and Detectors

**Generated:** 2026-10-29  
**Repository:** `D:\DS_RESEARCH\ppg-spark-vascular`  
**Protocol Invariant:** Training was **not run**. Results for split `'test'` and `AAMI` were **not read**.

---

## 1. Inventory of `src/`

Total files: **25 Python source files** (plus cached bytecode in `__pycache__/`).

| File | Size (Bytes) | Module Description / Functional Role |
|:---|---:|:---|
| `src/__init__.py` | 125 | Package initializer for `ppg-spark-vascular`. |
| `src/baselines.py` | 8,488 | Baseline demographic models (Logistic Regression, Random Forest on age, sex, BMI) for SBP classification. |
| `src/benchmark.py` | 2,646 | Scaling and throughput benchmark module for distributed Spark PPG processing. |
| `src/controls_w.py` | 15,203 | Robustness controls for Study W (within-subject label permutation shuffle, temporal drift control, nonlinear GBT HR control). |
| `src/exploratory_dev_models.py` | 14,516 | Exploratory development model comparison on training split (`split == 'train'`) evaluating pairwise interaction terms, HistGradientBoosting, and Random Forest. |
| `src/extract_features_c.py` | 9,372 | Batch feature extraction pipeline for cross-subject Study C development and CalFree validation cohorts. |
| `src/features.py` | 3,985 | Initial PySpark feature extraction definitions (Butterworth filter, VPG/APG derivatives, peak-to-peak interval). |
| `src/features_c.py` | 9,894 | Feature extraction, quality fields, and pre-registration median imputer (`PlanCImputer`) for single-segment cross-subject Study C. |
| `src/features_w.py` | 16,259 | Morphology feature extraction (`hr`, `systolic_rise_time`, `pulse_width`, `decay_ratio`) and per-subject centering for Study W. |
| `src/fiducials.py` | 21,952 | Canonical Elgendi et al. (2013) two-moving-average systolic peak detector, foot detection, turning points, and ABP reference agreement. |
| `src/gate_within_subject.py` | 10,461 | Gate check for physiological beat count plausibility (30–150 bpm) and within-subject heart rate correlation on training split. |
| `src/gate_within_subject_agreement.py` | 15,039 | Empirical beat-agreement gate evaluation (PPG vs invasive arterial catheter reference peaks). |
| `src/generate_prospective_report.py` | 14,121 | Formatted report generator for prefix-centering prospective control and amplitude invariance audit. |
| `src/ingest.py` | 10,755 | Subset ingestion module converting raw PulseDB/VitalDB MAT structures to partitioned Parquet tables. |
| `src/ingest_script.py` | 10,755 | Script entrypoint for dataset ingestion and chunked MAT-to-Parquet conversion. |
| `src/inspect_subset.py` | 2,122 | Utility for inspecting HDF5 metadata, field layouts, and signal dimensions of raw MAT subsets. |
| `src/prospective_and_scale_w.py` | 8,187 | Prospective prefix calibration control (first 60 segments) and PPG amplitude-scale invariance audit. |
| `src/run_audit.py` | 1,785 | Runner script executing prospective control and amplitude scale audits. |
| `src/run_dev_c.py` | 15,878 | 5-fold grouped cross-validation runner, bootstrap CI estimator, and model freezing pipeline for Study C. |
| `src/spark_smoke_test.py` | 1,178 | Smoke test script for PySpark local driver initialization. |
| `src/test_c.py` | 16,282 | Pre-registered single-pass test evaluation runner for Study C on `VitalDB_AAMI_Cal_Subset.mat`. |
| `src/test_w.py` | 20,638 | Pre-registered single-pass test evaluation runner for Study W on CalFree test cohort. |
| `src/train.py` | 3,564 | Baseline training script using PySpark MLlib / scikit-learn. |
| `src/train_c.py` | 14,599 | Cross-subject model training, 5-fold grouped CV (`GroupKFold`), and clustered subject bootstrap for Study C. |
| `src/train_w.py` | 37,739 | Development CV and nested evaluation for within-subject morphology Study W. |

---

## 2. Inventory of `results/`

Total files: **28 files** (21 in `results/`, 7 in `results/superseded/`).  
*(Per instructions: results for split `'test'` or AAMI were not opened or read).*

| File | Size (Bytes) | Category / Description |
|:---|---:|:---|
| `results/.gitkeep` | 56 | Git placeholder file. |
| `results/.test_locked_W` | 175 | Execution lock file permanently locking the CalFree test evaluation for Study W. |
| `results/bp_hist.png` | 42,658 | Histogram plot of invasive arterial catheter SBP and DBP distributions. |
| `results/dataset_report.md` | 2,011 | Ingestion audit summarizing subject counts, segment dimensions, and sampling rate (125 Hz). |
| `results/demographic_consistency.md` | 4,781 | Audit of demographic field alignment (`age`, `sex`, `height`, `weight`, `bmi`) between MAT and Parquet. |
| `results/dev_W.md` | 7,132 | Development 5-fold CV report for Study W (within-subject morphology vs HR comparator). |
| `results/dev_W_controls.md` | 5,227 | Sanity controls report for Study W (within-subject label permutation, time drift, GBT nonlinear HR). |
| `results/dev_W_nested.md` | 5,704 | Nested within-subject CV report (HR alone vs HR + morphology vs HR-residualized morphology). |
| `results/dev_W_prospective_and_scale.md` | 10,685 | Prospective prefix-centering control report and PPG amplitude-scale invariance audit. |
| `results/environment.md` | 2,625 | Compute environment audit (Windows, Python 3.13, PySpark 3.5, CPU, RAM). |
| `results/exploratory_dev_models.md` | 4,266 | Exploratory development model comparison on training split (interaction terms, HistGradientBoosting, RF). |
| `results/features_W_dev.md` | 2,596 | Statistical distribution and missingness audit for Study W development features. |
| `results/gate_within_subject.md` | 2,074 | Within-subject heart rate plausibility and Spearman correlation gate report. |
| `results/gate_within_subject_agreement.md` | 2,320 | Beat-to-beat agreement gate report (PPG vs invasive arterial catheter reference). |
| `results/mat_inspection.md` | 4,751 | Inspection report of HDF5 structures in raw PulseDB/VitalDB subsets. |
| `results/ppg_clipping.md` | 1,459 | Analysis of sensor rail clipping and flat-top saturation across PPG waveforms. |
| `results/ppg_range.md` | 1,514 | Analysis of PPG amplitude range and min-max normalization bounds. |
| `results/sample_waveforms.png` | 297,673 | Representative visual plots of raw PPG and ABP waveforms across subjects. |
| `results/source_verification.md` | 12,842 | Ingestion verification against source VitalDB subsets. |
| `results/source_verification_part2.md` | 8,825 | Extended ingestion verification and metadata parity audit. |
| `results/test_W.md` | 12,764 | Study W held-out CalFree test report *(unopened per prompt instructions)*. |
| `results/superseded/README.md` | 3,028 | Index documenting reasons for superseding earlier exploratory analyses. |
| `results/superseded/baseline_sanity.md` | 4,131 | Early baseline cross-subject demographic model checks. |
| `results/superseded/detector_agreement.md` | 2,649 | Initial beat detector agreement run under original asymmetric window. |
| `results/superseded/detector_agreement_A3.md` | 3,628 | Beat detector agreement evaluation under symmetric A3 window. |
| `results/superseded/detector_diagnostics.md` | 4,790 | Diagnostic analysis of peak alignment discrepancies. |
| `results/superseded/ptt_sign_check.md` | 8,627 | Cross-correlation lag and sign verification for Pulse Transit Time (PTT). |
| `results/superseded/validation_baselines.md` | 4,443 | Early cross-subject demographic baseline evaluations. |

---

## 3. Inventory of Feature Tables (`data/features*` & Ingested Tables)

### Table 3.1: `data/interim/features_w_train.parquet`
- **Location:** `data/interim/features_w_train.parquet`
- **File Size:** 11,942,232 bytes (~11.4 MB)
- **Total Rows:** **372,240**
- **Unique Subjects:** **1,034** (exactly 360 segments per subject)
- **Row Counts by `split`:**
  - `split == 'train'`: **372,240** (100.0%)
- **Row Counts by `file_split`:**
  - *Column `file_split` is not present in this table* (derived from the `train` partition of `train_file.parquet`).
- **Feature Columns & NaN Rates:**

| Column | Data Type | Non-Null Count | NaN / Null Count | NaN Rate (%) |
|:---|:---|---:|---:|---:|
| `segment_idx` | int32 | 372,240 | 0 | **0.0000%** |
| `subject_id` | string | 372,240 | 0 | **0.0000%** |
| `split` | string | 372,240 | 0 | **0.0000%** |
| `sbp` | float32 | 372,240 | 0 | **0.0000%** |
| `hr` | float32 | 372,240 | 0 | **0.0000%** |
| `systolic_rise_time` | float32 | 372,240 | 0 | **0.0000%** |
| `pulse_width` | float32 | 372,240 | 0 | **0.0000%** |
| `decay_ratio` | float32 | 372,240 | 0 | **0.0000%** |
| `delta_hr` | float32 | 372,240 | 0 | **0.0000%** |
| `delta_systolic_rise_time` | float32 | 372,240 | 0 | **0.0000%** |
| `delta_pulse_width` | float32 | 372,240 | 0 | **0.0000%** |
| `delta_decay_ratio` | float32 | 372,240 | 0 | **0.0000%** |

- **Per-Segment PPG Features Check:**
  - **Heart rate:** **YES** (`hr`, derived as $n_{\text{beats}} \times 6.0$, mean 75.69 bpm).
  - **Rise time:** **YES** (`systolic_rise_time`, duration from foot to systolic peak in seconds, mean 0.2209 s).
  - **Pulse width:** **YES** (`pulse_width`, full width at half maximum at 50% amplitude in seconds, mean 0.2341 s).
  - **Area ratio:** **NO** (not present).
  - **SDPPG ratios ($b/a, c/a, d/a, e/a$):** **NO** (not present).
  - **Notch:** **NO** (dicrotic notch time/amplitude is not stored as a standalone feature).
  - **Reflection index:** **NO** (not present).
  - *(Additional feature stored)*: `decay_ratio` (ratio of diastolic decay time to systolic rise time, mean 2.7147).
- **Per-Subject Centering / Baseline Window / `delta_` Prefix:**
  - **YES.** Features `delta_hr`, `delta_systolic_rise_time`, `delta_pulse_width`, and `delta_decay_ratio` use **per-subject session-level median centering** with a `delta_` prefix:
    $$\Delta X_{i, j} = X_{i, j} - \text{median}_{j=0..359}(X_{i, j})$$
    Centering was transductive across the full 360-segment recording for each subject.

---

### Table 3.2: `data/interim/features_w_calfree.parquet`
- **Location:** `data/interim/features_w_calfree.parquet`
- **File Size:** 1,608,515 bytes (~1.5 MB)
- **Total Rows:** **57,600**
- **Unique Subjects:** **144** (exactly 400 segments per subject)
- **Row Counts by `split`:**
  - *Column `split` is not stored in this interim table* (corresponds to CalFree test cohort).
- **Row Counts by `file_split`:**
  - *Column `file_split` is not stored in this interim table*.
- **Feature Columns & NaN Rates:**

| Column | Data Type | Non-Null Count | NaN / Null Count | NaN Rate (%) |
|:---|:---|---:|---:|---:|
| `segment_idx` | int32 | 57,600 | 0 | **0.0000%** |
| `subject_id` | string | 57,600 | 0 | **0.0000%** |
| `sbp` | float32 | 57,600 | 0 | **0.0000%** |
| `dbp` | float32 | 57,600 | 0 | **0.0000%** |
| `hr` | float32 | 57,600 | 0 | **0.0000%** |
| `systolic_rise_time` | float32 | 57,600 | 0 | **0.0000%** |
| `pulse_width` | float32 | 57,600 | 0 | **0.0000%** |
| `decay_ratio` | float32 | 57,600 | 0 | **0.0000%** |

- **Per-Segment PPG Features Check:**
  - **Heart rate:** **YES** (`hr`, bpm).
  - **Rise time:** **YES** (`systolic_rise_time`, seconds).
  - **Pulse width:** **YES** (`pulse_width`, seconds).
  - **Area ratio:** **NO** (not present).
  - **SDPPG ratios:** **NO** (not present).
  - **Notch:** **NO** (not present).
  - **Reflection index:** **NO** (not present).
  - *(Additional feature stored)*: `decay_ratio`.
- **Per-Subject Centering / Baseline Window / `delta_` Prefix:**
  - **NO.** There are **no `delta_` prefix features**, **no centered features**, and **no baseline window features** saved in this table. All columns represent absolute per-segment values.

---

### Table 3.3: Ingested Dataset Tables (`data/parquet/`)

For completeness, the two upstream ingested segment tables containing raw waveforms and patient vitals:

#### `data/parquet/train_file.parquet`
- **Location:** `data/parquet/train_file.parquet` (2,351,057,824 bytes, ~2.2 GB)
- **Total Rows:** **465,480** (1,293 subjects, exactly 360 segments per subject)
- **Columns (12):** `subject_id`, `segment_idx`, `file_split`, `split`, `age`, `sex`, `height`, `weight`, `bmi`, `sbp`, `dbp`, `ppg` (1,250 samples/segment at 125 Hz)
- **Row counts by `split`:**
  - `split == 'train'`: **372,240** (80.0%, 1,034 subjects)
  - `split == 'val'`: **93,240** (20.0%, 259 subjects)
- **Row counts by `file_split`:**
  - `file_split == 'train_file'`: **465,480** (100.0%)
- **NaN Rate:** **0.0000%** across all 12 columns.
- **Centering / Baseline / `delta_`:** **NO.** Stores raw uncentered demographic vitals and waveforms.

#### `data/parquet/calfree_test.parquet`
- **Location:** `data/parquet/calfree_test.parquet` (290,946,597 bytes, ~277 MB)
- **Total Rows:** **57,600** (144 subjects, exactly 400 segments per subject)
- **Columns (12):** `subject_id`, `segment_idx`, `file_split`, `split`, `age`, `sex`, `height`, `weight`, `bmi`, `sbp`, `dbp`, `ppg`
- **Row counts by `split`:**
  - `split == 'test'`: **57,600** (100.0%)
- **Row counts by `file_split`:**
  - `file_split == 'calfree_test'`: **57,600** (100.0%)
- **NaN Rate:** **0.0000%** across all 12 columns.
- **Centering / Baseline / `delta_`:** **NO.** Stores raw uncentered demographic vitals and waveforms.

---

## 4. Beat Detector Specification and Parameters

The beat detection algorithm implemented across all pipelines is the **two-moving-average method of Elgendi et al. (2013)**.

- **Primary Citation:** Elgendi M, Norton I, Brearley M, Abbott D, Lovell NH. *Systolic Peak Detection in Photoplethysmogram Signals during Rest and Exercise.* PLoS ONE 8(10): e76585 (2013). doi:10.1371/journal.pone.0076585.
- **Implementation File:** `src/fiducials.py::detect_peaks_elgendi`
- **Sampling Frequency ($f_s$):** **125 Hz**

### Exact Detector Parameters:
1. **$W_1$ (Peak Moving Average Window):**
   - Duration: **111 ms**
   - Window size in samples: $W_1 = \text{round}(0.111 \times 125) = \mathbf{14\text{ samples}}$
   - Implementation: Centered rolling average $\text{MA}_{\text{peak}}[n] = \frac{1}{W_1} \sum_{k = -W_1/2}^{W_1/2} x^2[n + k]$ on the squared signal.
2. **$W_2$ (Beat Moving Average Window):**
   - Duration: **667 ms**
   - Window size in samples: $W_2 = \text{round}(0.667 \times 125) = \mathbf{83\text{ samples}}$
   - Implementation: Centered rolling average $\text{MA}_{\text{beat}}[n] = \frac{1}{W_2} \sum_{k = -W_2/2}^{W_2/2} x^2[n + k]$ on the squared signal.
3. **$\beta$ (Threshold Offset Multiplier):**
   - Value: **0.02**
   - Dynamic threshold offset: $\alpha = \beta \times \bar{z}$, where $\bar{z} = \frac{1}{N}\sum_{n=1}^N x^2[n]$ is the mean of the squared signal.
4. **Decision Rule:**
   - Threshold signal: $\text{THR}_1[n] = \text{MA}_{\text{beat}}[n] + \alpha$.
   - Blocks of Interest: Continuous intervals where $\text{MA}_{\text{peak}}[n] > \text{THR}_1[n]$ with duration $\ge W_1$ (14 samples).
   - Peak selection: Discrete maximum of the raw signal $x[n]$ within each valid block of interest.
5. **Documented Filter Deviation:**
   - The upstream PulseDB database pre-filtered all raw signals with a 4th-order Chebyshev-II bandpass filter ($[0.5, 8.0]\text{ Hz}$, stopband ripple 20 dB).
   - Per pre-registration and study protocol (`PLAN_W.md` Section 5), the detector is applied directly to Channel 1 as stored without a cascaded secondary Butterworth filter, eliminating double-filtering attenuation and artificial phase distortion.
6. **Subsequent Landmark Extraction:**
   - **Foot:** Discrete minimum in the $300\text{ ms}$ ($37\text{ samples}$) window strictly preceding each detected systolic peak.
   - **Turning Point:** Discrete minimum between consecutive systolic peaks.
