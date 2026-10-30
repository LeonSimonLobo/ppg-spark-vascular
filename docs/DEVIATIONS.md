# Study C Pipeline Deviations & Methodology Evolution Log

This document records every modification, amendment, bug fix, and architectural refactor touching the Study C pipeline from initial project configuration to final release.

All git hashes, commit messages, dates, and classifications are derived directly from the git commit history and `PLAN_C.md`.

---

## 1. Commit Classification Overview

The pipeline progression is partitioned into three phases relative to pre-registration and test freeze:
1. **Pre-Development / Pre-Results:** Prior to commit `f1b9ed9` (includes Amendments C1–C3 and infrastructure fixes before results were generated).
2. **Freeze & Tag (`plan-c-locked`):** Commit `f1b9ed9`, where development cross-validation metrics, models, and pre-registration invariants were locked.
3. **Post-Freeze / Pre-Flight & Test Release:** Commits following `f1b9ed9` (dry run verifications, loader safety fixes, two-stage decoupled reporting, independent test evaluation).

---

## 2. Chronological Pipeline Deviations Table

| Date | Commit | Phase | Category | Description & Justification | Effect on Results |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **2026-10-29** | `584c5d4` | Pre-Results | Pre-Registration | **Initial PLAN_C Pre-Registration**: Defined cross-subject SBP classification on unseen AAMI_Cal subjects ($SBP \ge 120\text{ mmHg}$). | Baseline protocol establishment. |
| **2026-10-29** | `ed2e317` | Pre-Results | Protocol Audit | **Audit C Feature Fix**: Added `hr_ibi` and `ibi_sd` features; defined C1 model step as baseline demographics ($C0$) + heart rate (`hr_ibi`). | Ensured standard cardiovascular comparator step before vascular morphology features. |
| **2026-10-29** | `93d273d` | Pre-Results | Ingestion Audit | **Cohort Selection Fix**: Selected cohorts strictly by `file_split == 'train_file'` and `'calfree_test'` (never legacy `split`). | Prevented sample contamination across cohort boundaries. |
| **2026-10-29** | `16a561a` | Pre-Results | CV Methodology | **CV Subject Disjointness**: Implemented seeded subject permutation and disjoint fold assertions in 5-fold CV. | Guaranteed strict zero-leakage subject grouping across CV folds. |
| **2026-10-29** | `499f496` | Pre-Results | Security Gates | **Test Runner Security Gates**: Added strict 6-stage pre-flight security gates in `src/test_c.py` refusing execution if models or plan are tampered. | Safeguarded test set from accidental or repeated execution. |
| **2026-10-29** | `d1d0846` | Pre-Results | Protocol Amendment | **Amendment C1**: Added `C2m` morphology-only ladder step (`systolic_rise_time`, `pulse_width`, `decay_ratio`) between C1 and C2. | Enabled fine-grained ablation isolating pure morphology gain from signal quality indicators. |
| **2026-10-29** | `4245dc9` | Pre-Results | Pipeline | Added C2m morphology step to `src/features_c.py`. | Implementation of Amendment C1. |
| **2026-10-29** | `2f95b84` | Pre-Results | Pipeline | Added C2m morphology ablation and stepwise gain decomposition to `src/train_c.py`. | Implementation of Amendment C1. |
| **2026-10-29** | `ccceb61` | Pre-Results | Test Infrastructure | Added C2m ablation and 6-gate verification to `src/test_c.py`. | Implementation of Amendment C1. |
| **2026-10-29** | `43469e2` | Pre-Results | Scale Infrastructure | Implemented PySpark feature extraction, parity unit tests, and batch cohort processing. | Scaled extraction to 523,080 development waveforms. Verified bit-for-bit parity with NumPy. |
| **2026-10-29** | `1ab4290` | Pre-Results | Protocol Amendment | **Amendment C2**: Foot definition updated to inter-peak argmin (eliminating artificial 0.4s clip); removed constant zero `is_clipped` from `C2_COLS`; added auxiliary fields `beat_count`, `snr_ratio`, `ibi_sd` to C2. | Restored natural physiological rise time distribution; eliminated collinear zero-variance column. |
| **2026-10-29** | `8e42f6e` | Pre-Results | Signal Processing | Updated foot finding to inter-peak argmin in `src/features_c.py`. | Implementation of Amendment C2. |
| **2026-10-29** | `61f8325` | Pre-Results | Pipeline | Updated model training pipeline for auxiliary fields in `src/train_c.py`. | Implementation of Amendment C2. |
| **2026-10-29** | `3f90643` | Pre-Results | Protocol Amendment | **Amendment C3**: Standardized `PlanCImputer` training-median strategy and missingness indicator flags for robust handling of NaN values. | Zero data transduction across train/validation/test splits. |
| **2026-10-30** | `d74dd46` | Pre-Results | Bug Fix | **Dev Runner Crash Fix**: Fixed artifact loading order in `evaluate_on_calfree` (crash after step 4; no model or setting change). | Prevented crash during CalFree evaluation; zero change to models, weights, or features. |
| **2026-10-30** | `9c9826f` | Pre-Results | Bug Fix | **Console Encoding Fix**: Replaced Unicode mathematical symbols in `run_dev_c.py` table printer with ASCII to prevent Windows CP1252 crash. | Pure presentation fix; zero numerical impact. |
| **2026-10-30** | `f1b9ed9` | **FREEZE TAG** | Milestone | **Tag `plan-c-locked`**: Recorded development results, figures, metrics, and models README. | **DEVELOPMENT FROZEN**. All 9 model hashes and plan hash permanently locked. |
| **2026-10-30** | `f5f9234` | Post-Freeze | Ingestion Tooling | **Ingest `--only` Option**: Added `--only aami_cal` option to `src/ingest.py` to ingest AAMI cohort without modifying other files. | Descriptive audit only; zero impact on development data or frozen models. |
| **2026-10-30** | `b913dd1` | Post-Freeze | Safety / Output | **Save Predictions Parquet**: Added output-only code saving per-segment test predictions (`results/test_C_predictions.parquet`). | Enabled downstream auditing and decoupled reporting without repeated scoring. |
| **2026-10-30** | `58ff4fa` | Post-Freeze | Bug Fix | **Test Runner Loader Fix**: In `src/test_c.py`, correctly dereferenced MATLAB v7.3 HDF5 character references for `Gender` and mapped PPG to channel 1 (0=ECG, 1=PPG, 2=ABP). | Fixed metadata decoding prior to any test scoring; avoided feeding ECG signal to feature extractor. |
| **2026-10-30** | `354093b` | Post-Freeze | Verification | **CalFree Dry Run**: Added `src/dryrun_c_calfree.py` and `results/dryrun_C.md` validating in-process extraction and model scoring against development metrics to $< 10^{-4}$. | Verified end-to-end scoring pipeline parity on non-test data. Discovered 1-segment float32/float64 `ibi_sd` difference. |
| **2026-10-30** | `ef59cde` | Post-Freeze | Architecture | **Two-Stage Decoupled Evaluation**: Separated Stage 1 (single-pass locked scoring to parquet) from Stage 2 (`src/report_test_c.py`, reads parquet, runs bootstrap). | Ensured catastrophic failure during reporting does not invalidate locked test results. |
| **2026-10-30** | `eba4424` | Post-Freeze | Documentation | Appended factual corrections section to `results/test_c_inputs.md`. | Accurately recorded HDF5 schema findings (Gender string cell, SBP existence). |
| **2026-10-30** | `b345b26` | Post-Freeze | Presentation | **Neutral Benchmark Wording**: Removed "MET"/"NOT MET" wording from `report_test_c.py`; noted +4–5 point gain is an exploratory aspiration. | Presentation clarity only; zero numerical change. |
| **2026-10-30** | `beccac7` | Post-Freeze | Verification | **Scratch Clone Rehearsal**: Executed complete end-to-end rehearsal in scratch clone `ppg_scratch`; validated gates, scoring, report rebuild. | Confirmed zero defects in execution flow before unsealing test set. |
| **2026-10-30** | `acc1cd5` | Post-Freeze | CLI Fix | **Test Runner CLI Path Resolution**: Added `sys.path` repo_dir insertion and `if __name__ == '__main__': os.chdir(repo_dir)`. | Allowed direct CLI invocation from any directory without `ModuleNotFoundError`. |
| **2026-10-30** | `f1d2038` | Post-Freeze | Test Result | **AAMI_Cal Single-Pass Test Execution**: Committed unedited `results/test_C.md`, `test_C_diffs.csv`, `results/.test_locked_C`. | First and only execution of test scoring on unseen AAMI_Cal cohort. |
| **2026-10-30** | `463dbd4` | Post-Freeze | Data Release | Added `results/test_C_predictions.parquet` (3.43 MB) containing all 70,212 per-segment probabilities. | Full public auditability of test segment classifications. |
| **2026-10-30** | `01806ea` | Post-Freeze | Checksum | Added `results/test_C_predictions.sha256`. | Crytographic proof of immutable test predictions. |
| **2026-10-30** | `f1e5f15` | Post-Freeze | Reproducibility | Recorded final environment, locked manifest, and requirements. | Environment provenance and artifact validation. |
| **2026-10-30** | `16c5a1b` | Post-Freeze | Verification | **Independent Verification Engine**: Added `src/verify_test_C.py` and `results/verify_test_C.md` reading only predictions parquet with pure sklearn. | Independent confirmation: pooled point estimate agreed to $7.63 \times 10^{-17}$ ($< 10^{-6}$). |

---

## 3. Detailed Audit of Key Pipeline Deviations

### 1. Protocol Amendments (C1, C2, C3)
- **Amendment C1 (Morphology Ablation Step C2m):** Prior to development runs, the C2 step combined pulse morphology and signal quality. Amendment C1 inserted $C2m$ (pure pulse shape: rise time, pulse width, decay ratio) to distinguish physiological vascular shape signals from signal-quality and heart-rate variability features ($C2$). On the final test set, $C2m - C1$ provided $+6.05\%$ accuracy gain, demonstrating that morphology drives virtually all of the observed vascular advantage.
- **Amendment C2 (Foot Finding & Auxiliary Selection):** Preliminary inspection revealed that clamping rise times to $0.4\text{ s}$ distorted physiological distributions for prolonged arterial waveforms. The foot finding algorithm was refined to an unconstrained inter-peak minimum, restoring physiological log-normal rise times. `is_clipped` was removed from feature vectors because hardware clipping was zero across the cohorts. Auxiliary metrics `beat_count`, `snr_ratio`, and `ibi_sd` were allocated to the full $C2$ step.
- **Amendment C3 (Indicator-Flagged Training Imputer):** Missing waveform fiducials (e.g., noisy single-pulse segments) were handled via `PlanCImputer`, which calculates medians strictly on the training set and appends binary missingness indicators. No test or validation information is used during imputation.

### 2. Infrastructure & Safety Fixes
- **Dev Runner Crash Fix (`d74dd46`):** During initial development execution, the script encountered an `UnboundLocalError` when attempting to log CalFree metrics after model fitting had succeeded. The fix merely reordered the dictionary lookup. Zero weights, seeds, or parameters were modified.
- **Console Encoding Fix (`9c9826f`):** Windows command-line environments default to legacy code pages (e.g., CP1252), which throw `UnicodeEncodeError` when emitting UTF-8 mathematical glyphs like $\Delta$ and $\ge$. The display routine was adjusted to use ASCII equivalents.
- **Test Runner Metadata Loader Fix (`58ff4fa`):** The initial prototype of `src/test_c.py` assumed numeric encoding for demographic fields. Inspection of `VitalDB_AAMI_Cal_Subset.mat` revealed `Gender` is stored as HDF5 character object references, and channel 0 corresponds to ECG, channel 1 to PPG, and channel 2 to ABP. The loader was corrected to read channel 1 and resolve string references before any test data was evaluated.
- **Two-Stage Decoupled Evaluation (`ef59cde`):** To eliminate the risk of lock invalidation due to post-scoring crashes during lengthy bootstrap computations, `src/test_c.py` was separated into Stage 1 (single-pass locked scoring to parquet) and Stage 2 (`src/report_test_c.py`, reading parquet to produce reports). This ensured that once probabilities were computed and locked, statistical intervals could be calculated independently.
- **Neutral Report Wording (`b345b26`):** The target benchmark goal of $+4\text{--}5$ percentage points was designated as an exploratory aspiration rather than a binary acceptance threshold, avoiding misinterpretation of clinically relevant continuous gains.
- **Float32 vs Float64 Intermediate Accumulation on CalFree (`354093b`):** Across all 57,600 CalFree segments and 9 features ($518,400$ values), exactly one segment (`p000438_1`, segment 84) differed in `ibi_sd` by $0.000100$ between PySpark batch extraction and NumPy in-process extraction due to 32-bit floating-point accumulation in intermediate summation. All other 518,399 values were identical. This had zero impact on model predictions or thresholded classifications.
