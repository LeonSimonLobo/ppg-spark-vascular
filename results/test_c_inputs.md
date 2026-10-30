# Study C: AAMI Test Evaluation Input Audit (`src/test_c.py`)

**Date:** 2026-10-30  
**Target Script Audited:** `src/test_c.py` (audited purely by static analysis; script was **NOT** executed)  

---

## 1. Expected Input File Path

- **Primary Input Path:** `data/raw/VitalDB_AAMI_Cal_Subset.mat` (HDF5 / MATLAB v7.3 container).
- **Hardcoded / Default Constant:** Defined in `src/test_c.py` line 44:
  ```python
  MAT_PATH = "data/raw/VitalDB_AAMI_Cal_Subset.mat"
  ```
- **Auxiliary Paths Checked by Pre-Flight Gates:**
  - `PLAN_C.md` (pre-registration plan file)
  - `results/dev_C.md` (development report containing recorded plan hash and model hashes)
  - `models/*_frozen.pkl` (9 serialized model and imputer artifacts)
  - `results/.test_locked_C` (lock sentinel, must **not** exist prior to execution)
  - `results/test_C.md` (test report, must **not** exist prior to execution)

---

## 2. Expected Columns and Data Structures

`src/test_c.py` reads directly from the HDF5 group `f["Subset"]` inside `VitalDB_AAMI_Cal_Subset.mat`:

1. **Signals Array (`f["Subset"]["Signals"]`):**
   - Shape: `(1250, 3, N_segments)` at 125 Hz sampling rate (10-second segments).
   - Channel 0: ABP (invasive arterial blood pressure waveform in mmHg).
   - Channel 1: PPG (plethysmogram waveform).
   - Channel 2: ECG or secondary signal.

2. **Metadata Arrays:**
   - `Subject`: Object references pointing to ASCII character arrays resolved to strings (e.g., `'p000001'`).
   - `Age`: Array of `float32`.
   - `Sex`: Array of `float32` (encoded 1.0 for Male, 0.0 for Female).
   - `BMI`: Array of `float32`.
   - `SBP`: Array of `float32` (or extracted as `np.max(Signals[:, 0, :], axis=0)` if direct `SBP` dataset is absent).

3. **In-Process DataFrame Columns Constructed:**
   - Demographics & Labels:
     - `subject_id` (str)
     - `segment_idx` (int32)
     - `age` (float32)
     - `sex` (float32)
     - `bmi` (float32)
     - `sbp` (float32)
     - `y_120` (int, `sbp >= 120.0`)
     - `y_130` (int, `sbp >= 130.0`)
   - PPG Morphological & Quality Features:
     - `hr_ibi` (float32)
     - `hr` (float32)
     - `ibi_sd` (float32)
     - `systolic_rise_time` (float32)
     - `pulse_width` (float32)
     - `decay_ratio` (float32)
     - `beat_count` (float32)
     - `is_clipped` (float32)
     - `snr_ratio` (float32)

---

## 3. How `src/test_c.py` Obtains AAMI Features

- **Does it read a feature parquet?**
  **NO.** `src/test_c.py` does **NOT** read any parquet file (`data/parquet/aami_cal.parquet` or `data/features_c/aami_cal.parquet`).
- **Does it extract features in-process?**
  **YES.** `src/test_c.py` extracts features **entirely in-process** directly from `VitalDB_AAMI_Cal_Subset.mat`.
  - In `score_aami_cal_test` (lines 308–326):
    ```python
    chunk_size = 15000
    with h5py.File(mat_path, "r") as f:
        sig_ds = f["Subset"]["Signals"]
        for c_start in range(0, n_segs, chunk_size):
            c_end = min(n_segs, c_start + chunk_size)
            ppg_block = sig_ds[:, 1, c_start:c_end]
            block_res = extract_features_block(ppg_block, fs=125)
            # Fills hr_ibi, hr, ibi_sd, systolic_rise_time, pulse_width, decay_ratio, beat_count, is_clipped, snr_ratio
    ```
  - It then applies `PlanCImputer` in-process using frozen training medians (`artifacts["c0_lr"]["imputer_medians"]`), constructs feature subsets (`C0_COLS`, `C1_COLS`, `C2m_COLS`, `C2_COLS`), and scores with the unpickled models.

---

## 4. Implication for Test Preparation (Prompt Step 4)

Because `src/test_c.py` operates directly and exclusively on `VitalDB_AAMI_Cal_Subset.mat` in-process:
- Per prompt instructions: *"If test_c.py extracts in-process, skip the extraction and say so."*
- Feature extraction via Spark into `data/features_c/aami_cal.parquet` is skipped as unneeded by the single-pass test evaluator.
- We ingest `VitalDB_AAMI_Cal_Subset.mat` into `data/parquet/aami_cal.parquet` with `src/ingest.py` solely for dataset auditing and generating `results/aami_ingest_report.md` without modifying any test or feature extraction scripts.

---

## 5. Corrections & Audit Revisions

Following full programmatic inspection of the MATLAB v7.3 HDF5 structures in `VitalDB_AAMI_Cal_Subset.mat` and verification against `src/ingest.py` and `src/test_c.py`, the following factual errors in the initial static audit are corrected:

1. **Signals Channel Ordering:**
   - *Initial Audit Stated:* Channel 0 = ABP, Channel 1 = PPG, Channel 2 = ECG.
   - *Correction:* The true container layout is **Channel 0 = ECG**, **Channel 1 = PPG**, and **Channel 2 = ABP** (invasive arterial blood pressure waveform). Channel 1 remains the verified PPG channel extracted for all Study C features.
2. **Gender Field Name and Encoding:**
   - *Initial Audit Stated:* `Sex` exists as an array of `float32`.
   - *Correction:* There is no `Sex` key under `f["Subset"]`. The container field is **`Gender`**, which is an array of HDF5 object references pointing to MATLAB ASCII character arrays (`'M'` or `'F'`). These are dereferenced and converted to `1.0` (Male) and `0.0` (Female) per subject.
3. **Subject ID Format:**
   - *Initial Audit Stated:* Subject strings look like `'p000001'`.
   - *Correction:* Subject identifiers follow the format **`'p000238_1'`** (including subject ID and record/case suffix).
4. **Direct Existence of SBP Field:**
   - *Initial Audit Stated:* SBP is read from `'SBP'` or extracted via `np.max(Signals[:, 0, :], axis=0)` fallback.
   - *Correction:* The **`SBP` dataset directly exists** for all 70,212 segments in `VitalDB_AAMI_Cal_Subset.mat` (as well as all 57,600 segments in CalFree). No max-of-waveform fallback is needed or executed. (Furthermore, ABP is Channel 2, so the fallback in the legacy loader would have read ECG peaks).

