# Reproduction Guide: Study C

> Comprehensive, step-by-step instructions to reproduce Study C data ingestion, distributed feature extraction, development model training, and test verification.

---

## 1. Project Goal & Scientific Scope

Study C evaluates whether single-segment photoplethysmogram (PPG) pulse morphology features provide statistically significant classification gain for elevated systolic blood pressure ($SBP \ge 120\text{ mmHg}$) on **unseen patients** without individual calibration.

---

## 2. Data Sources & Licensing

1. **VitalDB-derived subsets** (`VitalDB_Train_Subset.mat`, `VitalDB_CalFree_Test_Subset.mat`, `VitalDB_AAMI_Cal_Subset.mat`):
   - **License:** Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International ([CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)).
2. **MIMIC-derived subsets** (where applicable):
   - **License:** Open Database License ([ODbL](https://opendatacommons.org/licenses/odbl/)).
3. **Source Distribution:** Kaggle dataset `weinanwangrutgers/pulsedb-balanced-training-and-testing`.

---

## 3. Literature Citations

- **PulseDB:** Wang W, Mohseni S, Kilgore A, Najafizadeh L. *PulseDB: A large-scale physiological dataset for cardiovascular waveform analysis*. **Frontiers in Digital Health**. 2022;4:1090854. doi:[10.3389/fdgth.2022.1090854](https://doi.org/10.3389/fdgth.2022.1090854).
- **PPG Pulse Feature Analysis:** Elgendi M, et al. *Analysis of photoplethysmogram signals to assess cardiovascular health*. **PLoS ONE**. 2013;8(10):e76585. doi:[10.1371/journal.pone.0076585](https://doi.org/10.1371/journal.pone.0076585).
- TODO: Additional literature citations to be inserted.

---

## 4. Environment Setup

### System Prerequisites
- **Operating System:** Windows 10/11, Linux, or macOS (x86_64)
- **Python:** Python 3.10+ (tested on Python 3.13.1 64-bit)
- **Java Runtime:** JDK 17 (required for Apache Spark / PySpark local execution)
- **Windows Spark Binaries (Windows only):** `winutils.exe` and `hadoop.dll` located under `C:\hadoop\bin`, with `HADOOP_HOME=C:\hadoop` set on `PATH`.

### Step 0: Clone & Install Dependencies

```powershell
git clone --no-hardlinks https://github.com/your-org/ppg-spark-vascular.git
cd ppg-spark-vascular

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies from lockfile
pip install -r requirements.lock.txt
```
*Expected Time:* ~1–2 minutes.

---

## 5. End-to-End Pipeline Reproduction

### Step 1: Raw Waveform Ingestion (`src/ingest.py`)
Ingests raw MATLAB v7.3 HDF5 files from `data/raw/` into compressed Apache Parquet cohorts under `data/parquet/`:

```powershell
python src/ingest.py
```
- **Inputs:** `data/raw/VitalDB_Train_Subset.mat`, `VitalDB_CalFree_Test_Subset.mat`, `VitalDB_AAMI_Cal_Subset.mat`.
- **Outputs:** `data/parquet/train_file.parquet` (465,480 rows), `data/parquet/calfree_test.parquet` (57,600 rows), `data/parquet/aami_cal.parquet` (70,212 rows).
- **Expected Time:** ~3–5 minutes.

### Step 2: Distributed Feature Extraction (`src/spark_extract_c.py`)
Extracts the canonical 20-column feature schema (heart rate, fiducial-based morphology, quality indicators) across cohorts using PySpark `mapInPandas` with zero-leakage within-segment processing:

```powershell
# Extract development training cohort
python src/spark_extract_c.py --cohort train_file --out data/features_c

# Extract CalFree prospective validation cohort
python src/spark_extract_c.py --cohort calfree_test --out data/features_c
```
- **Outputs:** `data/features_c/train_file.parquet` (465,480 rows across 64 partitions), `data/features_c/calfree_test.parquet` (57,600 rows).
- **Expected Time:** ~90 seconds for training set; ~50 seconds for CalFree.

### Step 3: Model Development & Cross-Validation (`src/run_dev_c.py`)
Performs 5-fold grouped cross-validation across 1,293 training subjects, evaluates on CalFree prospective validation, runs 2,000 subject bootstrap resamples, freezes all 9 final models into `models/`, and writes development reports:

```powershell
python src/run_dev_c.py
```
- **Outputs:**
  - `results/dev_C.md` (summary report)
  - `results/dev_C_metrics.csv` (pooled OOF and CalFree metrics across all 8 models)
  - `results/dev_C_diffs.csv` (paired difference bootstrap statistics)
  - `results/dev_C_cv_folds.csv` (per-fold metrics)
  - `models/c0_lr_frozen.pkl`, `models/c0_gbt_frozen.pkl`, `models/c1_lr_frozen.pkl`, `models/c1_gbt_frozen.pkl`, `models/c2m_lr_frozen.pkl`, `models/c2m_gbt_frozen.pkl`, `models/c2_lr_frozen.pkl`, `models/c2_gbt_frozen.pkl`, `models/c_imputer_frozen.pkl`
- **Expected Time:** ~10–15 minutes.

### Step 4: Pre-Registration Freeze Tag
The pre-registration freeze point is marked by git tag `plan-c-locked`:

```powershell
git tag plan-c-locked
```
All model weights, hyperparameters, and feature definitions are permanently frozen at this tag.

### Step 5: Held-Out Single-Pass Test Scoring (`src/test_c.py`)

> [!IMPORTANT]
> The single-pass test evaluation can only be executed **strictly once**. Pre-flight security gates verify `git status`, model SHA-256 hashes, and plan integrity. Once executed, the sentinel lock `results/.test_locked_C` is acquired and subsequent scoring attempts are permanently refused.

```powershell
python -u src/test_c.py
```
- **Inputs:** `data/raw/VitalDB_AAMI_Cal_Subset.mat`, frozen models in `models/`.
- **Outputs:**
  - `results/.test_locked_C` (lock sentinel)
  - `results/test_C_predictions.parquet` (all 70,212 per-segment probabilities)
  - `results/test_C_predictions.sha256` (cryptographic checksum)
  - `results/test_C.md` (official test evaluation report)
  - `results/test_C_diffs.csv` (paired hypothesis test results)
- **Expected Time:** ~2–3 minutes.

### Step 6: Decoupled Stage 2 Report Re-generation (`src/report_test_c.py`)
If reporting or bootstrap re-summarization is required without re-scoring the test data, Stage 2 reads only the saved predictions parquet:

```powershell
python src/report_test_c.py
```
- **Expected Time:** ~40 seconds.

### Step 7: Independent Test Verification (`src/verify_test_C.py`)
Recomputes all test metrics and paired differences using pure scikit-learn and numpy, reading exclusively from `results/test_C_predictions.parquet`:

```powershell
python src/verify_test_C.py
```
- **Verification Rule:** Recomputed point estimate for primary endpoint ($C2 - C0$, GBT) must agree with `results/test_C_diffs.csv` within $1 \times 10^{-6}$.
- **Expected Time:** ~30 seconds.
