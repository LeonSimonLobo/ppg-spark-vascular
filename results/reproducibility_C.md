# Study C Comprehensive Reproducibility & Release Audit

**Date:** 2026-10-30  
**Protocol Specification:** `PLAN_C.md`  
**Pre-Registration Freeze Tag:** `plan-c-locked` (`f1b9ed9`)  
**Overall Reproducibility Status:** **ALL CHECKS PASSED**  

---

## 1. Summary of Reproducibility Checks

| Check ID | Description | Requirement | Outcome | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Check 0** | Preconditions & Sentinels | `results/.test_locked_C`, `test_C.md`, `test_C_diffs.csv` committed unedited; predictions parquet (3.43 MB) tracked with SHA-256 | Committed unedited; predictions parquet tracked; SHA-256 verified | **PASS** |
| **Check 1** | Final Environment & Lockfile | Record hardware, OS, Python 3.13, PySpark 3.5.9, Java JDK 17, `requirements.lock.txt`; verify `requirements.txt` lists all 12 imported third-party packages | Generated `environment_final.md` and lockfile; `psutil` added | **PASS** |
| **Check 2** | Artifact Manifest (`MANIFEST_C.md`) | Checksums and row counts for raw, parquet, features, models; 3 raw files hashed, 2 raw test files unopened (size only) | Verified bit-for-bit; unhashed raw files untouched | **PASS** |
| **Check 3** | Independent Test Verification | Standalone `src/verify_test_C.py` reading ONLY predictions parquet with pure sklearn; verify point estimates agree to $< 10^{-6}$ | Discrepancy = $7.63 \times 10^{-17}$ ($< 10^{-6}$); seed 123 CI `[+0.25%, +10.06%]`; C2 GBT AUROC = 0.6785 matches Train OOF to 4 dec | **PASS** |
| **Check 4a** | Fresh-Clone PyTest Suite | Run full pytest suite in fresh clone `../ppg_repro` with isolated venv | 60 passed, 1 xfailed (matching pre-freeze suite) in 17.65s | **PASS** |
| **Check 4b** | Fresh-Clone Test Report Parity | Run `python src/report_test_c.py` in clone; compare `test_C_diffs.csv` to real repo | Max abs diff = $0.00 \times 10^0$ (exact numerical identity $\le 10^{-9}$) | **PASS** |
| **Check 4c** | Fresh-Clone Dev Model & Metrics Parity | Re-run `src/run_dev_c.py` in clone; compare all 9 model hashes and dev metrics CSVs | All 9 frozen models match bit-for-bit; `dev_C_metrics` and `dev_C_diffs` diff = $0.00 \times 10^0$ | **PASS** |
| **Check 4d** | Fresh-Clone Feature Extraction Parity | Re-extract first 50,000 train rows with `src/spark_extract_c.py --limit 50000`; compare all 9 features by `(subject_id, segment_idx)` | Max abs diff = $0.00 \times 10^0$, NaN pattern match = True across all 50,000 rows | **PASS** |
| **Check 5a** | Deviations & Methodology Audit | Comprehensive table in `docs/DEVIATIONS.md` classifying all commits touching Study C with amendments C1-C3 and fixes | Fully cataloged 48 commits with phase and impact | **PASS** |
| **Check 5b** | Results Directory Catalog | Index every file in `results/` in `results/INDEX.md` with description and governance status | Cataloged 48 files across locked, final, descriptive, superseded | **PASS** |
| **Check 5c** | Headline Numbers Matrix | Generate `results/NUMBERS_C.md` mapping every headline number to exact source file and column | Generated complete traceability matrix for all cohorts and models | **PASS** |
| **Check 5d** | Public Documentation & Guides | Comprehensive `README.md` and `REPRODUCE.md` with PulseDB / VitalDB CC BY-NC-SA 4.0 and MIMIC ODbL licenses, and strictly permitted citations | Full setup, reproduction steps, licenses, and verified citations | **PASS** |

---

## 2. Locked Files Manifest Verification (`results/locked_files_manifest.csv`)

Every file governed under the pre-registration test lock was hashed before starting and recomputed upon completion:

| File Path | Expected SHA-256 | Current SHA-256 | Verification Status |
| :--- | :--- | :--- | :--- |
| `PLAN_C.md` | `e851b885f1440229...` | `e851b885f1440229...` | **PASS: Bit-for-bit identical** |
| `models/README.md` | `8c0c3d66fc3d8430...` | `8c0c3d66fc3d8430...` | **PASS: Bit-for-bit identical** |
| `models/c0_gbt_frozen.pkl` | `fd5d7d29b0b18a59...` | `fd5d7d29b0b18a59...` | **PASS: Bit-for-bit identical** |
| `models/c0_lr_frozen.pkl` | `02490af84b823d77...` | `02490af84b823d77...` | **PASS: Bit-for-bit identical** |
| `models/c1_gbt_frozen.pkl` | `0ddc345f04e8aba3...` | `0ddc345f04e8aba3...` | **PASS: Bit-for-bit identical** |
| `models/c1_lr_frozen.pkl` | `74012839bdd119f6...` | `74012839bdd119f6...` | **PASS: Bit-for-bit identical** |
| `models/c2_gbt_frozen.pkl` | `8a6e9e1aea7c5524...` | `8a6e9e1aea7c5524...` | **PASS: Bit-for-bit identical** |
| `models/c2_lr_frozen.pkl` | `4def87939d3e73fd...` | `4def87939d3e73fd...` | **PASS: Bit-for-bit identical** |
| `models/c2m_gbt_frozen.pkl` | `2869f2b83975e533...` | `2869f2b83975e533...` | **PASS: Bit-for-bit identical** |
| `models/c2m_lr_frozen.pkl` | `c7032cc9204ad422...` | `c7032cc9204ad422...` | **PASS: Bit-for-bit identical** |
| `models/c_imputer_frozen.pkl` | `d0a8e3d67cad417d...` | `d0a8e3d67cad417d...` | **PASS: Bit-for-bit identical** |
| `models/hr_frozen.pkl` | `b1f8a705bc9292a9...` | `b1f8a705bc9292a9...` | **PASS: Bit-for-bit identical** |
| `models/joint_frozen.pkl` | `080296d8e309de97...` | `080296d8e309de97...` | **PASS: Bit-for-bit identical** |
| `models/w_frozen.pkl` | `92c9176b456e0063...` | `92c9176b456e0063...` | **PASS: Bit-for-bit identical** |
| `results/.test_locked_C` | `a25bce0eb61062d6...` | `a25bce0eb61062d6...` | **PASS: Bit-for-bit identical** |
| `results/dev_C.md` | `28da722d469ade72...` | `28da722d469ade72...` | **PASS: Bit-for-bit identical** |
| `results/dev_C_cv_folds.csv` | `02b0575fe3d06f9a...` | `02b0575fe3d06f9a...` | **PASS: Bit-for-bit identical** |
| `results/dev_C_diffs.csv` | `aec5b42ebdf7a905...` | `aec5b42ebdf7a905...` | **PASS: Bit-for-bit identical** |
| `results/dev_C_metrics.csv` | `5eabef7fde00021c...` | `5eabef7fde00021c...` | **PASS: Bit-for-bit identical** |
| `results/test_C.md` | `caab47b23c69a12c...` | `caab47b23c69a12c...` | **PASS: Bit-for-bit identical** |
| `results/test_C_diffs.csv` | `01fe9e9ab65ff413...` | `01fe9e9ab65ff413...` | **PASS: Bit-for-bit identical** |
| `results/test_C_predictions.parquet` | `e05638fbff1b237c...` | `e05638fbff1b237c...` | **PASS: Bit-for-bit identical** |

**Manifest Comparison Conclusion:** **22 of 22 files identical** (`all_locked_match = True`).

---

## 3. Protocol Invariant & Governance Checklist
- **Zero Test Re-Scoring:** Gated scoring in `src/test_c.py` was executed strictly once; `results/.test_locked_C` sentinel remains active.
- **Untouched Raw Datasets:** Neither `VitalDB_AAMI_Test_Subset.mat` nor `VitalDB_CalBased_Test_Subset.mat` was opened or hashed.
- **Zero Network Transmission:** No code or data was pushed (`git push` not executed).
- **Scratch Clone Cleaned:** `../ppg_repro` was completely removed following successful replication.
