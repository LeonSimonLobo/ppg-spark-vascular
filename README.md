# ppg-spark-vascular: Study C

> Rigorous Cross-Subject Evaluation of Photoplethysmogram (PPG) Vascular Morphology for Systolic Blood Pressure Classification at Scale.

---

## 1. Project Goal

The primary clinical and computational objective of **Study C** is to rigorously establish whether single-segment photoplethysmogram (PPG) pulse morphology features provide genuine, generalizable discriminative gain for elevated systolic blood pressure ($SBP \ge 120\text{ mmHg}$) across **unseen patients** without calibration.

Prior literature has frequently suffered from within-subject temporal autocorrelation, calibration leakage, or post-hoc threshold tuning. Study C addresses these vulnerabilities through:
- **Pre-registered hypothesis ladder:** Demographics baseline ($C0$), Demographics + Heart Rate ($C1$), Pure Pulse Morphology ($C2m$), and Full Morphology + Signal Quality ($C2$).
- **Strict single-pass held-out test evaluation:** Pre-registered in `PLAN_C.md`, enforced by cryptographic pre-flight gates, and permanently locked.
- **Distributed feature extraction:** Leveraging Apache Spark (PySpark) to ingest multi-gigabyte physiological recordings and compute pulse fiducials across 593,292 ten-second waveforms.

---

## 2. Key Findings

On the pre-registered, single-pass held-out test cohort (**VitalDB AAMI_Cal**, 116 unseen surgical/ICU patients, 70,212 segments):
- **Primary Endpoint ($C2 - C0$, GBT):** **+5.24%** paired accuracy gain (95% CI: `[+0.60%, +10.30%]`, $p < 0.05$) and **+0.1202** AUROC gain (`[+0.0493, +0.1952]`).
- **Ablation of Pulse Morphology ($C2m - C1$, GBT):** **+6.05%** paired accuracy gain (`[+0.37%, +12.35%]`) and **+0.1196** AUROC gain (`[+0.0499, +0.1963]`), demonstrating that pulse morphology accounts for the entire observed vascular performance advantage over heart rate alone.
- **Independent Verification:** Exact numerical point-estimate agreement ($7.63 \times 10^{-17}$, $< 10^{-6}$) confirmed via standalone scikit-learn recomputation in [`results/verify_test_C.md`](file:///D:/DS_RESEARCH/ppg-spark-vascular/results/verify_test_C.md).

---

## 3. Data Sources & Licensing

The datasets analyzed in this repository derive from the **PulseDB** benchmark:
- **VitalDB-derived subsets** (`VitalDB_Train_Subset.mat`, `VitalDB_CalFree_Test_Subset.mat`, `VitalDB_AAMI_Cal_Subset.mat`, `VitalDB_CalBased_Test_Subset.mat`, `VitalDB_AAMI_Test_Subset.mat`):
  - **License:** Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International ([CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)).
- **MIMIC-derived subsets** (where applicable):
  - **License:** Open Database License ([ODbL](https://opendatacommons.org/licenses/odbl/)).
- **Source Repository:** Kaggle dataset `weinanwangrutgers/pulsedb-balanced-training-and-testing`.

---

## 4. Citations

This project builds upon and cites the following formal literature:
1. **PulseDB Benchmark:**  
   Wang W, Mohseni S, Kilgore A, Najafizadeh L. *PulseDB: A large-scale physiological dataset for cardiovascular waveform analysis*. **Frontiers in Digital Health**. 2022;4:1090854. doi:[10.3389/fdgth.2022.1090854](https://doi.org/10.3389/fdgth.2022.1090854).
2. **PPG Pulse Feature Engineering:**  
   Elgendi M, et al. *Analysis of photoplethysmogram signals to assess cardiovascular health*. **PLoS ONE**. 2013;8(10):e76585. doi:[10.1371/journal.pone.0076585](https://doi.org/10.1371/journal.pone.0076585).
3. TODO: Additional clinical or algorithmic citations to be inserted.

---

## 5. Repository Structure

```text
ppg-spark-vascular/
├── PLAN_C.md                 # Pre-registered Study C protocol specification
├── requirements.txt          # Active project dependencies
├── requirements.lock.txt     # Exact pinned dependency lockfile
├── README.md                 # Project summary and overview
├── REPRODUCE.md              # Exact step-by-step reproduction instructions
├── docs/
│   └── DEVIATIONS.md         # Comprehensive audit log of all pipeline changes
├── models/                   # 9 frozen models and preprocessing artifacts
├── data/
│   ├── raw/                  # MATLAB v7.3 HDF5 source datasets
│   ├── parquet/              # Ingested multi-channel waveform parquets
│   └── features_c/           # Extracted 20-column feature parquets
├── results/
│   ├── MANIFEST_C.md         # Complete artifact and checksum manifest
│   ├── INDEX.md              # File catalog and governance classifications
│   ├── NUMBERS_C.md          # Comprehensive headline numbers traceability matrix
│   ├── dev_C.md              # Development 5-fold CV and CalFree validation report
│   ├── test_C.md             # Locked single-pass test evaluation report
│   ├── test_C_diffs.csv      # Test paired differences and bootstrap intervals
│   ├── test_C_predictions.parquet # Per-segment test model probabilities
│   ├── verify_test_C.md      # Independent sklearn verification report
│   └── reproducibility_C.md  # Fresh-clone and protocol reproducibility audit
└── src/
    ├── ingest.py             # Raw .mat ingestion to Parquet
    ├── features_c.py         # Pulse fiducial detection & PlanCImputer
    ├── spark_extract_c.py    # PySpark distributed feature extraction
    ├── run_dev_c.py          # 5-fold grouped CV and model freezer
    ├── test_c.py             # Pre-flight gated single-pass test scoring
    ├── report_test_c.py      # Standalone Stage 2 bootstrap reporting
    └── verify_test_C.py      # Independent test predictions verification
```

---

## 6. Pre-Registration & Test Governance

- **Pre-Registration Tag:** Git tag `plan-c-locked` marks the formal code and model freeze commit (`f1b9ed9`).
- **Single Evaluation Rule:** `src/test_c.py` executes under strict 6-stage cryptographic pre-flight gates. It writes `results/.test_locked_C` upon first run; any subsequent attempt is permanently refused by the security gates.
