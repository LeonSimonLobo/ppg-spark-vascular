# Study C Artifact & Checksum Manifest

Generated automatically by script.

- **Git HEAD**: `01806ea6987301d93e5b962429d2a5e6ba722624`
- **Pre-Registration Plan (`PLAN_C.md`) SHA-256**: `e851b885f144022920476b9de238304a52fade4fd46157404a8b3378c8bbb0f3` (13,995 bytes)
- **Held-Out Test Predictions (`results/test_C_predictions.parquet`) SHA-256**: `e05638fbff1b237c1746ecc4701275a713c2a398f7fd5fd714bd3d5c5e9c03be` (3,595,009 bytes)

---

## 1. Raw Subset Files (`data/raw/`)

| File Name | File Size (Bytes) | SHA-256 Checksum | Protocol Notes |
| :--- | :--- | :--- | :--- |
| `data/raw/VitalDB_Train_Subset.mat` | 13,797,698,523 | `20f19685b67a17fff6673743114cb17345a959d02367553ceea3cf560b14d002` | Hashed & verified |
| `data/raw/VitalDB_CalFree_Test_Subset.mat` | 1,706,947,648 | `f0dffa676987124cdca00d46ae88fd4d8bf7f6ac411e6cdb36b98ff8bcf4a00a` | Hashed & verified |
| `data/raw/VitalDB_AAMI_Cal_Subset.mat` | 2,078,737,367 | `bc68fc239d3809f20cda01bb55eeae7aee88912f8143e0c4cde18d9c4384dcde` | Hashed & verified |
| `data/raw/VitalDB_AAMI_Test_Subset.mat` | 19,806,481 | `UNOPENED / UNHASHED` | Stat size only (pre-registered rule) |
| `data/raw/VitalDB_CalBased_Test_Subset.mat` | 1,532,644,941 | `UNOPENED / UNHASHED` | Stat size only (pre-registered rule) |

---

## 2. Ingested Waveform Parquet Cohorts (`data/parquet/`)

| Cohort Parquet | Rows | Size (Bytes) | SHA-256 Checksum |
| :--- | :--- | :--- | :--- |
| `data/parquet/aami_cal.parquet` | 70,212 | 354,859,282 | `81f36a16532bfa53351528a5a43fd9d92ef41ae7a120abf15a13cfced0db87b0` |
| `data/parquet/calfree_test.parquet` | 57,600 | 290,946,597 | `b327aaaf6301b093d8d1e85dead05869335efc3af2ddb4b80ffcf79691a731ac` |
| `data/parquet/train_file.parquet` | 465,480 | 2,351,057,824 | `005d780666c017c0001033b79fa42648574f3a2b8fc8d618a14de88e60bdbe25` |

---

## 3. Study C Feature Sets (`data/features_c/`)

| Feature Parquet | Rows | Total Size (Bytes) | Composite SHA-256 |
| :--- | :--- | :--- | :--- |
| `data/features_c/calfree_test.parquet` | 57,600 | 2,631,762 | `f2a70582c9aa78437f0e3bdcbbadac1b48dda1413642dde3673a0115f6b1a444` |
| `data/features_c/train_file.parquet` | 465,480 | 18,071,394 | `2cfa2891d168364704b165a4a85f0c0c5a77c21316f019c2d1e1875a6b791535` |

---

## 4. Frozen Study C Models (`models/`)

| Model Artifact | Size (Bytes) | SHA-256 Checksum |
| :--- | :--- | :--- |
| `models/c0_lr_frozen.pkl` | 1,554 | `02490af84b823d77cfb05e416ea74982ad577cbb425cf1ebf38b532f5e7dd4b3` |
| `models/c0_gbt_frozen.pkl` | 286,776 | `fd5d7d29b0b18a592370d9d9885a5ed824efb5bfbdca043dc5e64cf907068af3` |
| `models/c1_lr_frozen.pkl` | 1,640 | `74012839bdd119f60619aee8c8ea07997125bb78eb2cf89e8556a22da948b0c5` |
| `models/c1_gbt_frozen.pkl` | 349,091 | `0ddc345f04e8aba38da90bbac8afaf26ce02c48bdeea78082a888931f840feb9` |
| `models/c2m_lr_frozen.pkl` | 1,921 | `c7032cc9204ad4225af592588d2334df1d94dd9bd9ba5f351876530d52c755d9` |
| `models/c2m_gbt_frozen.pkl` | 359,046 | `2869f2b83975e5335573904bb8c1a2fb5b412bbf3896537656c25da04265b54a` |
| `models/c2_lr_frozen.pkl` | 2,185 | `4def87939d3e73fdbfb28a2f0f94e669423c25a1f7d0fd2e7e1e0d5ddeb8fa8a` |
| `models/c2_gbt_frozen.pkl` | 367,364 | `8a6e9e1aea7c552455e4a9d1fa1b0883ceaab71a9ace1be018f8bc36af4f649f` |
| `models/c_imputer_frozen.pkl` | 340 | `d0a8e3d67cad417d1115f6b957ccd0ba901e1ac378e78eeec06b7df8d16bd641` |

---

## 5. Test Lock & Verification Integrity
- `results/.test_locked_C` and `results/test_C.md` were committed directly following execution.
- `results/test_C_predictions.parquet` SHA-256 bit-for-bit matches `results/test_C_predictions.sha256`.
- Neither `VitalDB_AAMI_Test_Subset.mat` nor `VitalDB_CalBased_Test_Subset.mat` was opened or hashed.
