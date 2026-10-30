# Frozen Models and Checksums

This directory holds trained model artifacts.

## Artifact Tracking Decision
- **Policy:** Binary model serialized files (`.pkl`) are execution artifacts. They are excluded from git version control (`models/*.pkl` is included in `.gitignore`) to prevent repository bloat and binary serialization inconsistencies across platforms.
- **Reproducibility Guarantee:** All mathematical definitions, architecture pipelines, hyperparameters, standardization parameters (means and scales), and learned linear coefficients are version-controlled in plain text in [`results/dev_W.md`](../results/dev_W.md).
- **Integrity Verification:** The SHA-256 cryptographic digest of the exact frozen artifact is recorded below.

---

## Model Inventory

| Model File | Target Task | Training Cohort | Architecture | SHA-256 Checksum |
|---|---|---|---|---|
| **`w_frozen.pkl`** | Primary Morphology | `VitalDB_Train_Subset.mat` training split (372,240 segments across 1,034 subjects) | `StandardScaler` + `LogisticRegression(solver='lbfgs', random_state=42)` on 3 centered morphology timing features | `92c9176b456e00630c45071afcdb1f5e0423d393ca2f6ad2ecf9357f5e640fdc` |
| **`hr_frozen.pkl`** | Non-Morphology HR Comparator | `VitalDB_Train_Subset.mat` training split (372,240 segments across 1,034 subjects) | `StandardScaler` + `LogisticRegression(solver='lbfgs', random_state=42)` on `delta_hr` | `b1f8a705bc9292a936476857b9eebc4c0d18ae5a1d59c0445bc84dcfe8a6eaf5` |
| **`joint_frozen.pkl`** | Joint Model (Sensitivity) | `VitalDB_Train_Subset.mat` training split (372,240 segments across 1,034 subjects) | `StandardScaler` + `LogisticRegression(solver='lbfgs', random_state=42)` on `delta_hr` + 3 morphology features | `080296d8e309de97ed4eb5c65c1c69405cbecdfc19a7b7a452d92872530b5982` |

---

## Re-generation
The frozen models are deterministically produced by running:
```bash
python -m src.train_w
```
with seed 42 on the development split.



## Study C: Frozen Cross-Subject Models (PLAN_C.md)
**Date:** 2026-10-30
**Cohort:** VitalDB Train development cohort (1,293 subjects, 465,480 segments, file_split == 'train_file')
**Target:** SBP >= 120 mmHg (Primary endpoint)
**Pre-registration:** `PLAN_C.md` (committed before AAMI_Cal access)

| Artifact Key | Path | SHA-256 Digest |
|---|---|---|
| **`c_imputer`** | `models\c_imputer_frozen.pkl` | `d0a8e3d67cad417d1115f6b957ccd0ba901e1ac378e78eeec06b7df8d16bd641` |
| **`C0_lr`** | `models\c0_lr_frozen.pkl` | `02490af84b823d77cfb05e416ea74982ad577cbb425cf1ebf38b532f5e7dd4b3` |
| **`C0_gbt`** | `models\c0_gbt_frozen.pkl` | `fd5d7d29b0b18a592370d9d9885a5ed824efb5bfbdca043dc5e64cf907068af3` |
| **`C1_lr`** | `models\c1_lr_frozen.pkl` | `74012839bdd119f60619aee8c8ea07997125bb78eb2cf89e8556a22da948b0c5` |
| **`C1_gbt`** | `models\c1_gbt_frozen.pkl` | `0ddc345f04e8aba38da90bbac8afaf26ce02c48bdeea78082a888931f840feb9` |
| **`C2m_lr`** | `models\c2m_lr_frozen.pkl` | `c7032cc9204ad4225af592588d2334df1d94dd9bd9ba5f351876530d52c755d9` |
| **`C2m_gbt`** | `models\c2m_gbt_frozen.pkl` | `2869f2b83975e5335573904bb8c1a2fb5b412bbf3896537656c25da04265b54a` |
| **`C2_lr`** | `models\c2_lr_frozen.pkl` | `4def87939d3e73fdbfb28a2f0f94e669423c25a1f7d0fd2e7e1e0d5ddeb8fa8a` |
| **`C2_gbt`** | `models\c2_gbt_frozen.pkl` | `8a6e9e1aea7c552455e4a9d1fa1b0883ceaab71a9ace1be018f8bc36af4f649f` |
