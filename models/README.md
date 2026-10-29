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
