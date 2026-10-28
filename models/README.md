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
| **`w_frozen.pkl`** | Within-subject $SBP \ge \text{median}(SBP_i)$ | `VitalDB_Train_Subset.mat` training split (372,240 segments across 1,034 subjects) | `StandardScaler` + `LogisticRegression(solver='lbfgs', random_state=42)` | `92c9176b456e00630c45071afcdb1f5e0423d393ca2f6ad2ecf9357f5e640fdc` |

---

## Re-generation
The frozen model is deterministically produced by running:
```bash
python -m src.train_w
```
with seed 42 on the development split.
