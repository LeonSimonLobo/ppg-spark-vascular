"""Execution runner for Study C development and CalFree validation (PLAN_C.md).

Pipeline steps:
1. Load data/interim/features_c_train.parquet (1,293 subjects, 465,480 segments)
   Assert file_split == 'train_file' across all rows.
2. Run 5-fold cross-validation with seeded subject permutation for primary target (SBP >= 120 mmHg)
3. Run 5-fold CV for sensitivity target (SBP >= 130 mmHg)
4. Compute clustered subject-level bootstrap percentile CIs (2,000 draws, seed 42)
5. Compute paired bootstrap differences: (C1 - C0), (C2m - C1), (C2 - C2m), (C2 - C0)
6. Fit and freeze final models on all development data to models/*_frozen.pkl
7. Evaluate frozen models on CalFree extra validation cohort (144 subjects, 57,600 segments)
   Assert file_split == 'calfree_test'.
8. Output results to results/dev_C.md and update models/README.md
"""

from pathlib import Path
import time
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.train_c import (
    compute_metrics,
    clustered_subject_bootstrap_metrics,
    train_and_eval_cv,
    fit_and_freeze_full_models,
    evaluate_on_calfree,
    LADDER_STEPS,
)


def run_development_pipeline(
    train_parquet: str = "data/interim/features_c_train.parquet",
    calfree_parquet: str = "data/interim/features_c_calfree.parquet",
    output_report: str = "results/dev_C.md",
    models_dir: str = "models",
    n_splits: int = 5,
    n_boot: int = 2000,
    seed: int = 42,
):
    print("=" * 70)
    print("STUDY C: Cross-Subject SBP Classification Development Pipeline")
    print("=" * 70)

    # 1. Load Development Data and Verify file_split
    print(f"Loading development features from {train_parquet}...")
    train_df = pd.read_parquet(train_parquet)
    n_train = len(train_df)
    n_subjs_train = train_df["subject_id"].nunique()
    print(f"Loaded {n_train:,} segments across {n_subjs_train:,} subjects.")

    # Strict file_split validation
    assert "file_split" in train_df.columns, "file_split column missing in development features!"
    assert (train_df["file_split"] == "train_file").all(), "All rows must have file_split == 'train_file'!"
    assert n_subjs_train == 1293, f"Expected exactly 1,293 subjects, got {n_subjs_train}"
    assert n_train == 465480, f"Expected exactly 465,480 segments, got {n_train}"

    train_df["y_120"] = (train_df["sbp"] >= 120.0).astype(int)
    train_df["y_130"] = (train_df["sbp"] >= 130.0).astype(int)
    prev_120_train = float(np.mean(train_df["y_120"]))
    prev_130_train = float(np.mean(train_df["y_130"]))
    print(f"SBP >= 120 prevalence: {prev_120_train * 100.0:.2f}%")
    print(f"SBP >= 130 prevalence: {prev_130_train * 100.0:.2f}%")

    # 2. 5-Fold Grouped CV on Primary Target (SBP >= 120) with Seeded Permutation
    print("\n" + "-" * 50)
    print("2. 5-Fold Grouped CV (Primary Target: SBP >= 120 mmHg)")
    print("-" * 50)
    oof_probs_120, pooled_metrics_120, fold_metrics_120 = train_and_eval_cv(
        train_df, target_col="y_120", n_splits=n_splits, seed=seed
    )

    # Bootstrap CIs on Primary Target OOF predictions
    print(f"Running {n_boot:,} subject-level bootstrap resamples on Primary OOF predictions...")
    subjs_train = train_df["subject_id"].to_numpy()
    y_120 = train_df["y_120"].to_numpy()
    model_cis_120, paired_cis_120 = clustered_subject_bootstrap_metrics(
        subjs_train, y_120, oof_probs_120, threshold=0.5, n_boot=n_boot, seed=seed
    )

    # 3. 5-Fold Grouped CV on Sensitivity Target (SBP >= 130)
    print("\n" + "-" * 50)
    print("3. 5-Fold Grouped CV (Sensitivity Target: SBP >= 130 mmHg)")
    print("-" * 50)
    oof_probs_130, pooled_metrics_130, fold_metrics_130 = train_and_eval_cv(
        train_df, target_col="y_130", n_splits=n_splits, seed=seed
    )
    y_130 = train_df["y_130"].to_numpy()
    model_cis_130, paired_cis_130 = clustered_subject_bootstrap_metrics(
        subjs_train, y_130, oof_probs_130, threshold=0.5, n_boot=n_boot, seed=seed
    )

    # 4. Freeze Final Models on all Development Data
    print("\n" + "-" * 50)
    print("4. Freezing Final Primary Models (Trained on all 465,480 Segments)")
    print("-" * 50)
    frozen_models_dict = fit_and_freeze_full_models(
        train_df, target_col="y_120", models_dir=models_dir, seed=seed
    )

    # Update models/README.md with hashes
    update_models_readme(frozen_models_dict)

    # 5. Extra Validation on CalFree
    print("\n" + "-" * 50)
    print("5. Evaluating Frozen Models on CalFree Extra Validation Cohort")
    print("-" * 50)
    calfree_df = pd.read_parquet(calfree_parquet)
    assert "file_split" in calfree_df.columns, "file_split column missing in CalFree features!"
    assert (calfree_df["file_split"] == "calfree_test").all(), "All CalFree rows must have file_split == 'calfree_test'!"
    assert calfree_df["subject_id"].nunique() == 144, "Expected exactly 144 CalFree subjects"
    assert len(calfree_df) == 57600, "Expected exactly 57,600 CalFree segments"

    calfree_df["y_120"] = (calfree_df["sbp"] >= 120.0).astype(int)
    calfree_df["y_130"] = (calfree_df["sbp"] >= 130.0).astype(int)

    cf_probs, cf_metrics, cf_model_cis, cf_paired_cis = evaluate_on_calfree(
        calfree_df, frozen_models_dict, target_col="y_120", seed=seed
    )

    # 6. Generate Markdown Report
    print(f"\nWriting comprehensive report to {output_report}...")
    generate_markdown_report(
        output_report=output_report,
        train_df=train_df,
        calfree_df=calfree_df,
        pooled_metrics_120=pooled_metrics_120,
        model_cis_120=model_cis_120,
        paired_cis_120=paired_cis_120,
        pooled_metrics_130=pooled_metrics_130,
        model_cis_130=model_cis_130,
        paired_cis_130=paired_cis_130,
        cf_metrics=cf_metrics,
        cf_model_cis=cf_model_cis,
        cf_paired_cis=cf_paired_cis,
        frozen_models_dict=frozen_models_dict,
    )
    print("Development and validation pipeline complete.")


def update_models_readme(frozen_dict: Dict[str, Tuple[str, str]]):
    """Record SHA-256 digests in models/README.md."""
    readme_path = Path("models/README.md")
    existing_content = readme_path.read_text(encoding="utf-8") if readme_path.exists() else "# Frozen Models\n"

    section_header = "## Study C: Frozen Cross-Subject Models (PLAN_C.md)"
    lines = [
        f"\n\n{section_header}",
        f"**Date:** 2026-10-29",
        f"**Cohort:** VitalDB Train development cohort (1,293 subjects, 465,480 segments, file_split == 'train_file')",
        f"**Target:** SBP >= 120 mmHg (Primary endpoint)",
        f"**Pre-registration:** `PLAN_C.md` (committed before AAMI_Cal access)",
        "",
        "| Artifact Key | Path | SHA-256 Digest |",
        "|---|---|---|",
    ]
    for k, (p, sha) in frozen_dict.items():
        lines.append(f"| **`{k}`** | `{p}` | `{sha}` |")

    # If section already exists, replace it, else append
    if section_header in existing_content:
        idx = existing_content.find(section_header)
        existing_content = existing_content[:idx].rstrip()

    new_content = existing_content + "\n" + "\n".join(lines) + "\n"
    readme_path.write_text(new_content, encoding="utf-8")
    print(f"Updated {readme_path}")


def generate_markdown_report(
    output_report: str,
    train_df: pd.DataFrame,
    calfree_df: pd.DataFrame,
    pooled_metrics_120: Dict[str, Dict[str, float]],
    model_cis_120: Dict[str, Dict[str, Tuple[float, float]]],
    paired_cis_120: Dict[str, Tuple[float, float]],
    pooled_metrics_130: Dict[str, Dict[str, float]],
    model_cis_130: Dict[str, Dict[str, Tuple[float, float]]],
    paired_cis_130: Dict[str, Tuple[float, float]],
    cf_metrics: Dict[str, Dict[str, float]],
    cf_model_cis: Dict[str, Dict[str, Tuple[float, float]]],
    cf_paired_cis: Dict[str, Tuple[float, float]],
    frozen_models_dict: Dict[str, Tuple[str, str]],
):
    n_train = len(train_df)
    n_subjs_train = train_df["subject_id"].nunique()
    n_cf = len(calfree_df)
    n_subjs_cf = calfree_df["subject_id"].nunique()

    p120_train = np.mean(train_df["y_120"]) * 100.0
    p130_train = np.mean(train_df["y_130"]) * 100.0
    p120_cf = np.mean(calfree_df["y_120"]) * 100.0

    plan_path = Path("PLAN_C.md")
    plan_hash = hashlib.sha256(plan_path.read_bytes()).hexdigest() if plan_path.exists() else "UNKNOWN"

    lines = [
        "# Study C Development & Cross-Cohort Validation Report",
        "",
        "**Date:** 2026-10-29  ",
        "**Specification:** `PLAN_C.md` (Cross-subject SBP classification on unseen subjects)  ",
        f"**Pre-Registered Plan Hash:** `{plan_hash}`  ",
        "**Final Test Boundary:** `VitalDB_AAMI_Cal_Subset.mat` is **PRISTINE and UNREAD**. Scored strictly once after models are frozen.  ",
        "",
        "---",
        "",
        "## 1. Study Setup & Data Splits",
        "",
        f"- **Development Cohort:** VitalDB Train file (`file_split == 'train_file'`, {n_subjs_train:,} subjects, {n_train:,} segments, exactly 360 seg/subj).",
        f"- **Development Protocol:** 5-fold cross-validation with seeded subject permutation and disjoint fold assertion.",
        f"- **Extra Validation Cohort:** CalFree (`file_split == 'calfree_test'`, {n_subjs_cf:,} subjects, {n_cf:,} segments, 400 seg/subj).",
        f"- **Primary Target Label:** $SBP \\ge 120\\text{{ mmHg}}$ (Development prevalence: **{p120_train:.2f}%**; CalFree: **{p120_cf:.2f}%**).",
        f"- **Sensitivity Target Label:** $SBP \\ge 130\\text{{ mmHg}}$ (Development prevalence: **{p130_train:.2f}%**).",
        "- **Decision Threshold:** Fixed standard threshold $0.5$ ($P(\\hat{y}=1) \\ge 0.5$).",
        "- **Bootstrap Protocol:** Clustered subject-level bootstrap with 2,000 resamples (seed 42).",
        "",
        "---",
        "",
        "## 2. Primary Endpoint (SBP $\\ge 120$ mmHg): 5-Fold Grouped CV Pooled OOF Results",
        "",
        "| Model Step | Classifier | Predictors | Accuracy [95% CI] | Balanced Acc [95% CI] | AUROC [95% CI] | Majority Baseline |",
        "|---|---|---|---|---|---|---|",
    ]

    ladder_desc = {
        "C0": "Age, Sex, BMI",
        "C1": "C0 + PPG Heart Rate (hr_ibi)",
        "C2m": "C1 + Morphology Only (rise_time, pulse_width, decay_ratio)",
        "C2": "C1 + Morphology + Auxiliary Fields (beat_count, snr_ratio, ibi_sd)",
    }

    for step in LADDER_STEPS:
        for fam, fname in [("lr", "Logistic Reg (regParam 0.01)"), ("gbt", "GBT (depth 5, iter 100)")]:
            k = f"{step}_{fam}"
            m = pooled_metrics_120[k]
            ci_acc = model_cis_120[k]["accuracy"]
            ci_bal = model_cis_120[k]["balanced_accuracy"]
            ci_auc = model_cis_120[k]["auroc"]
            lines.append(
                f"| **{step}** | {fname} | {ladder_desc[step]} | "
                f"**{m['accuracy'] * 100.0:.2f}%** [{ci_acc[0] * 100.0:.2f}%, {ci_acc[1] * 100.0:.2f}%] | "
                f"**{m['balanced_accuracy'] * 100.0:.2f}%** [{ci_bal[0] * 100.0:.2f}%, {ci_bal[1] * 100.0:.2f}%] | "
                f"**{m['auroc']:.4f}** [{ci_auc[0]:.4f}, {ci_auc[1]:.4f}] | "
                f"{m['majority_accuracy'] * 100.0:.2f}% |"
            )

    lines.extend([
        "",
        "### Stepwise Paired Gain Decomposition (5-Fold CV Pooled OOF)",
        "",
        "| Comparison | Model Family | Paired $\\Delta\\text{Accuracy}$ [95% CI] | Paired $\\Delta\\text{AUROC}$ [95% CI] |",
        "|---|---|---|---|",
    ])

    for fam, fname in [("lr", "Logistic Regression"), ("gbt", "GBT")]:
        d_acc_10 = paired_cis_120[f"delta_acc_C1_{fam}_minus_C0_{fam}"]
        d_auc_10 = paired_cis_120[f"delta_auc_C1_{fam}_minus_C0_{fam}"]
        m1 = pooled_metrics_120[f"C1_{fam}"]
        m0 = pooled_metrics_120[f"C0_{fam}"]
        lines.append(
            f"| **C1 vs C0 (Gain from Heart Rate IBI)** | {fname} | "
            f"**{(m1['accuracy'] - m0['accuracy']) * 100.0:+.2f}%** [{d_acc_10[0] * 100.0:+.2f}%, {d_acc_10[1] * 100.0:+.2f}%] | "
            f"**{(m1['auroc'] - m0['auroc']):+.4f}** [{d_auc_10[0]:+.4f}, {d_auc_10[1]:+.4f}] |"
        )

        d_acc_2m1 = paired_cis_120[f"delta_acc_C2m_{fam}_minus_C1_{fam}"]
        d_auc_2m1 = paired_cis_120[f"delta_auc_C2m_{fam}_minus_C1_{fam}"]
        m2m = pooled_metrics_120[f"C2m_{fam}"]
        lines.append(
            f"| **C2m vs C1 (Gain from Pulse Morphology Shape)** | {fname} | "
            f"**{(m2m['accuracy'] - m1['accuracy']) * 100.0:+.2f}%** [{d_acc_2m1[0] * 100.0:+.2f}%, {d_acc_2m1[1] * 100.0:+.2f}%] | "
            f"**{(m2m['auroc'] - m1['auroc']):+.4f}** [{d_auc_2m1[0]:+.4f}, {d_auc_2m1[1]:+.4f}] |"
        )

        d_acc_22m = paired_cis_120[f"delta_acc_C2_{fam}_minus_C2m_{fam}"]
        d_auc_22m = paired_cis_120[f"delta_auc_C2_{fam}_minus_C2m_{fam}"]
        m2 = pooled_metrics_120[f"C2_{fam}"]
        lines.append(
            f"| **C2 vs C2m (Gain from Signal Quality Fields)** | {fname} | "
            f"**{(m2['accuracy'] - m2m['accuracy']) * 100.0:+.2f}%** [{d_acc_22m[0] * 100.0:+.2f}%, {d_acc_22m[1] * 100.0:+.2f}%] | "
            f"**{(m2['auroc'] - m2m['auroc']):+.4f}** [{d_auc_22m[0]:+.4f}, {d_auc_22m[1]:+.4f}] |"
        )

        d_acc_20 = paired_cis_120[f"delta_acc_C2_{fam}_minus_C0_{fam}"]
        d_auc_20 = paired_cis_120[f"delta_auc_C2_{fam}_minus_C0_{fam}"]
        lines.append(
            f"| **C2 vs C0 (Total Pipeline Gain)** | {fname} | "
            f"**{(m2['accuracy'] - m0['accuracy']) * 100.0:+.2f}%** [{d_acc_20[0] * 100.0:+.2f}%, {d_acc_20[1] * 100.0:+.2f}%] | "
            f"**{(m2['auroc'] - m0['auroc']):+.4f}** [{d_auc_20[0]:+.4f}, {d_auc_20[1]:+.4f}] |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Sensitivity Analysis (SBP $\\ge 130$ mmHg): 5-Fold Grouped CV Pooled OOF Results",
        "",
        "| Model Step | Classifier | Accuracy [95% CI] | Balanced Acc [95% CI] | AUROC [95% CI] | Majority Baseline |",
        "|---|---|---|---|---|---|",
    ])

    for step in LADDER_STEPS:
        for fam, fname in [("lr", "Logistic Reg"), ("gbt", "GBT")]:
            k = f"{step}_{fam}"
            m = pooled_metrics_130[k]
            ci_acc = model_cis_130[k]["accuracy"]
            ci_bal = model_cis_130[k]["balanced_accuracy"]
            ci_auc = model_cis_130[k]["auroc"]
            lines.append(
                f"| **{step}** | {fname} | "
                f"**{m['accuracy'] * 100.0:.2f}%** [{ci_acc[0] * 100.0:.2f}%, {ci_acc[1] * 100.0:.2f}%] | "
                f"**{m['balanced_accuracy'] * 100.0:.2f}%** [{ci_bal[0] * 100.0:.2f}%, {ci_bal[1] * 100.0:.2f}%] | "
                f"**{m['auroc']:.4f}** [{ci_auc[0]:.4f}, {ci_auc[1]:.4f}] | "
                f"{m['majority_accuracy'] * 100.0:.2f}% |"
            )

    lines.extend([
        "",
        "---",
        "",
        "## 4. Extra Validation: CalFree Cohort ($N = 57,600$ segments, 144 subjects)",
        "",
        "Evaluated strictly using frozen models fit on the development set (zero refitting on CalFree).",
        "",
        "| Model Step | Classifier | Accuracy [95% CI] | Balanced Acc [95% CI] | AUROC [95% CI] | Majority Baseline |",
        "|---|---|---|---|---|---|",
    ])

    for step in LADDER_STEPS:
        for fam, fname in [("lr", "Logistic Reg"), ("gbt", "GBT")]:
            k = f"{step}_{fam}"
            m = cf_metrics[k]
            ci_acc = cf_model_cis[k]["accuracy"]
            ci_bal = cf_model_cis[k]["balanced_accuracy"]
            ci_auc = cf_model_cis[k]["auroc"]
            lines.append(
                f"| **{step}** | {fname} | "
                f"**{m['accuracy'] * 100.0:.2f}%** [{ci_acc[0] * 100.0:.2f}%, {ci_acc[1] * 100.0:.2f}%] | "
                f"**{m['balanced_accuracy'] * 100.0:.2f}%** [{ci_bal[0] * 100.0:.2f}%, {ci_bal[1] * 100.0:.2f}%] | "
                f"**{m['auroc']:.4f}** [{ci_auc[0]:.4f}, {ci_auc[1]:.4f}] | "
                f"{m['majority_accuracy'] * 100.0:.2f}% |"
            )

    lines.extend([
        "",
        "### CalFree Paired Differences ($C2 - C0$ Endpoint Preview)",
        "",
        "| Comparison | Model Family | Paired $\\Delta\\text{Accuracy}$ [95% CI] | Paired $\\Delta\\text{AUROC}$ [95% CI] |",
        "|---|---|---|---|",
    ])

    for fam, fname in [("lr", "Logistic Regression"), ("gbt", "GBT")]:
        d_acc_20 = cf_paired_cis[f"delta_acc_C2_{fam}_minus_C0_{fam}"]
        d_auc_20 = cf_paired_cis[f"delta_auc_C2_{fam}_minus_C0_{fam}"]
        m2 = cf_metrics[f"C2_{fam}"]
        m0 = cf_metrics[f"C0_{fam}"]
        lines.append(
            f"| **C2 vs C0** | {fname} | "
            f"**{(m2['accuracy'] - m0['accuracy']) * 100.0:+.2f}%** [{d_acc_20[0] * 100.0:+.2f}%, {d_acc_20[1] * 100.0:+.2f}%] | "
            f"**{(m2['auroc'] - m0['auroc']):+.4f}** [{d_auc_20[0]:+.4f}, {d_auc_20[1]:+.4f}] |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 5. Frozen Model Artifacts & SHA-256 Checksums",
        "",
        "All models below are frozen and ready for the single-pass evaluation on `AAMI_Cal`:",
        "",
        "| Model Key | Path | SHA-256 Digest |",
        "|---|---|---|",
    ])

    for k, (p, sha) in frozen_models_dict.items():
        lines.append(f"| **`{k}`** | `{p}` | `{sha}` |")

    lines.extend([
        "",
        "---",
        "",
        "## 6. Pre-Registration Invariant Status",
        "",
        "- `PLAN_C.md` pre-registration and Amendment C1 committed to git.",
        "- `VitalDB_AAMI_Cal_Subset.mat` has **NOT** been opened or read.",
        "- Pre-scoring lock `results/.test_locked_C` must be created before running single-pass test script.",
    ])

    out_p = Path(output_report)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report written to {output_report}")
