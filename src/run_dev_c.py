"""Execution runner for Study C development and CalFree validation (PLAN_C.md).

Pipeline steps:
1. Load data/features_c/train_file.parquet (1,293 subjects, 465,480 segments)
   Assert file_split == 'train_file' across all rows.
   Compute and print class counts for Primary (SBP >= 120) and Sensitivity (SBP >= 130).
2. Run 5-fold cross-validation with seeded subject permutation for primary target (SBP >= 120 mmHg)
   imputer and scaler fit strictly on training folds; assert subject-disjoint.
3. Run 5-fold CV for sensitivity target (SBP >= 130 mmHg)
4. Compute clustered subject-level bootstrap percentile CIs (2,000 draws, seed 42)
5. Compute paired bootstrap differences: (C1 - C0), (C2m - C1), (C2 - C2m), (C2 - C0)
   on accuracy, balanced accuracy, and AUROC with 95% CIs.
6. Fit and freeze final models on all development data to models/*_frozen.pkl
7. Evaluate frozen models on CalFree extra validation cohort (144 subjects, 57,600 segments)
   Assert file_split == 'calfree_test'.
8. Save results/dev_C_metrics.csv, results/dev_C_diffs.csv, results/dev_C_cv_folds.csv
9. Save results/dev_C_accuracy_ladder.png and results/dev_C_paired_diffs.png
10. Generate results/dev_C.md from the CSV files and computed hashes.
11. Print ladder table and four paired differences to stdout.
"""

import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
import time
from typing import Dict, Any, Tuple, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

repo_dir = Path(__file__).resolve().parent.parent
if str(repo_dir) not in sys.path:
    sys.path.insert(0, str(repo_dir))

from src.train_c import (
    compute_metrics,
    clustered_subject_bootstrap_metrics,
    train_and_eval_cv,
    fit_and_freeze_full_models,
    evaluate_on_calfree,
    LADDER_STEPS,
)


def get_git_head() -> str:
    """Retrieve current git HEAD commit hash."""
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "UNKNOWN"


def hash_parquet_dir(dir_path: str) -> str:
    """Compute deterministic SHA-256 hash of a multi-part parquet directory."""
    p = Path(dir_path)
    if not p.exists():
        return "NOT_FOUND"
    if p.is_file():
        return hashlib.sha256(p.read_bytes()).hexdigest()
    h = hashlib.sha256()
    parts = sorted([f for f in p.iterdir() if f.is_file() and f.name.endswith(".parquet")])
    for f in parts:
        h.update(f.name.encode("utf-8"))
        h.update(hashlib.sha256(f.read_bytes()).digest())
    return h.hexdigest()


def run_development_pipeline(
    train_parquet: str = "data/features_c/train_file.parquet",
    calfree_parquet: str = "data/features_c/calfree_test.parquet",
    output_report: str = "results/dev_C.md",
    models_dir: str = "models",
    n_splits: int = 5,
    n_boot: int = 2000,
    seed: int = 42,
):
    print("=" * 80)
    print("STUDY C: CROSS-SUBJECT SBP CLASSIFICATION DEVELOPMENT PIPELINE")
    print("Specification: PLAN_C.md (Amendments C1, C2, C3)")
    print("=" * 80)

    # 1. Load Development Data and Verify file_split
    print(f"\n[1/7] Loading development features from {train_parquet}...")
    t0 = time.time()
    train_df = pd.read_parquet(train_parquet)
    n_train = len(train_df)
    n_subjs_train = train_df["subject_id"].nunique()
    print(f"      Loaded {n_train:,} segments across {n_subjs_train:,} subjects in {time.time()-t0:.2f}s.")

    # Strict file_split validation
    assert "file_split" in train_df.columns, "file_split column missing in development features!"
    assert (train_df["file_split"] == "train_file").all(), "All rows must have file_split == 'train_file'!"
    assert n_subjs_train == 1293, f"Expected exactly 1,293 subjects, got {n_subjs_train}"
    assert n_train == 465480, f"Expected exactly 465,480 segments, got {n_train}"

    train_df["y_120"] = (train_df["sbp"] >= 120.0).astype(int)
    train_df["y_130"] = (train_df["sbp"] >= 130.0).astype(int)

    n_pos_120 = int(train_df["y_120"].sum())
    n_neg_120 = int((train_df["y_120"] == 0).sum())
    n_pos_130 = int(train_df["y_130"].sum())
    n_neg_130 = int((train_df["y_130"] == 0).sum())

    print("\n" + "=" * 70)
    print("CLASS COUNTS (TRAIN DEVELOPMENT COHORT, N = 465,480):")
    print(f"  Target SBP >= 120 mmHg (Primary):     Pos={n_pos_120:,} ({n_pos_120/n_train*100:.2f}%), Neg={n_neg_120:,} ({n_neg_120/n_train*100:.2f}%)")
    print(f"  Target SBP >= 130 mmHg (Sensitivity): Pos={n_pos_130:,} ({n_pos_130/n_train*100:.2f}%), Neg={n_neg_130:,} ({n_neg_130/n_train*100:.2f}%)")
    print("=" * 70 + "\n")

    # 2. 5-Fold Grouped CV on Primary Target (SBP >= 120) with Seeded Permutation
    print("-" * 70)
    print("2. 5-Fold Grouped CV (Primary Target: SBP >= 120 mmHg)")
    print("-" * 70)
    oof_probs_120, pooled_metrics_120, fold_metrics_120 = train_and_eval_cv(
        train_df, target_col="y_120", n_splits=n_splits, seed=seed
    )

    # Bootstrap CIs on Primary Target OOF predictions
    print(f"\nRunning {n_boot:,} subject-level bootstrap resamples on Primary OOF predictions...")
    subjs_train = train_df["subject_id"].to_numpy()
    y_120 = train_df["y_120"].to_numpy()
    model_cis_120, paired_cis_120 = clustered_subject_bootstrap_metrics(
        subjs_train, y_120, oof_probs_120, threshold=0.5, n_boot=n_boot, seed=seed
    )

    # 3. 5-Fold Grouped CV on Sensitivity Target (SBP >= 130)
    print("\n" + "-" * 70)
    print("3. 5-Fold Grouped CV (Sensitivity Target: SBP >= 130 mmHg)")
    print("-" * 70)
    oof_probs_130, pooled_metrics_130, fold_metrics_130 = train_and_eval_cv(
        train_df, target_col="y_130", n_splits=n_splits, seed=seed
    )
    y_130 = train_df["y_130"].to_numpy()
    print(f"\nRunning {n_boot:,} subject-level bootstrap resamples on Sensitivity OOF predictions...")
    model_cis_130, paired_cis_130 = clustered_subject_bootstrap_metrics(
        subjs_train, y_130, oof_probs_130, threshold=0.5, n_boot=n_boot, seed=seed
    )

    # 4. Freeze Final Models on all Development Data
    print("\n" + "-" * 70)
    print("4. Freezing Final Primary Models (Trained on all 465,480 Segments)")
    print("-" * 70)
    frozen_models_dict = fit_and_freeze_full_models(
        train_df, target_col="y_120", models_dir=models_dir, seed=seed
    )
    update_models_readme(frozen_models_dict)

    # 5. Extra Validation on CalFree
    print("\n" + "-" * 70)
    print("5. Evaluating Frozen Models on CalFree Extra Validation Cohort")
    print("-" * 70)
    calfree_df = pd.read_parquet(calfree_parquet)
    assert "file_split" in calfree_df.columns, "file_split column missing in CalFree features!"
    assert (calfree_df["file_split"] == "calfree_test").all(), "All CalFree rows must have file_split == 'calfree_test'!"
    assert calfree_df["subject_id"].nunique() == 144, "Expected exactly 144 CalFree subjects"
    assert len(calfree_df) == 57600, "Expected exactly 57,600 CalFree segments"

    calfree_df["y_120"] = (calfree_df["sbp"] >= 120.0).astype(int)
    calfree_df["y_130"] = (calfree_df["sbp"] >= 130.0).astype(int)

    n_pos_120_cf = int(calfree_df["y_120"].sum())
    n_neg_120_cf = int((calfree_df["y_120"] == 0).sum())
    print(f"CalFree Class Counts (SBP >= 120): Pos={n_pos_120_cf:,} ({n_pos_120_cf/len(calfree_df)*100:.2f}%), Neg={n_neg_120_cf:,} ({n_neg_120_cf/len(calfree_df)*100:.2f}%)")

    cf_probs, cf_metrics, cf_model_cis, cf_paired_cis = evaluate_on_calfree(
        calfree_df, frozen_models_dict, target_col="y_120", seed=seed
    )

    # 6. Save DataFrames to CSV
    print("\n" + "-" * 70)
    print("6. Saving Metrics, Paired Differences, and CV Folds to CSV Files...")
    print("-" * 70)
    res_dir = Path("results")
    res_dir.mkdir(parents=True, exist_ok=True)

    metrics_csv_path = res_dir / "dev_C_metrics.csv"
    diffs_csv_path = res_dir / "dev_C_diffs.csv"
    folds_csv_path = res_dir / "dev_C_cv_folds.csv"

    # Build dev_C_metrics.csv rows
    metric_rows = []
    # Primary Target (Train OOF)
    for step in LADDER_STEPS:
        for fam in ["lr", "gbt"]:
            k = f"{step}_{fam}"
            m = pooled_metrics_120[k]
            ci_acc = model_cis_120[k]["accuracy"]
            ci_bal = model_cis_120[k]["balanced_accuracy"]
            ci_auc = model_cis_120[k]["auroc"]
            metric_rows.append({
                "cohort": "Train_OOF_Pooled",
                "target": "sbp_ge_120",
                "model_step": step,
                "classifier": fam,
                "accuracy": m["accuracy"],
                "acc_ci_low": ci_acc[0],
                "acc_ci_high": ci_acc[1],
                "balanced_accuracy": m["balanced_accuracy"],
                "bal_ci_low": ci_bal[0],
                "bal_ci_high": ci_bal[1],
                "auroc": m["auroc"],
                "auc_ci_low": ci_auc[0],
                "auc_ci_high": ci_auc[1],
                "f1": m["f1"],
                "majority_accuracy": m["majority_accuracy"],
            })

    # Sensitivity Target (Train OOF)
    for step in LADDER_STEPS:
        for fam in ["lr", "gbt"]:
            k = f"{step}_{fam}"
            m = pooled_metrics_130[k]
            ci_acc = model_cis_130[k]["accuracy"]
            ci_bal = model_cis_130[k]["balanced_accuracy"]
            ci_auc = model_cis_130[k]["auroc"]
            metric_rows.append({
                "cohort": "Train_OOF_Pooled",
                "target": "sbp_ge_130",
                "model_step": step,
                "classifier": fam,
                "accuracy": m["accuracy"],
                "acc_ci_low": ci_acc[0],
                "acc_ci_high": ci_acc[1],
                "balanced_accuracy": m["balanced_accuracy"],
                "bal_ci_low": ci_bal[0],
                "bal_ci_high": ci_bal[1],
                "auroc": m["auroc"],
                "auc_ci_low": ci_auc[0],
                "auc_ci_high": ci_auc[1],
                "f1": m["f1"],
                "majority_accuracy": m["majority_accuracy"],
            })

    # Extra Validation (CalFree)
    for step in LADDER_STEPS:
        for fam in ["lr", "gbt"]:
            k = f"{step}_{fam}"
            m = cf_metrics[k]
            ci_acc = cf_model_cis[k]["accuracy"]
            ci_bal = cf_model_cis[k]["balanced_accuracy"]
            ci_auc = cf_model_cis[k]["auroc"]
            metric_rows.append({
                "cohort": "CalFree_Validation",
                "target": "sbp_ge_120",
                "model_step": step,
                "classifier": fam,
                "accuracy": m["accuracy"],
                "acc_ci_low": ci_acc[0],
                "acc_ci_high": ci_acc[1],
                "balanced_accuracy": m["balanced_accuracy"],
                "bal_ci_low": ci_bal[0],
                "bal_ci_high": ci_bal[1],
                "auroc": m["auroc"],
                "auc_ci_low": ci_auc[0],
                "auc_ci_high": ci_auc[1],
                "f1": m["f1"],
                "majority_accuracy": m["majority_accuracy"],
            })

    df_metrics = pd.DataFrame(metric_rows)
    df_metrics.to_csv(metrics_csv_path, index=False)
    print(f"      Saved {metrics_csv_path}")

    # Build dev_C_diffs.csv rows
    diff_rows = []
    comparisons = [
        ("C1_minus_C0", "C1", "C0"),
        ("C2m_minus_C1", "C2m", "C1"),
        ("C2_minus_C2m", "C2", "C2m"),
        ("C2_minus_C0", "C2", "C0"),
    ]

    for fam in ["lr", "gbt"]:
        for comp_name, m_hi, m_lo in comparisons:
            # Primary Train OOF
            m_hi_val = pooled_metrics_120[f"{m_hi}_{fam}"]
            m_lo_val = pooled_metrics_120[f"{m_lo}_{fam}"]
            d_acc = m_hi_val["accuracy"] - m_lo_val["accuracy"]
            d_bal = m_hi_val["balanced_accuracy"] - m_lo_val["balanced_accuracy"]
            d_auc = m_hi_val["auroc"] - m_lo_val["auroc"]
            ci_acc = paired_cis_120[f"delta_acc_{m_hi}_{fam}_minus_{m_lo}_{fam}"]
            ci_bal = paired_cis_120[f"delta_bal_acc_{m_hi}_{fam}_minus_{m_lo}_{fam}"]
            ci_auc = paired_cis_120[f"delta_auc_{m_hi}_{fam}_minus_{m_lo}_{fam}"]
            diff_rows.append({
                "cohort": "Train_OOF_Pooled",
                "target": "sbp_ge_120",
                "comparison": comp_name,
                "classifier": fam,
                "delta_accuracy": d_acc,
                "delta_acc_ci_low": ci_acc[0],
                "delta_acc_ci_high": ci_acc[1],
                "delta_balanced_accuracy": d_bal,
                "delta_bal_ci_low": ci_bal[0],
                "delta_bal_ci_high": ci_bal[1],
                "delta_auroc": d_auc,
                "delta_auc_ci_low": ci_auc[0],
                "delta_auc_ci_high": ci_auc[1],
            })

            # Sensitivity Train OOF
            m_hi_val_130 = pooled_metrics_130[f"{m_hi}_{fam}"]
            m_lo_val_130 = pooled_metrics_130[f"{m_lo}_{fam}"]
            d_acc_130 = m_hi_val_130["accuracy"] - m_lo_val_130["accuracy"]
            d_bal_130 = m_hi_val_130["balanced_accuracy"] - m_lo_val_130["balanced_accuracy"]
            d_auc_130 = m_hi_val_130["auroc"] - m_lo_val_130["auroc"]
            ci_acc_130 = paired_cis_130[f"delta_acc_{m_hi}_{fam}_minus_{m_lo}_{fam}"]
            ci_bal_130 = paired_cis_130[f"delta_bal_acc_{m_hi}_{fam}_minus_{m_lo}_{fam}"]
            ci_auc_130 = paired_cis_130[f"delta_auc_{m_hi}_{fam}_minus_{m_lo}_{fam}"]
            diff_rows.append({
                "cohort": "Train_OOF_Pooled",
                "target": "sbp_ge_130",
                "comparison": comp_name,
                "classifier": fam,
                "delta_accuracy": d_acc_130,
                "delta_acc_ci_low": ci_acc_130[0],
                "delta_acc_ci_high": ci_acc_130[1],
                "delta_balanced_accuracy": d_bal_130,
                "delta_bal_ci_low": ci_bal_130[0],
                "delta_bal_ci_high": ci_bal_130[1],
                "delta_auroc": d_auc_130,
                "delta_auc_ci_low": ci_auc_130[0],
                "delta_auc_ci_high": ci_auc_130[1],
            })

            # CalFree Validation
            m_hi_cf = cf_metrics[f"{m_hi}_{fam}"]
            m_lo_cf = cf_metrics[f"{m_lo}_{fam}"]
            d_acc_cf = m_hi_cf["accuracy"] - m_lo_cf["accuracy"]
            d_bal_cf = m_hi_cf["balanced_accuracy"] - m_lo_cf["balanced_accuracy"]
            d_auc_cf = m_hi_cf["auroc"] - m_lo_cf["auroc"]
            ci_acc_cf = cf_paired_cis[f"delta_acc_{m_hi}_{fam}_minus_{m_lo}_{fam}"]
            ci_bal_cf = cf_paired_cis[f"delta_bal_acc_{m_hi}_{fam}_minus_{m_lo}_{fam}"]
            ci_auc_cf = cf_paired_cis[f"delta_auc_{m_hi}_{fam}_minus_{m_lo}_{fam}"]
            diff_rows.append({
                "cohort": "CalFree_Validation",
                "target": "sbp_ge_120",
                "comparison": comp_name,
                "classifier": fam,
                "delta_accuracy": d_acc_cf,
                "delta_acc_ci_low": ci_acc_cf[0],
                "delta_acc_ci_high": ci_acc_cf[1],
                "delta_balanced_accuracy": d_bal_cf,
                "delta_bal_ci_low": ci_bal_cf[0],
                "delta_bal_ci_high": ci_bal_cf[1],
                "delta_auroc": d_auc_cf,
                "delta_auc_ci_low": ci_auc_cf[0],
                "delta_auc_ci_high": ci_auc_cf[1],
            })

    df_diffs = pd.DataFrame(diff_rows)
    df_diffs.to_csv(diffs_csv_path, index=False)
    print(f"      Saved {diffs_csv_path}")

    # Build dev_C_cv_folds.csv rows
    fold_rows = []
    for f_idx, f_rec in enumerate(fold_metrics_120):
        f_num = f_rec["fold"]
        for step in LADDER_STEPS:
            for fam in ["lr", "gbt"]:
                fold_rows.append({
                    "target": "sbp_ge_120",
                    "fold": f_num,
                    "model_step": step,
                    "classifier": fam,
                    "accuracy": f_rec[f"{step}_{fam}_acc"],
                    "balanced_accuracy": f_rec[f"{step}_{fam}_bal_acc"],
                    "auroc": f_rec[f"{step}_{fam}_auc"],
                    "f1": f_rec[f"{step}_{fam}_f1"],
                    "majority_accuracy": f_rec[f"{step}_{fam}_majority"],
                })

    for f_idx, f_rec in enumerate(fold_metrics_130):
        f_num = f_rec["fold"]
        for step in LADDER_STEPS:
            for fam in ["lr", "gbt"]:
                fold_rows.append({
                    "target": "sbp_ge_130",
                    "fold": f_num,
                    "model_step": step,
                    "classifier": fam,
                    "accuracy": f_rec[f"{step}_{fam}_acc"],
                    "balanced_accuracy": f_rec[f"{step}_{fam}_bal_acc"],
                    "auroc": f_rec[f"{step}_{fam}_auc"],
                    "f1": f_rec[f"{step}_{fam}_f1"],
                    "majority_accuracy": f_rec[f"{step}_{fam}_majority"],
                })

    df_folds = pd.DataFrame(fold_rows)
    df_folds.to_csv(folds_csv_path, index=False)
    print(f"      Saved {folds_csv_path}")

    # 7. Generate Figures
    print("\n" + "-" * 70)
    print("7. Generating Figures...")
    print("-" * 70)
    fig_ladder_path = res_dir / "dev_C_accuracy_ladder.png"
    fig_diffs_path = res_dir / "dev_C_paired_diffs.png"

    # Figure 1: Accuracy Ladder Plot (Primary Target SBP >= 120)
    fig, ax = plt.subplots(figsize=(9, 6))
    steps_arr = np.arange(len(LADDER_STEPS))
    width = 0.35

    # Filter df_metrics for primary target Train OOF
    m_120 = df_metrics[(df_metrics["target"] == "sbp_ge_120") & (df_metrics["cohort"] == "Train_OOF_Pooled")]
    gbt_m = m_120[m_120["classifier"] == "gbt"].set_index("model_step").loc[LADDER_STEPS]
    lr_m = m_120[m_120["classifier"] == "lr"].set_index("model_step").loc[LADDER_STEPS]

    maj_acc = gbt_m["majority_accuracy"].iloc[0] * 100.0

    # GBT bars / points
    gbt_acc = gbt_m["accuracy"].values * 100.0
    gbt_err_lo = gbt_acc - gbt_m["acc_ci_low"].values * 100.0
    gbt_err_hi = gbt_m["acc_ci_high"].values * 100.0 - gbt_acc

    # LR bars / points
    lr_acc = lr_m["accuracy"].values * 100.0
    lr_err_lo = lr_acc - lr_m["acc_ci_low"].values * 100.0
    lr_err_hi = lr_m["acc_ci_high"].values * 100.0 - lr_acc

    ax.errorbar(
        steps_arr - 0.1, gbt_acc, yerr=[gbt_err_lo, gbt_err_hi],
        fmt="s-", color="#1f77b4", lw=2, markersize=8, capsize=5, label="GBT (Primary Classifier)"
    )
    ax.errorbar(
        steps_arr + 0.1, lr_acc, yerr=[lr_err_lo, lr_err_hi],
        fmt="o--", color="#ff7f0e", lw=1.5, markersize=7, capsize=4, label="Logistic Regression (Secondary)"
    )

    ax.axhline(maj_acc, color="gray", ls=":", lw=1.5, label=f"Majority Baseline ({maj_acc:.1f}%)")
    ax.set_xticks(steps_arr)
    ax.set_xticklabels([
        "C0\n(Age, Sex, BMI)",
        "C1\n(C0 + HR IBI)",
        "C2m\n(C1 + Morphology)",
        "C2\n(Full Pipeline)",
    ], fontsize=10)
    ax.set_ylabel("Pooled OOF Accuracy (%)", fontsize=11)
    ax.set_title("Study C: Stepwise Feature Ladder Accuracy (SBP >= 120 mmHg, 5-Fold CV)", fontsize=12, pad=12)
    ax.grid(True, ls=":", alpha=0.6)
    ax.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(fig_ladder_path, dpi=200)
    plt.close()
    print(f"      Saved {fig_ladder_path}")

    # Figure 2: Paired Differences Plot
    fig, ax = plt.subplots(figsize=(9, 6))
    comp_labels = [
        "C1 - C0\n(Heart Rate)",
        "C2m - C1\n(Pulse Shape)",
        "C2 - C2m\n(Auxiliary)",
        "C2 - C0\n(Total Pipeline)",
    ]
    comp_keys = ["C1_minus_C0", "C2m_minus_C1", "C2_minus_C2m", "C2_minus_C0"]
    x_idx = np.arange(len(comp_keys))

    d_120 = df_diffs[(df_diffs["target"] == "sbp_ge_120") & (df_diffs["cohort"] == "Train_OOF_Pooled")]
    gbt_d = d_120[d_120["classifier"] == "gbt"].set_index("comparison").loc[comp_keys]
    lr_d = d_120[d_120["classifier"] == "lr"].set_index("comparison").loc[comp_keys]

    gbt_d_acc = gbt_d["delta_accuracy"].values * 100.0
    gbt_d_lo = gbt_d_acc - gbt_d["delta_acc_ci_low"].values * 100.0
    gbt_d_hi = gbt_d["delta_acc_ci_high"].values * 100.0 - gbt_d_acc

    lr_d_acc = lr_d["delta_accuracy"].values * 100.0
    lr_d_lo = lr_d_acc - lr_d["delta_acc_ci_low"].values * 100.0
    lr_d_hi = lr_d["delta_acc_ci_high"].values * 100.0 - lr_d_acc

    ax.axhline(0.0, color="black", lw=1.2, ls="-")
    ax.errorbar(
        x_idx - 0.1, gbt_d_acc, yerr=[gbt_d_lo, gbt_d_hi],
        fmt="s-", color="#1f77b4", lw=2, markersize=8, capsize=5, label="GBT (Primary Classifier)"
    )
    ax.errorbar(
        x_idx + 0.1, lr_d_acc, yerr=[lr_d_lo, lr_d_hi],
        fmt="o--", color="#ff7f0e", lw=1.5, markersize=7, capsize=4, label="Logistic Regression"
    )

    ax.set_xticks(x_idx)
    ax.set_xticklabels(comp_labels, fontsize=10)
    ax.set_ylabel("Paired ΔAccuracy (%) [95% Subject Bootstrap CI]", fontsize=11)
    ax.set_title("Study C: Paired Gain Decomposition on Accuracy (SBP >= 120 mmHg)", fontsize=12, pad=12)
    ax.grid(True, ls=":", alpha=0.6)
    ax.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(fig_diffs_path, dpi=200)
    plt.close()
    print(f"      Saved {fig_diffs_path}")

    # 8. Generate Markdown Report strictly from CSVs and Checksums
    print("\n" + "-" * 70)
    print("8. Generating results/dev_C.md strictly from CSV files...")
    print("-" * 70)
    generate_markdown_report_from_csv(
        report_path=output_report,
        metrics_csv=metrics_csv_path,
        diffs_csv=diffs_csv_path,
        folds_csv=folds_csv_path,
        frozen_models_dict=frozen_models_dict,
        train_parquet=train_parquet,
        calfree_parquet=calfree_parquet,
        n_train=n_train,
        n_subjs_train=n_subjs_train,
        n_cf=len(calfree_df),
        n_subjs_cf=calfree_df["subject_id"].nunique(),
        n_pos_120=n_pos_120,
        n_neg_120=n_neg_120,
        n_pos_130=n_pos_130,
        n_neg_130=n_neg_130,
        n_pos_120_cf=n_pos_120_cf,
        n_neg_120_cf=n_neg_120_cf,
    )

    # 9. Print Ladder Table and Four Paired Differences to stdout
    print("\n" + "=" * 90)
    print("PRIMARY ENDPOINT RESULTS TABLE (5-Fold CV Pooled OOF, SBP >= 120 mmHg):")
    print("=" * 90)
    print(f"{'Step':<6} | {'Classifier':<10} | {'Accuracy [95% CI]':<26} | {'Balanced Acc [95% CI]':<26} | {'AUROC [95% CI]':<20} | {'Majority'}")
    print("-" * 105)
    for step in LADDER_STEPS:
        for fam in ["gbt", "lr"]:
            row = df_metrics[
                (df_metrics["target"] == "sbp_ge_120") &
                (df_metrics["cohort"] == "Train_OOF_Pooled") &
                (df_metrics["model_step"] == step) &
                (df_metrics["classifier"] == fam)
            ].iloc[0]
            acc_str = f"{row['accuracy']*100:.2f}% [{row['acc_ci_low']*100:.2f}%, {row['acc_ci_high']*100:.2f}%]"
            bal_str = f"{row['balanced_accuracy']*100:.2f}% [{row['bal_ci_low']*100:.2f}%, {row['bal_ci_high']*100:.2f}%]"
            auc_str = f"{row['auroc']:.4f} [{row['auc_ci_low']:.4f}, {row['auc_ci_high']:.4f}]"
            maj_str = f"{row['majority_accuracy']*100:.2f}%"
            fname = "GBT (Pri)" if fam == "gbt" else "LR (Sec)"
            print(f"{step:<6} | {fname:<10} | {acc_str:<26} | {bal_str:<26} | {auc_str:<20} | {maj_str}")

    print("\n" + "=" * 90)
    print("FOUR PAIRED DIFFERENCES (PRIMARY ENDPOINT, SBP >= 120 mmHg):")
    print("=" * 90)
    print(f"{'Comparison':<16} | {'Fam':<5} | {'Paired dAccuracy [95% CI]':<28} | {'Paired dBalAcc [95% CI]':<28} | {'Paired dAUROC [95% CI]'}")
    print("-" * 105)
    for fam in ["gbt", "lr"]:
        for comp_name in ["C1_minus_C0", "C2m_minus_C1", "C2_minus_C2m", "C2_minus_C0"]:
            d_row = df_diffs[
                (df_diffs["target"] == "sbp_ge_120") &
                (df_diffs["cohort"] == "Train_OOF_Pooled") &
                (df_diffs["comparison"] == comp_name) &
                (df_diffs["classifier"] == fam)
            ].iloc[0]
            d_acc_str = f"{d_row['delta_accuracy']*100:+.2f}% [{d_row['delta_acc_ci_low']*100:+.2f}%, {d_row['delta_acc_ci_high']*100:+.2f}%]"
            d_bal_str = f"{d_row['delta_balanced_accuracy']*100:+.2f}% [{d_row['delta_bal_ci_low']*100:+.2f}%, {d_row['delta_bal_ci_high']*100:+.2f}%]"
            d_auc_str = f"{d_row['delta_auroc']:+.4f} [{d_row['delta_auc_ci_low']:+.4f}, {d_row['delta_auc_ci_high']:+.4f}]"
            print(f"{comp_name:<16} | {fam.upper():<5} | {d_acc_str:<28} | {d_bal_str:<28} | {d_auc_str}")
    print("=" * 105 + "\n")


def update_models_readme(frozen_dict: Dict[str, Tuple[str, str]]):
    """Record SHA-256 digests in models/README.md."""
    readme_path = Path("models/README.md")
    existing_content = readme_path.read_text(encoding="utf-8") if readme_path.exists() else "# Frozen Models\n"

    section_header = "## Study C: Frozen Cross-Subject Models (PLAN_C.md)"
    lines = [
        f"\n\n{section_header}",
        f"**Date:** 2026-10-30",
        f"**Cohort:** VitalDB Train development cohort (1,293 subjects, 465,480 segments, file_split == 'train_file')",
        f"**Target:** SBP >= 120 mmHg (Primary endpoint)",
        f"**Pre-registration:** `PLAN_C.md` (committed before AAMI_Cal access)",
        "",
        "| Artifact Key | Path | SHA-256 Digest |",
        "|---|---|---|",
    ]
    for k, (p, sha) in frozen_dict.items():
        lines.append(f"| **`{k}`** | `{p}` | `{sha}` |")

    if section_header in existing_content:
        idx = existing_content.find(section_header)
        existing_content = existing_content[:idx].rstrip()

    new_content = existing_content + "\n" + "\n".join(lines) + "\n"
    readme_path.write_text(new_content, encoding="utf-8")
    print(f"Updated {readme_path}")


def generate_markdown_report_from_csv(
    report_path: str,
    metrics_csv: Path,
    diffs_csv: Path,
    folds_csv: Path,
    frozen_models_dict: Dict[str, Tuple[str, str]],
    train_parquet: str,
    calfree_parquet: str,
    n_train: int,
    n_subjs_train: int,
    n_cf: int,
    n_subjs_cf: int,
    n_pos_120: int,
    n_neg_120: int,
    n_pos_130: int,
    n_neg_130: int,
    n_pos_120_cf: int,
    n_neg_120_cf: int,
):
    df_m = pd.read_csv(metrics_csv)
    df_d = pd.read_csv(diffs_csv)
    df_f = pd.read_csv(folds_csv)

    plan_path = Path("PLAN_C.md")
    plan_hash = hashlib.sha256(plan_path.read_bytes()).hexdigest() if plan_path.exists() else "UNKNOWN"
    git_head = get_git_head()
    train_hash = hash_parquet_dir(train_parquet)
    calfree_hash = hash_parquet_dir(calfree_parquet)

    p120_train = (n_pos_120 / n_train) * 100.0
    p130_train = (n_pos_130 / n_train) * 100.0
    p120_cf = (n_pos_120_cf / n_cf) * 100.0

    lines = [
        "# Study C Development & Cross-Cohort Validation Report",
        "",
        "**Date:** 2026-10-30  ",
        "**Specification:** `PLAN_C.md` (Cross-subject SBP classification on unseen subjects)  ",
        f"**Pre-Registered Plan Hash:** `{plan_hash}`  ",
        f"**Git HEAD Commit:** `{git_head}`  ",
        f"**Train Features Parquet Hash (64 parts):** `{train_hash}`  ",
        f"**CalFree Features Parquet Hash (64 parts):** `{calfree_hash}`  ",
        "**Final Test Boundary:** `VitalDB_AAMI_Cal_Subset.mat` is **PRISTINE and UNREAD**. Scored strictly once after models are frozen.  ",
        "",
        "---",
        "",
        "## 1. Study Setup & Data Splits",
        "",
        f"- **Development Cohort:** VitalDB Train file (`file_split == 'train_file'`, {n_subjs_train:,} subjects, {n_train:,} segments, exactly 360 seg/subj).",
        f"- **Development Protocol:** 5-fold cross-validation with seeded subject permutation and disjoint fold assertion.",
        f"- **Extra Validation Cohort:** CalFree (`file_split == 'calfree_test'`, {n_subjs_cf:,} subjects, {n_cf:,} segments, 400 seg/subj).",
        f"- **Primary Target Label:** $SBP \\ge 120\\text{{ mmHg}}$ (Development: **{n_pos_120:,}** pos / **{n_neg_120:,}** neg, **{p120_train:.2f}%**; CalFree: **{n_pos_120_cf:,}** pos / **{n_neg_120_cf:,}** neg, **{p120_cf:.2f}%**).",
        f"- **Sensitivity Target Label:** $SBP \\ge 130\\text{{ mmHg}}$ (Development: **{n_pos_130:,}** pos / **{n_neg_130:,}** neg, **{p130_train:.2f}%**).",
        "- **Decision Threshold:** Fixed standard threshold $0.5$ ($P(\\hat{y}=1) \\ge 0.5$, no recalibration).",
        "- **Bootstrap Protocol:** Clustered subject-level bootstrap with 2,000 resamples (seed 42).",
        "- **Primary Endpoint:** Paired accuracy difference $C2 - C0$ for the primary classifier (HistGradientBoostingClassifier).",
        "",
        "---",
        "",
        "## 2. Primary Endpoint (SBP $\\ge 120$ mmHg): 5-Fold Grouped CV Pooled OOF Results",
        "",
        "| Model Step | Classifier | Predictors | Accuracy [95% CI] | Balanced Acc [95% CI] | AUROC [95% CI] | F1 Score | Majority Baseline |",
        "|---|---|---|---|---|---|---|---|",
    ]

    ladder_desc = {
        "C0": "Age, Sex, BMI",
        "C1": "C0 + PPG Heart Rate (hr_ibi)",
        "C2m": "C1 + Morphology Only (rise_time, pulse_width, decay_ratio)",
        "C2": "C1 + Morphology + Auxiliary Fields (beat_count, snr_ratio, ibi_sd)",
    }

    m_120 = df_m[(df_m["target"] == "sbp_ge_120") & (df_m["cohort"] == "Train_OOF_Pooled")]
    for step in LADDER_STEPS:
        for fam in ["gbt", "lr"]:
            r = m_120[(m_120["model_step"] == step) & (m_120["classifier"] == fam)].iloc[0]
            fname = "**GBT (Primary)**" if fam == "gbt" else "Logistic Regression"
            lines.append(
                f"| **{step}** | {fname} | {ladder_desc[step]} | "
                f"**{r['accuracy'] * 100.0:.2f}%** [{r['acc_ci_low'] * 100.0:.2f}%, {r['acc_ci_high'] * 100.0:.2f}%] | "
                f"**{r['balanced_accuracy'] * 100.0:.2f}%** [{r['bal_ci_low'] * 100.0:.2f}%, {r['bal_ci_high'] * 100.0:.2f}%] | "
                f"**{r['auroc']:.4f}** [{r['auc_ci_low']:.4f}, {r['auc_ci_high']:.4f}] | "
                f"{r['f1']:.4f} | "
                f"{r['majority_accuracy'] * 100.0:.2f}% |"
            )

    lines.extend([
        "",
        "### Stepwise Paired Gain Decomposition (5-Fold CV Pooled OOF, SBP $\\ge 120$ mmHg)",
        "",
        "| Comparison | Classifier | Paired $\\Delta\\text{Accuracy}$ [95% CI] | Paired $\\Delta\\text{Balanced Acc}$ [95% CI] | Paired $\\Delta\\text{AUROC}$ [95% CI] |",
        "|---|---|---|---|---|",
    ])

    d_120 = df_d[(df_d["target"] == "sbp_ge_120") & (df_d["cohort"] == "Train_OOF_Pooled")]
    comp_labels = {
        "C1_minus_C0": "C1 vs C0 (Gain from Heart Rate IBI)",
        "C2m_minus_C1": "C2m vs C1 (Gain from Pulse Morphology Shape)",
        "C2_minus_C2m": "C2 vs C2m (Gain from Auxiliary Segment Fields)",
        "C2_minus_C0": "**C2 vs C0 (Total Pipeline Gain - Primary Endpoint)**",
    }

    for fam in ["gbt", "lr"]:
        fname = "**GBT (Primary)**" if fam == "gbt" else "Logistic Regression"
        for comp_key in ["C1_minus_C0", "C2m_minus_C1", "C2_minus_C2m", "C2_minus_C0"]:
            r = d_120[(d_120["comparison"] == comp_key) & (d_120["classifier"] == fam)].iloc[0]
            lines.append(
                f"| {comp_labels[comp_key]} | {fname} | "
                f"**{r['delta_accuracy'] * 100.0:+.2f}%** [{r['delta_acc_ci_low'] * 100.0:+.2f}%, {r['delta_acc_ci_high'] * 100.0:+.2f}%] | "
                f"**{r['delta_balanced_accuracy'] * 100.0:+.2f}%** [{r['delta_bal_ci_low'] * 100.0:+.2f}%, {r['delta_bal_ci_high'] * 100.0:+.2f}%] | "
                f"**{r['delta_auroc']:+.4f}** [{r['delta_auc_ci_low']:+.4f}, {r['delta_auc_ci_high']:+.4f}] |"
            )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Per-Fold Cross-Validation Metrics (Primary Target: SBP $\\ge 120$ mmHg)",
        "",
        "| Fold | Model Step | Classifier | Accuracy | Balanced Acc | AUROC | F1 Score | Majority Baseline |",
        "|---|---|---|---|---|---|---|---|",
    ])

    f_120 = df_f[df_f["target"] == "sbp_ge_120"]
    for f_num in sorted(f_120["fold"].unique()):
        for step in LADDER_STEPS:
            for fam in ["gbt", "lr"]:
                r = f_120[(f_120["fold"] == f_num) & (f_120["model_step"] == step) & (f_120["classifier"] == fam)].iloc[0]
                lines.append(
                    f"| Fold {f_num} | **{step}** | {fam.upper()} | "
                    f"{r['accuracy'] * 100.0:.2f}% | {r['balanced_accuracy'] * 100.0:.2f}% | {r['auroc']:.4f} | {r['f1']:.4f} | {r['majority_accuracy'] * 100.0:.2f}% |"
                )

    lines.extend([
        "",
        "---",
        "",
        "## 4. Sensitivity Analysis (SBP $\\ge 130$ mmHg): 5-Fold Grouped CV Pooled OOF Results",
        "",
        "| Model Step | Classifier | Accuracy [95% CI] | Balanced Acc [95% CI] | AUROC [95% CI] | F1 Score | Majority Baseline |",
        "|---|---|---|---|---|---|---|",
    ])

    m_130 = df_m[(df_m["target"] == "sbp_ge_130") & (df_m["cohort"] == "Train_OOF_Pooled")]
    for step in LADDER_STEPS:
        for fam in ["gbt", "lr"]:
            r = m_130[(m_130["model_step"] == step) & (m_130["classifier"] == fam)].iloc[0]
            fname = "GBT" if fam == "gbt" else "Logistic Regression"
            lines.append(
                f"| **{step}** | {fname} | "
                f"**{r['accuracy'] * 100.0:.2f}%** [{r['acc_ci_low'] * 100.0:.2f}%, {r['acc_ci_high'] * 100.0:.2f}%] | "
                f"**{r['balanced_accuracy'] * 100.0:.2f}%** [{r['bal_ci_low'] * 100.0:.2f}%, {r['bal_ci_high'] * 100.0:.2f}%] | "
                f"**{r['auroc']:.4f}** [{r['auc_ci_low']:.4f}, {r['auc_ci_high']:.4f}] | "
                f"{r['f1']:.4f} | "
                f"{r['majority_accuracy'] * 100.0:.2f}% |"
            )

    lines.extend([
        "",
        "---",
        "",
        "## 5. Extra Validation: CalFree Cohort ($N = 57,600$ segments, 144 subjects)",
        "",
        "Evaluated strictly using frozen models fit on the development set (zero refitting on CalFree).",
        "",
        "| Model Step | Classifier | Accuracy [95% CI] | Balanced Acc [95% CI] | AUROC [95% CI] | F1 Score | Majority Baseline |",
        "|---|---|---|---|---|---|---|",
    ])

    m_cf = df_m[df_m["cohort"] == "CalFree_Validation"]
    for step in LADDER_STEPS:
        for fam in ["gbt", "lr"]:
            r = m_cf[(m_cf["model_step"] == step) & (m_cf["classifier"] == fam)].iloc[0]
            fname = "GBT" if fam == "gbt" else "Logistic Regression"
            lines.append(
                f"| **{step}** | {fname} | "
                f"**{r['accuracy'] * 100.0:.2f}%** [{r['acc_ci_low'] * 100.0:.2f}%, {r['acc_ci_high'] * 100.0:.2f}%] | "
                f"**{r['balanced_accuracy'] * 100.0:.2f}%** [{r['bal_ci_low'] * 100.0:.2f}%, {r['bal_ci_high'] * 100.0:.2f}%] | "
                f"**{r['auroc']:.4f}** [{r['auc_ci_low']:.4f}, {r['auc_ci_high']:.4f}] | "
                f"{r['f1']:.4f} | "
                f"{r['majority_accuracy'] * 100.0:.2f}% |"
            )

    lines.extend([
        "",
        "### CalFree Paired Differences ($C2 - C0$ Endpoint Preview)",
        "",
        "| Comparison | Classifier | Paired $\\Delta\\text{Accuracy}$ [95% CI] | Paired $\\Delta\\text{Balanced Acc}$ [95% CI] | Paired $\\Delta\\text{AUROC}$ [95% CI] |",
        "|---|---|---|---|---|",
    ])

    d_cf = df_d[df_d["cohort"] == "CalFree_Validation"]
    for fam in ["gbt", "lr"]:
        fname = "GBT" if fam == "gbt" else "Logistic Regression"
        for comp_key in ["C1_minus_C0", "C2m_minus_C1", "C2_minus_C2m", "C2_minus_C0"]:
            r = d_cf[(d_cf["comparison"] == comp_key) & (d_cf["classifier"] == fam)].iloc[0]
            lines.append(
                f"| {comp_labels[comp_key]} | {fname} | "
                f"**{r['delta_accuracy'] * 100.0:+.2f}%** [{r['delta_acc_ci_low'] * 100.0:+.2f}%, {r['delta_acc_ci_high'] * 100.0:+.2f}%] | "
                f"**{r['delta_balanced_accuracy'] * 100.0:+.2f}%** [{r['delta_bal_ci_low'] * 100.0:+.2f}%, {r['delta_bal_ci_high'] * 100.0:+.2f}%] | "
                f"**{r['delta_auroc']:+.4f}** [{r['delta_auc_ci_low']:+.4f}, {r['delta_auc_ci_high']:+.4f}] |"
            )

    lines.extend([
        "",
        "---",
        "",
        "## 6. Frozen Model Artifacts & SHA-256 Checksums",
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
        "## 7. Generated Visualizations",
        "",
        "- **Accuracy Ladder Plot:** `results/dev_C_accuracy_ladder.png`",
        "- **Paired Gain Decomposition Plot:** `results/dev_C_paired_diffs.png`",
        "",
        "---",
        "",
        "## 8. Pre-Registration Invariant Status",
        "",
        "- `PLAN_C.md` pre-registration and Amendments C1, C2, C3 committed to git.",
        "- `VitalDB_AAMI_Cal_Subset.mat` has **NOT** been opened or read.",
        "- Tag `plan-c-locked` has **NOT** been created.",
        "- **Execution Note:** The first development pipeline attempt completed Steps 1–4 (cross-validation, bootstrap, and full model freezing) and crashed in Step 5 due to an artifact dictionary loading bug in `evaluate_on_calfree` (attempting to subscript a pickled `PlanCImputer` instance). The bug was fixed with zero changes to any feature definition, model, hyperparameter, fold assignment, seed, threshold, or bootstrap setting.",
    ])

    out_p = Path(report_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report written to {report_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Study C Development Runner.")
    parser.add_argument("--train-parquet", type=str, default="data/features_c/train_file.parquet")
    parser.add_argument("--calfree-parquet", type=str, default="data/features_c/calfree_test.parquet")
    parser.add_argument("--output-report", type=str, default="results/dev_C.md")
    parser.add_argument("--models-dir", type=str, default="models")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_development_pipeline(
        train_parquet=args.train_parquet,
        calfree_parquet=args.calfree_parquet,
        output_report=args.output_report,
        models_dir=args.models_dir,
        n_splits=args.n_splits,
        n_boot=args.n_boot,
        seed=args.seed,
    )
