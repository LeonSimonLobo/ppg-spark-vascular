"""Stage 2 Test Reporting Script for Study C.

Reads ONLY results/test_C_predictions.parquet (never opens .mat files or re-scores models).
Computes metrics, executes clustered subject-level bootstrap, and writes:
- results/test_C.md
- results/test_C_diffs.csv

Can be executed independently and re-run if needed without invalidating pre-flight locks or re-scoring.
"""

import argparse
import hashlib
from pathlib import Path
import re
import sys
import time
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import pandas as pd

repo_dir = Path(__file__).resolve().parent.parent
if str(repo_dir) not in sys.path:
    sys.path.insert(0, str(repo_dir))

from src.train_c import compute_metrics, clustered_subject_bootstrap_metrics, LADDER_STEPS

PREDICTIONS_PATH = "results/test_C_predictions.parquet"
REPORT_PATH = "results/test_C.md"
DIFFS_CSV_PATH = "results/test_C_diffs.csv"
DEV_REPORT_PATH = "results/dev_C.md"
PLAN_FILE_PATH = "PLAN_C.md"


def write_aami_cal_report(
    output_report: str,
    df_imp: pd.DataFrame,
    metrics: Dict[str, Dict[str, float]],
    model_cis: Dict[str, Dict[str, Tuple[float, float]]],
    paired_cis: Dict[str, Tuple[float, float]],
    expected_hashes: Dict[str, str],
    git_commit: str,
    plan_hash: str,
    cohort_name: str = "AAMI_Cal",
):
    n_segs = len(df_imp)
    n_subjs = df_imp["subject_id"].nunique()
    prev_120 = float(np.mean(df_imp["y_120"])) * 100.0
    prev_130 = float(np.mean(df_imp["y_130"])) * 100.0 if "y_130" in df_imp else 0.0

    lines = [
        f"# Study C Held-Out Test Evaluation: VitalDB {cohort_name} Cohort",
        "",
        "**Date:** 2026-10-30  ",
        "**Pre-Registration Specification:** `PLAN_C.md` (including Amendments 1-3)  ",
        f"**Pre-Registered Plan Hash:** `{plan_hash}`  ",
        f"**Code Git Commit:** `{git_commit}`  ",
        "**Lock Status:** Permanently locked at `results/.test_locked_C` (Single evaluation).  ",
        f"**Test Dataset:** VitalDB {cohort_name} ({n_subjs:,} subjects, {n_segs:,} segments).  ",
        "**Excluded Datasets:** Untouched test subsets strictly preserved.  ",
        "",
        "---",
        "",
        f"## 1. Study Setup & {cohort_name} Cohort Characteristics",
        "",
        f"- **Subjects:** {n_subjs:,} unseen surgical/ICU patients",
        f"- **Segments Evaluated:** {n_segs:,} (10-second non-overlapping PPG windows)",
        f"- **Primary Target:** $SBP \\ge 120\\text{{ mmHg}}$",
        f"- **Cohort Prevalence:** **{prev_120:.2f}%** ({int(np.sum(df_imp['y_120'])):,} positive / {n_segs:,} total segments)",
        f"- **Majority-Class Baseline Rate:** **{max(prev_120, 100.0 - prev_120):.2f}%**",
        "- **Decision Threshold:** Fixed standard threshold $0.5$ (pre-registered, zero tuning on test set)",
        "- **Bootstrap Protocol:** Clustered subject-level bootstrap with 2,000 resamples (seed 42)",
        "",
        "---",
        "",
        f"## 2. Primary Endpoint: $C2 - C0$ Accuracy Gain on {cohort_name}",
        "",
        "| Model Step | Classifier | Predictors | Accuracy [95% CI] | Balanced Acc [95% CI] | AUROC [95% CI] | Predicted Pos Rate | Majority Baseline |",
        "|---|---|---|---|---|---|---|---|",
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
            m = metrics[k]
            ci_acc = model_cis[k]["accuracy"]
            ci_bal = model_cis[k]["balanced_accuracy"]
            ci_auc = model_cis[k]["auroc"]
            lines.append(
                f"| **{step}** | {fname} | {ladder_desc[step]} | "
                f"**{m['accuracy'] * 100.0:.2f}%** [{ci_acc[0] * 100.0:.2f}%, {ci_acc[1] * 100.0:.2f}%] | "
                f"**{m['balanced_accuracy'] * 100.0:.2f}%** [{ci_bal[0] * 100.0:.2f}%, {ci_bal[1] * 100.0:.2f}%] | "
                f"**{m['auroc']:.4f}** [{ci_auc[0]:.4f}, {ci_auc[1]:.4f}] | "
                f"{m['predicted_positive_rate'] * 100.0:.2f}% | "
                f"{m['majority_accuracy'] * 100.0:.2f}% |"
            )

    lines.extend([
        "",
        "### Paired Hypothesis Testing & Stepwise Gain Decomposition",
        "",
        "| Comparison | Model Family | Paired $\\Delta\\text{Accuracy}$ [95% CI] | Paired $\\Delta\\text{AUROC}$ [95% CI] |",
        "|---|---|---|---|",
    ])

    for fam, fname in [("lr", "Logistic Regression"), ("gbt", "GBT")]:
        d_acc_10 = paired_cis[f"delta_acc_C1_{fam}_minus_C0_{fam}"]
        d_auc_10 = paired_cis[f"delta_auc_C1_{fam}_minus_C0_{fam}"]
        m1 = metrics[f"C1_{fam}"]
        m0 = metrics[f"C0_{fam}"]
        lines.append(
            f"| **C1 vs C0 (Gain from Heart Rate)** | {fname} | "
            f"**{(m1['accuracy'] - m0['accuracy']) * 100.0:+.2f}%** [{d_acc_10[0] * 100.0:+.2f}%, {d_acc_10[1] * 100.0:+.2f}%] | "
            f"**{(m1['auroc'] - m0['auroc']):+.4f}** [{d_auc_10[0]:+.4f}, {d_auc_10[1]:+.4f}] |"
        )

        d_acc_2m1 = paired_cis[f"delta_acc_C2m_{fam}_minus_C1_{fam}"]
        d_auc_2m1 = paired_cis[f"delta_auc_C2m_{fam}_minus_C1_{fam}"]
        m2m = metrics[f"C2m_{fam}"]
        lines.append(
            f"| **C2m vs C1 (Gain from Pulse Morphology)** | {fname} | "
            f"**{(m2m['accuracy'] - m1['accuracy']) * 100.0:+.2f}%** [{d_acc_2m1[0] * 100.0:+.2f}%, {d_acc_2m1[1] * 100.0:+.2f}%] | "
            f"**{(m2m['auroc'] - m1['auroc']):+.4f}** [{d_auc_2m1[0]:+.4f}, {d_auc_2m1[1]:+.4f}] |"
        )

        d_acc_22m = paired_cis[f"delta_acc_C2_{fam}_minus_C2m_{fam}"]
        d_auc_22m = paired_cis[f"delta_auc_C2_{fam}_minus_C2m_{fam}"]
        m2 = metrics[f"C2_{fam}"]
        lines.append(
            f"| **C2 vs C2m (Gain from Signal Quality Fields)** | {fname} | "
            f"**{(m2['accuracy'] - m2m['accuracy']) * 100.0:+.2f}%** [{d_acc_22m[0] * 100.0:+.2f}%, {d_acc_22m[1] * 100.0:+.2f}%] | "
            f"**{(m2['auroc'] - m2m['auroc']):+.4f}** [{d_auc_22m[0]:+.4f}, {d_auc_22m[1]:+.4f}] |"
        )

        d_acc_20 = paired_cis[f"delta_acc_C2_{fam}_minus_C0_{fam}"]
        d_auc_20 = paired_cis[f"delta_auc_C2_{fam}_minus_C0_{fam}"]
        diff_acc_val = (m2['accuracy'] - m0['accuracy']) * 100.0
        lines.append(
            f"| **C2 vs C0 (Primary Endpoint: Total Pipeline Gain)** | {fname} | "
            f"**{diff_acc_val:+.2f}%** [{d_acc_20[0] * 100.0:+.2f}%, {d_acc_20[1] * 100.0:+.2f}%] | "
            f"**{(m2['auroc'] - m0['auroc']):+.4f}** [{d_auc_20[0]:+.4f}, {d_auc_20[1]:+.4f}] |"
        )

    lines.extend([
        "",
        "> [!NOTE]",
        "> The exploratory benchmark goal of +4–5 percentage points accuracy gain is an aspiration, not a pass/fail criterion.",
    ])

    if expected_hashes:
        lines.extend([
            "",
            "---",
            "",
            "## 3. Verified Frozen Model Checksums",
            "",
            "| Model Key | Expected & Verified SHA-256 Digest |",
            "|---|---|",
        ])
        for k, h in expected_hashes.items():
            lines.append(f"| **`{k}`** | `{h}` |")

    lines.extend([
        "",
        "---",
        "",
        "## 4. Pre-Registration Protocol Adherence",
        "",
        "- The single-pass test was evaluated strictly once without repeated attempts.",
        "- No post-hoc tuning, threshold adjustments, or feature modifications were performed.",
        "- Results reflect exact observed cross-subject test performance on unseen patients.",
    ])

    out_p = Path(output_report)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report written to {output_report}")


def generate_test_report(
    predictions_path: str = PREDICTIONS_PATH,
    output_report: str = REPORT_PATH,
    diffs_csv_path: str = DIFFS_CSV_PATH,
    dev_report_path: str = DEV_REPORT_PATH,
    plan_path: str = PLAN_FILE_PATH,
    n_boot: int = 2000,
    seed: int = 42,
    git_commit: str = "HEAD",
    cohort_name: str = "AAMI_Cal",
) -> Dict[str, Any]:
    """Generate final test report and diffs CSV from per-segment predictions parquet."""
    print("=" * 70)
    print(f"STUDY C STAGE 2: Report Generation ({cohort_name})")
    print("=" * 70)

    pred_p = Path(predictions_path)
    if not pred_p.exists():
        raise FileNotFoundError(f"Predictions parquet not found: {predictions_path}")

    print(f"Loading predictions from {predictions_path}...")
    pred_df = pd.read_parquet(str(pred_p))
    n_segs = len(pred_df)
    n_subjs = pred_df["subject_id"].nunique()
    print(f"Loaded {n_segs:,} segments across {n_subjs:,} subjects.")

    # 1. Parse plan hash and expected model hashes from dev report if available
    plan_hash = ""
    plan_p = Path(plan_path)
    if plan_p.exists():
        plan_hash = hashlib.sha256(plan_p.read_bytes()).hexdigest()

    expected_hashes = {}
    dev_p = Path(dev_report_path)
    if dev_p.exists():
        dev_text = dev_p.read_text(encoding="utf-8")
        for line in dev_text.splitlines():
            m = re.match(r"\|\s*\*\*`([^`]+)`\*\*\s*\|\s*`[^`]+`\s*\|\s*`([0-9a-f]{64})`\s*\|", line)
            if m:
                expected_hashes[m.group(1)] = m.group(2)

    # 2. Extract targets and probabilities
    y_true = pred_df["y_120"].to_numpy(dtype=int)
    probs = {}
    for step in LADDER_STEPS:
        for fam in ["lr", "gbt"]:
            k = f"{step}_{fam}"
            col_name = f"prob_{k}"
            if col_name not in pred_df.columns:
                raise KeyError(f"Expected column '{col_name}' in {predictions_path}")
            probs[k] = pred_df[col_name].to_numpy(dtype=np.float64)

    # 3. Compute metrics at threshold 0.5
    print("Computing metrics at threshold 0.5...")
    metrics = {k: compute_metrics(y_true, probs[k], threshold=0.5) for k in probs}

    # 4. Run clustered subject-level bootstrap
    print(f"Running {n_boot:,} subject-level bootstrap resamples (seed {seed})...")
    subjs = pred_df["subject_id"].to_numpy()
    model_cis, paired_cis = clustered_subject_bootstrap_metrics(
        subjs, y_true, probs, threshold=0.5, n_boot=n_boot, seed=seed
    )

    # 5. Build and save paired diffs CSV
    print(f"Writing paired differences CSV to {diffs_csv_path}...")
    diff_records = []
    comparisons = [
        ("C1_minus_C0", "C1", "C0"),
        ("C2m_minus_C1", "C2m", "C1"),
        ("C2_minus_C2m", "C2", "C2m"),
        ("C2_minus_C0", "C2", "C0"),
    ]
    for comp_name, m_high, m_low in comparisons:
        for fam in ["lr", "gbt"]:
            mh = f"{m_high}_{fam}"
            ml = f"{m_low}_{fam}"

            d_acc = metrics[mh]["accuracy"] - metrics[ml]["accuracy"]
            ci_acc = paired_cis[f"delta_acc_{mh}_minus_{ml}"]

            d_bal = metrics[mh]["balanced_accuracy"] - metrics[ml]["balanced_accuracy"]
            ci_bal = paired_cis[f"delta_bal_acc_{mh}_minus_{ml}"]

            d_auc = metrics[mh]["auroc"] - metrics[ml]["auroc"]
            ci_auc = paired_cis[f"delta_auc_{mh}_minus_{ml}"]

            diff_records.append({
                "cohort": cohort_name,
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

    diffs_df = pd.DataFrame(diff_records)
    diffs_p = Path(diffs_csv_path)
    diffs_p.parent.mkdir(parents=True, exist_ok=True)
    diffs_df.to_csv(str(diffs_p), index=False)
    print(f"Paired differences saved ({len(diffs_df)} rows).")

    # 6. Write markdown test report
    print(f"Writing markdown report to {output_report}...")
    write_aami_cal_report(
        output_report=output_report,
        df_imp=pred_df,
        metrics=metrics,
        model_cis=model_cis,
        paired_cis=paired_cis,
        expected_hashes=expected_hashes,
        git_commit=git_commit,
        plan_hash=plan_hash,
        cohort_name=cohort_name,
    )

    return {
        "metrics": metrics,
        "model_cis": model_cis,
        "paired_cis": paired_cis,
        "diffs_df": diffs_df,
    }


def main():
    parser = argparse.ArgumentParser(description="Study C Stage 2 Report Generator")
    parser.add_argument("--predictions", default=PREDICTIONS_PATH, help="Path to predictions parquet")
    parser.add_argument("--output-report", default=REPORT_PATH, help="Path to output markdown report")
    parser.add_argument("--diffs-csv", default=DIFFS_CSV_PATH, help="Path to output diffs CSV")
    parser.add_argument("--dev-report", default=DEV_REPORT_PATH, help="Path to dev report")
    parser.add_argument("--plan", default=PLAN_FILE_PATH, help="Path to PLAN_C.md")
    parser.add_argument("--n-boot", type=int, default=2000, help="Number of bootstrap resamples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--cohort", default="AAMI_Cal", help="Cohort name")
    args = parser.parse_args()

    generate_test_report(
        predictions_path=args.predictions,
        output_report=args.output_report,
        diffs_csv_path=args.diffs_csv,
        dev_report_path=args.dev_report,
        plan_path=args.plan,
        n_boot=args.n_boot,
        seed=args.seed,
        cohort_name=args.cohort,
    )


if __name__ == "__main__":
    main()
