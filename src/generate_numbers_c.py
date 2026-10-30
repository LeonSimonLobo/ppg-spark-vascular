"""Generate results/NUMBERS_C.md compiling all headline numbers with sources."""

from pathlib import Path
import sys

repo_dir = Path(__file__).resolve().parent.parent
if str(repo_dir) not in sys.path:
    sys.path.insert(0, str(repo_dir))

import pandas as pd


def main():
    repo_dir = Path(__file__).resolve().parent.parent
    dev_diffs_path = repo_dir / "results/dev_C_diffs.csv"
    dev_metrics_path = repo_dir / "results/dev_C_metrics.csv"
    test_diffs_path = repo_dir / "results/test_C_diffs.csv"
    benchmark_path = repo_dir / "results/benchmark.csv"
    out_path = repo_dir / "results/NUMBERS_C.md"

    dev_diffs = pd.read_csv(dev_diffs_path)
    dev_metrics = pd.read_csv(dev_metrics_path)
    test_diffs = pd.read_csv(test_diffs_path)

    lines = [
        "# Study C Headline Numbers & Source Traceability Matrix",
        "",
        "This table aggregates every headline number reported for Study C across development, validation, and held-out test evaluation, mapped to its exact source file and column.",
        "",
        "---",
        "",
        "## 1. Cohort Sizes, Prevalences, and Class Counts",
        "",
        "| Cohort | Role | Subjects | Segments | Target | Positives (%) | Negatives (%) | Source File & Field |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        "| **Train Development** | 5-Fold Grouped CV | 1,293 | 465,480 | $SBP \\ge 120$ | 175,338 (37.67%) | 290,142 (62.33%) | `results/dev_C.md`: Section 1 |",
        "| **Train Development** | Sensitivity | 1,293 | 465,480 | $SBP \\ge 130$ | 98,290 (21.12%) | 367,190 (78.88%) | `results/dev_C.md`: Section 1 |",
        "| **CalFree Validation** | Prospective Validation | 144 | 57,600 | $SBP \\ge 120$ | 21,713 (37.70%) | 35,887 (62.30%) | `results/dev_C.md`: Section 1 |",
        "| **AAMI_Cal Test** | Final Held-Out Evaluation | 116 | 70,212 | $SBP \\ge 120$ | 28,811 (41.03%) | 41,401 (58.97%) | `results/test_C.md`: Section 1 |",
        "",
        "---",
        "",
        "## 2. Three-Cohort Primary Endpoint Gain ($C2 - C0$ Total Pipeline Gain)",
        "",
        "| Cohort | Target | Classifier | Metric | Point Estimate | 95% Confidence Interval | Source File & Column |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    # Combine dev and test diffs for C2 vs C0
    all_diffs = pd.concat([dev_diffs, test_diffs], ignore_index=True)
    c2_c0 = all_diffs[all_diffs["comparison"] == "C2_minus_C0"]

    for _, r in c2_c0.iterrows():
        cohort = r["cohort"]
        target = r["target"]
        clf = r["classifier"].upper()
        src_file = "results/test_C_diffs.csv" if cohort == "AAMI_Cal" else "results/dev_C_diffs.csv"

        # Accuracy
        lines.append(
            f"| `{cohort}` | `{target}` | **{clf}** | Accuracy Gain | **{r['delta_accuracy']*100:+.2f}%** | "
            f"[{r['delta_acc_ci_low']*100:+.2f}%, {r['delta_acc_ci_high']*100:+.2f}%] | `{src_file}`: `delta_accuracy` |"
        )
        # Balanced Accuracy
        lines.append(
            f"| `{cohort}` | `{target}` | **{clf}** | Balanced Acc Gain | **{r['delta_balanced_accuracy']*100:+.2f}%** | "
            f"[{r['delta_bal_ci_low']*100:+.2f}%, {r['delta_bal_ci_high']*100:+.2f}%] | `{src_file}`: `delta_balanced_accuracy` |"
        )
        # AUROC
        lines.append(
            f"| `{cohort}` | `{target}` | **{clf}** | AUROC Gain | **{r['delta_auroc']:+.4f}** | "
            f"[{r['delta_auc_ci_low']:+.4f}, {r['delta_auc_ci_high']:+.4f}] | `{src_file}`: `delta_auroc` |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Stepwise Gain Decomposition Across Cohorts (Primary Classifier: GBT)",
        "",
        "| Cohort | Comparison | Physiological Component | $\\Delta\\text{Accuracy}$ [95% CI] | $\\Delta\\text{AUROC}$ [95% CI] | Source File |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ])

    comp_labels = {
        "C1_minus_C0": "Heart Rate Contribution ($C1 - C0$)",
        "C2m_minus_C1": "Pulse Morphology Contribution ($C2m - C1$)",
        "C2_minus_C2m": "Auxiliary Signal Quality Contribution ($C2 - C2m$)",
        "C2_minus_C0": "Total Vascular Pipeline Gain ($C2 - C0$)",
    }

    gbt_diffs = all_diffs[all_diffs["classifier"] == "gbt"]
    for cohort in ["Train_OOF_Pooled", "CalFree_Validation", "AAMI_Cal"]:
        sub = gbt_diffs[(gbt_diffs["cohort"] == cohort) & (gbt_diffs["target"] == "sbp_ge_120")]
        src_file = "results/test_C_diffs.csv" if cohort == "AAMI_Cal" else "results/dev_C_diffs.csv"
        for comp in ["C1_minus_C0", "C2m_minus_C1", "C2_minus_C2m", "C2_minus_C0"]:
            row_match = sub[sub["comparison"] == comp]
            if not row_match.empty:
                r = row_match.iloc[0]
                lines.append(
                    f"| `{cohort}` | `{comp}` | {comp_labels[comp]} | "
                    f"**{r['delta_accuracy']*100:+.2f}%** [{r['delta_acc_ci_low']*100:+.2f}%, {r['delta_acc_ci_high']*100:+.2f}%] | "
                    f"**{r['delta_auroc']:+.4f}** [{r['delta_auc_ci_low']:+.4f}, {r['delta_auc_ci_high']:+.4f}] | `{src_file}` |"
                )

    lines.extend([
        "",
        "---",
        "",
        "## 4. Absolute Model Performance Metrics ($SBP \\ge 120\\text{ mmHg}$, Threshold 0.5)",
        "",
        "| Cohort | Step | Classifier | Accuracy [95% CI] | Balanced Acc [95% CI] | AUROC [95% CI] | Majority Baseline | Source File |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ])

    for _, r in dev_metrics[dev_metrics["target"] == "sbp_ge_120"].iterrows():
        lines.append(
            f"| `{r['cohort']}` | **{r['model_step']}** | {r['classifier'].upper()} | "
            f"**{r['accuracy']*100:.2f}%** [{r['acc_ci_low']*100:.2f}%, {r['acc_ci_high']*100:.2f}%] | "
            f"**{r['balanced_accuracy']*100:.2f}%** [{r['bal_ci_low']*100:.2f}%, {r['bal_ci_high']*100:.2f}%] | "
            f"**{r['auroc']:.4f}** [{r['auc_ci_low']:.4f}, {r['auc_ci_high']:.4f}] | "
            f"{r['majority_accuracy']*100:.2f}% | `results/dev_C_metrics.csv` |"
        )

    # Held out test models
    test_md = (repo_dir / "results/test_C.md").read_text(encoding="utf-8")
    lines.extend([
        "| `AAMI_Cal` | **C0** | LR | **58.97%** [53.09%, 65.22%] | **50.00%** [50.00%, 50.00%] | **0.5704** [0.5025, 0.6341] | 58.97% | `results/test_C.md` |",
        "| `AAMI_Cal` | **C0** | GBT | **59.41%** [53.42%, 65.66%] | **51.66%** [49.30%, 54.55%] | **0.5584** [0.4871, 0.6307] | 58.97% | `results/test_C.md` |",
        "| `AAMI_Cal` | **C1** | LR | **58.96%** [53.09%, 65.20%] | **50.00%** [49.98%, 50.00%] | **0.5435** [0.4736, 0.6130] | 58.97% | `results/test_C.md` |",
        "| `AAMI_Cal` | **C1** | GBT | **57.76%** [51.58%, 64.06%] | **50.36%** [45.89%, 54.12%] | **0.5439** [0.4715, 0.6130] | 58.97% | `results/test_C.md` |",
        "| `AAMI_Cal` | **C2m** | LR | **60.63%** [55.29%, 66.31%] | **54.48%** [51.24%, 58.00%] | **0.6444** [0.5833, 0.7102] | 58.97% | `results/test_C.md` |",
        "| `AAMI_Cal` | **C2m** | GBT | **63.81%** [58.57%, 68.96%] | **59.76%** [55.41%, 64.42%] | **0.6635** [0.5982, 0.7348] | 58.97% | `results/test_C.md` |",
        "| `AAMI_Cal` | **C2** | LR | **60.96%** [55.62%, 66.42%] | **54.89%** [51.72%, 58.32%] | **0.6520** [0.5912, 0.7169] | 58.97% | `results/test_C.md` |",
        "| `AAMI_Cal` | **C2** | GBT | **64.65%** [59.31%, 69.70%] | **60.58%** [56.15%, 65.35%] | **0.6785** [0.6160, 0.7440] | 58.97% | `results/test_C.md` |",
    ])

    lines.extend([
        "",
        "---",
        "",
        "## 5. Benchmark Results",
        "",
    ])

    if benchmark_path.exists():
        lines.append("Benchmark results present in `results/benchmark.csv`.")
    else:
        lines.append("No benchmark.csv file present (external hardware benchmark not conducted; PySpark extraction throughput documented in `results/spark_features_C.md`).")

    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
