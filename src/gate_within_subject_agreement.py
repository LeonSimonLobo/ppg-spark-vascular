"""Training-split beat-agreement gate evaluation.

Computes the share of segments with |PPG beats - ABP beats| <= 1,
median and IQR of the difference by SBP class, subject-level bootstrap CIs (2,000 resamples),
and HR-only AUROC on the training-split cohort (VitalDB_Train_Subset.mat).
"""

from pathlib import Path
import time
from typing import Dict, Any, Tuple
import h5py
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score

from src.fiducials import detect_peaks_elgendi


def fast_auc_with_ties(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Fast AUROC calculation handling ties, identical to roc_auc_score."""
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score_sorted = y_score[desc_score_indices]
    y_true_sorted = y_true[desc_score_indices]
    distinct_value_indices = np.where(np.diff(y_score_sorted))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
    tps = np.cumsum(y_true_sorted)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    tps = np.r_[0, tps]
    fps = np.r_[0, fps]
    if fps[-1] == 0 or tps[-1] == 0:
        return 0.5
    fpr = fps / fps[-1]
    tpr = tps / tps[-1]
    return float(np.trapezoid(tpr, fpr))


def compute_abp_beats_for_slice(
    sig_ds: h5py.Dataset,
    start: int,
    end: int,
    fs: int = 125,
) -> np.ndarray:
    """Compute ABP beat counts for a contiguous slice of segments."""
    block_len = end - start
    abp_beats = np.zeros(block_len, dtype=np.int16)
    # Channel 2 is ABP
    abp_block = sig_ds[:, 2, start:end]
    for j in range(block_len):
        abp = abp_block[:, j]
        r = np.max(abp) - np.min(abp)
        if r > 1e-6:
            abp_scaled = (abp - np.min(abp)) / r
        else:
            abp_scaled = np.zeros_like(abp)
        pks = detect_peaks_elgendi(abp_scaled, fs=fs)
        abp_beats[j] = len(pks)
    return abp_beats


def compute_agreement_metrics(
    train_df: pd.DataFrame,
    ppg_beats: np.ndarray,
    abp_beats: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Calculate beat agreement, class splits, subject bootstrap CIs, and HR AUROC."""
    n_train = len(train_df)
    n_subjs = train_df["subject_id"].nunique()

    # Beat counts and differences
    diff = ppg_beats.astype(int) - abp_beats.astype(int)
    abs_diff = np.abs(diff)
    agree = (abs_diff <= 1).astype(int)

    train_df = train_df.copy()
    train_df["ppg_beats"] = ppg_beats
    train_df["abp_beats"] = abp_beats
    train_df["diff"] = diff
    train_df["abs_diff"] = abs_diff
    train_df["agree"] = agree

    # SBP labels
    sbp = train_df["sbp"].values
    y_130 = (sbp >= 130).astype(int)
    y_120 = (sbp >= 120).astype(int)
    y_140 = (sbp >= 140).astype(int)

    # Centered HR
    hr = ppg_beats * 6.0  # 10s window -> bpm
    train_df["hr"] = hr
    sub_med_hr = train_df.groupby("subject_id")["hr"].transform("median")
    delta_hr = (hr - sub_med_hr).values
    train_df["delta_hr"] = delta_hr

    # Point estimates: overall
    total_segments = n_train
    pass_count = int(np.sum(agree))
    pass_pct = float(pass_count / total_segments * 100.0)
    pass_rule_80 = bool(pass_pct >= 80.0)

    # Class stats helper
    def get_class_stats(mask):
        sub_diff = diff[mask]
        sub_agree = agree[mask]
        n_seg = len(sub_diff)
        n_pass = int(np.sum(sub_agree))
        pct = float(n_pass / n_seg * 100.0) if n_seg > 0 else 0.0
        med = float(np.median(sub_diff)) if n_seg > 0 else 0.0
        q1 = float(np.percentile(sub_diff, 25)) if n_seg > 0 else 0.0
        q3 = float(np.percentile(sub_diff, 75)) if n_seg > 0 else 0.0
        iqr = q3 - q1
        return {
            "n_total": n_seg,
            "n_pass": n_pass,
            "pass_pct": pct,
            "median_diff": med,
            "q1_diff": q1,
            "q3_diff": q3,
            "iqr_diff": iqr,
        }

    stats_ge130 = get_class_stats(y_130 == 1)
    stats_lt130 = get_class_stats(y_130 == 0)
    stats_ge120 = get_class_stats(y_120 == 1)
    stats_lt120 = get_class_stats(y_120 == 0)
    stats_ge140 = get_class_stats(y_140 == 1)
    stats_lt140 = get_class_stats(y_140 == 0)

    # Point estimate: HR-only AUROC
    hr_auc = float(roc_auc_score(y_130, delta_hr))

    # Subject-level bootstrap
    # Each subject has exactly 360 segments in order
    n_seg_per_subj = 360
    assert n_train == n_subjs * n_seg_per_subj, "Subjects must have exactly 360 segments."

    agree_by_subj = agree.reshape(n_subjs, n_seg_per_subj)
    y_130_by_subj = y_130.reshape(n_subjs, n_seg_per_subj)
    delta_hr_by_subj = delta_hr.reshape(n_subjs, n_seg_per_subj)

    rng = np.random.default_rng(seed)
    boot_agree_overall = np.zeros(n_boot, dtype=np.float64)
    boot_agree_ge130 = np.zeros(n_boot, dtype=np.float64)
    boot_agree_lt130 = np.zeros(n_boot, dtype=np.float64)
    boot_hr_auc = np.zeros(n_boot, dtype=np.float64)

    for b in range(n_boot):
        sub_sample = rng.choice(n_subjs, size=n_subjs, replace=True)
        # Resampled matrices
        b_agree = agree_by_subj[sub_sample].ravel()
        b_y130 = y_130_by_subj[sub_sample].ravel()
        b_dhr = delta_hr_by_subj[sub_sample].ravel()

        boot_agree_overall[b] = np.mean(b_agree) * 100.0

        mask_ge130 = (b_y130 == 1)
        if np.any(mask_ge130):
            boot_agree_ge130[b] = np.mean(b_agree[mask_ge130]) * 100.0
        else:
            boot_agree_ge130[b] = np.nan

        mask_lt130 = (b_y130 == 0)
        if np.any(mask_lt130):
            boot_agree_lt130[b] = np.mean(b_agree[mask_lt130]) * 100.0
        else:
            boot_agree_lt130[b] = np.nan

        boot_hr_auc[b] = fast_auc_with_ties(b_y130, b_dhr)

    # 95% CIs
    ci_agree_overall = (
        float(np.percentile(boot_agree_overall, 2.5)),
        float(np.percentile(boot_agree_overall, 97.5)),
    )
    ci_agree_ge130 = (
        float(np.nanpercentile(boot_agree_ge130, 2.5)),
        float(np.nanpercentile(boot_agree_ge130, 97.5)),
    )
    ci_agree_lt130 = (
        float(np.nanpercentile(boot_agree_lt130, 2.5)),
        float(np.nanpercentile(boot_agree_lt130, 97.5)),
    )
    ci_hr_auc = (
        float(np.percentile(boot_hr_auc, 2.5)),
        float(np.percentile(boot_hr_auc, 97.5)),
    )

    return {
        "n_train": n_train,
        "n_subjs": n_subjs,
        "total_segments": total_segments,
        "pass_count": pass_count,
        "pass_pct": pass_pct,
        "pass_rule_80": pass_rule_80,
        "ci_agree_overall": ci_agree_overall,
        "stats_ge130": stats_ge130,
        "stats_lt130": stats_lt130,
        "ci_agree_ge130": ci_agree_ge130,
        "ci_agree_lt130": ci_agree_lt130,
        "stats_ge120": stats_ge120,
        "stats_lt120": stats_lt120,
        "stats_ge140": stats_ge140,
        "stats_lt140": stats_lt140,
        "hr_auc": hr_auc,
        "ci_hr_auc": ci_hr_auc,
    }


def run_agreement_gate(
    mat_path: str = "data/raw/VitalDB_Train_Subset.mat",
    parquet_path: str = "data/parquet/train_file.parquet",
    ppg_beats_path: str = "data/interim/train_n_beats.npy",
    abp_beats_path: str = "data/interim/train_n_abp_beats.npy",
    output_report_path: str = "results/gate_within_subject_agreement.md",
    chunk_size: int = 15000,
    n_boot: int = 2000,
) -> Dict[str, Any]:
    """Execute training-split beat-agreement gate and save report."""
    print(f"Loading metadata from {parquet_path}...")
    table = pq.read_table(
        parquet_path,
        columns=["segment_idx", "subject_id", "split", "sbp"],
    )
    df = table.to_pandas()
    train_df = df[df["split"] == "train"].copy().reset_index(drop=True)
    n_train = len(train_df)
    n_subjs = train_df["subject_id"].nunique()
    print(f"Cohort: {n_train:,} segments across {n_subjs:,} subjects.")

    # 1. Load PPG beats
    p_ppg = Path(ppg_beats_path)
    if not p_ppg.exists():
        raise FileNotFoundError(f"PPG beat cache not found at {ppg_beats_path}. Run PROMPT 2 gate first.")
    print(f"Loading PPG beat counts from {ppg_beats_path}...")
    ppg_beats = np.load(p_ppg)
    assert len(ppg_beats) == n_train, f"PPG beats len {len(ppg_beats)} != n_train {n_train}"

    # 2. Load or compute ABP beats
    p_abp = Path(abp_beats_path)
    if p_abp.exists():
        print(f"Loading cached ABP beat counts from {abp_beats_path}...")
        abp_beats = np.load(p_abp)
    else:
        print(f"Computing ABP beat counts from {mat_path}...")
        train_indices = train_df["segment_idx"].values
        diffs = np.diff(train_indices)
        breaks = np.where(diffs > 1)[0]
        starts = [train_indices[0]] + [train_indices[b + 1] for b in breaks]
        ends = [train_indices[b] + 1 for b in breaks] + [train_indices[-1] + 1]
        slices = list(zip(starts, ends))

        abp_beats = np.zeros(n_train, dtype=np.int16)
        curr_idx = 0
        last_print = 0
        t0 = time.time()
        with h5py.File(mat_path, "r") as f:
            sig_ds = f["Subset"]["Signals"]
            for s_idx, (start, end) in enumerate(slices):
                for c_start in range(start, end, chunk_size):
                    c_end = min(end, c_start + chunk_size)
                    c_len = c_end - c_start
                    block = compute_abp_beats_for_slice(sig_ds, c_start, c_end)
                    abp_beats[curr_idx : curr_idx + c_len] = block
                    curr_idx += c_len

                if curr_idx - last_print >= 50000 or curr_idx == n_train:
                    elapsed = time.time() - t0
                    speed = curr_idx / (elapsed + 1e-6)
                    pct = curr_idx / n_train * 100.0
                    print(f"Processed ABP {curr_idx:,}/{n_train:,} segments ({pct:.1f}%) in {elapsed:.1f}s ({speed:.1f} seg/s)...")
                    last_print = curr_idx

        p_abp.parent.mkdir(parents=True, exist_ok=True)
        np.save(p_abp, abp_beats)
        print(f"Cached ABP beats to {abp_beats_path}")

    assert len(abp_beats) == n_train, f"ABP beats len {len(abp_beats)} != n_train {n_train}"

    # 3. Compute metrics & bootstrap
    print(f"Computing agreement metrics and running {n_boot:,} subject bootstrap resamples...")
    metrics = compute_agreement_metrics(train_df, ppg_beats, abp_beats, n_boot=n_boot)

    overall_pass_str = "PASS" if metrics["pass_rule_80"] else "FAIL"
    status_summary = (
        "**PASS**: The training-split beat agreement (|PPG beats − ABP beats| ≤ 1) exceeds the 80% pre-registered threshold."
        if metrics["pass_rule_80"]
        else "**FAIL**: The training-split beat agreement falls below the 80% threshold."
    )

    report = rf"""# Training-Split Beat-Agreement Gate (PLAN_W.md Section 7)

**Date:** 2026-10-29  
**Cohort:** `VitalDB_Train_Subset.mat` training split (`split == 'train'`)  
**Scope:** {metrics['n_subjs']:,} subjects | {metrics['n_train']:,} segments (10 s at 125 Hz; 360 segments per subject)  
**CalFree / Test Status:** Pristine (0 test segments loaded or evaluated)  
**Agreement Criterion:** $|n_{{\text{{PPG}}}} - n_{{\text{{ABP}}}}| \le 1$ beat per 10 s segment  
**Threshold:** $\ge 80.0\%$ passing segments  

---

## 1. Overall Beat Agreement

- **Passing Segments ($|n_{{\text{{PPG}}}} - n_{{\text{{ABP}}}}| \le 1$):** **{metrics['pass_pct']:.2f}%** ({metrics['pass_count']:,} / {metrics['total_segments']:,})
- **95% Subject-Level Bootstrap CI (2,000 resamples):** **[{metrics['ci_agree_overall'][0]:.2f}%, {metrics['ci_agree_overall'][1]:.2f}%]**
- **Threshold Rule ($\ge 80.0\%$):** **{overall_pass_str}** ({metrics['pass_pct']:.2f}% vs. 80.0%)

---

## 2. Agreement by SBP Class

### Primary Split ($SBP \ge 130$ mmHg):
| Subgroup | Total Segments | Passing Segments ($|d| \le 1$) | Pass Rate | 95% Bootstrap CI | Median Diff ($n_{{\text{{PPG}}}} - n_{{\text{{ABP}}}}$) | IQR of Diff ($Q_1, Q_3$) |
|---|---|---|---|---|---|---|
| **$SBP \ge 130$ mmHg** | {metrics['stats_ge130']['n_total']:,} | {metrics['stats_ge130']['n_pass']:,} | **{metrics['stats_ge130']['pass_pct']:.2f}%** | [{metrics['ci_agree_ge130'][0]:.2f}%, {metrics['ci_agree_ge130'][1]:.2f}%] | {metrics['stats_ge130']['median_diff']:+.1f} | {metrics['stats_ge130']['iqr_diff']:.1f} ({metrics['stats_ge130']['q1_diff']:+.1f}, {metrics['stats_ge130']['q3_diff']:+.1f}) |
| **$SBP < 130$ mmHg** | {metrics['stats_lt130']['n_total']:,} | {metrics['stats_lt130']['n_pass']:,} | **{metrics['stats_lt130']['pass_pct']:.2f}%** | [{metrics['ci_agree_lt130'][0]:.2f}%, {metrics['ci_agree_lt130'][1]:.2f}%] | {metrics['stats_lt130']['median_diff']:+.1f} | {metrics['stats_lt130']['iqr_diff']:.1f} ({metrics['stats_lt130']['q1_diff']:+.1f}, {metrics['stats_lt130']['q3_diff']:+.1f}) |

### Sensitivity Cut-Points:
| Cut-Point | Class | Total Segments | Passing Segments | Pass Rate | Median Diff | IQR of Diff |
|---|---|---|---|---|---|---|
| **120 mmHg** | $SBP \ge 120$ | {metrics['stats_ge120']['n_total']:,} | {metrics['stats_ge120']['n_pass']:,} | {metrics['stats_ge120']['pass_pct']:.2f}% | {metrics['stats_ge120']['median_diff']:+.1f} | {metrics['stats_ge120']['iqr_diff']:.1f} ({metrics['stats_ge120']['q1_diff']:+.1f}, {metrics['stats_ge120']['q3_diff']:+.1f}) |
| | $SBP < 120$ | {metrics['stats_lt120']['n_total']:,} | {metrics['stats_lt120']['n_pass']:,} | {metrics['stats_lt120']['pass_pct']:.2f}% | {metrics['stats_lt120']['median_diff']:+.1f} | {metrics['stats_lt120']['iqr_diff']:.1f} ({metrics['stats_lt120']['q1_diff']:+.1f}, {metrics['stats_lt120']['q3_diff']:+.1f}) |
| **140 mmHg** | $SBP \ge 140$ | {metrics['stats_ge140']['n_total']:,} | {metrics['stats_ge140']['n_pass']:,} | {metrics['stats_ge140']['pass_pct']:.2f}% | {metrics['stats_ge140']['median_diff']:+.1f} | {metrics['stats_ge140']['iqr_diff']:.1f} ({metrics['stats_ge140']['q1_diff']:+.1f}, {metrics['stats_ge140']['q3_diff']:+.1f}) |
| | $SBP < 140$ | {metrics['stats_lt140']['n_total']:,} | {metrics['stats_lt140']['n_pass']:,} | {metrics['stats_lt140']['pass_pct']:.2f}% | {metrics['stats_lt140']['median_diff']:+.1f} | {metrics['stats_lt140']['iqr_diff']:.1f} ({metrics['stats_lt140']['q1_diff']:+.1f}, {metrics['stats_lt140']['q3_diff']:+.1f}) |

---

## 3. Heart Rate Baseline AUROC on the Same Subjects

- **Feature:** Within-subject centered heart rate ($\Delta HR_{{i, j}} = HR_{{i, j}} - \text{{median}}(HR_i)$)
- **Target:** Acute hypertensive state ($SBP \ge 130$ mmHg)
- **Point Estimate AUROC:** **{metrics['hr_auc']:.4f}**
- **95% Subject-Level Bootstrap CI (2,000 resamples):** **[{metrics['ci_hr_auc'][0]:.4f}, {metrics['ci_hr_auc'][1]:.4f}]**

---

## 4. Decision & Gate Summary

- **Beat Agreement ($\ge 80\%$):** **{overall_pass_str}** ({metrics['pass_pct']:.2f}%)
- **Conclusion:** {status_summary}
"""

    out_file = Path(output_report_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(report, encoding="utf-8")
    print(f"Agreement gate report written to {output_report_path}")

    return metrics


if __name__ == "__main__":
    run_agreement_gate()
