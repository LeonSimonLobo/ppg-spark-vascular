"""PROMPT 2 Gate: Within-subject heart rate signal and beat plausibility on training-split subjects."""

from pathlib import Path
import time
from typing import Dict, Any
import h5py
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from src.fiducials import detect_peaks_elgendi


def compute_gate_metrics(
    train_df: pd.DataFrame,
    hr_arr: np.ndarray,
) -> Dict[str, Any]:
    """Compute plausibility and within-subject association metrics."""
    n_train = len(train_df)
    n_subjs = train_df["subject_id"].nunique()

    # 1. Beat plausibility (5 to 25 beats per 10s segment -> 30 to 150 bpm)
    valid_beats_mask = (hr_arr >= 30.0) & (hr_arr <= 150.0)
    plausible_pct = float(np.mean(valid_beats_mask)) * 100.0
    plausible_pass = bool(plausible_pct >= 95.0)

    # 2. Within-subject centring
    train_df = train_df.copy()
    train_df["hr"] = hr_arr
    sub_med_hr = train_df.groupby("subject_id")["hr"].transform("median")
    sub_med_sbp = train_df.groupby("subject_id")["sbp"].transform("median")

    train_df["delta_hr"] = train_df["hr"] - sub_med_hr
    train_df["delta_sbp"] = train_df["sbp"] - sub_med_sbp
    train_df["label_130"] = (train_df["sbp"] >= 130).astype(int)
    train_df["label_rel"] = (train_df["sbp"] >= sub_med_sbp).astype(int)

    # Spearman correlation: overall pooled delta_hr vs delta_sbp
    spearman_corr, spearman_p = spearmanr(train_df["delta_hr"], train_df["delta_sbp"])

    # Per-subject Spearman correlation
    def subj_spearman(g):
        d_hr = g["delta_hr"]
        d_sbp = g["delta_sbp"]
        if d_hr.std() > 1e-5 and d_sbp.std() > 1e-5:
            r, _ = spearmanr(d_hr, d_sbp)
            return r
        return np.nan

    subj_corrs = (
        train_df.groupby("subject_id")[["delta_hr", "delta_sbp"]]
        .apply(subj_spearman)
        .dropna()
    )
    median_subj_corr = float(subj_corrs.median())
    mean_subj_corr = float(subj_corrs.mean())
    n_valid_subj = len(subj_corrs)
    n_pos_subj = int(np.sum(subj_corrs > 0))
    pos_subj_share = float(np.mean(subj_corrs > 0)) * 100.0

    # Logistic regression on delta_hr -> SBP >= 130
    scaler = StandardScaler()
    X_dhr = scaler.fit_transform(train_df[["delta_hr"]])
    clf = LogisticRegression(random_state=42, solver="lbfgs")
    clf.fit(X_dhr, train_df["label_130"])
    pred_p = clf.predict_proba(X_dhr)[:, 1]
    auc_130 = float(roc_auc_score(train_df["label_130"], pred_p))
    coef_hr = float(clf.coef_[0, 0])

    # Also on relative within-subject excursion (delta_sbp >= 0)
    clf_rel = LogisticRegression(random_state=42, solver="lbfgs")
    clf_rel.fit(X_dhr, train_df["label_rel"])
    pred_p_rel = clf_rel.predict_proba(X_dhr)[:, 1]
    auc_rel = float(roc_auc_score(train_df["label_rel"], pred_p_rel))
    coef_rel = float(clf_rel.coef_[0, 0])

    signal_pass = bool((spearman_corr > 0) and (coef_hr > 0) and (auc_130 > 0.50))
    gate_passed = bool(plausible_pass and signal_pass)

    return {
        "n_train": n_train,
        "n_subjs": n_subjs,
        "n_valid_beats": int(valid_beats_mask.sum()),
        "plausible_pct": plausible_pct,
        "plausible_pass": plausible_pass,
        "hr_median": float(np.median(hr_arr)),
        "hr_iqr": float(np.percentile(hr_arr, 75) - np.percentile(hr_arr, 25)),
        "hr_q1": float(np.percentile(hr_arr, 25)),
        "hr_q3": float(np.percentile(hr_arr, 75)),
        "hr_min": float(np.min(hr_arr)),
        "hr_max": float(np.max(hr_arr)),
        "spearman_corr": float(spearman_corr),
        "spearman_p": float(spearman_p),
        "median_subj_corr": median_subj_corr,
        "mean_subj_corr": mean_subj_corr,
        "n_valid_subj": n_valid_subj,
        "n_pos_subj": n_pos_subj,
        "pos_subj_share": pos_subj_share,
        "coef_hr": coef_hr,
        "auc_130": auc_130,
        "coef_rel": coef_rel,
        "auc_rel": auc_rel,
        "signal_pass": signal_pass,
        "gate_passed": gate_passed,
    }


def run_gate(
    mat_path: str = "data/raw/VitalDB_Train_Subset.mat",
    parquet_path: str = "data/parquet/train_file.parquet",
    output_report_path: str = "results/gate_within_subject.md",
    chunk_size: int = 15000,
) -> Dict[str, Any]:
    """Execute within-subject gate check on training split only."""
    print(f"Loading metadata from {parquet_path}...")
    table = pq.read_table(
        parquet_path,
        columns=["segment_idx", "subject_id", "split", "sbp"],
    )
    df = table.to_pandas()
    train_df = df[df["split"] == "train"].copy().reset_index(drop=True)
    n_train = len(train_df)
    n_subjs = train_df["subject_id"].nunique()
    print(f"Training split cohort: {n_train:,} segments across {n_subjs:,} subjects.")

    train_indices = train_df["segment_idx"].values
    diffs = np.diff(train_indices)
    breaks = np.where(diffs > 1)[0]
    slice_starts = [train_indices[0]] + [train_indices[b + 1] for b in breaks]
    slice_ends = [train_indices[b] + 1 for b in breaks] + [train_indices[-1] + 1]
    slices = list(zip(slice_starts, slice_ends))
    print(f"Identified {len(slices)} contiguous HDF5 slices to read.")

    n_beats_arr = np.zeros(n_train, dtype=np.int16)

    cache_path = Path("data/interim/train_n_beats.npy")
    if cache_path.exists():
        print(f"Loading cached beat counts from {cache_path}...")
        n_beats_arr = np.load(cache_path)
    else:
        t0 = time.time()
        curr_idx = 0
        last_print = 0
        with h5py.File(mat_path, "r") as f:
            sig_ds = f["Subset"]["Signals"]
            for s_idx, (start, end) in enumerate(slices):
                block_len = end - start
                for c_start in range(start, end, chunk_size):
                    c_end = min(end, c_start + chunk_size)
                    c_len = c_end - c_start
                    # Read channel 1 (PPG)
                    ppg_block = sig_ds[:, 1, c_start:c_end]
                    for j in range(c_len):
                        ppg = ppg_block[:, j]
                        pks = detect_peaks_elgendi(ppg)
                        n_beats_arr[curr_idx + j] = len(pks)
                    curr_idx += c_len

                if curr_idx - last_print >= 50000 or curr_idx == n_train:
                    elapsed = time.time() - t0
                    speed = curr_idx / (elapsed + 1e-6)
                    pct = curr_idx / n_train * 100.0
                    print(f"Processed {curr_idx:,}/{n_train:,} segments ({pct:.1f}%) in {elapsed:.1f}s ({speed:.1f} seg/s)...")
                    last_print = curr_idx
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, n_beats_arr)
        print(f"Cached beat counts to {cache_path}")

    # Compute HR
    hr_arr = n_beats_arr * 6.0  # 10s window -> bpm

    metrics = compute_gate_metrics(train_df, hr_arr)

    status_text = (
        "**PASS**: The within-subject heart-rate comparator and PPG beat detection meet all pre-registered criteria on the training split. Progression to within-subject morphology modeling is authorized."
        if metrics["gate_passed"]
        else "**FAIL / BLOCKED**: Gate criteria not met on training-split subjects. Execution stopped."
    )

    report = rf"""# Within-Subject Gate Evaluation (PROMPT 2)

**Date:** 2026-10-29  
**Cohort:** `VitalDB_Train_Subset.mat` training split only (`split == 'train'`)  
**Scope:** {metrics['n_subjs']:,} subjects | {metrics['n_train']:,} segments (10 s at 125 Hz; exactly 360 segments per subject)  
**CalFree / Test Status:** Pristine (0 test segments loaded or scored)  

---

## 1. Criterion 1: Physiological Beat Count Plausibility

- **Rule:** $\ge 95\%$ of segments must yield heart rate in [30, 150] bpm (5 to 25 detected beats per 10 s).
- **Segments within bounds:** **{metrics['plausible_pct']:.2f}%** ({metrics['n_valid_beats']:,} / {metrics['n_train']:,})
- **Heart Rate Distribution (bpm):**
  - Median: {metrics['hr_median']:.1f} bpm
  - IQR: {metrics['hr_iqr']:.1f} bpm (Q1: {metrics['hr_q1']:.1f}, Q3: {metrics['hr_q3']:.1f})
  - Min / Max: {int(metrics['hr_min'])} bpm / {int(metrics['hr_max'])} bpm
- **Criterion 1 Status:** **{'PASS' if metrics['plausible_pass'] else 'FAIL'}** ({metrics['plausible_pct']:.2f}% vs. 95.0% threshold)

---

## 2. Criterion 2: Within-Subject Heart Rate Signal

- **Centered Feature:** $\Delta HR = HR - \text{{median}}(HR_i)$ (bpm)
- **Centered Target:** $\Delta SBP = SBP - \text{{median}}(SBP_i)$ (mmHg)

### Association Metrics:
| Metric | Value | Threshold / Expected | Pass |
|---|---|---|---|
| **Pooled Spearman Correlation ($r_s$)** | **{metrics['spearman_corr']:+.4f}** ($p = {metrics['spearman_p']:.2e}$) | $r_s > 0$ | {'Yes' if metrics['spearman_corr'] > 0 else 'No'} |
| **Per-Subject Median Correlation** | **{metrics['median_subj_corr']:+.4f}** (Mean: {metrics['mean_subj_corr']:+.4f}) | Positive | {'Yes' if metrics['median_subj_corr'] > 0 else 'No'} |
| **Subjects with Positive Correlation** | **{metrics['pos_subj_share']:.2f}%** ({metrics['n_pos_subj']}/{metrics['n_valid_subj']}) | $> 50\%$ | {'Yes' if metrics['pos_subj_share'] > 50 else 'No'} |
| **Logistic Regression Slope ($\beta_{{HR}}$)** | **{metrics['coef_hr']:+.4f}** (std coef) | $\beta_{{HR}} > 0$ | {'Yes' if metrics['coef_hr'] > 0 else 'No'} |
| **Classification AUROC ($SBP \ge 130$)** | **{metrics['auc_130']:.4f}** | AUROC $> 0.50$ | {'Yes' if metrics['auc_130'] > 0.50 else 'No'} |
| **Classification AUROC (Within-Subject $\Delta SBP \ge 0$)** | **{metrics['auc_rel']:.4f}** ($\beta = {metrics['coef_rel']:+.4f}$) | AUROC $> 0.50$ | {'Yes' if metrics['auc_rel'] > 0.50 else 'No'} |

---

## 3. Gate Decision Summary

- **Criterion 1 (Plausibility $\ge 95\%$):** **{'PASS' if metrics['plausible_pass'] else 'FAIL'}** ({metrics['plausible_pct']:.2f}%)
- **Criterion 2 (Within-Subject Signal):** **{'PASS' if metrics['signal_pass'] else 'FAIL'}** (AUROC = {metrics['auc_130']:.4f}, $r_s = {metrics['spearman_corr']:+.4f}$, $\beta = {metrics['coef_hr']:+.4f}$)
- **Overall Gate Outcome:** {status_text}
"""

    out_file = Path(output_report_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(report, encoding="utf-8")
    print(f"Gate report written to {output_report_path}")

    return metrics


if __name__ == "__main__":
    run_gate()
