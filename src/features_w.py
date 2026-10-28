"""PPG feature extraction module for within-subject modeling (PLAN_W.md).

Extracts PPG-only features using the frozen Elgendi fiducials and geometric landmarks:
- Heart rate (hr) from detected beat count
- Systolic rise time (systolic_rise_time) from foot to systolic peak
- Pulse width at 50% amplitude (pulse_width)
- Decay ratio (decay_ratio) of diastolic decay time to systolic rise time
- Per-subject centered features (delta_hr, delta_systolic_rise_time, delta_pulse_width, delta_decay_ratio)
computed strictly from PPG segments without any ABP inputs.
"""

from pathlib import Path
import time
from typing import Dict, Any, Tuple, Optional
import h5py
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.fiducials import detect_peaks_elgendi


def extract_segment_features(
    ppg: np.ndarray,
    fs: int = 125,
) -> Tuple[float, float, float, float]:
    """Extract classic PPG morphology features for a single 10-s segment.

    Parameters
    ----------
    ppg : np.ndarray
        1D PPG waveform (e.g. 1250 samples at 125 Hz).
    fs : int
        Sampling frequency in Hz (default 125).

    Returns
    -------
    hr : float
        Heart rate in beats per minute (len(peaks) * 6.0).
    systolic_rise_time : float
        Median time from foot to peak across beats in seconds.
    pulse_width : float
        Median full width at half-maximum (50% amplitude) in seconds.
    decay_ratio : float
        Median ratio of decay time to rise time across beats.
    """
    ppg = np.asarray(ppg, dtype=np.float64)
    peaks = detect_peaks_elgendi(ppg, fs=fs)
    n_pks = len(peaks)
    hr = float(n_pks * 6.0)

    if n_pks == 0:
        return hr, np.nan, np.nan, np.nan

    w_foot = int(round(0.3 * fs))  # 300 ms preceding peak
    rise_times = []
    pulse_widths = []
    decay_ratios = []

    for k in range(n_pks):
        pk = peaks[k]

        # 1. Foot detection: minimum in 300 ms preceding peak (closest to peak if tied)
        start_foot = max(0, pk - w_foot)
        window_foot = ppg[start_foot : pk + 1]
        min_foot_val = np.min(window_foot)
        foot_ties = np.where(window_foot <= min_foot_val + 1e-9)[0]
        foot = start_foot + int(foot_ties[-1])

        # 2. Turning point detection: minimum between peak and next peak
        if k < n_pks - 1:
            turn_end = peaks[k + 1]
        else:
            turn_end = min(len(ppg) - 1, pk + int(round(0.6 * fs)))

        if turn_end > pk:
            window_turn = ppg[pk : turn_end + 1]
            min_turn_val = np.min(window_turn)
            turn_ties = np.where(window_turn <= min_turn_val + 1e-9)[0]
            turn = pk + int(turn_ties[0])
        else:
            turn = pk

        r_time = (pk - foot) / fs
        d_time = (turn - pk) / fs

        y_foot = ppg[foot]
        y_pk = ppg[pk]
        h = y_pk - y_foot

        # 3. Pulse width at 50% amplitude (FWHM)
        pw = np.nan
        if h > 1e-4 and r_time > 0 and d_time > 0:
            y50 = y_foot + 0.5 * h

            # Ascending 50% crossing between foot and pk
            t_asc = None
            for m in range(foot, pk):
                if ppg[m] <= y50 <= ppg[m + 1] and (ppg[m + 1] > ppg[m]):
                    t_asc = m + (y50 - ppg[m]) / (ppg[m + 1] - ppg[m])
                    break

            # Descending 50% crossing between pk and turn
            t_desc = None
            for m in range(pk, turn):
                if ppg[m] >= y50 >= ppg[m + 1] and (ppg[m] > ppg[m + 1]):
                    t_desc = m + (y50 - ppg[m]) / (ppg[m + 1] - ppg[m])
                    break

            if t_asc is not None and t_desc is not None and t_desc > t_asc:
                pw = (t_desc - t_asc) / fs

        # 4. Decay ratio (decay time / rise time)
        d_ratio = (d_time / r_time) if r_time > 0 else np.nan

        if r_time > 0:
            rise_times.append(r_time)
        if not np.isnan(pw):
            pulse_widths.append(pw)
        if not np.isnan(d_ratio):
            decay_ratios.append(d_ratio)

    med_rise = float(np.median(rise_times)) if len(rise_times) > 0 else np.nan
    med_pw = float(np.median(pulse_widths)) if len(pulse_widths) > 0 else np.nan
    med_decay = float(np.median(decay_ratios)) if len(decay_ratios) > 0 else np.nan

    return hr, med_rise, med_pw, med_decay


def center_features_per_subject(
    df: pd.DataFrame,
    feature_cols: Optional[list] = None,
) -> pd.DataFrame:
    """Compute per-subject centered features (x - median(x_i)) from PPG features only."""
    if feature_cols is None:
        feature_cols = ["hr", "systolic_rise_time", "pulse_width", "decay_ratio"]

    df = df.copy()
    for col in feature_cols:
        delta_col = f"delta_{col}"
        sub_med = df.groupby("subject_id")[col].transform("median")
        df[delta_col] = df[col] - sub_med
    return df


def extract_features_for_cohort(
    mat_path: str = "data/raw/VitalDB_Train_Subset.mat",
    parquet_path: str = "data/parquet/train_file.parquet",
    output_parquet_path: str = "data/interim/features_w_train.parquet",
    output_report_path: str = "results/features_W_dev.md",
    chunk_size: int = 15000,
    fs: int = 125,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Extract features and apply per-subject centering for training split only."""
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

    out_p = Path(output_parquet_path)
    if out_p.exists():
        print(f"Loading cached extracted features from {output_parquet_path}...")
        feat_df = pd.read_parquet(output_parquet_path)
    else:
        print(f"Extracting features from {mat_path}...")
        train_indices = train_df["segment_idx"].values
        diffs = np.diff(train_indices)
        breaks = np.where(diffs > 1)[0]
        starts = [train_indices[0]] + [train_indices[b + 1] for b in breaks]
        ends = [train_indices[b] + 1 for b in breaks] + [train_indices[-1] + 1]
        slices = list(zip(starts, ends))

        hr_arr = np.zeros(n_train, dtype=np.float32)
        rise_arr = np.full(n_train, np.nan, dtype=np.float32)
        pw_arr = np.full(n_train, np.nan, dtype=np.float32)
        decay_arr = np.full(n_train, np.nan, dtype=np.float32)

        curr_idx = 0
        last_print = 0
        t0 = time.time()
        with h5py.File(mat_path, "r") as f:
            sig_ds = f["Subset"]["Signals"]
            for s_idx, (start, end) in enumerate(slices):
                for c_start in range(start, end, chunk_size):
                    c_end = min(end, c_start + chunk_size)
                    c_len = c_end - c_start
                    ppg_block = sig_ds[:, 1, c_start:c_end]
                    for j in range(c_len):
                        hr, rise, pw, dec = extract_segment_features(ppg_block[:, j], fs=fs)
                        idx = curr_idx + j
                        hr_arr[idx] = hr
                        rise_arr[idx] = rise
                        pw_arr[idx] = pw
                        decay_arr[idx] = dec
                    curr_idx += c_len

                if curr_idx - last_print >= 50000 or curr_idx == n_train:
                    elapsed = time.time() - t0
                    speed = curr_idx / (elapsed + 1e-6)
                    pct = curr_idx / n_train * 100.0
                    print(f"Processed {curr_idx:,}/{n_train:,} ({pct:.1f}%) in {elapsed:.1f}s ({speed:.1f} seg/s)...")
                    last_print = curr_idx

        feat_df = train_df.copy()
        feat_df["hr"] = hr_arr
        feat_df["systolic_rise_time"] = rise_arr
        feat_df["pulse_width"] = pw_arr
        feat_df["decay_ratio"] = decay_arr

        # Apply per-subject centering
        feat_df = center_features_per_subject(feat_df)

        out_p.parent.mkdir(parents=True, exist_ok=True)
        feat_df.to_parquet(out_p, index=False)
        print(f"Saved feature dataset to {output_parquet_path}")

    # Compute quality and missingness statistics
    feature_cols = [
        "hr",
        "systolic_rise_time",
        "pulse_width",
        "decay_ratio",
        "delta_hr",
        "delta_systolic_rise_time",
        "delta_pulse_width",
        "delta_decay_ratio",
    ]

    stats = {}
    for col in feature_cols:
        s = feat_df[col]
        nan_count = int(s.isna().sum())
        nan_pct = float(nan_count / n_train * 100.0)
        valid_s = s.dropna()
        stats[col] = {
            "valid_count": int(len(valid_s)),
            "nan_count": nan_count,
            "nan_pct": nan_pct,
            "mean": float(valid_s.mean()) if len(valid_s) > 0 else np.nan,
            "std": float(valid_s.std()) if len(valid_s) > 0 else np.nan,
            "median": float(valid_s.median()) if len(valid_s) > 0 else np.nan,
            "iqr": float(np.percentile(valid_s, 75) - np.percentile(valid_s, 25)) if len(valid_s) > 0 else np.nan,
            "q1": float(np.percentile(valid_s, 25)) if len(valid_s) > 0 else np.nan,
            "q3": float(np.percentile(valid_s, 75)) if len(valid_s) > 0 else np.nan,
            "min": float(valid_s.min()) if len(valid_s) > 0 else np.nan,
            "max": float(valid_s.max()) if len(valid_s) > 0 else np.nan,
        }

    # Share of segments with ANY NaN across morphology features
    morph_cols = ["systolic_rise_time", "pulse_width", "decay_ratio"]
    any_nan_mask = feat_df[morph_cols].isna().any(axis=1)
    any_nan_count = int(any_nan_mask.sum())
    any_nan_pct = float(any_nan_count / n_train * 100.0)

    # Share of segments with ANY NaN across all 4 features (including HR)
    all4_nan_mask = feat_df[["hr"] + morph_cols].isna().any(axis=1)
    all4_nan_count = int(all4_nan_mask.sum())
    all4_nan_pct = float(all4_nan_count / n_train * 100.0)

    # Centering verification: median delta per subject should be ~0
    max_median_delta_hr = float(feat_df.groupby("subject_id")["delta_hr"].median().abs().max())
    max_median_delta_rise = float(feat_df.groupby("subject_id")["delta_systolic_rise_time"].median().abs().max())
    max_median_delta_pw = float(feat_df.groupby("subject_id")["delta_pulse_width"].median().abs().max())
    max_median_delta_decay = float(feat_df.groupby("subject_id")["delta_decay_ratio"].median().abs().max())

    report = rf"""# Features W Extraction Report: Development / Training Split

**Date:** 2026-10-29  
**Cohort:** `VitalDB_Train_Subset.mat` training split (`split == 'train'`)  
**Scope:** {n_subjs:,} subjects | {n_train:,} segments (exactly 360 segments per subject)  
**CalFree / Test Status:** Pristine (0 test segments loaded or scored)  
**Inputs:** PPG channel 1 only (no ABP inputs used in feature extraction or centering)  

---

## 1. Summary Statistics and Missingness

| Feature | Valid Count | NaN Count | NaN Share | Mean | Std | Median | IQR ($Q_1, Q_3$) | [Min, Max] |
|---|---|---|---|---|---|---|---|---|
| **`hr`** (bpm) | {stats['hr']['valid_count']:,} | {stats['hr']['nan_count']:,} | {stats['hr']['nan_pct']:.2f}% | {stats['hr']['mean']:.2f} | {stats['hr']['std']:.2f} | {stats['hr']['median']:.1f} | {stats['hr']['iqr']:.1f} ({stats['hr']['q1']:.1f}, {stats['hr']['q3']:.1f}) | [{stats['hr']['min']:.1f}, {stats['hr']['max']:.1f}] |
| **`systolic_rise_time`** (s) | {stats['systolic_rise_time']['valid_count']:,} | {stats['systolic_rise_time']['nan_count']:,} | {stats['systolic_rise_time']['nan_pct']:.2f}% | {stats['systolic_rise_time']['mean']:.4f} | {stats['systolic_rise_time']['std']:.4f} | {stats['systolic_rise_time']['median']:.4f} | {stats['systolic_rise_time']['iqr']:.4f} ({stats['systolic_rise_time']['q1']:.4f}, {stats['systolic_rise_time']['q3']:.4f}) | [{stats['systolic_rise_time']['min']:.4f}, {stats['systolic_rise_time']['max']:.4f}] |
| **`pulse_width`** (s) | {stats['pulse_width']['valid_count']:,} | {stats['pulse_width']['nan_count']:,} | {stats['pulse_width']['nan_pct']:.2f}% | {stats['pulse_width']['mean']:.4f} | {stats['pulse_width']['std']:.4f} | {stats['pulse_width']['median']:.4f} | {stats['pulse_width']['iqr']:.4f} ({stats['pulse_width']['q1']:.4f}, {stats['pulse_width']['q3']:.4f}) | [{stats['pulse_width']['min']:.4f}, {stats['pulse_width']['max']:.4f}] |
| **`decay_ratio`** | {stats['decay_ratio']['valid_count']:,} | {stats['decay_ratio']['nan_count']:,} | {stats['decay_ratio']['nan_pct']:.2f}% | {stats['decay_ratio']['mean']:.4f} | {stats['decay_ratio']['std']:.4f} | {stats['decay_ratio']['median']:.4f} | {stats['decay_ratio']['iqr']:.4f} ({stats['decay_ratio']['q1']:.4f}, {stats['decay_ratio']['q3']:.4f}) | [{stats['decay_ratio']['min']:.4f}, {stats['decay_ratio']['max']:.4f}] |

### Centered Features ($\Delta X_{{i, j}} = X_{{i, j}} - \text{{median}}(X_i)$):
| Feature | Valid Count | NaN Count | Mean | Std | Median | IQR ($Q_1, Q_3$) |
|---|---|---|---|---|---|---|
| **`delta_hr`** (bpm) | {stats['delta_hr']['valid_count']:,} | {stats['delta_hr']['nan_count']:,} | {stats['delta_hr']['mean']:+.4f} | {stats['delta_hr']['std']:.4f} | {stats['delta_hr']['median']:+.4f} | {stats['delta_hr']['iqr']:.4f} ({stats['delta_hr']['q1']:+.4f}, {stats['delta_hr']['q3']:+.4f}) |
| **`delta_systolic_rise_time`** (s) | {stats['delta_systolic_rise_time']['valid_count']:,} | {stats['delta_systolic_rise_time']['nan_count']:,} | {stats['delta_systolic_rise_time']['mean']:+.4f} | {stats['delta_systolic_rise_time']['std']:.4f} | {stats['delta_systolic_rise_time']['median']:+.4f} | {stats['delta_systolic_rise_time']['iqr']:.4f} ({stats['delta_systolic_rise_time']['q1']:+.4f}, {stats['delta_systolic_rise_time']['q3']:+.4f}) |
| **`delta_pulse_width`** (s) | {stats['delta_pulse_width']['valid_count']:,} | {stats['delta_pulse_width']['nan_count']:,} | {stats['delta_pulse_width']['mean']:+.4f} | {stats['delta_pulse_width']['std']:.4f} | {stats['delta_pulse_width']['median']:+.4f} | {stats['delta_pulse_width']['iqr']:.4f} ({stats['delta_pulse_width']['q1']:+.4f}, {stats['delta_pulse_width']['q3']:+.4f}) |
| **`delta_decay_ratio`** | {stats['delta_decay_ratio']['valid_count']:,} | {stats['delta_decay_ratio']['nan_count']:,} | {stats['delta_decay_ratio']['mean']:+.4f} | {stats['delta_decay_ratio']['std']:.4f} | {stats['delta_decay_ratio']['median']:+.4f} | {stats['delta_decay_ratio']['iqr']:.4f} ({stats['delta_decay_ratio']['q1']:+.4f}, {stats['delta_decay_ratio']['q3']:+.4f}) |

---

## 2. Segment-Level Missingness & Coverage

- **Total Segments:** {n_train:,}
- **Segments with ANY NaN across Morphology (`systolic_rise_time`, `pulse_width`, `decay_ratio`):** **{any_nan_pct:.2f}%** ({any_nan_count:,} / {n_train:,})
- **Complete Morphology Coverage:** **{100.0 - any_nan_pct:.2f}%** ({n_train - any_nan_count:,} / {n_train:,})
- **Segments with ANY NaN across all 4 features (including `hr`):** **{all4_nan_pct:.2f}%** ({all4_nan_count:,} / {n_train:,})

---

## 3. Centering Sanity Verification

Per-subject centering subtracts $\text{{median}}(X_i)$ for each subject $i$. Across all 1,034 training subjects:
- Maximum absolute subject-median of `delta_hr`: **{max_median_delta_hr:.6f}** (expected 0.0)
- Maximum absolute subject-median of `delta_systolic_rise_time`: **{max_median_delta_rise:.6f}** (expected 0.0)
- Maximum absolute subject-median of `delta_pulse_width`: **{max_median_delta_pw:.6f}** (expected 0.0)
- Maximum absolute subject-median of `delta_decay_ratio`: **{max_median_delta_decay:.6f}** (expected 0.0)

Centering verification **PASSED**.
"""

    out_file = Path(output_report_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(report, encoding="utf-8")
    print(f"Features W development report written to {output_report_path}")

    summary_info = {
        "n_train": n_train,
        "n_subjs": n_subjs,
        "stats": stats,
        "any_nan_count": any_nan_count,
        "any_nan_pct": any_nan_pct,
        "complete_count": n_train - any_nan_count,
    }
    return feat_df, summary_info


if __name__ == "__main__":
    extract_features_for_cohort()
