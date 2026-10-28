"""PPG and ABP fiducial detection and agreement validation.

Implements the frozen fiducial detection method specified in PLAN.md:
- Elgendi et al. (2013) two-moving-average method on Signals channel 1 as stored
  with frozen parameters: W1 = 111 ms, W2 = 667 ms, beta = 0.02.
- Foot: PPG minimum in 300 ms preceding each systolic peak.
- Turning points: minima between consecutive systolic peaks.
- ABP reference: same detector on channel 2 (min-max scaled for thresholding).
- PTT offset: lag 0 to 60 samples maximizing cross-correlation as in ingest.py.
- Agreement: PPG peak matched if within +/-50 ms of PTT-corrected ABP peak.
  A segment passes if at least 80% of beats are matched.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np


def detect_peaks_elgendi(
    signal: np.ndarray,
    fs: int = 125,
    w1_ms: float = 111.0,
    w2_ms: float = 667.0,
    beta: float = 0.02,
) -> np.ndarray:
    """Detect systolic peaks using the frozen Elgendi et al. (2013) algorithm.

    Parameters
    ----------
    signal : np.ndarray
        1D array containing the physiological waveform.
    fs : int
        Sampling frequency in Hz (default 125).
    w1_ms : float
        First moving average window in milliseconds (default 111 ms).
    w2_ms : float
        Second moving average window in milliseconds (default 667 ms).
    beta : float
        Threshold offset multiplier (default 0.02).

    Returns
    -------
    np.ndarray
        1D array of integer sample indices of detected systolic peaks.
    """
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 1 or len(signal) == 0:
        return np.empty(0, dtype=int)

    # Window widths in samples
    w1 = int(round(w1_ms * fs / 1000.0))
    w2 = int(round(w2_ms * fs / 1000.0))
    w1 = max(1, w1)
    w2 = max(w1, w2)

    # Signal squaring (applied as stored)
    z = signal ** 2

    # Event-related moving averages (boxcar smoothing with 'same' boundary mode)
    ma1 = np.convolve(z, np.ones(w1) / w1, mode="same")
    ma2 = np.convolve(z, np.ones(w2) / w2, mode="same")

    # Adaptive threshold
    threshold = ma2 + beta * np.mean(z)

    # Blocks of interest where MA_peak > threshold
    blocks = ma1 > threshold

    peaks = []
    in_block = False
    start = 0
    n = len(signal)

    for i in range(n):
        if blocks[i] and not in_block:
            in_block = True
            start = i
        elif not blocks[i] and in_block:
            in_block = False
            # Block must be at least W1 wide to qualify as a valid beat wave
            if (i - start) >= w1:
                pk = start + int(np.argmax(signal[start:i]))
                peaks.append(pk)

    # Handle block active at end of signal
    if in_block and (n - start) >= w1:
        pk = start + int(np.argmax(signal[start:n]))
        peaks.append(pk)

    return np.array(peaks, dtype=int)


def detect_feet(
    signal: np.ndarray,
    peaks: np.ndarray,
    fs: int = 125,
    window_ms: float = 300.0,
) -> np.ndarray:
    """Detect pulse onset / foot as the minimum in the window before each peak.

    Parameters
    ----------
    signal : np.ndarray
        1D waveform.
    peaks : np.ndarray
        Indices of detected systolic peaks.
    fs : int
        Sampling frequency in Hz.
    window_ms : float
        Window duration in milliseconds before peak to search (default 300 ms).

    Returns
    -------
    np.ndarray
        1D array of integer indices of detected feet (same length as peaks).
    """
    signal = np.asarray(signal, dtype=np.float64)
    peaks = np.asarray(peaks, dtype=int)
    if len(peaks) == 0:
        return np.empty(0, dtype=int)

    w = int(round(window_ms * fs / 1000.0))
    feet = []
    for pk in peaks:
        start = max(0, pk - w)
        foot = start + int(np.argmin(signal[start : pk + 1]))
        feet.append(foot)

    return np.array(feet, dtype=int)


def detect_turning_points(
    signal: np.ndarray,
    peaks: np.ndarray,
) -> np.ndarray:
    """Detect turning points as minima between consecutive systolic peaks.

    Parameters
    ----------
    signal : np.ndarray
        1D waveform.
    peaks : np.ndarray
        Indices of detected systolic peaks.

    Returns
    -------
    np.ndarray
        1D array of integer indices of turning points (length len(peaks) - 1).
    """
    signal = np.asarray(signal, dtype=np.float64)
    peaks = np.asarray(peaks, dtype=int)
    if len(peaks) < 2:
        return np.empty(0, dtype=int)

    turns = []
    for i in range(len(peaks) - 1):
        p0, p1 = peaks[i], peaks[i + 1]
        turn = p0 + int(np.argmin(signal[p0 : p1 + 1]))
        turns.append(turn)

    return np.array(turns, dtype=int)


def compute_ptt_offset(
    ppg: np.ndarray,
    abp: np.ndarray,
    max_lag: Optional[int] = None,
    min_lag: Optional[int] = None,
    W: Optional[int] = None,
) -> Tuple[int, float]:
    """Compute PTT offset lag maximizing cross-correlation between PPG and ABP.

    Under Amendment A3, uses a symmetric window [-W, W) with W = 37 samples (~300 ms at 125 Hz).
    For legacy calls with max_lag=60 and no min_lag/W, preserves the original [0, 60) window.

    Sign Convention:
    Aligns ABP(t + lag) with PPG(t).
    - When lag >= 0: correlates ppg[:-lag] with abp[lag:].
    - When lag < 0: correlates ppg[k:] with abp[:-k], where k = -lag > 0.
    Physiological pulse arrival (ABP leading PPG) corresponds to lag < 0.

    Parameters
    ----------
    ppg : np.ndarray
        PPG waveform (channel 1).
    abp : np.ndarray
        ABP waveform (channel 2).
    max_lag : Optional[int]
        Upper lag bound (exclusive).
    min_lag : Optional[int]
        Lower lag bound (inclusive).
    W : Optional[int]
        Symmetric window half-width: lag in [-W, W). Defaults to 37 under A3.

    Returns
    -------
    best_lag : int
        Lag in [min_lag, max_lag) yielding maximum cross-correlation.
    max_corr : float
        Maximum cross-correlation coefficient at best_lag.
    """
    if W is not None:
        low = -int(W)
        high = int(W)
    elif min_lag is not None and max_lag is not None:
        low = int(min_lag)
        high = int(max_lag)
    elif max_lag is not None and min_lag is None:
        if max_lag == 60:
            low = 0
            high = 60
        else:
            low = -int(max_lag)
            high = int(max_lag)
    else:
        low = -37
        high = 37

    ppg = np.asarray(ppg, dtype=np.float64)
    abp = np.asarray(abp, dtype=np.float64)

    ppg_c = ppg - np.mean(ppg)
    abp_c = abp - np.mean(abp)

    best_lag = low
    best_corr = -2.0

    for lag in range(low, high):
        if lag >= 0:
            a_sub = abp_c[lag:]
            p_sub = ppg_c[:-lag or None]
        else:
            k = -lag
            a_sub = abp_c[:-k]
            p_sub = ppg_c[k:]
        norm = np.linalg.norm(a_sub) * np.linalg.norm(p_sub) + 1e-12
        r = float(np.dot(p_sub, a_sub) / norm)
        if r > best_corr:
            best_corr = r
            best_lag = lag

    return best_lag, float(best_corr)


def evaluate_beat_agreement(
    ppg_peaks: np.ndarray,
    abp_peaks: np.ndarray,
    ptt_offset: int,
    fs: int = 125,
    tol_ms: float = 50.0,
) -> Tuple[int, int, float, bool]:
    """Evaluate whether PPG peaks match PTT-corrected ABP reference peaks.

    A PPG peak matches if within +/-tol_ms (+/-tol_samples) of any
    PTT-corrected ABP peak (abp_peaks - ptt_offset).

    Parameters
    ----------
    ppg_peaks : np.ndarray
        Indices of PPG systolic peaks.
    abp_peaks : np.ndarray
        Indices of ABP systolic peaks.
    ptt_offset : int
        Lag in samples from compute_ptt_offset.
    fs : int
        Sampling frequency in Hz (default 125).
    tol_ms : float
        Tolerance window in milliseconds (default +/-50 ms).

    Returns
    -------
    n_ppg : int
        Count of PPG peaks detected.
    matched_count : int
        Count of PPG peaks matched to ABP reference.
    matched_fraction : float
        matched_count / n_ppg (0.0 if n_ppg == 0).
    passes : bool
        True if matched_fraction >= 0.80.
    """
    ppg_peaks = np.asarray(ppg_peaks, dtype=int)
    abp_peaks = np.asarray(abp_peaks, dtype=int)

    n_ppg = len(ppg_peaks)
    if n_ppg == 0:
        return 0, 0, 0.0, False

    if len(abp_peaks) == 0:
        return n_ppg, 0, 0.0, False

    tol_samples = tol_ms * fs / 1000.0  # 6.25 samples at 125 Hz
    # Under ingest.py convention (ppg[t] ~ abp[t + lag]),
    # the PTT-corrected ABP peak position is abp_peak - ptt_offset.
    abp_peaks_corrected = abp_peaks.astype(float) - float(ptt_offset)

    matched = 0
    for p_pk in ppg_peaks:
        min_dist = np.min(np.abs(p_pk - abp_peaks_corrected))
        if min_dist <= tol_samples:
            matched += 1

    matched_fraction = matched / float(n_ppg)
    passes = bool(matched_fraction >= 0.80)
    return n_ppg, matched, matched_fraction, passes


def process_segment(
    ppg: np.ndarray,
    abp: np.ndarray,
    fs: int = 125,
    W: int = 37,
    tol_ms: float = 50.0,
) -> Dict[str, Any]:
    """Process a single 10-s segment for PPG fiducials and ABP agreement.

    Parameters
    ----------
    ppg : np.ndarray
        1250-sample PPG signal (channel 1).
    abp : np.ndarray
        1250-sample ABP signal (channel 2).
    fs : int
        Sampling frequency in Hz (default 125).

    Returns
    -------
    dict
        Dictionary containing detected peaks, feet, PTT offset, correlation,
        matched beat count, fraction, and pass/fail boolean.
    """
    ppg = np.asarray(ppg, dtype=np.float64)
    abp = np.asarray(abp, dtype=np.float64)

    # 1. Detect PPG fiducials on signal as stored
    ppg_peaks = detect_peaks_elgendi(ppg, fs=fs)
    ppg_feet = detect_feet(ppg, ppg_peaks, fs=fs)
    ppg_turns = detect_turning_points(ppg, ppg_peaks)

    # 2. Detect ABP reference fiducials (min-max scaled for Elgendi thresholding)
    abp_range = np.max(abp) - np.min(abp)
    if abp_range > 1e-6:
        abp_scaled = (abp - np.min(abp)) / abp_range
    else:
        abp_scaled = np.zeros_like(abp)
    abp_peaks = detect_peaks_elgendi(abp_scaled, fs=fs)
    abp_feet = detect_feet(abp, abp_peaks, fs=fs)

    # 3. Compute PTT offset lag (Amendment A3: symmetric window [-W, W))
    best_lag, max_corr = compute_ptt_offset(ppg, abp, W=W)

    # 4. Evaluate agreement
    n_ppg, matched, frac, passes = evaluate_beat_agreement(
        ppg_peaks, abp_peaks, best_lag, fs=fs, tol_ms=tol_ms
    )

    return {
        "ppg_peaks": ppg_peaks,
        "ppg_feet": ppg_feet,
        "ppg_turns": ppg_turns,
        "abp_peaks": abp_peaks,
        "abp_feet": abp_feet,
        "ptt_offset": best_lag,
        "max_corr": max_corr,
        "n_ppg_beats": n_ppg,
        "n_abp_beats": len(abp_peaks),
        "matched_beats": matched,
        "matched_fraction": frac,
        "passes_rule": passes,
    }


def run_validation(
    mat_path: str = "data/raw/VitalDB_Train_Subset.mat",
    parquet_path: str = "data/parquet/train_file.parquet",
    output_report_path: str = "results/detector_agreement_A3.md",
    fs: int = 125,
    W: int = 37,
) -> Dict[str, Any]:
    """Run detector validation across all validation subjects in the training set under Amendment A3."""
    import time
    from pathlib import Path
    import h5py
    import pandas as pd
    import pyarrow.parquet as pq

    print(f"Loading validation set metadata from {parquet_path}...")
    table = pq.read_table(parquet_path, columns=["segment_idx", "subject_id", "split", "sbp", "dbp"])
    df = table.to_pandas()
    val_mask = df["split"] == "val"
    val_df = df[val_mask].copy().reset_index(drop=True)
    val_indices = val_df["segment_idx"].values
    n_val = len(val_df)
    n_val_subj = len(val_df["subject_id"].unique())
    print(f"Validation cohort: {n_val} segments across {n_val_subj} subjects.")

    # Group contiguous indices to minimize HDF5 I/O overhead
    diffs = np.diff(val_indices)
    breaks = np.where(diffs > 1)[0]
    slice_starts = [val_indices[0]] + [val_indices[b + 1] for b in breaks]
    slice_ends = [val_indices[b] + 1 for b in breaks] + [val_indices[-1] + 1]
    slices = list(zip(slice_starts, slice_ends))
    print(f"Identified {len(slices)} contiguous HDF5 slices to read.")

    n_ppg_arr = np.zeros(n_val, dtype=np.int16)
    n_abp_arr = np.zeros(n_val, dtype=np.int16)
    ptt_arr = np.zeros(n_val, dtype=np.int16)
    corr_arr = np.zeros(n_val, dtype=np.float32)
    matched_arr = np.zeros(n_val, dtype=np.int16)
    frac_arr = np.zeros(n_val, dtype=np.float32)
    passes_arr = np.zeros(n_val, dtype=bool)

    t0 = time.time()
    curr_idx = 0
    with h5py.File(mat_path, "r") as f:
        sig_ds = f["Subset"]["Signals"]
        for s_idx, (start, end) in enumerate(slices):
            block_len = end - start
            block = sig_ds[:, :, start:end]
            for i in range(block_len):
                ppg = block[:, 1, i]
                abp = block[:, 2, i]
                res = process_segment(ppg, abp, fs=fs, W=W)

                out_pos = curr_idx + i
                n_ppg_arr[out_pos] = res["n_ppg_beats"]
                n_abp_arr[out_pos] = res["n_abp_beats"]
                ptt_arr[out_pos] = res["ptt_offset"]
                corr_arr[out_pos] = res["max_corr"]
                matched_arr[out_pos] = res["matched_beats"]
                frac_arr[out_pos] = res["matched_fraction"]
                passes_arr[out_pos] = res["passes_rule"]

            curr_idx += block_len
            if (s_idx + 1) % 25 == 0 or (s_idx + 1) == len(slices):
                elapsed = time.time() - t0
                speed = curr_idx / (elapsed + 1e-6)
                print(f"Processed {curr_idx}/{n_val} segments ({s_idx+1}/{len(slices)} slices) in {elapsed:.1f}s ({speed:.1f} seg/s)...")

    val_df["n_ppg"] = n_ppg_arr
    val_df["n_abp"] = n_abp_arr
    val_df["ptt_offset"] = ptt_arr
    val_df["max_corr"] = corr_arr
    val_df["matched_beats"] = matched_arr
    val_df["matched_fraction"] = frac_arr
    val_df["passes_rule"] = passes_arr

    # Compute key statistics
    overall_pass_share = float(np.mean(passes_arr))
    q10 = float(np.percentile(frac_arr, 10))
    q25 = float(np.percentile(frac_arr, 25))
    median_frac = float(np.median(frac_arr))
    q75 = float(np.percentile(frac_arr, 75))
    q90 = float(np.percentile(frac_arr, 90))
    iqr_frac = q75 - q25

    # Boundary hits
    bound_neg = int(np.sum(ptt_arr == -W))
    bound_pos = int(np.sum(ptt_arr == W - 1))
    bound_neg_pct = bound_neg / n_val * 100.0
    bound_pos_pct = bound_pos / n_val * 100.0

    # Lag histogram
    hist_bins = [
        (-W, -30), (-30, -20), (-20, -10), (-10, 0),
        (0, 10), (10, 20), (20, 30), (30, W)
    ]
    hist_rows = []
    for low_b, high_b in hist_bins:
        cnt = int(np.sum((ptt_arr >= low_b) & (ptt_arr < high_b)))
        pct = cnt / n_val * 100.0
        hist_rows.append(f"| [{low_b}, {high_b}) samples ({low_b/fs*1000:.0f} to {high_b/fs*1000:.0f} ms) | {cnt:,} | {pct:.2f}% |")
    hist_table = "\n".join(hist_rows)

    # By class: SBP >= 130 vs < 130
    mask_ge_130 = val_df["sbp"] >= 130
    n_ge_130 = int(mask_ge_130.sum())
    pass_ge_130 = int(val_df.loc[mask_ge_130, "passes_rule"].sum())
    share_ge_130 = pass_ge_130 / n_ge_130 if n_ge_130 > 0 else 0.0

    n_lt_130 = int((~mask_ge_130).sum())
    pass_lt_130 = int(val_df.loc[~mask_ge_130, "passes_rule"].sum())
    share_lt_130 = pass_lt_130 / n_lt_130 if n_lt_130 > 0 else 0.0

    # Sensitivity thresholds 120 and 140
    mask_ge_120 = val_df["sbp"] >= 120
    share_ge_120 = float(val_df.loc[mask_ge_120, "passes_rule"].mean()) if mask_ge_120.sum() > 0 else 0.0
    share_lt_120 = float(val_df.loc[~mask_ge_120, "passes_rule"].mean()) if (~mask_ge_120).sum() > 0 else 0.0

    mask_ge_140 = val_df["sbp"] >= 140
    share_ge_140 = float(val_df.loc[mask_ge_140, "passes_rule"].mean()) if mask_ge_140.sum() > 0 else 0.0
    share_lt_140 = float(val_df.loc[~mask_ge_140, "passes_rule"].mean()) if (~mask_ge_140).sum() > 0 else 0.0

    # Failure rule evaluation
    rule_passed = overall_pass_share >= 0.80
    decision_text = (
        "**PASS**: Validation agreement meets the 80% threshold. Step 1 (classic morphology) is unblocked."
        if rule_passed
        else "**Step 1 and later blocked**: Validation agreement is below the 80% threshold. No morphology gain is claimed, and Steps 0a and 0 remain the reported models."
    )

    # Format Markdown Report
    report = f"""# Detector Validation on Validation Subjects (Amendment A3)

**Date:** 2026-10-29  
**Amendment:** A3 (Symmetric PTT window [-W, W) with W = {W} samples / ~300 ms at 125 Hz)  
**Split:** VitalDB_Train_Subset.mat validation cohort (seed 42, 20% of training subjects)  
**Subjects:** {n_val_subj} subjects | **Segments:** {n_val:,} segments (10 s at 125 Hz)  
**Detector Configuration:**
- Method: Elgendi et al. (2013) two-moving-average method on Signals channel 1 as stored (no extra filter)
- Parameters (frozen): W1 = 111 ms (14 samples), W2 = 667 ms (83 samples), beta = 0.02 (unchanged)
- Foot detection: minimum in 300 ms window before each systolic peak
- Turning points: minimum between consecutive systolic peaks
- ABP Reference: channel 2 (min-max scaled for thresholding, same Elgendi parameters)
- PTT Search Window (Amendment A3): [{-W}, {W}) samples (~{-W/fs*1000:.0f} ms to +{(W-1)/fs*1000:.0f} ms)
- Beat Agreement Rule: PPG peak matched if within +/-50 ms (+/-6.25 samples) of PTT-corrected ABP peak; segment passes if matched beat fraction >= 80% (unchanged)

---

## 1. Headline Agreement Results

| Metric | Result |
|---|---|
| **Validation Segments Passing (>= 80% Match)** | **{overall_pass_share * 100:.2f}%** ({int(passes_arr.sum()):,}/{n_val:,}) |
| **Matched-Beat Fraction Median** | **{median_frac * 100:.1f}%** |
| **Matched-Beat Fraction IQR** | **{iqr_frac * 100:.1f}%** (Q1: {q25 * 100:.1f}%, Q3: {q75 * 100:.1f}%) |
| **Matched-Beat Percentiles** | P10: {q10 * 100:.1f}%, P90: {q90 * 100:.1f}%, Min: {float(np.min(frac_arr)) * 100:.1f}%, Max: {float(np.max(frac_arr)) * 100:.1f}% |
| **Mean Matched-Beat Fraction** | {float(np.mean(frac_arr)) * 100:.2f}% (SD: {float(np.std(frac_arr)) * 100:.2f}%) |

---

## 2. Agreement by Class (SBP Labels)

| SBP Threshold / Subgroup | Total Segments | Passing Segments | Passing Share |
|---|---|---|---|
| **Primary: SBP >= 130 mmHg** (Hypertensive range) | {n_ge_130:,} | {pass_ge_130:,} | **{share_ge_130 * 100:.2f}%** |
| **Primary: SBP < 130 mmHg** (Normotensive range) | {n_lt_130:,} | {pass_lt_130:,} | **{share_lt_130 * 100:.2f}%** |
| Sensitivity: SBP >= 120 mmHg | {int(mask_ge_120.sum()):,} | {int(val_df.loc[mask_ge_120, 'passes_rule'].sum()):,} | {share_ge_120 * 100:.2f}% |
| Sensitivity: SBP < 120 mmHg | {int((~mask_ge_120).sum()):,} | {int(val_df.loc[~mask_ge_120, 'passes_rule'].sum()):,} | {share_lt_120 * 100:.2f}% |
| Sensitivity: SBP >= 140 mmHg | {int(mask_ge_140.sum()):,} | {int(val_df.loc[mask_ge_140, 'passes_rule'].sum()):,} | {share_ge_140 * 100:.2f}% |
| Sensitivity: SBP < 140 mmHg | {int((~mask_ge_140).sum()):,} | {int(val_df.loc[~mask_ge_140, 'passes_rule'].sum()):,} | {share_lt_140 * 100:.2f}% |

---

## 3. PTT Lag Distribution and Boundary Hits

### Boundary Hits
- **Lower Boundary (lag = {-W} / {-W/fs*1000:.0f} ms):** {bound_neg:,} segments ({bound_neg_pct:.2f}%)
- **Upper Boundary (lag = {W-1} / {(W-1)/fs*1000:.0f} ms):** {bound_pos:,} segments ({bound_pos_pct:.2f}%)
- **Total Boundary Hits:** {bound_neg + bound_pos:,} segments ({(bound_neg + bound_pos)/n_val*100:.2f}%)

### Lag Histogram
| Lag Interval | Segments | Share (%) |
|---|---|---|
{hist_table}

---

## 4. Signal & Physiological Characteristics

| Feature / Metric | Median | IQR (Q1 - Q3) | Range [Min, Max] |
|---|---|---|---|
| PPG Beats per Segment | {float(np.median(n_ppg_arr)):.1f} | {float(np.percentile(n_ppg_arr, 75) - np.percentile(n_ppg_arr, 25)):.1f} ({float(np.percentile(n_ppg_arr, 25)):.1f} - {float(np.percentile(n_ppg_arr, 75)):.1f}) | [{int(np.min(n_ppg_arr))}, {int(np.max(n_ppg_arr))}] |
| ABP Reference Beats per Segment | {float(np.median(n_abp_arr)):.1f} | {float(np.percentile(n_abp_arr, 75) - np.percentile(n_abp_arr, 25)):.1f} ({float(np.percentile(n_abp_arr, 25)):.1f} - {float(np.percentile(n_abp_arr, 75)):.1f}) | [{int(np.min(n_abp_arr))}, {int(np.max(n_abp_arr))}] |
| PTT Offset (samples) | {float(np.median(ptt_arr)):.1f} | {float(np.percentile(ptt_arr, 75) - np.percentile(ptt_arr, 25)):.1f} ({float(np.percentile(ptt_arr, 25)):.1f} - {float(np.percentile(ptt_arr, 75)):.1f}) | [{int(np.min(ptt_arr))}, {int(np.max(ptt_arr))}] |
| PTT Offset (ms) | {float(np.median(ptt_arr)) / fs * 1000:.1f} ms | {(float(np.percentile(ptt_arr, 75)) - float(np.percentile(ptt_arr, 25))) / fs * 1000:.1f} ms | [{float(np.min(ptt_arr)) / fs * 1000:.1f}, {float(np.max(ptt_arr)) / fs * 1000:.1f}] ms |
| Peak PPG-ABP Correlation | {float(np.median(corr_arr)):.3f} | {float(np.percentile(corr_arr, 75) - np.percentile(corr_arr, 25)):.3f} ({float(np.percentile(corr_arr, 25)):.3f} - {float(np.percentile(corr_arr, 75)):.3f}) | [{float(np.min(corr_arr)):.3f}, {float(np.max(corr_arr)):.3f}] |

---

## 5. Detector Failure Rule Evaluation

- **Criterion:** Pass share must be >= 80.0% of validation segments.
- **Measured Pass Share:** **{overall_pass_share * 100:.2f}%**
- **Status:** {decision_text}
"""

    out_file = Path(output_report_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(report, encoding="utf-8")
    print(f"Detector validation report written to {output_report_path}")

    return {
        "overall_pass_share": overall_pass_share,
        "median_frac": median_frac,
        "iqr_frac": iqr_frac,
        "share_ge_130": share_ge_130,
        "share_lt_130": share_lt_130,
        "rule_passed": rule_passed,
    }


if __name__ == "__main__":
    run_validation()

