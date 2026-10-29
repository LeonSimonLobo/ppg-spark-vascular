"""Feature extraction and preprocessing module for cross-subject SBP classification (PLAN_C.md).

Features are extracted strictly from the single 10-s PPG segment and patient demographics.
No per-subject normalization, centering, or transduction is applied.

Feature Ladder (Amendment C1):
- C0: Demographics (age, sex, bmi)
- C1: C0 + PPG heart rate from inter-beat interval (hr_ibi)
- C2m: C1 + morphology only (systolic_rise_time, pulse_width, decay_ratio)
- C2: C1 + morphology + quality fields (systolic_rise_time, pulse_width, decay_ratio,
      beat_count, is_clipped, snr_ratio, ibi_sd)

Missing values are imputed using training-set medians accompanied by binary missingness indicators.
No rows are dropped.
"""

from pathlib import Path
import time
from typing import Dict, Any, Tuple, Optional, List
import h5py
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.fiducials import detect_peaks_elgendi


# Canonical feature column lists
C0_COLS = ["age", "sex", "bmi"]
C1_COLS = C0_COLS + ["hr_ibi"]
C2m_COLS = C1_COLS + [
    "systolic_rise_time",
    "pulse_width",
    "decay_ratio",
]
C2_COLS = C1_COLS + [
    "systolic_rise_time",
    "pulse_width",
    "decay_ratio",
    "beat_count",
    "is_clipped",
    "snr_ratio",
    "ibi_sd",
]

ALL_EXTRACTED_COLS = [
    "age",
    "sex",
    "bmi",
    "hr_ibi",
    "hr",
    "ibi_sd",
    "systolic_rise_time",
    "pulse_width",
    "decay_ratio",
    "beat_count",
    "is_clipped",
    "snr_ratio",
]


def extract_single_segment_features(
    ppg: np.ndarray,
    fs: int = 125,
) -> Dict[str, float]:
    """Extract morphology and quality metrics from a single 10-s PPG waveform.

    Parameters
    ----------
    ppg : np.ndarray
        1D PPG waveform (e.g., 1250 samples at 125 Hz).
    fs : int
        Sampling frequency in Hz (default 125).

    Returns
    -------
    dict with keys:
        hr_ibi, hr, ibi_sd, systolic_rise_time, pulse_width, decay_ratio,
        beat_count, is_clipped, snr_ratio
    """
    ppg = np.asarray(ppg, dtype=np.float64)
    n_samples = len(ppg)

    # 1. Quality: Rail clipping / saturation check (>1% samples within 1e-4 of rail)
    frac_min = np.mean(ppg <= 1e-4)
    frac_max = np.mean(ppg >= 1.0 - 1e-4)
    is_clipped = float((frac_min > 0.01) or (frac_max > 0.01))

    # 2. Quality: SNR / relative physiological band energy [0.5, 8.0] Hz
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs)
    psd = np.abs(np.fft.rfft(ppg)) ** 2
    phys_mask = (freqs >= 0.5) & (freqs <= 8.0)
    non_dc_mask = freqs > 0
    phys_energy = np.sum(psd[phys_mask])
    total_energy = np.sum(psd[non_dc_mask])
    snr_ratio = float(phys_energy / (total_energy + 1e-12))

    # 3. Peak detection
    peaks = detect_peaks_elgendi(ppg, fs=fs)
    n_pks = len(peaks)
    beat_count = float(n_pks)

    if n_pks == 0:
        return {
            "hr_ibi": np.nan,
            "hr": np.nan,
            "ibi_sd": np.nan,
            "systolic_rise_time": np.nan,
            "pulse_width": np.nan,
            "decay_ratio": np.nan,
            "beat_count": beat_count,
            "is_clipped": is_clipped,
            "snr_ratio": snr_ratio,
        }

    # Heart rate from beat count (coarse n_pks x 6)
    hr = float(n_pks * 6.0)

    # Continuous heart rate from median IBI and IBI standard deviation
    if n_pks >= 2:
        ibi_sec = np.diff(peaks) / float(fs)
        med_ibi = float(np.median(ibi_sec))
        hr_ibi = float(60.0 / med_ibi) if med_ibi > 0 else np.nan
        ibi_sd = float(np.std(ibi_sec, ddof=1)) if len(ibi_sec) > 1 else 0.0
    else:
        hr_ibi = np.nan
        ibi_sd = np.nan

    # 4. Morphology features
    w_foot = int(round(0.3 * fs))  # 300 ms preceding peak
    rise_times = []
    pulse_widths = []
    decay_ratios = []

    for k in range(n_pks):
        pk = peaks[k]

        # Foot: minimum in 300 ms preceding peak
        start_foot = max(0, pk - w_foot)
        window_foot = ppg[start_foot : pk + 1]
        min_foot_val = np.min(window_foot)
        foot_ties = np.where(window_foot <= min_foot_val + 1e-9)[0]
        foot = start_foot + int(foot_ties[-1])

        # Turning point: minimum between peak and next peak
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

        # Pulse width at 50% amplitude (FWHM)
        pw = np.nan
        if h > 1e-4 and r_time > 0 and d_time > 0:
            y50 = y_foot + 0.5 * h

            # Ascending 50% crossing
            t_asc = None
            for m in range(foot, pk):
                if ppg[m] <= y50 <= ppg[m + 1] and (ppg[m + 1] > ppg[m]):
                    t_asc = m + (y50 - ppg[m]) / (ppg[m + 1] - ppg[m])
                    break

            # Descending 50% crossing
            t_desc = None
            for m in range(pk, turn):
                if ppg[m] >= y50 >= ppg[m + 1] and (ppg[m] > ppg[m + 1]):
                    t_desc = m + (y50 - ppg[m]) / (ppg[m + 1] - ppg[m])
                    break

            if t_asc is not None and t_desc is not None and t_desc > t_asc:
                pw = (t_desc - t_asc) / fs

        # Decay ratio: time ratio (decay duration / rise duration)
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

    return {
        "hr_ibi": hr_ibi,
        "hr": hr,
        "ibi_sd": ibi_sd,
        "systolic_rise_time": med_rise,
        "pulse_width": med_pw,
        "decay_ratio": med_decay,
        "beat_count": beat_count,
        "is_clipped": is_clipped,
        "snr_ratio": snr_ratio,
    }


def extract_features_block(
    ppg_block: np.ndarray,
    fs: int = 125,
) -> Dict[str, np.ndarray]:
    """Extract features for a 2D block of segments [n_samples, n_segments].

    Vectorizes FFT and rail clipping across columns for high throughput.
    """
    n_samples, n_segs = ppg_block.shape

    # 1. Vectorized Clipping
    min_rail = np.sum(ppg_block <= 1e-4, axis=0)
    max_rail = np.sum(ppg_block >= 1.0 - 1e-4, axis=0)
    thresh = int(round(0.01 * n_samples))
    is_clipped_arr = ((min_rail > thresh) | (max_rail > thresh)).astype(np.float32)

    # 2. Vectorized SNR
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs)
    phys_mask = (freqs >= 0.5) & (freqs <= 8.0)
    non_dc_mask = freqs > 0
    psds = np.abs(np.fft.rfft(ppg_block, axis=0)) ** 2
    phys_e = np.sum(psds[phys_mask, :], axis=0)
    tot_e = np.sum(psds[non_dc_mask, :], axis=0)
    snr_arr = (phys_e / (tot_e + 1e-12)).astype(np.float32)

    # 3. Peak detection & Morphology loop
    hr_ibi_arr = np.full(n_segs, np.nan, dtype=np.float32)
    hr_arr = np.full(n_segs, np.nan, dtype=np.float32)
    ibi_sd_arr = np.full(n_segs, np.nan, dtype=np.float32)
    rise_arr = np.full(n_segs, np.nan, dtype=np.float32)
    pw_arr = np.full(n_segs, np.nan, dtype=np.float32)
    decay_arr = np.full(n_segs, np.nan, dtype=np.float32)
    beat_arr = np.zeros(n_segs, dtype=np.float32)

    for j in range(n_segs):
        sig = ppg_block[:, j]
        feats = extract_single_segment_features(sig, fs=fs)
        hr_ibi_arr[j] = feats["hr_ibi"]
        hr_arr[j] = feats["hr"]
        ibi_sd_arr[j] = feats["ibi_sd"]
        rise_arr[j] = feats["systolic_rise_time"]
        pw_arr[j] = feats["pulse_width"]
        decay_arr[j] = feats["decay_ratio"]
        beat_arr[j] = feats["beat_count"]

    return {
        "hr_ibi": hr_ibi_arr,
        "hr": hr_arr,
        "ibi_sd": ibi_sd_arr,
        "systolic_rise_time": rise_arr,
        "pulse_width": pw_arr,
        "decay_ratio": decay_arr,
        "beat_count": beat_arr,
        "is_clipped": is_clipped_arr,
        "snr_ratio": snr_arr,
    }


class PlanCImputer:
    """Pre-registration compliant median imputer with missingness indicator columns.

    Fits feature medians strictly on training folds/cohort.
    For each feature in cols, replaces NaN with training median and
    appends `is_missing_<col>` (1.0 for imputed values, 0.0 otherwise).
    Guarantees no rows are dropped.
    """

    def __init__(self, cols: Optional[List[str]] = None):
        self.cols = list(cols) if cols is not None else list(ALL_EXTRACTED_COLS)
        self.medians_: Dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> "PlanCImputer":
        for col in self.cols:
            if col in df.columns:
                series = df[col].dropna()
                med = float(series.median()) if len(series) > 0 else 0.0
                self.medians_[col] = med
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()
        for col in self.cols:
            if col in df_out.columns:
                is_missing = df_out[col].isna().astype(np.float32)
                med = self.medians_.get(col, 0.0)
                df_out[col] = df_out[col].fillna(med).astype(np.float32)
                df_out[f"is_missing_{col}"] = is_missing
        return df_out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


def get_feature_matrix(
    df: pd.DataFrame,
    ladder_step: str = "C2",
) -> Tuple[np.ndarray, List[str]]:
    """Return numpy array and feature names for a specific ladder step.

    Parameters
    ----------
    df : pd.DataFrame
        Transformed DataFrame containing features and `is_missing_*` indicator columns.
    ladder_step : str
        'C0', 'C1', 'C2m', or 'C2'.

    Returns
    -------
    X : np.ndarray
    feature_names : List[str]
    """
    if ladder_step == "C0":
        base_cols = list(C0_COLS)
    elif ladder_step == "C1":
        base_cols = list(C1_COLS)
    elif ladder_step == "C2m":
        base_cols = list(C2m_COLS)
    elif ladder_step == "C2":
        base_cols = list(C2_COLS)
    else:
        raise ValueError(f"Unknown ladder step: {ladder_step}. Expected 'C0', 'C1', 'C2m', or 'C2'.")

    # Include base cols + their corresponding missingness indicator columns
    cols_to_use = []
    for col in base_cols:
        cols_to_use.append(col)
        indicator_col = f"is_missing_{col}"
        if indicator_col in df.columns:
            cols_to_use.append(indicator_col)

    X = df[cols_to_use].to_numpy(dtype=np.float32)
    return X, cols_to_use
