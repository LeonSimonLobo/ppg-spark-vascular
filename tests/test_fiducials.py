"""Unit tests for frozen Elgendi fiducial detection and agreement pipeline."""

import numpy as np
import pytest
from src.fiducials import (
    detect_peaks_elgendi,
    detect_feet,
    detect_turning_points,
    compute_ptt_offset,
    evaluate_beat_agreement,
    process_segment,
)


def generate_synthetic_pulse_train(
    fs: int = 125,
    duration_s: float = 10.0,
    bpm: float = 75.0,
    noise_std: float = 0.01,
    seed: int = 42,
):
    """Generate a synthetic PPG waveform with known peak positions.

    Each beat consists of an asymmetric pulse: sharp systolic upstroke (20% of beat)
    and slower diastolic decay (80% of beat).
    """
    rng = np.random.default_rng(seed)
    n_samples = int(round(duration_s * fs))
    t = np.arange(n_samples) / fs

    period_s = 60.0 / bpm
    period_samples = int(round(period_s * fs))  # 100 samples at 75 bpm, 125 Hz

    signal = np.zeros(n_samples, dtype=np.float64)
    true_peaks = []
    true_feet = []

    # Place pulses starting from first period
    upstroke_samples = int(round(0.20 * period_samples))  # 20 samples
    decay_samples = period_samples - upstroke_samples     # 80 samples

    start = upstroke_samples
    while start + decay_samples < n_samples:
        pk = start
        ft = start - upstroke_samples
        true_peaks.append(pk)
        true_feet.append(ft)

        # Upstroke: half sine
        u_idx = np.arange(ft, pk)
        signal[u_idx] = np.sin(0.5 * np.pi * (u_idx - ft) / upstroke_samples)

        # Decay: cosine decay
        d_idx = np.arange(pk, min(n_samples, pk + decay_samples))
        signal[d_idx] = np.cos(0.5 * np.pi * (d_idx - pk) / decay_samples)

        start += period_samples

    # Normalize to [0, 1] as in PulseDB stored PPG
    sig_range = np.max(signal) - np.min(signal)
    if sig_range > 1e-12:
        signal = (signal - np.min(signal)) / sig_range

    if noise_std > 0:
        signal += rng.normal(0, noise_std, size=n_samples)
        signal = np.clip(signal, 0.0, 1.0)

    return signal, np.array(true_peaks), np.array(true_feet)


def test_synthetic_pulse_detection():
    """Verify that detect_peaks_elgendi recovers true peaks within +/-2 samples."""
    fs = 125
    signal, true_peaks, true_feet = generate_synthetic_pulse_train(
        fs=fs, duration_s=10.0, bpm=75.0, noise_std=0.01, seed=42
    )

    detected_peaks = detect_peaks_elgendi(signal, fs=fs)
    assert len(detected_peaks) == len(true_peaks), (
        f"Expected {len(true_peaks)} peaks, detected {len(detected_peaks)}"
    )

    for true_pk, det_pk in zip(true_peaks, detected_peaks):
        assert abs(true_pk - det_pk) <= 2, (
            f"True peak {true_pk} differs from detected {det_pk} by >2 samples"
        )


def test_foot_and_turning_points_detection():
    """Verify foot and turning point detection on synthetic pulse train."""
    fs = 125
    signal, true_peaks, true_feet = generate_synthetic_pulse_train(
        fs=fs, duration_s=10.0, bpm=75.0, noise_std=0.00, seed=42
    )

    detected_peaks = detect_peaks_elgendi(signal, fs=fs)
    feet = detect_feet(signal, detected_peaks, fs=fs, window_ms=300.0)
    turns = detect_turning_points(signal, detected_peaks)

    assert len(feet) == len(detected_peaks)
    assert len(turns) == len(detected_peaks) - 1

    # Each foot must precede its corresponding peak
    for ft, pk in zip(feet, detected_peaks):
        assert ft <= pk
        assert signal[ft] <= signal[pk]

    # Each turning point must lie between consecutive peaks
    for i, trn in enumerate(turns):
        assert detected_peaks[i] <= trn <= detected_peaks[i + 1]


def test_ptt_offset_and_agreement():
    """Verify PTT offset and beat agreement on a synthetically delayed pair."""
    fs = 125
    ppg, ppg_peaks, _ = generate_synthetic_pulse_train(
        fs=fs, duration_s=10.0, bpm=75.0, noise_std=0.0, seed=42
    )

    # Simulate ABP: abp[t + 35] matches ppg[t], i.e. abp[35:] = ppg[:-35]
    known_lag = 35
    abp = np.zeros_like(ppg)
    abp[known_lag:] = ppg[:-known_lag] * 40.0 + 80.0  # ABP in mmHg (80 to 120 mmHg)
    abp[:known_lag] = np.mean(abp[known_lag:])

    best_lag, max_corr = compute_ptt_offset(ppg, abp, max_lag=60)
    assert best_lag == known_lag, f"Expected lag {known_lag}, got {best_lag}"
    assert max_corr > 0.90, f"Expected high correlation, got {max_corr}"

    res = process_segment(ppg, abp, fs=fs)
    assert res["passes_rule"] is True
    assert res["matched_fraction"] >= 0.80


def test_empty_and_flat_signals():
    """Verify robust handling of empty, flat, or constant signals."""
    flat = np.zeros(1250)
    peaks = detect_peaks_elgendi(flat)
    assert len(peaks) == 0

    feet = detect_feet(flat, peaks)
    assert len(feet) == 0

    turns = detect_turning_points(flat, peaks)
    assert len(turns) == 0

    empty = np.array([])
    assert len(detect_peaks_elgendi(empty)) == 0
