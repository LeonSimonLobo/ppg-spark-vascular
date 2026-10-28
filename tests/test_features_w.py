"""Unit tests for within-subject PPG feature extraction and centering."""

import numpy as np
import pandas as pd
import pytest

from src.features_w import extract_segment_features, center_features_per_subject


def test_extract_segment_features_synthetic():
    fs = 125
    n_samples = 1250
    ppg = np.full(n_samples, 0.1, dtype=np.float64)

    # 10 beats with exact rise time = 20 samples (0.16s), decay time = 60 samples (0.48s)
    # Total period = 100 samples (0.80s) -> HR = 75 bpm
    known_rise_s = 20.0 / fs  # 0.160 s
    known_decay_s = 60.0 / fs  # 0.480 s
    known_decay_ratio = 60.0 / 20.0  # 3.0
    known_pw_s = 40.0 / fs  # 0.320 s (FWHM: 10 samples on rise + 30 samples on decay)

    for b in range(10):
        start = 50 + b * 100
        pk = start + 20
        turn = start + 80
        # Linear rise from 0.1 to 1.0
        ppg[start:pk] = np.linspace(0.1, 1.0, 20)
        # Linear decay from 1.0 to 0.1
        ppg[pk:turn] = np.linspace(1.0, 0.1, 60)

    hr, rise_time, pw, decay_ratio = extract_segment_features(ppg, fs=fs)

    assert hr == 60.0  # 10 beats in 10 s -> 60 bpm
    assert np.isclose(rise_time, known_rise_s, atol=0.015)
    assert np.isclose(pw, known_pw_s, atol=0.015)
    assert np.isclose(decay_ratio, known_decay_ratio, atol=0.20)


def test_center_features_per_subject():
    # 2 subjects, each with 20 segments
    n_seg = 20
    df = pd.DataFrame({
        "subject_id": ["sub_1"] * n_seg + ["sub_2"] * n_seg,
        "hr": np.concatenate([np.random.normal(70, 5, n_seg), np.random.normal(90, 8, n_seg)]),
        "systolic_rise_time": np.concatenate([np.random.normal(0.15, 0.02, n_seg), np.random.normal(0.25, 0.03, n_seg)]),
        "pulse_width": np.concatenate([np.random.normal(0.30, 0.03, n_seg), np.random.normal(0.40, 0.04, n_seg)]),
        "decay_ratio": np.concatenate([np.random.normal(2.5, 0.3, n_seg), np.random.normal(3.5, 0.5, n_seg)]),
    })

    centered = center_features_per_subject(df)

    for col in ["hr", "systolic_rise_time", "pulse_width", "decay_ratio"]:
        delta_col = f"delta_{col}"
        assert delta_col in centered.columns
        for sub in ["sub_1", "sub_2"]:
            sub_deltas = centered[centered["subject_id"] == sub][delta_col]
            assert np.isclose(sub_deltas.median(), 0.0, atol=1e-6)


def test_extract_segment_features_flat_signal():
    fs = 125
    ppg = np.zeros(1250, dtype=np.float64)
    hr, rise_time, pw, decay_ratio = extract_segment_features(ppg, fs=fs)
    assert hr == 0.0
    assert np.isnan(rise_time)
    assert np.isnan(pw)
    assert np.isnan(decay_ratio)
