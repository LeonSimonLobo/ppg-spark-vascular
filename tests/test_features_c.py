"""Unit tests for single-segment feature extraction and imputation for PLAN_C."""

import numpy as np
import pandas as pd
import pytest

from src.features_c import (
    extract_single_segment_features,
    extract_features_block,
    PlanCImputer,
    get_feature_matrix,
    C0_COLS,
    C1_COLS,
    C2m_COLS,
    C2_COLS,
)


def generate_synthetic_ppg(
    freq_hz: float = 1.2,
    fs: int = 125,
    duration_s: float = 10.0,
) -> np.ndarray:
    """Generate synthetic asymmetric PPG-like waveform."""
    t = np.linspace(0, duration_s, int(fs * duration_s), endpoint=False)
    # Fundamental + harmonics for pulse shape
    y = np.sin(2 * np.pi * freq_hz * t) + 0.3 * np.sin(4 * np.pi * freq_hz * t - 0.5)
    # Min-max scale to [0, 1]
    y = (y - np.min(y)) / (np.max(y) - np.min(y))
    return y


def test_extract_single_segment_features_normal():
    """Test feature extraction on clean synthetic PPG signal."""
    sig = generate_synthetic_ppg(freq_hz=1.2, fs=125)
    feats = extract_single_segment_features(sig, fs=125)

    assert "hr_ibi" in feats
    assert "hr" in feats
    assert "ibi_sd" in feats
    assert "systolic_rise_time" in feats
    assert "pulse_width" in feats
    assert "decay_ratio" in feats
    assert "beat_count" in feats
    assert "is_clipped" in feats
    assert "snr_ratio" in feats

    assert 60.0 <= feats["hr"] <= 85.0
    assert 60.0 <= feats["hr_ibi"] <= 85.0
    assert feats["beat_count"] >= 10.0
    assert 0.1 <= feats["pulse_width"] <= 0.6
    assert feats["is_clipped"] == 0.0
    assert feats["snr_ratio"] > 0.8


def test_extract_single_segment_features_flat_signal():
    """Test feature extraction on flat / dead zero signal (zero beats detected)."""
    flat = np.zeros(1250, dtype=np.float64)
    feats = extract_single_segment_features(flat, fs=125)

    assert feats["beat_count"] == 0.0
    assert np.isnan(feats["hr_ibi"])
    assert np.isnan(feats["hr"])
    assert np.isnan(feats["ibi_sd"])
    assert np.isnan(feats["systolic_rise_time"])
    assert np.isnan(feats["pulse_width"])
    assert np.isnan(feats["decay_ratio"])


def test_extract_single_segment_features_clipped():
    """Test detection of clipped / saturated waveforms."""
    clipped = np.ones(1250, dtype=np.float64)  # 100% at rail
    feats = extract_single_segment_features(clipped, fs=125)
    assert feats["is_clipped"] == 1.0


def test_extract_features_block():
    """Test vectorized block extraction matches segment-wise extraction."""
    sig1 = generate_synthetic_ppg(freq_hz=1.0, fs=125)
    sig2 = generate_synthetic_ppg(freq_hz=1.5, fs=125)
    block = np.column_stack([sig1, sig2])

    block_feats = extract_features_block(block, fs=125)
    assert len(block_feats["hr_ibi"]) == 2
    assert len(block_feats["hr"]) == 2
    assert block_feats["beat_count"][0] < block_feats["beat_count"][1]


def test_plan_c_imputer_and_ladder_matrix():
    """Test median imputation, missingness indicator creation, and feature ladder matrices."""
    train_df = pd.DataFrame({
        "age": [50.0, 60.0, 70.0],
        "sex": [1.0, 0.0, 1.0],
        "bmi": [22.0, 26.0, 24.0],
        "hr_ibi": [71.5, np.nan, 83.2],
        "hr": [72.0, np.nan, 84.0],
        "ibi_sd": [0.02, np.nan, 0.03],
        "systolic_rise_time": [0.20, np.nan, 0.24],
        "pulse_width": [0.25, 0.30, np.nan],
        "decay_ratio": [2.5, np.nan, 3.5],
        "beat_count": [12.0, 0.0, 14.0],
        "is_clipped": [0.0, 0.0, 0.0],
        "snr_ratio": [0.95, 0.50, 0.98],
    })

    imputer = PlanCImputer()
    imputer.fit(train_df)

    # Impute training data
    df_imputed = imputer.transform(train_df)
    assert len(df_imputed) == len(train_df)  # No rows dropped!
    assert not df_imputed.isnull().any().any()  # No NaNs remain

    # Check that indicator columns were appended
    assert "is_missing_hr_ibi" in df_imputed.columns
    assert "is_missing_pulse_width" in df_imputed.columns
    assert df_imputed["is_missing_hr_ibi"].iloc[1] == 1.0
    assert df_imputed["is_missing_hr_ibi"].iloc[0] == 0.0

    # Test ladder matrices
    X_c0, names_c0 = get_feature_matrix(df_imputed, "C0")
    X_c1, names_c1 = get_feature_matrix(df_imputed, "C1")
    X_c2m, names_c2m = get_feature_matrix(df_imputed, "C2m")
    X_c2, names_c2 = get_feature_matrix(df_imputed, "C2")

    assert X_c0.shape[0] == 3
    assert X_c1.shape[0] == 3
    assert X_c2m.shape[0] == 3
    assert X_c2.shape[0] == 3

    assert X_c0.shape[1] < X_c1.shape[1] < X_c2m.shape[1] < X_c2.shape[1]
    assert "age" in names_c0
    assert "hr_ibi" in names_c1
    assert "systolic_rise_time" in names_c2m
    assert "decay_ratio" in names_c2m
    assert "snr_ratio" not in names_c2m
    assert "systolic_rise_time" in names_c2
    assert "snr_ratio" in names_c2
