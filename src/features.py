"""Feature extraction module for ppg-spark-vascular.

Performs PPG signal pre-processing (bandpass filtering, derivative calculations:
VPG and APG), morphological peak/trough detection, and computation of vascular
biomarkers (Augmentation Index, Stiffness Index, Reflection Index, Crest Time).
"""

import argparse
import os
from typing import Dict, List, Optional

import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, pandas_udf
from pyspark.sql.types import ArrayType, DoubleType, MapType, StringType
from scipy.signal import butter, filtfilt, find_peaks


def butter_bandpass_filter(data: np.ndarray, lowcut: float = 0.5, highcut: float = 8.0, fs: float = 125.0, order: int = 3) -> np.ndarray:
    """Apply zero-phase Butterworth bandpass filter to raw PPG signal."""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, data)


def compute_derivatives(ppg_signal: np.ndarray, fs: float) -> Dict[str, np.ndarray]:
    """Compute first (VPG) and second (APG) derivatives of the filtered PPG waveform."""
    dt = 1.0 / fs
    vpg = np.gradient(ppg_signal, dt)  # Velocity Plethysmogram
    apg = np.gradient(vpg, dt)         # Acceleration Plethysmogram
    return {"ppg": ppg_signal, "vpg": vpg, "apg": apg}


def extract_vascular_features(ppg_segment: List[float], fs: float = 125.0) -> Dict[str, float]:
    """Extract clinical vascular markers from a single PPG pulse or pulse window:
    - Peak-to-peak interval (Heart Rate / Pulse rate variability)
    - Crest time (time from foot to systolic peak)
    - Stiffness Index (SI = body_height / delta_T)
    - Reflection Index (RI = diastolic peak / systolic peak * 100)
    - APG indices: a, b, c, d, e wave ratios (b/a, c/a, d/a, e/a)
    """
    signal = np.array(ppg_segment, dtype=np.float64)
    if len(signal) < int(fs * 0.5):
        return {}

    filtered = butter_bandpass_filter(signal, fs=fs)
    derivatives = compute_derivatives(filtered, fs)

    # Placeholder feature dictionary
    features = {
        "mean_amplitude": float(np.mean(filtered)),
        "std_amplitude": float(np.std(filtered)),
        "systolic_peak_count": float(len(find_peaks(filtered, distance=int(fs * 0.4))[0])),
        # TODO: Implement complete landmark detection (systolic peak, dicrotic notch, diastolic peak)
        # TODO: Compute arterial stiffness index (SI) and reflection index (RI)
    }
    return features


def run_feature_pipeline(input_parquet: str, output_parquet: str, spark: Optional[SparkSession] = None):
    """Load ingested PPG Parquet data, run distributed feature extraction using PySpark,
    and save resulting feature matrix.
    """
    if spark is None:
        spark = SparkSession.builder.appName("PPG-Feature-Extraction").getOrCreate()

    print(f"[Features] Reading ingested data from: {input_parquet}")
    print(f"[Features] Output feature matrix destination: {output_parquet}")

    # TODO: Load df = spark.read.parquet(input_parquet)
    # TODO: Apply Pandas UDF or Spark ML Transformers to extract vascular features across partitions
    # TODO: df_features.write.mode("overwrite").parquet(output_parquet)
    print(f"[Features] Feature extraction pipeline completed.")


def parse_args():
    parser = argparse.ArgumentParser(description="Extract vascular biomarkers and features from PPG signals.")
    parser.add_argument("--input-parquet", type=str, default="data/processed/ppg_raw.parquet", help="Path to ingested PPG Parquet dataset")
    parser.add_argument("--output-parquet", type=str, default="data/processed/ppg_features.parquet", help="Path to save extracted features")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_feature_pipeline(input_parquet=args.input_parquet, output_parquet=args.output_parquet)
