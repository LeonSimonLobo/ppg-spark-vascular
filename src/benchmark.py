"""Performance and scaling benchmark module for ppg-spark-vascular (Study C).

Measures PySpark distributed feature extraction throughput and scaling across:
- Core counts: {1, 2, 4, 8} (capped by machine physical cores)
- Partition counts: {4, 16, 64}
- Dataset fractions: {25%, 50%, 100%} of VitalDB Train development cohort
- 3 recorded repeats per configuration (median reported) + 1 unrecorded warm-up
- Baseline: Single-process pandas run on 5% sample, extrapolated to 100% [ESTIMATE]

NOTE: In accordance with PySpark constraints (one active SparkContext per JVM),
every Spark configuration executes in a separate isolated subprocess.
"""

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Dict, Any, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil

# Ensure repo root is on sys.path
REPO_DIR = Path(__file__).resolve().parent.parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))


def collect_machine_specs() -> Dict[str, Any]:
    """Capture machine hardware specs, OS, and software environment."""
    phys_cores = psutil.cpu_count(logical=False) or os.cpu_count() or 1
    logical_cores = psutil.cpu_count(logical=True) or os.cpu_count() or 1
    total_ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 2)

    import pyspark

    return {
        "platform": platform.platform(),
        "system": platform.system(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "physical_cores": phys_cores,
        "logical_cores": logical_cores,
        "total_ram_gb": total_ram_gb,
        "python_version": sys.version.split()[0],
        "pyspark_version": pyspark.__version__,
    }


def benchmark_pandas_single_process(
    input_parquet: str = "data/parquet/train_file.parquet",
    sample_fraction: float = 0.05,
    seed: int = 42,
) -> Dict[str, Any]:
    """Time single-process pandas feature extraction on a 5% sample and extrapolate to 100%."""
    import pyarrow.parquet as pq
    from src.features_c import extract_features_block

    print(f"\n[Baseline] Running single-process Pandas benchmark on {sample_fraction:.1%} sample...")
    table = pq.read_table(input_parquet)
    total_rows = len(table)
    sample_size = int(round(total_rows * sample_fraction))

    rng = np.random.default_rng(seed)
    indices = rng.choice(total_rows, size=sample_size, replace=False)
    sample_table = table.take(indices)
    pdf = sample_table.to_pandas()

    t0 = time.perf_counter()
    ppg_mat = np.array(list(pdf["ppg"]), dtype=np.float32).T
    _ = extract_features_block(ppg_mat, fs=125)
    elapsed = time.perf_counter() - t0

    throughput = sample_size / (elapsed + 1e-6)
    extrapolated_sec = elapsed / sample_fraction

    print(f"[Baseline] Pandas 5% ({sample_size:,} rows): {elapsed:.2f}s ({throughput:.1f} rows/s)")
    print(f"[Baseline] Extrapolated 100% ({total_rows:,} rows) runtime: {extrapolated_sec:.2f}s [ESTIMATE]")

    return {
        "mode": "pandas_single_process",
        "sample_rows": sample_size,
        "sample_fraction": sample_fraction,
        "sample_elapsed_sec": round(elapsed, 3),
        "throughput_rows_per_sec": round(throughput, 1),
        "total_rows": total_rows,
        "extrapolated_100pct_sec": round(extrapolated_sec, 3),
        "is_estimate": True,
    }


def run_spark_subprocess(
    cores: int,
    partitions: int,
    fraction: float,
    out_dir: str = "data/benchmark_tmp",
) -> float:
    """Execute spark_extract_c.py in an isolated subprocess and return elapsed seconds."""
    cmd = [
        sys.executable,
        "-m", "src.spark_extract_c",
        "--cores", str(cores),
        "--partitions", str(partitions),
        "--limit", str(fraction),
        "--out", out_dir,
        "--cohort", "train_file",
        "--report", "",  # Suppress markdown reporting during benchmark runs
    ]

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0

    if proc.returncode != 0:
        print(f"Error in subprocess (cores={cores}, parts={partitions}, frac={fraction}):")
        print(proc.stderr)
        raise RuntimeError(f"Subprocess failed with return code {proc.returncode}")

    return elapsed


def run_full_benchmark(
    output_dir: str = "results",
    tmp_data_dir: str = "data/benchmark_tmp",
):
    """Run full scaling benchmark grid and save metrics, machine info, and figures."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(tmp_data_dir, exist_ok=True)

    # 1. Machine specs
    specs = collect_machine_specs()
    machine_path = os.path.join(output_dir, "machine.json")
    with open(machine_path, "w", encoding="utf-8") as f:
        json.dump(specs, f, indent=2)
    print(f"[Benchmark] Saved machine specifications to {machine_path}")

    # 2. Pandas single-process baseline
    pandas_base = benchmark_pandas_single_process(
        input_parquet="data/parquet/train_file.parquet",
        sample_fraction=0.05,
    )

    # 3. Parameter Grid
    phys_cores = specs["physical_cores"]
    core_candidates = [1, 2, 4, 8]
    cores_list = [c for c in core_candidates if c <= phys_cores]
    partitions_list = [4, 16, 64]
    fractions_list = [0.25, 0.50, 1.00]

    print("\n" + "=" * 70)
    print("STARTING PYSPARK SCALING BENCHMARK")
    print(f"Cores: {cores_list} (physical: {phys_cores})")
    print(f"Partitions: {partitions_list}")
    print(f"Fractions: {[f'{int(f*100)}%' for f in fractions_list]}")
    print("Repeats: 1 warm-up (unrecorded) + 3 recorded runs (median reported)")
    print("=" * 70)

    # 4. Warm-up run (not recorded)
    print("\n[Warm-up] Executing unrecorded warm-up run (cores=1, parts=4, frac=0.25)...")
    _ = run_spark_subprocess(cores=1, partitions=4, fraction=0.25, out_dir=tmp_data_dir)
    print("[Warm-up] Complete.")

    records = []
    total_train_rows = pandas_base["total_rows"]

    for frac in fractions_list:
        target_rows = int(round(total_train_rows * frac))
        for parts in partitions_list:
            for c in cores_list:
                print(f"\n[Benchmark] Testing config: cores={c}, partitions={parts}, fraction={frac:.0%} ({target_rows:,} rows)...")
                repeats = []
                for rep in range(1, 4):
                    el = run_spark_subprocess(cores=c, partitions=parts, fraction=frac, out_dir=tmp_data_dir)
                    repeats.append(el)
                    print(f"  - Repeat {rep}: {el:.2f}s ({target_rows / el:.1f} rows/s)")

                med_sec = float(np.median(repeats))
                throughput = target_rows / med_sec

                records.append({
                    "cores": c,
                    "partitions": parts,
                    "train_fraction": frac,
                    "train_rows": target_rows,
                    "repeat_1_sec": round(repeats[0], 3),
                    "repeat_2_sec": round(repeats[1], 3),
                    "repeat_3_sec": round(repeats[2], 3),
                    "median_sec": round(med_sec, 3),
                    "throughput_rows_per_sec": round(throughput, 1),
                })

    df_bench = pd.DataFrame(records)

    # Compute speedups
    # Speedup relative to 1-core within same fraction and partition count
    speedups = []
    for idx, row in df_bench.iterrows():
        base_1c = df_bench[
            (df_bench["cores"] == 1) &
            (df_bench["partitions"] == row["partitions"]) &
            (df_bench["train_fraction"] == row["train_fraction"])
        ]
        if len(base_1c) > 0:
            t1 = base_1c["median_sec"].iloc[0]
            sp = t1 / row["median_sec"]
        else:
            sp = 1.0
        speedups.append(round(sp, 2))
    df_bench["speedup_vs_1core_same_part"] = speedups

    # Save benchmark.csv
    csv_path = os.path.join(output_dir, "benchmark.csv")
    df_bench.to_csv(csv_path, index=False)
    print(f"\n[Benchmark] Results saved to {csv_path}")

    # 5. Generate Figures
    generate_benchmark_figures(df_bench, pandas_base, output_dir=output_dir)

    # Cleanup temp data
    import shutil
    if os.path.exists(tmp_data_dir):
        shutil.rmtree(tmp_data_dir, ignore_errors=True)

    print("\nBenchmark complete.")


def generate_benchmark_figures(
    df_bench: pd.DataFrame,
    pandas_base: Dict[str, Any],
    output_dir: str = "results",
):
    """Plot runtime vs cores and speedup vs cores."""
    os.makedirs(output_dir, exist_ok=True)

    # Figure 1: Runtime vs Cores (at 100% data)
    fig1, ax1 = plt.subplots(figsize=(8, 5), dpi=300)
    df_100 = df_bench[df_bench["train_fraction"] == 1.0]

    markers = ["o", "s", "^"]
    for i, parts in enumerate(sorted(df_100["partitions"].unique())):
        sub = df_100[df_100["partitions"] == parts].sort_values("cores")
        ax1.plot(sub["cores"], sub["median_sec"], marker=markers[i % len(markers)], linewidth=2, label=f"Spark ({parts} partitions)")

    # Baseline horizontal line
    t_base = pandas_base["extrapolated_100pct_sec"]
    ax1.axhline(t_base, color="red", linestyle="--", linewidth=1.5, label=f"Pandas 1-Process Baseline ({t_base:.1f}s, extrapolated)")

    ax1.set_xlabel("Executor Cores", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Median Wall-Clock Runtime (s)", fontsize=11, fontweight="bold")
    ax1.set_title("Study C: Feature Extraction Runtime vs Cores (100% Train Cohort, 465k Segments)", fontsize=12, pad=12)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(frameon=True)
    fig1.tight_layout()

    fig1_path = os.path.join(output_dir, "benchmark_runtime_vs_cores.png")
    fig1.savefig(fig1_path)
    plt.close(fig1)
    print(f"[Benchmark] Saved {fig1_path}")

    # Figure 2: Speedup vs Cores (at 100% data)
    fig2, ax2 = plt.subplots(figsize=(8, 5), dpi=300)
    for i, parts in enumerate(sorted(df_100["partitions"].unique())):
        sub = df_100[df_100["partitions"] == parts].sort_values("cores")
        ax2.plot(sub["cores"], sub["speedup_vs_1core_same_part"], marker=markers[i % len(markers)], linewidth=2, label=f"Spark ({parts} partitions)")

    # Ideal linear speedup
    cores_vals = sorted(df_100["cores"].unique())
    ax2.plot(cores_vals, cores_vals, color="gray", linestyle="--", linewidth=1.5, label="Ideal Linear Speedup")

    ax2.set_xlabel("Executor Cores", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Speedup ($T_1 / T_p$)", fontsize=11, fontweight="bold")
    ax2.set_title("Study C: Scaling Speedup vs Cores (100% Train Cohort)", fontsize=12, pad=12)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(frameon=True)
    fig2.tight_layout()

    fig2_path = os.path.join(output_dir, "benchmark_speedup_vs_cores.png")
    fig2.savefig(fig2_path)
    plt.close(fig2)
    print(f"[Benchmark] Saved {fig2_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Distributed PPG feature extraction scaling benchmark.")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory for reports and figures.")
    parser.add_argument("--run", action="store_true", help="Execute the full scaling benchmark suite.")
    parser.add_argument("--dry-run", action="store_true", help="Print configuration without running.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.run:
        run_full_benchmark(output_dir=args.output_dir)
    else:
        specs = collect_machine_specs()
        print("[Benchmark Module] Configured and validated.")
        print(f"Machine detected: {specs['physical_cores']} physical cores, {specs['logical_cores']} logical cores, {specs['total_ram_gb']} GB RAM.")
        print("Benchmark execution deferred per instruction: 'Do not run the benchmark yet.'")
        print("To execute when authorized: python -m src.benchmark --run")
