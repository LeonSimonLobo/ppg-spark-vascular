"""Spark session manager for distributed PPG feature extraction and processing.

Configured for Windows local multi-core execution with PyArrow optimization,
same-drive spill directories, and worker PYTHONPATH propagation.
"""

import os
from pathlib import Path
import sys

repo_dir = Path(__file__).resolve().parent.parent
if str(repo_dir) not in sys.path:
    sys.path.insert(0, str(repo_dir))

from pyspark.sql import SparkSession


def get_spark(
    cores: str = "*",
    mem: str = "6g",
    app_name: str = "ppg-spark-vascular",
) -> SparkSession:
    """Create or retrieve a configured local SparkSession.

    Parameters
    ----------
    cores : str or int
        Number of local executor cores (e.g., "*", 1, 2, 4, 8).
    mem : str
        Driver memory allocation (e.g., "6g", "4g").
    app_name : str
        Spark application name.

    Returns
    -------
    SparkSession
    """
    # 1. Force python worker executable to match current interpreter
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    # 2. Repo and temp directory on same drive as data/
    repo_dir = Path(__file__).resolve().parent.parent
    data_dir = repo_dir / "data"
    spark_tmp_dir = repo_dir / ".spark_tmp"
    spark_tmp_dir.mkdir(parents=True, exist_ok=True)

    # 3. Propagate src/ and repo directory to PYTHONPATH for driver and worker processes
    repo_str = str(repo_dir.resolve())
    current_pp = os.environ.get("PYTHONPATH", "")
    if repo_str not in current_pp.split(os.pathsep):
        os.environ["PYTHONPATH"] = f"{repo_str}{os.pathsep}{current_pp}".rstrip(os.pathsep)

    builder = (
        SparkSession.builder
        .appName(app_name)
        .master(f"local[{cores}]")
        .config("spark.driver.memory", str(mem))
        .config("spark.sql.shuffle.partitions", "64")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", "2000")
        .config("spark.local.dir", str(spark_tmp_dir.resolve()))
        .config("spark.executorEnv.PYTHONPATH", repo_str)
        .config("spark.executorEnv.PYSPARK_PYTHON", sys.executable)
    )

    spark = builder.getOrCreate()
    return spark


if __name__ == "__main__":
    spark = get_spark(cores=2, mem="2g", app_name="spark_session_selftest")
    print(f"SparkSession created: master={spark.sparkContext.master}, app={spark.sparkContext.appName}")
    spark.stop()
