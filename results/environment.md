# Environment Report

Generated for project **ppg-spark-vascular**.

## 1. Operating System
- **OS**: Microsoft Windows 11 Home Single Language
- **Version**: 10.0.26300 (Build 26300)
- **Architecture**: 64-bit (x86_64)

## 2. Python Version
- **Default Python**: Python 3.13.1 (64-bit)
  - Executable: `C:\Users\mhask\AppData\Local\Programs\Python\Python313\python.exe`
- **Secondary Installed Python**: Python 3.10.7 (64-bit)
  - Executable: `C:\Users\mhask\AppData\Local\Programs\Python\Python310\python.exe`

## 3. Java Version (`java -version`)
- **Version output**:
  ```text
  java version "17.0.12" 2024-07-16 LTS
  Java(TM) SE Runtime Environment (build 17.0.12+8-LTS-286)
  Java HotSpot(TM) 64-Bit Server VM (build 17.0.12+8-LTS-286, mixed mode, sharing)
  ```
- **JAVA_HOME**: `C:\Java\jdk-17`

## 4. CPU Core Count
- **Processor**: 12th Gen Intel(R) Core(TM) i5-12450H @ 2.00 GHz
- **Physical Cores**: 8 (4 Performance cores + 4 Efficient cores)
- **Logical Processors**: 12 threads

## 5. Total RAM
- **Total Physical Memory**: 16,802,910,208 bytes (15.65 GiB / ~16.80 GB)

## 6. Free Disk Space
- **Repo Drive (`D:`)**:
  - Total: 311.97 GB
  - Used: 124.95 GB
  - Free: 187.01 GB
- **System Drive (`C:`)**:
  - Total: 162.99 GB
  - Used: 154.93 GB
  - Free: 8.06 GB

## 7. Docker Status
- **Installed**: No (`docker` command is not recognized in system PATH).
- **Running**: No (Docker daemon/service is not running).

## 8. PySpark Status & Parquet Reading
- **Installed**: `pyspark` is **not installed** in the active Python environment (`ModuleNotFoundError: No module named 'pyspark'`).
- **Local Session**: Cannot start a PySpark session until the package is installed.
- **Reading Parquet**: Cannot test Spark Parquet reading without `pyspark`. (Direct Parquet reading via `pyarrow` 22.0.0 and `pandas` 2.2.3 is available and functional).

## 9. Windows Spark Local Mode Requirements (`winutils` / `HADOOP_HOME`)
- **HADOOP_HOME**: Configured as `C:\hadoop`.
- **`winutils.exe`**: Present at `C:\hadoop\bin\winutils.exe` (112,640 bytes).
- **`hadoop.dll`**: Present at `C:\hadoop\bin\hadoop.dll` (87,040 bytes).
- **PATH**: `C:\hadoop\bin` is present in system `PATH`.
- **Missing Elements**:
  - Native Windows Hadoop binaries (`winutils.exe` and `hadoop.dll`) and Java JDK 17 are already installed and configured.
  - The missing dependency is purely the `pyspark` Python package.
  - Note: PySpark releases generally support Python versions up to 3.12 (Python 3.10 is installed on this machine and can serve as the Spark Python interpreter if Python 3.13 encounters wheel/bytecode incompatibilities).
