# ppg-spark-vascular

> Distributed photoplethysmogram (PPG) signal processing and vascular biomarker assessment at scale using Apache Spark.

---

## 🎯 Project Goal

<!-- PLACEHOLDER: PROJECT GOAL -->
*Briefly describe the clinical and computational goals of the project.*

- **Clinical Objective:** Analyze large-scale photoplethysmography (PPG) waveform datasets to extract physiological and morphological biomarkers of vascular health, arterial stiffness (e.g., Stiffness Index, Reflection Index, Crest Time), and vascular aging.
- **Computational Objective:** Overcome single-node memory and runtime bottlenecks by leveraging Apache Spark (PySpark) to ingest multi-gigabyte/terabyte physiological recordings (`.mat`, `.h5`), parallelize digital signal processing / feature extraction, and benchmark distributed ML workflows against traditional single-node implementations.

---

## 📂 Project Structure

```text
ppg-spark-vascular/
├── .gitignore               # Excludes data/, *.mat, *.parquet, __pycache__, etc.
├── docker-compose.yml       # Containerized Spark cluster and Jupyter environment
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation and workflow guide
├── notebooks/               # Jupyter notebooks for prototyping and visualization
├── paper/                   # Research manuscripts, LaTeX sources, and references
├── results/                 # Metrics, benchmark logs, and SHAP visualizations
└── src/                     # Core production pipeline source code
    ├── __init__.py
    ├── ingest.py            # Raw data ingestion (.mat/.h5 -> Spark Parquet)
    ├── features.py          # Distributed PPG signal filtering and feature extraction
    ├── train.py             # ML model training, evaluation, and SHAP explainability
    └── benchmark.py         # Performance, throughput, and scaling benchmarks
```

---

## ⚙️ Setup

<!-- PLACEHOLDER: SETUP -->

### Prerequisites
- Python 3.10+
- Java 8 or 11 (required for local PySpark execution)
- Docker & Docker Compose (optional, for cluster deployment)

### 1. Local Python Environment

Create and activate a virtual environment:

```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Dockerized Spark Cluster Setup (Alternative)

To spin up a local Spark cluster (Master + Worker) and Jupyter Lab via Docker:

```bash
docker-compose up -d
```

- **Spark Master UI:** `http://localhost:8080`
- **Jupyter Lab:** `http://localhost:8888`
- To stop the cluster: `docker-compose down`

---

## 🚀 How to Run Each Step

<!-- PLACEHOLDER: HOW TO RUN EACH STEP -->

### Step 1: Data Ingestion (`src/ingest.py`)
Parses raw physiological waveform files (`.mat`, `.h5`, or CSV) from the raw data directory, extracts PPG signals with metadata, partitions the dataset, and saves it into optimized Parquet format.

```bash
python src/ingest.py --input-dir data/raw --output-parquet data/processed/ppg_raw.parquet
```

### Step 2: Distributed Feature Extraction (`src/features.py`)
Applies zero-phase Butterworth bandpass filtering, derives velocity (VPG) and acceleration (APG) waveforms, and computes morphological vascular indices (Systolic Peak, Dicrotic Notch, Stiffness Index, Reflection Index).

```bash
python src/features.py --input-parquet data/processed/ppg_raw.parquet --output-parquet data/processed/ppg_features.parquet
```

### Step 3: Model Training & Interpretability (`src/train.py`)
Trains predictive models (e.g., Random Forest, Gradient Boosted Trees) on vascular features to estimate arterial stiffness and cardiovascular outcomes. Computes SHAP values for model explainability.

```bash
python src/train.py --feature-parquet data/processed/ppg_features.parquet --target label_vascular_stiffness --model-type random_forest --output-dir results/
```

### Step 4: Scaling & Latency Benchmarks (`src/benchmark.py`)
Profiles runtime, throughput (signals processed per second), cluster scaling efficiency, and memory consumption. Outputs summary metrics to `results/`.

```bash
python src/benchmark.py --output-dir results/
```

---

## 📓 Notebooks

Prototyping, exploratory data analysis (EDA), and interactive signal plotting are maintained in [`notebooks/`](notebooks/).

---

## 📊 Results & Artifacts

Model weights, evaluation plots, SHAP explainability summaries, and benchmark reports are written to [`results/`](results/).

---

## 📄 Research & Publications

Manuscript drafts, LaTeX files, and related literature references (e.g., optical cutaneous microvascular assessment) are organized in [`paper/`](paper/).
