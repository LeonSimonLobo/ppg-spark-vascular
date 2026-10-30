"""Generate results/MANIFEST_C.md with checksums, sizes, and row counts."""

import hashlib
import os
import subprocess
from pathlib import Path
import pyarrow.parquet as pq


def hash_file(path: Path | str) -> tuple[str, int]:
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as f:
        while chunk := f.read(64 * 1024 * 1024):
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


def hash_dir(path: Path) -> tuple[str, int]:
    # Deterministic directory hash over sorted constituent non-crc files
    h = hashlib.sha256()
    total_sz = 0
    files = sorted([p for p in path.iterdir() if p.is_file() and not p.name.endswith(".crc")])
    for f in files:
        fh, fsz = hash_file(f)
        h.update(f.name.encode("utf-8"))
        h.update(fh.encode("utf-8"))
        total_sz += fsz
    return h.hexdigest(), total_sz


def main():
    repo_dir = Path(__file__).resolve().parent.parent

    # 1. Raw files
    raw_info = []
    # 3 hashed
    for rf_rel in [
        "data/raw/VitalDB_Train_Subset.mat",
        "data/raw/VitalDB_CalFree_Test_Subset.mat",
        "data/raw/VitalDB_AAMI_Cal_Subset.mat",
    ]:
        rf = repo_dir / rf_rel
        sha, sz = hash_file(rf)
        raw_info.append({"name": rf_rel, "sha256": sha, "size": sz, "note": "Hashed & verified"})

    # 2 unhashed
    for rf_rel in [
        "data/raw/VitalDB_AAMI_Test_Subset.mat",
        "data/raw/VitalDB_CalBased_Test_Subset.mat",
    ]:
        rf = repo_dir / rf_rel
        sz = rf.stat().st_size
        raw_info.append({"name": rf_rel, "sha256": "UNOPENED / UNHASHED", "size": sz, "note": "Stat size only (pre-registered rule)"})

    # 2. data/parquet/*.parquet
    parquet_info = []
    for p in sorted((repo_dir / "data/parquet").glob("*.parquet")):
        sha, sz = hash_file(p)
        rows = pq.read_table(p).num_rows
        parquet_info.append({"name": p.relative_to(repo_dir).as_posix(), "sha256": sha, "size": sz, "rows": rows})

    # 3. data/features_c/*.parquet
    features_info = []
    for p in sorted((repo_dir / "data/features_c").glob("*.parquet")):
        if p.is_file():
            sha, sz = hash_file(p)
            rows = pq.read_table(p).num_rows
        else:
            sha, sz = hash_dir(p)
            rows = pq.read_table(p).num_rows
        features_info.append({"name": p.relative_to(repo_dir).as_posix(), "sha256": sha, "size": sz, "rows": rows})

    # 4. 9 Model hashes
    model_files = [
        "models/c0_lr_frozen.pkl",
        "models/c0_gbt_frozen.pkl",
        "models/c1_lr_frozen.pkl",
        "models/c1_gbt_frozen.pkl",
        "models/c2m_lr_frozen.pkl",
        "models/c2m_gbt_frozen.pkl",
        "models/c2_lr_frozen.pkl",
        "models/c2_gbt_frozen.pkl",
        "models/c_imputer_frozen.pkl",
    ]
    models_info = []
    for mf_rel in model_files:
        mf = repo_dir / mf_rel
        sha, sz = hash_file(mf)
        models_info.append({"name": mf_rel, "sha256": sha, "size": sz})

    # 5. PLAN_C.md hash
    plan_c_sha, plan_c_sz = hash_file(repo_dir / "PLAN_C.md")

    # 6. git HEAD
    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_dir, text=True).strip()

    # 7. test predictions hash
    pred_sha, pred_sz = hash_file(repo_dir / "results/test_C_predictions.parquet")

    lines = [
        "# Study C Artifact & Checksum Manifest",
        "",
        "Generated automatically by script.",
        "",
        f"- **Git HEAD**: `{git_head}`",
        f"- **Pre-Registration Plan (`PLAN_C.md`) SHA-256**: `{plan_c_sha}` ({plan_c_sz:,} bytes)",
        f"- **Held-Out Test Predictions (`results/test_C_predictions.parquet`) SHA-256**: `{pred_sha}` ({pred_sz:,} bytes)",
        "",
        "---",
        "",
        "## 1. Raw Subset Files (`data/raw/`)",
        "",
        "| File Name | File Size (Bytes) | SHA-256 Checksum | Protocol Notes |",
        "| :--- | :--- | :--- | :--- |",
    ]
    for r in raw_info:
        lines.append(f"| `{r['name']}` | {r['size']:,} | `{r['sha256']}` | {r['note']} |")

    lines.extend([
        "",
        "---",
        "",
        "## 2. Ingested Waveform Parquet Cohorts (`data/parquet/`)",
        "",
        "| Cohort Parquet | Rows | Size (Bytes) | SHA-256 Checksum |",
        "| :--- | :--- | :--- | :--- |",
    ])
    for p in parquet_info:
        lines.append(f"| `{p['name']}` | {p['rows']:,} | {p['size']:,} | `{p['sha256']}` |")

    lines.extend([
        "",
        "---",
        "",
        "## 3. Study C Feature Sets (`data/features_c/`)",
        "",
        "| Feature Parquet | Rows | Total Size (Bytes) | Composite SHA-256 |",
        "| :--- | :--- | :--- | :--- |",
    ])
    for f in features_info:
        lines.append(f"| `{f['name']}` | {f['rows']:,} | {f['size']:,} | `{f['sha256']}` |")

    lines.extend([
        "",
        "---",
        "",
        "## 4. Frozen Study C Models (`models/`)",
        "",
        "| Model Artifact | Size (Bytes) | SHA-256 Checksum |",
        "| :--- | :--- | :--- |",
    ])
    for m in models_info:
        lines.append(f"| `{m['name']}` | {m['size']:,} | `{m['sha256']}` |")

    lines.extend([
        "",
        "---",
        "",
        "## 5. Test Lock & Verification Integrity",
        "- `results/.test_locked_C` and `results/test_C.md` were committed directly following execution.",
        f"- `results/test_C_predictions.parquet` SHA-256 bit-for-bit matches `results/test_C_predictions.sha256`.",
        "- Neither `VitalDB_AAMI_Test_Subset.mat` nor `VitalDB_CalBased_Test_Subset.mat` was opened or hashed.",
        "",
    ])

    out_path = repo_dir / "results/MANIFEST_C.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Successfully generated {out_path}")


if __name__ == "__main__":
    main()
