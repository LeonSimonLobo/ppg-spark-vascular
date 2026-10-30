"""Generate results/INDEX.md cataloging every file in results/ with status."""

from pathlib import Path

DESCRIPTIONS = {
    ".gitkeep": ("Git tracking anchor file for results directory", "descriptive"),
    ".test_locked_C": ("Pre-flight lock sentinel created upon execution of Study C single-pass test evaluation", "locked"),
    ".test_locked_W": ("Legacy lock sentinel for within-subject Study W test evaluation", "superseded"),
    "aami_ingest_report.md": ("Descriptive audit of ingested AAMI_Cal dataset, demographics, and NaN rates", "descriptive"),
    "audit_C.md": ("Formal pre-registration protocol audit for Study C cross-subject pipeline", "final"),
    "audit_C2.md": ("Second protocol audit verifying Amendments C1 and C2 implementations and unit tests", "final"),
    "bp_hist.png": ("Histogram visualization of blood pressure distributions across raw cohorts", "descriptive"),
    "dataset_report.md": ("Initial ingestion summary report of raw waveform files and record counts", "descriptive"),
    "demographic_consistency.md": ("Audit report validating subject age, sex, and BMI distributions across cohorts", "descriptive"),
    "dev_C.md": ("Formal development report recording 5-fold CV metrics, CalFree validation, and model hashes", "locked"),
    "dev_C_accuracy_ladder.png": ("Visual ladder chart illustrating classification accuracy across C0, C1, C2m, and C2", "final"),
    "dev_C_cv_folds.csv": ("Per-fold validation metrics and subject counts across 5-fold cross-validation", "locked"),
    "dev_C_diffs.csv": ("Paired difference hypothesis testing metrics and bootstrap confidence intervals for dev cohort", "locked"),
    "dev_C_metrics.csv": ("Consolidated metrics table covering Train OOF and CalFree validation across all 8 models", "locked"),
    "dev_C_paired_diffs.png": ("Forest plot visual of paired bootstrap accuracy and AUROC differences across model families", "final"),
    "dev_W.md": ("Legacy development report for within-subject Study W pipeline", "superseded"),
    "dev_W_controls.md": ("Robustness controls report evaluating within-subject label permutation", "superseded"),
    "dev_W_nested.md": ("Nested cross-validation metrics for legacy within-subject pipeline", "superseded"),
    "dev_W_prospective_and_scale.md": ("Prospective evaluation report for within-subject model scaling", "superseded"),
    "dryrun_C.md": ("Dry-run verification report validating in-process extraction and scoring parity on CalFree", "final"),
    "environment.md": ("Initial baseline system and dependency audit report", "superseded"),
    "environment_final.md": ("Final system environment, hardware, PySpark, and dependency lock specification", "final"),
    "exploratory_dev_models.md": ("Initial exploratory modeling benchmarking tree-based classifiers on raw features", "descriptive"),
    "features_C_sanity.md": ("Sanity check report on physiological plausibility and distributions of Study C features", "final"),
    "features_W_dev.md": ("Feature engineering report for within-subject Study W pipeline", "superseded"),
    "gate_within_subject.md": ("Diagnostic gate check for within-subject sample correlation", "superseded"),
    "gate_within_subject_agreement.md": ("Inter-rater agreement evaluation for within-subject segmentation", "superseded"),
    "inventory.md": ("Repository code and script inventory audit", "descriptive"),
    "locked_files_manifest.csv": ("Cryptographic SHA-256 and byte-size manifest of all locked files in Study C", "locked"),
    "MANIFEST_C.md": ("Formal release manifest containing SHA-256 digests, sizes, and row counts of all project assets", "final"),
    "mat_inspection.md": ("Structural inspection report of MATLAB v7.3 HDF5 datasets and internal hierarchies", "descriptive"),
    "morphology_check_C.md": ("Validation audit confirming unconstrained rise time distributions and foot identification", "final"),
    "morphology_check_C.png": ("Visual waveform inspection demonstrating detected pulse feet and systolic rise times", "final"),
    "ppg_clipping.md": ("Clipping audit verifying zero amplitude saturation across ingested PPG segments", "descriptive"),
    "ppg_range.md": ("Dynamic range audit of normalized photoplethysmogram waveforms", "descriptive"),
    "rehearsal_C.md": ("Rehearsal audit report validating full test gates and Stage 2 reporting in scratch clone", "final"),
    "sample_waveforms.png": ("Exemplar 10-second multi-channel physiological waveforms: ECG, PPG, ABP", "descriptive"),
    "source_verification.md": ("Source checksum and metadata verification of downloaded PulseDB subsets", "descriptive"),
    "source_verification_part2.md": ("Secondary integrity verification confirming file sizes and archive CRCs", "descriptive"),
    "spark_features_C.md": ("Report detailing PySpark distributed feature extraction throughput and NaN frequencies", "final"),
    "spark_parity_C.md": ("Parity report confirming numerical identity between PySpark and NumPy implementations", "final"),
    "test_C.md": ("Official single-pass held-out test evaluation report on VitalDB AAMI_Cal cohort", "locked"),
    "test_C_diffs.csv": ("Primary endpoint and stepwise paired difference bootstrap statistics on held-out test cohort", "locked"),
    "test_c_inputs.md": ("Audit of test input schema, HDF5 datasets, and channel mapping corrections", "final"),
    "test_C_predictions.parquet": ("Complete per-segment model probabilities for all 70,212 held-out test segments", "locked"),
    "test_C_predictions.sha256": ("SHA-256 cryptographic checksum of test_C_predictions.parquet", "locked"),
    "test_W.md": ("Test report for legacy within-subject Study W evaluation", "superseded"),
    "verify_test_C.md": ("Independent mathematical verification of test results reading only predictions parquet", "final"),
}


def main():
    repo_dir = Path(__file__).resolve().parent.parent
    results_dir = repo_dir / "results"
    out_file = results_dir / "INDEX.md"

    files = sorted([p for p in results_dir.iterdir() if p.name != "superseded" and p.name != "INDEX.md"])

    lines = [
        "# Results Directory Catalog (`results/INDEX.md`)",
        "",
        "This index documents every file in `results/`, classifying its role, contents, and governance status.",
        "",
        "## Governance Status Legend",
        "- **`locked`**: Frozen artifact representing pre-registered hypotheses, models, or single-pass test results. Must never be altered.",
        "- **`final`**: Authoritative, script-verified release documentation or verification report for Study C.",
        "- **`descriptive`**: Exploratory, diagnostic, or exploratory dataset summaries.",
        "- **`superseded`**: Historical reports preserved from earlier development phases or superseded protocols.",
        "",
        "---",
        "",
        "| File Name | Governance Status | Description |",
        "| :--- | :--- | :--- |",
    ]

    for p in files:
        name = p.name
        desc, status = DESCRIPTIONS.get(name, ("Artifact file in results directory", "descriptive"))
        lines.append(f"| [`{name}`](file:///{p.as_posix()}) | **`{status}`** | {desc} |")

    lines.append("")
    out_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_file} with {len(files)} cataloged files.")


if __name__ == "__main__":
    main()
