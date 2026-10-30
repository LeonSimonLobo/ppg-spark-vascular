# Results Directory Catalog (`results/INDEX.md`)

This index documents every file in `results/`, classifying its role, contents, and governance status.

## Governance Status Legend
- **`locked`**: Frozen artifact representing pre-registered hypotheses, models, or single-pass test results. Must never be altered.
- **`final`**: Authoritative, script-verified release documentation or verification report for Study C.
- **`descriptive`**: Exploratory, diagnostic, or exploratory dataset summaries.
- **`superseded`**: Historical reports preserved from earlier development phases or superseded protocols.

---

| File Name | Governance Status | Description |
| :--- | :--- | :--- |
| [`.gitkeep`](file:///D:/DS_RESEARCH/ppg-spark-vascular/results/.gitkeep) | **`descriptive`** | Git tracking anchor file for results directory |
| [`.test_locked_C`](file:///D:/DS_RESEARCH/ppg-spark-vascular/results/.test_locked_C) | **`locked`** | Pre-flight lock sentinel created upon execution of Study C single-pass test evaluation |
| [`.test_locked_W`](file:///D:/DS_RESEARCH/ppg-spark-vascular/results/.test_locked_W) | **`superseded`** | Legacy lock sentinel for within-subject Study W test evaluation |
| [`aami_ingest_report.md`](file:///D:/DS_RESEARCH/ppg-spark-vascular/results/aami_ingest_report.md) | **`descriptive`** | Descriptive audit of ingested AAMI_Cal dataset, demographics, and NaN rates |
| [`audit_C.md`](file:///D:/DS_RESEARCH/ppg-spark-vascular/results/audit_C.md) | **`final`** | Formal pre-registration protocol audit for Study C cross-subject pipeline |
| [`audit_C2.md`](file:///D:/DS_RESEARCH/ppg-spark-vascular/results/audit_C2.md) | **`final`** | Second protocol audit verifying Amendments C1 and C2 implementations and unit tests |
| [`bp_hist.png`](file:///D:/DS_RESEARCH/ppg-spark-vascular/results/bp_hist.png) | **`descriptive`** | Histogram visualization of blood pressure distributions across raw cohorts |
| [`dataset_report.md`](file:///D:/DS_RESEARCH/ppg-spark-vascular/results/dataset_report.md) | **`descriptive`** | Initial ingestion summary report of raw waveform files and record counts |
| [`demographic_consistency.md`](file:///D:/DS_RESEARCH/ppg-spark-vascular/results/demographic_consistency.md) | **`descriptive`** | Audit report validating subject age, sex, and BMI distributions across cohorts |
| [`dev_C.md`](file:///D:/DS_RESEARCH/ppg-spark-vascular/results/dev_C.md) | **`locked`** | Formal development report recording 5-fold CV metrics, CalFree validation, and model hashes |
| [`dev_C_accuracy_ladder.png`](file:///D:/DS_RESEARCH/ppg-spark-vascular/results/dev_C_accuracy_ladder.png) | **`final`** | Visual ladder chart illustrating classification accuracy across C0, C1, C2m, and C2 |
| [`dev_C_cv_folds.csv`](file:///D:/DS_RESEARCH/ppg-spark-vascular/results/dev_C_cv_folds.csv) | **`locked`** | Per-fold validation metrics and subject counts across 5-fold cross-validation |
| [`dev_C_diffs.csv`](file:///D:/DS_RESEARCH/ppg-spark-vascular/results/dev_C_diffs.csv) | **`locked`** | Paired difference hypothesis testing metrics and bootstrap confidence intervals for dev cohort |
| [`dev_C_metrics.csv`](file:///D:/DS_RESEARCH/ppg-spark-vascular/results/dev_C_metrics.csv) | **`locked`** | Consolidated metrics table covering Train OOF and CalFree validation across all 8 models |
| [`dev_C_paired_diffs.png`](file:///D:/DS_RESEARCH/ppg-spark-vascular/results/dev_C_paired_diffs.png) | **`final`** | Forest plot visual of paired bootstrap accuracy and AUROC differences across model families |
| [`dev_W.md`](file:///D:/DS_RESEARCH/ppg-spark-vascular/results/dev_W.md) | **`superseded`** | Legacy development report for within-subject Study W pipeline |
| [`dev_W_controls.md`](file:///D:/DS_RESEARCH/ppg-spark-vascular/results/dev_W_controls.md) | **`superseded`** | Robustness controls report evaluating within-subject label permutation |
| [`dev_W_nested.md`](file:///D:/DS_RESEARCH/ppg-spark-vascular/results/dev_W_nested.md) | **`superseded`** | Nested cross-validation metrics for legacy within-subject pipeline |
| [`dev_W_prospective_and_scale.md`](file:///D:/DS_RESEARCH/ppg-spark-vascular/results/dev_W_prospective_and_scale.md) | **`superseded`** | Prospective evaluation report for within-subject model scaling |
| [`dryrun_C.md`](file:///D:/DS_RESEARCH/ppg-spark-vascular/results/dryrun_C.md) | **`final`** | Dry-run verification report validating in-process extraction and scoring parity on CalFree |
| [`environment.md`](file:///D:/DS_RESEARCH/ppg-spark-vascular/results/environment.md) | **`superseded`** | Initial baseline system and dependency audit report |
| [`environment_final.md`](file:///D:/DS_RESEARCH/ppg-spark-vascular/results/environment_final.md) | **`final`** | Final system environment, hardware, PySpark, and dependency lock specification |
| [`exploratory_dev_models.md`](file:///D:/DS_RESEARCH/ppg-spark-vascular/results/exploratory_dev_models.md) | **`descriptive`** | Initial exploratory modeling benchmarking tree-based classifiers on raw features |
| [`features_C_sanity.md`](file:///D:/DS_RESEARCH/ppg-spark-vascular/results/features_C_sanity.md) | **`final`** | Sanity check report on physiological plausibility and distributions of Study C features |
| [`features_W_dev.md`](file:///D:/DS_RESEARCH/ppg-spark-vascular/results/features_W_dev.md) | **`superseded`** | Feature engineering report for within-subject Study W pipeline |
| [`gate_within_subject.md`](file:///D:/DS_RESEARCH/ppg-spark-vascular/results/gate_within_subject.md) | **`superseded`** | Diagnostic gate check for within-subject sample correlation |
| [`gate_within_subject_agreement.md`](file:///D:/DS_RESEARCH/ppg-spark-vascular/results/gate_within_subject_agreement.md) | **`superseded`** | Inter-rater agreement evaluation for within-subject segmentation |
| [`inventory.md`](file:///D:/DS_RESEARCH/ppg-spark-vascular/results/inventory.md) | **`descriptive`** | Repository code and script inventory audit |
| [`locked_files_manifest.csv`](file:///D:/DS_RESEARCH/ppg-spark-vascular/results/locked_files_manifest.csv) | **`locked`** | Cryptographic SHA-256 and byte-size manifest of all locked files in Study C |
| [`MANIFEST_C.md`](file:///D:/DS_RESEARCH/ppg-spark-vascular/results/MANIFEST_C.md) | **`final`** | Formal release manifest containing SHA-256 digests, sizes, and row counts of all project assets |
| [`mat_inspection.md`](file:///D:/DS_RESEARCH/ppg-spark-vascular/results/mat_inspection.md) | **`descriptive`** | Structural inspection report of MATLAB v7.3 HDF5 datasets and internal hierarchies |
| [`morphology_check_C.md`](file:///D:/DS_RESEARCH/ppg-spark-vascular/results/morphology_check_C.md) | **`final`** | Validation audit confirming unconstrained rise time distributions and foot identification |
| [`morphology_check_C.png`](file:///D:/DS_RESEARCH/ppg-spark-vascular/results/morphology_check_C.png) | **`final`** | Visual waveform inspection demonstrating detected pulse feet and systolic rise times |
| [`ppg_clipping.md`](file:///D:/DS_RESEARCH/ppg-spark-vascular/results/ppg_clipping.md) | **`descriptive`** | Clipping audit verifying zero amplitude saturation across ingested PPG segments |
| [`ppg_range.md`](file:///D:/DS_RESEARCH/ppg-spark-vascular/results/ppg_range.md) | **`descriptive`** | Dynamic range audit of normalized photoplethysmogram waveforms |
| [`rehearsal_C.md`](file:///D:/DS_RESEARCH/ppg-spark-vascular/results/rehearsal_C.md) | **`final`** | Rehearsal audit report validating full test gates and Stage 2 reporting in scratch clone |
| [`sample_waveforms.png`](file:///D:/DS_RESEARCH/ppg-spark-vascular/results/sample_waveforms.png) | **`descriptive`** | Exemplar 10-second multi-channel physiological waveforms: ECG, PPG, ABP |
| [`source_verification.md`](file:///D:/DS_RESEARCH/ppg-spark-vascular/results/source_verification.md) | **`descriptive`** | Source checksum and metadata verification of downloaded PulseDB subsets |
| [`source_verification_part2.md`](file:///D:/DS_RESEARCH/ppg-spark-vascular/results/source_verification_part2.md) | **`descriptive`** | Secondary integrity verification confirming file sizes and archive CRCs |
| [`spark_features_C.md`](file:///D:/DS_RESEARCH/ppg-spark-vascular/results/spark_features_C.md) | **`final`** | Report detailing PySpark distributed feature extraction throughput and NaN frequencies |
| [`spark_parity_C.md`](file:///D:/DS_RESEARCH/ppg-spark-vascular/results/spark_parity_C.md) | **`final`** | Parity report confirming numerical identity between PySpark and NumPy implementations |
| [`test_C.md`](file:///D:/DS_RESEARCH/ppg-spark-vascular/results/test_C.md) | **`locked`** | Official single-pass held-out test evaluation report on VitalDB AAMI_Cal cohort |
| [`test_C_diffs.csv`](file:///D:/DS_RESEARCH/ppg-spark-vascular/results/test_C_diffs.csv) | **`locked`** | Primary endpoint and stepwise paired difference bootstrap statistics on held-out test cohort |
| [`test_c_inputs.md`](file:///D:/DS_RESEARCH/ppg-spark-vascular/results/test_c_inputs.md) | **`final`** | Audit of test input schema, HDF5 datasets, and channel mapping corrections |
| [`test_C_predictions.parquet`](file:///D:/DS_RESEARCH/ppg-spark-vascular/results/test_C_predictions.parquet) | **`locked`** | Complete per-segment model probabilities for all 70,212 held-out test segments |
| [`test_C_predictions.sha256`](file:///D:/DS_RESEARCH/ppg-spark-vascular/results/test_C_predictions.sha256) | **`locked`** | SHA-256 cryptographic checksum of test_C_predictions.parquet |
| [`test_W.md`](file:///D:/DS_RESEARCH/ppg-spark-vascular/results/test_W.md) | **`superseded`** | Test report for legacy within-subject Study W evaluation |
| [`verify_test_C.md`](file:///D:/DS_RESEARCH/ppg-spark-vascular/results/verify_test_C.md) | **`final`** | Independent mathematical verification of test results reading only predictions parquet |
