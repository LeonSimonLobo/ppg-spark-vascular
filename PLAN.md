## Research question
On subject-disjoint evaluation, we measure how much quality-aware PPG morphology adds to an age, sex and BMI baseline for classifying hypertensive-range SBP, and how that gain depends on signal filtering, feature set and training size.

## Target and label
Per-segment label: SBP >= 130 (primary). Sensitivity checks at 120 and 140.
Proxy for vascular state, not microvascular function. Label source: ICU and surgical arterial-line BP in VitalDB, not a hypertension diagnosis.
Sex coding: 1 = male, 0 = female (ingest.py).

## Data and verified provenance
- Files: VitalDB_Train_Subset.mat and VitalDB_CalFree_Test_Subset.mat, byte-identical to the Kaggle supplementary subsets (CRC-32 match on all members; the 20,849-byte gap between Kaggle's listed total and the member sum is unexplained and does not affect identity).
- Source: PulseDB (Wang et al., 2023, Front Digit Health, doi:10.3389/fdgth.2022.1090854).
- Sampling rate 125 Hz, quoted from Section 2.2: "downsampled from 500 Hz to 125 Hz".
- Segments: 10-s non-overlapping, quoted from Section 2.5.
- PPG_F: min-max normalized per the README. For 12.35% of CalFree segments the peak is below 1.0, and the cause is unresolved. Features use dimensionless ratios and time intervals only.
- PPG filter: 4th-order Chebyshev-II, [0.5, 8] Hz, applied in the source (Section 2.4.2). No second filter is applied in this study.
- Full run: 465,480 training segments (1,293 subjects); 57,600 test segments (144 subjects). Source: results/dataset_report.md. The earlier 3,000-segment run (commit 5753906) was a test run, not a result.
- PPG channel: index 1, chosen by correlation with ABP (ch1 0.95 train, 0.94 test). Correlation method: as in the ingest commit.

## Splits
- Train/validation: VitalDB_Train_Subset.mat. Validation is 20% of subjects, grouped by subject_id, seed 42.
- Test: VitalDB_CalFree_Test_Subset.mat. Subject overlap with training: 0.
- Test set used exactly once for the final comparison.
- All ladder steps, hyperparameters and quality thresholds are chosen on validation only.

## Fiducials (Option A, frozen before any detector runs)
- Systolic peaks and turning points: Elgendi et al. (2013) two-moving-average method, applied to Signals channel 1 as stored, with no additional filtering.
- Frozen Elgendi parameters: systolic peak window W1 = 111 ms; beat window W2 = 667 ms; offset beta = 0.02 (Elgendi et al., 2013 defaults).
- Foot: the PPG minimum in the 300 ms before each systolic peak.
- ABP reference (validation only, never a feature): ABP peaks and feet from channel 2. The ABP filter is undocumented; this is stated as a limitation.
- PTT offset per segment: the lag from 0 to 60 samples that maximizes PPG-ABP cross-correlation, as in ingest.py.
- Agreement: a PPG peak is matched if it falls within +/-50 ms of a PTT-corrected ABP peak. A segment passes if at least 80% of its beats are matched. Agreement is measured on validation subjects only and reported as a result.
- Published PulseDB fiducials are not used. Segment_Files (about 143 GB) was not downloaded. A later cross-check against it is optional.

## Demographic rule
- Age, sex and BMI are per-subject medians of the segment values.
- BMI is the median of segment BMI.
- p003938_1 is kept. Sensitivity analysis excludes it.
- Subjects with varying age (11) or BMI (17) are listed in results/demographic_consistency.md.

## Ladder (one change per step; every step scored on the same test subjects)
- Step 0a: age only, logistic regression (standardized inputs)
- Step 0: age, sex, BMI, logistic regression (standardized inputs)
- Step 1: classic morphology features, random forest (default), no filtering
- Step 2: step 1 plus signal-quality filter (threshold chosen on validation; report rejected share per class)
- Step 3: step 2 plus extended features (SDPPG b/a and e/a, notch depth, reflection index), random forest (default)
- Step 4a: step 3 with GBT tuned on validation
- Step 4b: step 4a trained on 25, 50, 75 and 100% of training subjects (learning curve); subjects (not segments) are subsampled at 25, 50, 75 and 100% with seed 42; the same subjects are used for all morphology steps at each fraction.

## Detector failure rule
If fewer than 80% of validation segments pass the agreement check, Step 1 and all later morphology steps are reported as blocked, no morphology gain is claimed, and Steps 0a and 0 are still reported. No alternative detector may be tried after seeing agreement results.

## Test-set policy
The test set is scored once, in a single run, for Steps 0a and 0 only, because morphology steps are blocked by the failure rule. The test set is spent after that run and cannot be reused for any later detector or model.

## Metrics
Headline: AUROC and balanced accuracy, with 95% subject-level bootstrap CIs (2000 resamples over test subjects).
Secondary: F1 at a threshold chosen on validation.
Plain accuracy only with prevalence stated. Positive prevalence at 130: 12.5% of subjects, 21.1% of segments.

## Known limitations
- Test has 18 positive subjects at threshold 130; CIs will be wide.
- Between-subject share of SBP variance is about 40%, so age, sex and BMI cannot explain most within-subject variation.
- Label is a proxy from arterial-line BP, not a hypertension diagnosis.
- Raw .mat files contain no fiducial arrays (results/mat_inspection.md). Fiducials are derived from PPG by the method above; this has NOT yet been validated.
- The 12.35% of CalFree segments with peak below 1.0 is unexplained (results/source_verification_part2.md).
- The ABP reference filter is undocumented.
- Segment_Files (about 143 GB) was not used.

## Claim status
No result is claimed until it is produced by a run recorded in results/. No target gain is stated. A null or negative result is reported as such.
