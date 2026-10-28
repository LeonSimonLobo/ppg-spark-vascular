# [SUPERSEDED] Between-Subject Pre-Registration Plan (PLAN.md)

> [!NOTE]
> **Superceded Status:** This document is superseded for the within-subject question by [PLAN_W.md](PLAN_W.md) (amendments A2 to A4 and W1 retained below for the record).
> All between-subject morphology and demographic baseline evaluations are inactive; CalFree is reserved exclusively for the within-subject study in PLAN_W.md.

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

## Fiducials (Option A, frozen before any detector runs; amended under A3)
- Systolic peaks and turning points: Elgendi et al. (2013) two-moving-average method, applied to Signals channel 1 as stored, with no additional filtering.
- Frozen Elgendi parameters: systolic peak window W1 = 111 ms; beat window W2 = 667 ms; offset beta = 0.02 (Elgendi et al., 2013 defaults). Unchanged under amendment A3.
- Foot: the PPG minimum in the 300 ms before each systolic peak.
- ABP reference (validation only, never a feature): ABP peaks and feet from channel 2. The ABP filter is undocumented; this is stated as a limitation.
- PTT offset per segment (Amendment A3):
  - The original search window [0, 60) samples is withdrawn. The initial validation agreement result of 68.93% remains recorded in results/detector_agreement.md, marked as superseded by Amendment A3, because the [0, 60) window is one-sided and cannot represent the physiological timing where ABP leads PPG (results/ptt_sign_check.md).
  - Fixed sign convention: cross-correlation aligns ABP(t + lag) with PPG(t). Because arterial pulse waves reach the central/radial arterial catheter before the peripheral digital capillary bed (t_ABP < t_PPG), physiological transit delays correspond to negative lags (lag < 0).
  - The new window is symmetric: [-W, W), with W = 37 samples (about 300 ms at 125 Hz: 37 / 125 = 296 ms).
  - Physiological basis for 300 ms: Pulse transit time (PTT) and pulse arrival time (PAT) from central arteries to peripheral digits typically range from 100 to 300 ms across normotensive and hypertensive states (Allen, 2007, "Photoplethysmography and its application in clinical physiological measurement", Physiol. Meas. 28: R1–R39; Peter et al., 2014, "Pulse transit time: a review of physiological and technical considerations", Physiol. Meas. 35: R1–R29). A window of [-37, 37) samples fully spans -296 ms to +288 ms, capturing physiological pulse arrival delays without aliasing into adjacent cardiac cycles at physiological heart rates.
  - Disclosure: Inspection of validation agreement results and diagnostic boundary hits (results/detector_diagnostics.md, results/ptt_sign_check.md) directly informed the discovery of this window and sign error.
- Agreement: a PPG peak is matched if it falls within +/-50 ms of a PTT-corrected ABP peak (abp_peak - ptt_offset). A segment passes if at least 80% of its beats are matched. The 80% threshold and all Elgendi parameters remain unchanged. Agreement is measured on validation subjects only and reported as a result.
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

## Test-set policy (Amended under A4: Steps 0a/0 test run cancelled)
The Steps 0a and 0 test run on CalFree is cancelled and recorded as cancelled. 
- Reason: Morphology ladder steps are blocked by the detector failure rule (validation agreement of 44.09% under Amendment A3 remained below the 80% threshold). Furthermore, the validation baseline sanity check (results/baseline_sanity.md) revealed that the subject-level demographic signal (age vs. SBP) is absent in validation (Spearman r_s = +0.0145, p = 0.82; subject AUROC = 0.4327) due to intraoperative hemodynamic interventions decoupling operative blood pressure from chronic demographic features. Scoring Steps 0a and 0 alone would exhaust the held-out test set without testing any morphology hypothesis.
- CalFree Reservation: VitalDB_CalFree_Test_Subset.mat is kept pristine (zero test segments or subjects loaded or scored) and is reserved for the within-subject claim (PLAN_W.md).

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
