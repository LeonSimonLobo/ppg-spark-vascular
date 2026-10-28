# Detector Validation on Validation Subjects (Frozen Elgendi)

**Date:** 2026-10-29  
**Split:** VitalDB_Train_Subset.mat validation cohort (seed 42, 20% of training subjects)  
**Subjects:** 259 subjects | **Segments:** 93,240 segments (10 s at 125 Hz)  
**Detector Configuration:**
- Method: Elgendi et al. (2013) two-moving-average method on Signals channel 1 as stored (no extra filter)
- Parameters (frozen): W1 = 111 ms (14 samples), W2 = 667 ms (83 samples), beta = 0.02
- Foot detection: minimum in 300 ms window before each systolic peak
- Turning points: minimum between consecutive systolic peaks
- ABP Reference: channel 2 (min-max scaled for thresholding, same Elgendi parameters)
- PTT Lag: optimal cross-correlation lag in [0, 60) samples (~0 to 480 ms)
- Beat Agreement Rule: PPG peak matched if within +/-50 ms (+/-6.25 samples) of PTT-corrected ABP peak; segment passes if matched beat fraction >= 80%

---

## 1. Headline Agreement Results

| Metric | Result |
|---|---|
| **Validation Segments Passing (>= 80% Match)** | **68.93%** (64,271/93,240) |
| **Matched-Beat Fraction Median** | **91.7%** |
| **Matched-Beat Fraction IQR** | **36.4%** (Q1: 63.6%, Q3: 100.0%) |
| **Mean Matched-Beat Fraction** | 73.32% (SD: 35.19%) |

---

## 2. Agreement by Class (SBP Labels)

| SBP Threshold / Subgroup | Total Segments | Passing Segments | Passing Share |
|---|---|---|---|
| **Primary: SBP >= 130 mmHg** (Hypertensive range) | 17,783 | 11,890 | **66.86%** |
| **Primary: SBP < 130 mmHg** (Normotensive range) | 75,457 | 52,381 | **69.42%** |
| Sensitivity: SBP >= 120 mmHg | 33,045 | 22,629 | 68.48% |
| Sensitivity: SBP < 120 mmHg | 60,195 | 41,642 | 69.18% |
| Sensitivity: SBP >= 140 mmHg | 8,409 | 5,445 | 64.75% |
| Sensitivity: SBP < 140 mmHg | 84,831 | 58,826 | 69.34% |

---

## 3. Signal & Physiological Characteristics

| Feature / Metric | Median | IQR (Q1 - Q3) | Range [Min, Max] |
|---|---|---|---|
| PPG Beats per Segment | 12.0 | 3.0 (11.0 - 14.0) | [7, 26] |
| ABP Reference Beats per Segment | 12.0 | 3.0 (11.0 - 14.0) | [7, 26] |
| PTT Offset (samples) | 44.0 | 29.0 (30.0 - 59.0) | [0, 59] |
| PTT Offset (ms) | 352.0 ms | 232.0 ms | [0.0, 472.0] ms |
| Peak PPG-ABP Correlation | 0.946 | 0.087 (0.878 - 0.965) | [-0.399, 0.995] |

---

## 4. Detector Failure Rule Evaluation

- **Criterion:** Pass share must be >= 80.0% of validation segments.
- **Measured Pass Share:** **68.93%**
- **Status:** **Step 1 and later blocked**: Validation agreement is below the 80% threshold. No morphology gain is claimed, and Steps 0a and 0 remain the reported models.
