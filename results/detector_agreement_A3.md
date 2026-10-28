# Detector Validation on Validation Subjects (Amendment A3)

**Date:** 2026-10-29  
**Amendment:** A3 (Symmetric PTT window [-W, W) with W = 37 samples / ~300 ms at 125 Hz)  
**Split:** VitalDB_Train_Subset.mat validation cohort (seed 42, 20% of training subjects)  
**Subjects:** 259 subjects | **Segments:** 93,240 segments (10 s at 125 Hz)  
**Detector Configuration:**
- Method: Elgendi et al. (2013) two-moving-average method on Signals channel 1 as stored (no extra filter)
- Parameters (frozen): W1 = 111 ms (14 samples), W2 = 667 ms (83 samples), beta = 0.02 (unchanged)
- Foot detection: minimum in 300 ms window before each systolic peak
- Turning points: minimum between consecutive systolic peaks
- ABP Reference: channel 2 (min-max scaled for thresholding, same Elgendi parameters)
- PTT Search Window (Amendment A3): [-37, 37) samples (~-296 ms to +288 ms)
- Beat Agreement Rule: PPG peak matched if within +/-50 ms (+/-6.25 samples) of PTT-corrected ABP peak; segment passes if matched beat fraction >= 80% (unchanged)

---

## 1. Headline Agreement Results

| Metric | Result |
|---|---|
| **Validation Segments Passing (>= 80% Match)** | **44.09%** (41,110/93,240) |
| **Matched-Beat Fraction Median** | **41.7%** |
| **Matched-Beat Fraction IQR** | **93.3%** (Q1: 0.0%, Q3: 93.3%) |
| **Matched-Beat Percentiles** | P10: 0.0%, P90: 100.0%, Min: 0.0%, Max: 100.0% |
| **Mean Matched-Beat Fraction** | 47.50% (SD: 45.32%) |

---

## 2. Agreement by Class (SBP Labels)

| SBP Threshold / Subgroup | Total Segments | Passing Segments | Passing Share |
|---|---|---|---|
| **Primary: SBP >= 130 mmHg** (Hypertensive range) | 17,783 | 7,941 | **44.66%** |
| **Primary: SBP < 130 mmHg** (Normotensive range) | 75,457 | 33,169 | **43.96%** |
| Sensitivity: SBP >= 120 mmHg | 33,045 | 15,260 | 46.18% |
| Sensitivity: SBP < 120 mmHg | 60,195 | 25,850 | 42.94% |
| Sensitivity: SBP >= 140 mmHg | 8,409 | 3,771 | 44.84% |
| Sensitivity: SBP < 140 mmHg | 84,831 | 37,339 | 44.02% |

---

## 3. PTT Lag Distribution and Boundary Hits

### Boundary Hits
- **Lower Boundary (lag = -37 / -296 ms):** 29,650 segments (31.80%)
- **Upper Boundary (lag = 36 / 288 ms):** 32,472 segments (34.83%)
- **Total Boundary Hits:** 62,122 segments (66.63%)

### Lag Histogram
| Lag Interval | Segments | Share (%) |
|---|---|---|
| [-37, -30) samples (-296 to -240 ms) | 29,650 | 31.80% |
| [-30, -20) samples (-240 to -160 ms) | 18 | 0.02% |
| [-20, -10) samples (-160 to -80 ms) | 40 | 0.04% |
| [-10, 0) samples (-80 to 0 ms) | 865 | 0.93% |
| [0, 10) samples (0 to 80 ms) | 2,521 | 2.70% |
| [10, 20) samples (80 to 160 ms) | 6,197 | 6.65% |
| [20, 30) samples (160 to 240 ms) | 11,709 | 12.56% |
| [30, 37) samples (240 to 296 ms) | 42,240 | 45.30% |

---

## 4. Signal & Physiological Characteristics

| Feature / Metric | Median | IQR (Q1 - Q3) | Range [Min, Max] |
|---|---|---|---|
| PPG Beats per Segment | 12.0 | 3.0 (11.0 - 14.0) | [7, 26] |
| ABP Reference Beats per Segment | 12.0 | 3.0 (11.0 - 14.0) | [7, 26] |
| PTT Offset (samples) | 26.0 | 73.0 (-37.0 - 36.0) | [-37, 36] |
| PTT Offset (ms) | 208.0 ms | 584.0 ms | [-296.0, 288.0] ms |
| Peak PPG-ABP Correlation | 0.687 | 0.662 (0.283 - 0.945) | [-0.172, 0.993] |

---

## 5. Detector Failure Rule Evaluation

- **Criterion:** Pass share must be >= 80.0% of validation segments.
- **Measured Pass Share:** **44.09%**
- **Status:** **Step 1 and later blocked**: Validation agreement is below the 80% threshold. No morphology gain is claimed, and Steps 0a and 0 remain the reported models.
