# PPG Channel Clipping Analysis

Evaluation of potential waveform saturation, flat-topping, or sensor clipping in `VitalDB_CalFree_Test_Subset.mat` on the PPG channel (`Signals[:, 1, :]`).

---

## 1. Summary Statistics (57,600 segments, 1,250 samples/segment)

| Metric | Minimum Value ($0.0$) | Maximum Value ($1.0$) |
| :--- | :--- | :--- |
| **Median Fraction of Samples** | **0.000800** (0.080%, exactly 1 sample) | **0.000800** (0.080%, exactly 1 sample) |
| **95th Percentile Fraction** | **0.000800** (0.080%, exactly 1 sample) | **0.000800** (0.080%, exactly 1 sample) |
| **Maximum Fraction Observed** | **0.000800** (0.080%, exactly 1 sample) | **0.000800** (0.080%, exactly 1 sample) |
| **Segments with Fraction > 1.0%** | **0** (0.0%) | **0** (0.0%) |

---

## 2. Key Findings

1. **Absence of Clipping**:
   - Across all 57,600 segments, **0 segments (0.0%)** have either a 0.0 fraction or 1.0 fraction exceeding 1.0% (which would be $\ge 13$ samples).
   - In every single segment, exactly **1 sample** equals 0.0 (the discrete trough of the 10-second segment) and at most **1 sample** equals 1.0 (the discrete crest).
   - There are **zero segments** exhibiting flat-top saturation, sensor rail clipping, or bottom clipping.
2. **Signal Properties**:
   - The PPG signals represent smooth, unclipped physiological continuous waveforms whose extrema were mapped to $[0, 1]$ via segment-wise min-max scaling during PulseDB subset construction.
