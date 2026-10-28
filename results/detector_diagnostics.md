# Exploratory Detector Diagnostics (Validation Cohort Only)

> **[EXPLORATORY]**  
> This diagnostic analysis is post-hoc, strictly exploratory, and performed exclusively on the validation cohort of `VitalDB_Train_Subset.mat`. As pre-registered in `PLAN.md`, this analysis is **not used to change any result**, modify the frozen parameters, or unblock Step 1. The detector failure outcome remains frozen.

---

## 1. Subject-Level Matched-Beat Fraction and Pass Share
- **Cohort:** 259 validation subjects (93,240 segments)
- **Subject-Level Median Matched-Beat Fraction:**
  - **Median:** 0.9167 (91.67%)
  - **Min:** 0.0000 (0.0%)
  - **Max:** 1.0000 (100.0%)
  - **Percentiles:** 10th: 0.000 | 25th: 0.777 | 50th: 0.917 | 75th: 0.931 | 90th: 1.000
- **Subject-Level Pass Share (fraction of segments with matched fraction $\ge 80\%$):**
  - **Median:** 0.7972 (79.72%)
  - **Min:** 0.0000 (0.0%)
  - **Max:** 1.0000 (100.0%)
  - **Subjects with pass share < 50%:** **66 / 259 subjects (25.48%)**
  - **Subjects with pass share < 80%:** 130 / 259 subjects (50.19%)
- **Subject Pass Share Distribution:**
  - `[0.0, 0.2)`: 32 subjects (12.4%)
  - `[0.2, 0.4)`: 19 subjects (7.3%)
  - `[0.4, 0.6)`: 29 subjects (11.2%)
  - `[0.6, 0.8)`: 50 subjects (19.3%)
  - `[0.8, 1.0]`: 129 subjects (49.8%)

---

## 2. PTT Lag Analysis and Search Boundary Hits
The pre-registered PTT offset search window is constrained to $[0, 60)$ samples (0 to 59 samples, corresponding to 0 to 472.0 ms at 125 Hz).

- **Boundary Hits Count:**
  - **Lag 0 hits (lower boundary):** 2,119 segments (2.27%)
  - **Lag 59 hits (upper boundary):** 23,806 segments (25.53%)
  - **Total boundary hits (lag 0 or 59):** **25,925 segments (27.80%)**
- **Impact of Boundary Hits on Agreement:**
  - **Pass share at boundary (lag 0 or 59):** **28.35%** (7,351 / 25,925 segments)
  - **Pass share strictly interior ($1 \le \text{lag} \le 58$):** **84.56%** (56,920 / 67,315 segments)
- **Share of Overall Failures Caused by Boundary Segments:**
  - Of all 28,969 non-passing segments, **18,574 (64.12%)** occurred on segments hitting lag 0 or lag 59.
- **Top Lag Frequencies:**
  - Lag 59: 23,806 segments (25.53%)
  - Lag 0: 2,119 segments (2.27%)
  - Lag 36: 1,828 segments (1.96%)
  - Lag 39: 1,813 segments (1.94%)
  - Lag 43: 1,810 segments (1.94%)
  - Lag 40: 1,768 segments (1.90%)
  - Lag 38: 1,761 segments (1.89%)
  - Lag 37: 1,733 segments (1.86%)
  - Lag 45: 1,723 segments (1.85%)
  - Lag 44: 1,722 segments (1.85%)

---

## 3. Beat-Count Sanity: PPG Beats vs. ABP Reference Beats
Per-segment difference is defined as `PPG beats - ABP beats`.

- **Beat Difference Summary:**
  - **Median:** 0.00 beats
  - **IQR:** 1.00 beat (Q1: -1.00, Q3: 0.00)
  - **Mean difference:** -0.031 beats (SD: 0.983)
- **Difference Concordance:**
  - **Exact match ($\Delta = 0$):** 49.60% of segments (46,245 / 93,240)
  - **Within $\pm 1$ beat:** **97.32%** of segments (90,741 / 93,240)
  - **Large discrepancy ($|\Delta| \ge 3$ beats):** 1.26% of segments (1,173 / 93,240)

---

## 4. ABP Detector Alone
- **Beats per 10-s segment:**
  - Median: 12.0 beats (corresponding to 72.0 bpm)
  - IQR: 3.0 beats (Q1: 11.0, Q3: 14.0, corresponding to 66.0–84.0 bpm)
  - Range: [7, 26] beats (corresponding to 42.0–156.0 bpm)
- **Segments outside [5, 25] beats (<30 bpm or >150 bpm):**
  - **3 out of 93,240 segments (0.003%)**
- **PPG detector comparison:**
  - Median: 12.0 beats, IQR: 3.0 beats, Range: [7, 26] beats
  - Segments outside [5, 25] beats: 3 out of 93,240 segments (0.003%)

---

## 5. Failure Pattern Assessment

The failure mode is highly systematic rather than uniformly distributed across beats or segments. First, beat-count agreement between PPG and ABP detectors is remarkably high (97.32% of segments agree within $\pm 1$ beat, with a median difference of 0.00), and ABP beat counts fall almost entirely within physiological bounds (only 0.003% outside 5–25 beats), showing that individual peak-trough identification per se is functioning reliably. Instead, failure is concentrated in the PTT alignment stage: over one-quarter of all segments (25.53%) hit the upper search boundary at lag 59 (472 ms). When the lag is interior ($1 \le \text{lag} \le 58$), the pass rate is 84.56%, whereas at the search boundary the pass rate collapses to 28.35%, directly accounting for 64.12% of all failed segments. At the subject level, failure is heavily clustered: 25.48% of subjects have pass rates below 50% (and 12.4% have pass rates under 20%), while nearly half the cohort (49.8%) achieves pass rates above 80%. Thus, failure is driven predominantly by a subset of subjects whose true pulse transit time or inter-signal phase delay exceeds the 60-sample search window, causing boundary clipping and phase misalignment.
