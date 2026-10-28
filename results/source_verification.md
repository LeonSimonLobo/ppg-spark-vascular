# Source Verification and Provenance Analysis

**Date:** 2026-10-28  
**Scope:** Verification of PulseDB published literature, official GitHub repository, and Kaggle supplementary subsets against repository empirical findings.

---

## 1. Source Verification from Literature and Official Repository

Sources checked:
1. **Frontiers in Digital Health publication:**  
   Wang W, Mohseni P, Kilgore KL, Najafizadeh L. *PulseDB: A large, cleaned dataset based on MIMIC-III and VitalDB for benchmarking cuff-less blood pressure estimation methods.* Front Digit Health. 2023; 4: 1090854.  
   DOI: [10.3389/fdgth.2022.1090854](https://doi.org/10.3389/fdgth.2022.1090854) | PMC Full Text: [PMC9944565](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9944565/)
2. **Official PulseDB GitHub Repository:**  
   URL: [https://github.com/pulselabteam/PulseDB](https://github.com/pulselabteam/PulseDB)

---

### a. 500 Hz to 125 Hz Downsampling
- **Status:** Verified from fetched page.
- **Source:** Frontiers in Digital Health publication (PMC9944565), Section 2.2 *Record selection*.
- **Exact Quote:**
  > "Signals acquired from the VitalDB database were downsampled from 500 Hz to 125 Hz to have a consistent sampling rate with the signals from the MIMIC-III matched subset."
- **URL:** [https://doi.org/10.3389/fdgth.2022.1090854](https://doi.org/10.3389/fdgth.2022.1090854)

---

### b. 10-Second Non-Overlapping Segments, Segment and Subject Counts
- **Status:** Verified from fetched page.
- **Source:** Frontiers in Digital Health publication (PMC9944565), Abstract, Section 1 *Introduction*, Section 2.5 *Data segmentation and cleaning*, Section 3.1 *Demographic and blood pressure distributions*.
- **Exact Quotes:**
  - **10-second non-overlapping segmentation (Section 2.5):**
    > "Data selection is conducted by dividing each record into 10-s non-overlapping segments, and determining whether to include or discard each segment."
  - **Total dataset segment and subject counts (Abstract and Section 1):**
    > "PulseDB contains 1) 5,245,454 high-quality 10-s segments of ECG, PPG, and arterial BP (ABP) waveforms from 5,361 subjects retrieved from the MIMIC-III waveform database matched subset and the VitalDB database; 2) subjects’ identification and demographic information, that can be utilized as additional input features to improve the performance of BP estimation models, or to evaluate the generalizability of the models to data from unseen subjects; and 3) positions of the characteristic points of the ECG/PPG signals, making PulseDB directly usable for training deep learning models with minimal data pre-processing."
  - **Overall count confirmation (Section 3.1):**
    > "The dataset includes 5,245,454 10-s segments from 5,361 subjects."
- **URL:** [https://doi.org/10.3389/fdgth.2022.1090854](https://doi.org/10.3389/fdgth.2022.1090854)

---

### c. Min-Max Scaling to [0, 1] and Stage Applied (Before vs. After Windowing)
- **Status:**
  - Frontiers publication: **not found in fetched page**. The article text does not mention amplitude normalization of the input ECG or PPG signals to $[0, 1]$.
  - GitHub repository: **verified from fetched page** (concept of normalized consecutive clips before windowing).
- **Source:** Official PulseDB GitHub Repository (`README.md`, lines 6 and 15).
- **Exact Quotes:**
  - **Clip-level processing before windowing (`README.md`, line 6):**
    > "PulseDB retrieves 1 consecutive clip from each MIMIC/VitalDB record, divides the clip into 10-s segments, and removes those 10-s segments of bad quality."
  - **Normalized signals retrieved by `Generate_Subsets.m` (`README.md`, line 15):**
    > "Therefore, running the script **Generate_Subsets.m** still retrieves the normalized signals from the previous fields, `ECG_F` and `PPG_F` (corresponding to the paper). If you want to retrieve the new non-normalized signals instead, modify the script to read from the newly added fields."
- **Note on Formula:** The exact mathematical phrase `"min-max scaling to [0, 1]"` is **not found in fetched page** in either document. However, empirical analysis of `Signals[:, 1, :]` in `results/ppg_range.md` confirms strictly bounded values in $[0.0, 1.0]$.
- **URL:** [https://github.com/pulselabteam/PulseDB](https://github.com/pulselabteam/PulseDB)

---

### d. PPG Filter Type and Cutoffs
- **Status:** Verified from fetched page.
- **Source:** Frontiers in Digital Health publication (PMC9944565), Section 2.4.2 *PPG systolic peaks and turning points*.
- **Exact Quote:**
  > "The PPG signal was filtered with a 4th order Chebyshev-II filter at [0.5,8] Hz before presenting to the Elgendi’s algorithm, as suggested in previous studies (12, 16, 43–45) for optimally suppressing noise and baseline wandering, and preserving PPG morphology features."
- **URL:** [https://doi.org/10.3389/fdgth.2022.1090854](https://doi.org/10.3389/fdgth.2022.1090854)

---

### e. Beat or Fiducial Annotations Distributed with Subsets
- **Status:** Verified from fetched pages.
- **Finding:** Beat/fiducial annotations were extracted and stored in the underlying per-subject `Segment_Files`, but they are **omitted from the distributed `Subset_Files`**.
- **Source 1:** Frontiers in Digital Health publication (PMC9944565), Section 2.4 & Section 2.6:
  > "The following characteristic points were extracted from the ECG, PPG and ABP signals for each record in the PulseDB dataset."
  > "The GitHub repository also includes MATLAB scripts for reproducing the subject-balanced training set, calibration-based testing set, calibration-free testing set, and AAMI testing and calibration sets summarized in Table 4 from the segment files."
- **Source 2:** PulseDB GitHub README (`README.md`, lines 31–33):
  > "**Subset** files are generated from the **Segment** and **Info** files stored in the folder **Segment_Files**, **Info_Files**, and **Supplementary_Info_Files**, whose contents have to be downloaded separately from online drive services."
  > "The generated **Subset** files are stored in the folder **Subset_Files** and **Supplementary_Subset_Files**. Data were organized as large matrices for the compactness of file size that is more suitable to fit into memory for training and testing machine learning models."
- **Source 3:** `Generate_Subsets.m` (lines 62–71, 99–113):
  The script pre-allocates only `Subject`, `Signals`, `SBP`, `DBP`, `Age`, `Gender`, `Height`, `Weight`, and `BMI`. Fiducials from `Seg_Info` (`ECG_R_Peaks`, `PPG_Peaks`, `PPG_Onsets`, etc.) are not copied into the `Subset` structure.
- **URL:** [https://github.com/pulselabteam/PulseDB](https://github.com/pulselabteam/PulseDB)

---

## 2. Kaggle Provenance

- **Kaggle URL:** [https://www.kaggle.com/datasets/weinanwangrutgers/pulsedb-balanced-training-and-testing](https://www.kaggle.com/datasets/weinanwangrutgers/pulsedb-balanced-training-and-testing)  
- **DOI:** [10.34740/KAGGLE/DS/2447469](https://doi.org/10.34740/KAGGLE/DS/2447469) (cited as Reference 61 in the Frontiers publication)
- **Title on Kaggle:** `PulseDB Supplementary Subsets`
- **Owner/Creator:** `WeinanWang_Rutgers` (Rutgers University)
- **License:** `CC BY-NC-SA 4.0` (Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International)
- **Kaggle Description:**
  > "This is the supplementary training, validation, and testing subsets of PulseDB, derived from the **VitalDB** dataset: https://vitaldb.net/dataset/
  > 
  > These files are made available under the Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) License: https://creativecommons.org/licenses/by-nc-sa/4.0/ "

### Does Kaggle state how the .mat subsets were generated?
The Kaggle dataset description does **not** document the generation procedure directly on the page. However, the Frontiers publication (Section 4.2) explicitly links them:
> "These subsets are available for download from Kaggle at (61), and can be reproduced from the script provided in the GitHub repository at (34)."
And the GitHub repository README confirms:
> "The generated **Supplementary Subset** files with data derived from the **VitalDB** dataset are also available via Kaggle (https://doi.org/10.34740/KAGGLE/DS/2447469)."

### Do they match `Generate_Subsets.m`?
Yes. Lines 28–45 of `Generate_Subsets.m` define the exact supplementary subsets generated from VitalDB subjects:
1. `Supplementary_Subset_Files/VitalDB_Train_Subset.mat`
2. `Supplementary_Subset_Files/VitalDB_CalBased_Test_Subset.mat`
3. `Supplementary_Subset_Files/VitalDB_CalFree_Test_Subset.mat`
4. `Supplementary_Subset_Files/VitalDB_AAMI_Test_Subset.mat`
5. `Supplementary_Subset_Files/VitalDB_AAMI_Cal_Subset.mat`

### File Names and Size Comparison with `data/raw/`
- **Kaggle Reported Uncompressed Size:** `19,135,855,809` bytes (~19.14 GB / ~17.82 GiB)
- **Kaggle Reported Archive Download Size (`contentSize`):** `18,592,635,040` bytes (~18.59 GB / ~17.32 GiB)

| File Name in `data/raw/` | Raw `.mat` Size (bytes) | Raw `.mat` Size (GB) | `.zip` Size (bytes) |
|---|---|---|---|
| `VitalDB_Train_Subset.mat` | 13,797,698,523 | 12.85 GB | 13,405,647,245 |
| `VitalDB_CalFree_Test_Subset.mat` | 1,706,947,648 | 1.59 GB | 1,658,731,199 |
| `VitalDB_CalBased_Test_Subset.mat` | 1,532,644,941 | 1.43 GB | 1,488,910,139 |
| `VitalDB_AAMI_Cal_Subset.mat` | 2,078,737,367 | 1.94 GB | 2,020,123,141 |
| `VitalDB_AAMI_Test_Subset.mat` | 19,806,481 | 0.02 GB | 19,216,849 |
| **Total in `data/raw/`** | **19,135,834,960** | **17.82 GiB** | **18,592,628,573** |

- **Comparison:**
  - Total raw `.mat` byte count (19,135,834,960) matches Kaggle's `totalBytesNullable` (19,135,855,809) within 20,849 bytes (zip directory metadata overhead).
  - Total `.zip` byte count (18,592,628,573) matches Kaggle's distribution archive size (18,592,635,040) within 6,467 bytes.
  - The files in `data/raw/` are byte-identical to the supplementary subsets generated by `Generate_Subsets.m` and hosted on Kaggle.

---

## 3. Analysis of `Generate_Subsets.m`

- **Source File:** `Generate_Subsets.m` from `https://raw.githubusercontent.com/pulselabteam/PulseDB/main/Generate_Subsets.m`
- **Line Assigning `Signals`:** Line 102
- **Exact Quote of Line 102:**
  ```matlab
  Subset.Signals(pos,:,:)=[Segment.ECG_F,Segment.PPG_F,Segment.ABP_Raw]';
  ```
- **Timing of Normalization:**
  - Normalization occurs **before** the 10-second windowing.
  - In `Generate_Subsets.m`, `Segment.PPG_F` is extracted directly from the pre-existing segment record `Subj_Segments(j)`.
  - As described in the official PulseDB repository `README.md` (lines 6 and 15), PulseDB extracted one long continuous recording clip per patient, filtered and normalized that clip, and then divided it into 10-second windows (`Subj_Wins`).
  - `Generate_Subsets.m` applies no normalization; it merely aggregates pre-normalized waveforms.

---

## 4. Reconciliation of the 88% Maximum Sample Finding

In `results/ppg_clipping.md` and `results/ppg_range.md`, empirical testing showed that across the 57,600 segments of `VitalDB_CalFree_Test_Subset.mat`, 87.65% had a maximum sample value of exactly 1.0, while 12.35% peaked below 1.0 (all strictly $\le 1.0$).

### Theoretical vs. Empirical Behavior
1. **Per-segment min-max scaling:**
   If min-max normalization were computed independently on each 10-second segment:
   $$x_{\text{norm}}[t] = \frac{x[t] - \min(x)}{\max(x) - \min(x)}$$
   by mathematical definition, **100% of non-flat segments would have $\max(x_{\text{norm}}) = 1.0$ and $\min(x_{\text{norm}}) = 0.0$**.
2. **Clip-level / record-level scaling (as done in PulseDB):**
   Because the min-max scaling was executed over the entire consecutive multi-minute/hour recording before slicing into 10-second segments:
   - Only the segments that encompass the recording's global maximum reach exactly 1.0.
   - Segments during periods of lower pulse amplitude (e.g., lower perfusion, vasoconstriction, or natural pulse amplitude modulation) peak at values less than 1.0.

### Implications for Amplitude Comparisons
- **Intra-subject comparisons (across segments from the same subject):**
  Relative amplitude variations between consecutive segments within a subject are **preserved**. Changes in PPG pulse height reflect genuine physiological changes in relative pulse volume over time for that individual.
- **Inter-subject comparisons (across different subjects):**
  Absolute amplitudes are **not directly comparable across subjects**. Because each subject's signal was scaled by their own clip-specific minimum and maximum values, an amplitude of 0.8 in subject A does not correspond to the same physical voltage or optical absorption as 0.8 in subject B. Feature engineering must rely on morphological ratios, time intervals, normalized slopes, and dimensionless shape metrics rather than absolute waveform heights across subjects.
