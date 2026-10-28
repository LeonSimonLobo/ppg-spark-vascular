# Source Verification and Provenance Analysis (Part 2)

**Date:** 2026-10-28  
**Scope:** Verification of Kaggle archive checksums, Segment_Files availability, variable names, alignment mapping, and amplitude distribution.

---

## 1. Kaggle Byte and Checksum Check

- **Archive Dataset:** `weinanwangrutgers/pulsedb-balanced-training-and-testing` (PulseDB Supplementary Subsets)
- **Source URL:** [https://www.kaggle.com/datasets/weinanwangrutgers/pulsedb-balanced-training-and-testing](https://www.kaggle.com/datasets/weinanwangrutgers/pulsedb-balanced-training-and-testing) (DOI: `10.34740/KAGGLE/DS/2447469`)
- **Verification Method:** Downloaded file verification / local archive verification (`data/raw/*.zip` and `data/raw/*.mat`).

### Member-by-Member Comparison

For each `.mat` file, the CRC-32 was read directly from the archive member header (`ZipInfo.CRC`) and compared against the computed CRC-32 of the matching raw file in `data/raw/` (calculated via `zlib.crc32` in 64 MB streaming blocks).

| Archive Member Name | Raw File Size (bytes) | Zip Member Size (bytes) | Size Match | Raw File CRC-32 | Zip Member CRC-32 | CRC-32 Match |
|---|---|---|---|---|---|---|
| `VitalDB_AAMI_Cal_Subset.mat` | 2,078,737,367 | 2,078,737,367 | **Yes (Exact)** | `0x9fe15aa1` | `0x9fe15aa1` | **Yes (Exact)** |
| `VitalDB_AAMI_Test_Subset.mat` | 19,806,481 | 19,806,481 | **Yes (Exact)** | `0x8dd3a5f6` | `0x8dd3a5f6` | **Yes (Exact)** |
| `VitalDB_CalBased_Test_Subset.mat` | 1,532,644,941 | 1,532,644,941 | **Yes (Exact)** | `0xeee0dff7` | `0xeee0dff7` | **Yes (Exact)** |
| `VitalDB_CalFree_Test_Subset.mat` | 1,706,947,648 | 1,706,947,648 | **Yes (Exact)** | `0x67c5e8f6` | `0x67c5e8f6` | **Yes (Exact)** |
| `VitalDB_Train_Subset.mat` | 13,797,698,523 | 13,797,698,523 | **Yes (Exact)** | `0xa8ecd1e8` | `0xa8ecd1e8` | **Yes (Exact)** |

### Totals Comparison
- **Sum of member sizes in zip archives:** `19,135,834,960` bytes
- **Sum of raw .mat file sizes in `data/raw/`:** `19,135,834,960` bytes (matches exactly)
- **Kaggle API uncompressed total (`totalBytesNullable`):** `19,135,855,809` bytes
- **Comparison:**
  - `19,135,834,960` bytes vs. `19,135,855,809` bytes
  - **Difference:** `20,849` bytes
  - **Structure Check:** The local zip files contain solely the single `.mat` file members (no other file entries exist inside them). Therefore, the zip structure itself does not account for this `20,849`-byte discrepancy. These two numbers **do not match**.

---

## 2. Segment_Files and Info_Files Check

- **Sources:**
  - Frontiers publication: [https://doi.org/10.3389/fdgth.2022.1090854](https://doi.org/10.3389/fdgth.2022.1090854) (Section 2, Section 4.2)
  - GitHub Repository README & Guides: [https://github.com/pulselabteam/PulseDB](https://github.com/pulselabteam/PulseDB)
  - Supplementary Materials: [https://raw.githubusercontent.com/pulselabteam/PulseDB/main/Supplementary%20Materials.pdf](https://raw.githubusercontent.com/pulselabteam/PulseDB/main/Supplementary%20Materials.pdf)
- **Verification Method:** Verified from fetched pages and downloaded official PDF.

### Download Locations
1. **Box Shared Links:**
   - Root / Compressed parts: `https://rutgers.box.com/s/sw3c51fr5oybz6mhqsphh5zg8ibxw800`
   - `Segment_Files`: `https://rutgers.box.com/s/ezmiotu66dtnahxoirnukn25nrcylybc`
   - `Supplementary_Info_Files`: `https://rutgers.box.com/s/ntgymk9xg3o2pdc34h210vno4s2biuoh`
   - `Info_Files`: `https://rutgers.box.com/s/b0q4v7kiqwbw3o9mczey9o615sjo2qsp`
2. **Google Drive Shared Links:**
   - Root: `https://drive.google.com/drive/folders/10mz4mfBo6NczPNbbjX0a9tAKQSMugBjV?usp=sharing`
   - `Segment_Files`: `https://drive.google.com/drive/folders/1-WKcJgO9lAjY2HBepRGDqyF3xjFADr8-?usp=sharing`
   - `Supplementary_Info_Files`: `https://drive.google.com/drive/folders/1mm1a_hPd8jMtRSowQ1TKi51Jyes_x8Fb?usp=sharing`
   - `Info_Files`: `https://drive.google.com/drive/folders/1-0IAdIHT9AXpQ8I1Z-xS5PqB3Yeo-bNY?usp=sharing`
3. **OneDrive Shared Links:**
   - Root: `https://rutgersconnect-my.sharepoint.com/:f:/g/personal/ww329_soe_rutgers_edu/EqalUqc2s_dEqbhgugkUW1MBeNQIUEntgsGM67atFfivbg?e=csitkl`
   - `Segment_Files`: `https://rutgersconnect-my.sharepoint.com/:f:/g/personal/ww329_soe_rutgers_edu/EiJIJdNUH5FEkjHozQLiz2IBX5XD8T1RdVGJwGT4MIIA1g?e=0B4d8K`
   - `Supplementary_Info_Files`: `https://rutgersconnect-my.sharepoint.com/:f:/g/personal/ww329_soe_rutgers_edu/EsRbMMcydjtOtXUb9930TCEBvpd3ZUTLo7L6p_tmzQH80Q?e=KWej99`
   - `Info_Files`: `https://rutgersconnect-my.sharepoint.com/:f:/g/personal/ww329_soe_rutgers_edu/EpdTCcDihmBIrQpYAuCbV7sBgFX7H-NovUu0hSWNDWRcDQ?e=jbm1jN`

### File Names and Sizes
- **`Supplementary_Info_Files`:**
  - Files: `VitalDB_Train_Info.mat`, `VitalDB_CalBased_Test_Info.mat`, `VitalDB_CalFree_Test_Info.mat`, `VitalDB_AAMI_Test_Info.mat`, `VitalDB_AAMI_Cal_Info.mat`.
- **`Info_Files`:**
  - Files: `Train_Info.mat`, `CalBased_Test_Info.mat`, `CalFree_Test_Info.mat`, `AAMI_Test_Info.mat`, `AAMI_Cal_Info.mat`, `PulseDB_Info.mat`.
- **`Segment_Files`:**
  - Stored as multi-part archives for two subfolders: `PulseDB_Vital` and `PulseDB_MIMIC`.
  - `PulseDB_Vital` files on Box:
    - `PulseDB_Vital.zip.001` through `PulseDB_Vital.zip.009`: `15,728,640,000` bytes each (~14.65 GiB each).
    - `PulseDB_Vital.zip.010`: `1,499,811,157` bytes (~1.40 GiB).
    - **Total size of `PulseDB_Vital`:** `143,057,571,157` bytes (~**143.06 GB** / 133.23 GiB).
  - `PulseDB_MIMIC` files on Box:
    - Split into 16 parts (`PulseDB_MIMIC.zip.001` through `PulseDB_MIMIC.zip.016`).
- **Download Action:** Because the archive size for `PulseDB_Vital` alone is ~143 GB (far exceeding the 5 GB threshold), the full archive was not downloaded, and download was stopped as instructed.

### Characteristic Points and Quoted Variable Names
VitalDB segment files contain an array of structures (`Subj_Wins`), with one structure per 10-s segment. The characteristic points are explicitly documented in `Supplementary Materials.pdf` (page 1, lines 38–40):
> "• ECG RPeaks, PPG SPeaks, PPG Turns, ABP SPeaks, ABP Turns: Positions of extracted characteristic points, specified as indices. For example, indexing ECG Raw using ECG RPeaks identify the amplitudes of all R-peaks of the ECG signal in this segment."

**Quoted variable names from the source:**
- `PPG SPeaks` (systolic peaks of the PPG signal)
- `PPG Turns` (turning points / minima between consecutive systolic peaks, representing onsets)
- `ECG RPeaks` (R-peaks of the ECG signal)
- `ABP SPeaks` (systolic peaks of the ABP signal)
- `ABP Turns` (turning points / diastolic minima of the ABP signal)

---

## 3. Alignment Check

- **Status:** Verified from code structure and subset metadata; raw segment file download not performed due to archive size.
- **Source:** `Generate_Subsets.m` ([https://raw.githubusercontent.com/pulselabteam/PulseDB/main/Generate_Subsets.m](https://raw.githubusercontent.com/pulselabteam/PulseDB/main/Generate_Subsets.m)) and `data/raw/VitalDB_CalFree_Test_Subset.mat`.
- **Verification Details:**
  1. In `Generate_Subsets.m` (lines 75–113), the subset generator processes subjects in sorted alphabetical order:
     ```matlab
     Subjects = unique({Info.Subj_Name});
     ```
  2. For each subject, the script reads `Subj_Segments = Segments_File.Subj_Wins` and extracts segments indexed by `Selected_IDX`:
     ```matlab
     Selected_IDX = [Info(strcmp({Info.Subj_Name}, Subj_Name)).Subj_SegIDX];
     for j = Selected_IDX
         Segment = Subj_Segments(j);
         Subset.Subject{pos} = Subj_Name;
         Subset.Signals(pos,:,:) = [Segment.ECG_F, Segment.PPG_F, Segment.ABP_Raw]';
         pos = pos + 1;
     end
     ```
  3. In `data/raw/VitalDB_CalFree_Test_Subset.mat`, inspection confirms that the first 400 segments belong to subject `p000053_1`, which is the first subject alphabetically in the calibration-free test cohort.
  4. Direct sample-by-sample matching of `Signals` against the raw subject file `p000053.mat` could not be executed on disk because `PulseDB_Vital` is 143 GB (> 5 GB limit) and was not downloaded. The deterministic mapping from `Subj_Wins` to `Subset.Signals` is verified via the MATLAB generation code.

---

## 4. Reconciliation of the 12.35% Finding

- **Empirical Observation:** In `results/ppg_clipping.md`, 87.65% of segments in `VitalDB_CalFree_Test_Subset.mat` reach a maximum sample value of exactly 1.0, while 12.35% peak below 1.0.
- **Source Finding:** In the fetched source documents (the Frontiers publication, GitHub repository README, `Generate_Subsets.m`, and `Supplementary Materials.pdf`), there is no statement or explanation addressing why 12.35% of `PPG_F` segments in the subset peak below 1.0.
- **Conclusion:** Because this cannot be explained from the source text, per instructions, no explanation is offered.
