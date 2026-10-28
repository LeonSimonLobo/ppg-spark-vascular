# Raw MATLAB (.mat) File Inspection Report

Inspection of `data/raw/VitalDB_Train_Subset.mat` and `data/raw/VitalDB_CalFree_Test_Subset.mat` using `h5py`.

---

## 1. Top-Level Keys and `Subset` Keys

### `data/raw/VitalDB_Train_Subset.mat`
- **File Size**: 13.80 GB (13,797,698,523 bytes)
- **Top-Level HDF5 Keys**: `['#refs#', 'Subset']`
- **Top-Level HDF5 Attributes (`f.attrs`)**: None (`{}`)
- **Group `/Subset` Attributes**:
  - `MATLAB_class`: `b'struct'`
  - `MATLAB_fields`: `['Subject', 'Signals', 'SBP', 'DBP', 'Age', 'Gender', 'Height', 'Weight', 'BMI']`
- **Datasets under `/Subset`**:

| Key | Shape | Dtype | Chunking | MATLAB Class | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`Age`** | `(1, 465480)` | `float64` | `(1, 8166)` | `double` | Subject age in years |
| **`BMI`** | `(1, 465480)` | `float64` | `(1, 8166)` | `double` | Body Mass Index ($kg/m^2$) |
| **`DBP`** | `(1, 465480)` | `float64` | `(1, 8166)` | `double` | Diastolic Blood Pressure (mmHg) |
| **`Gender`** | `(1, 465480)` | `object` (HDF5 references) | `(1, 8166)` | `cell` | Cell array of char ('M'/'F') |
| **`Height`** | `(1, 465480)` | `float64` | `(1, 8166)` | `double` | Subject height in cm |
| **`SBP`** | `(1, 465480)` | `float64` | `(1, 8166)` | `double` | Systolic Blood Pressure (mmHg) |
| **`Signals`** | `(1250, 3, 465480)` | `float64` | `(1250, 3, 2)` | `double` | 3-channel waveforms (ECG, PPG, ABP) |
| **`Subject`** | `(1, 465480)` | `object` (HDF5 references) | `(1, 8166)` | `cell` | Cell array of subject ID strings |
| **`Weight`** | `(1, 465480)` | `float64` | `(1, 8166)` | `double` | Subject weight in kg |

---

### `data/raw/VitalDB_CalFree_Test_Subset.mat`
- **File Size**: 1.71 GB (1,706,947,648 bytes)
- **Top-Level HDF5 Keys**: `['#refs#', 'Subset']`
- **Top-Level HDF5 Attributes (`f.attrs`)**: None (`{}`)
- **Group `/Subset` Attributes**:
  - `MATLAB_class`: `b'struct'`
  - `MATLAB_fields`: `['Subject', 'Signals', 'SBP', 'DBP', 'Age', 'Gender', 'Height', 'Weight', 'BMI']`
- **Datasets under `/Subset`**:

| Key | Shape | Dtype | Chunking | MATLAB Class | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`Age`** | `(1, 57600)` | `float64` | `(1, 7200)` | `double` | Subject age in years |
| **`BMI`** | `(1, 57600)` | `float64` | `(1, 7200)` | `double` | Body Mass Index ($kg/m^2$) |
| **`DBP`** | `(1, 57600)` | `float64` | `(1, 7200)` | `double` | Diastolic Blood Pressure (mmHg) |
| **`Gender`** | `(1, 57600)` | `object` (HDF5 references) | `(1, 7200)` | `cell` | Cell array of char ('M'/'F') |
| **`Height`** | `(1, 57600)` | `float64` | `(1, 7200)` | `double` | Subject height in cm |
| **`SBP`** | `(1, 57600)` | `float64` | `(1, 7200)` | `double` | Systolic Blood Pressure (mmHg) |
| **`Signals`** | `(1250, 3, 57600)` | `float64` | `(1250, 3, 2)` | `double` | 3-channel waveforms (ECG, PPG, ABP) |
| **`Subject`** | `(1, 57600)` | `object` (HDF5 references) | `(1, 7200)` | `cell` | Cell array of subject ID strings |
| **`Weight`** | `(1, 57600)` | `float64` | `(1, 7200)` | `double` | Subject weight in kg |

---

## 2. Beat Fiducial Points Search

Specifically searched for keys and structures containing: `"Fid"`, `"Peak"`, `"Onset"`, `"Foot"`, `"Notch"`, `"Dicrotic"`, `"Sys"`, `"Dia"`, or `"Fiducial"`.

- **Result**: **No beat fiducial points exist in either file.**
- Neither per-segment nor per-beat fiducial arrays are present in the raw `.mat` files. The only keys present under `/Subset` are the 9 listed in Section 1.

---

## 3. Sampling Rate and Segment Duration

- **Stored in file**: **No.** Neither sampling rate nor segment duration is stored as a dataset, attribute, or metadata field in either `.mat` file.
- **Source of values (125 Hz, 10 s)**:
  - The waveform dimension in `Signals` has 1,250 samples per segment (`shape[0] = 1250`).
  - The 125 Hz sampling rate and 10-second duration come from the ingest constants in `src/ingest.py` (line 13: `L, FS = 1250, 125`) and the project data contract (`contract.md`, line 7):
    > `ppg: list<float32> (1250 samples, 125 Hz, PPG only, per-segment normalized)`
  - At 125 Hz, 1,250 samples corresponds to exactly $\frac{1250}{125} = 10.0$ seconds.

---

## 4. Demographic and Clinical Fields

- The demographic and clinical fields present are strictly: `Age`, `Gender`, `Height`, `Weight`, `BMI`, `SBP`, and `DBP`.
- **No other demographic fields exist** in the files (no ethnicity, race, comorbidities, or medication records).

---

## 5. Dataset Attributes

- **File root attributes (`f.attrs`)**: Empty (`{}`).
- **Dataset attributes**:
  - Each dataset under `/Subset` carries two metadata attributes:
    - `H5PATH = b'/Subset'`
    - `MATLAB_class` specifying the original MATLAB type (`b'double'` or `b'cell'`).
