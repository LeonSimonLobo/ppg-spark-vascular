# Study C Hardening & Audit Report (Audit C2)

**Date:** 2026-10-29  
**Repository:** `D:\DS_RESEARCH\ppg-spark-vascular`  
**Git HEAD:** `3141b9d`  
**Scope:** Verification of Study C feature definitions, Amendment C1, pre-flight security gating, and test suite verification against `PLAN_C.md`.

---

## (a) Exact Code Definitions & Units from `src/features_c.py`

All single-segment features in Study C are computed strictly from the isolated 10-second PPG waveform ($f_s = 125\text{ Hz}$, 1,250 samples) and patient demographics without any per-subject normalization or cross-segment transduction.

| Feature Identifier | Code Definition & Mathematical Formula | Measurement Units | Implementation Location in `src/features_c.py` |
| :--- | :--- | :--- | :--- |
| **`hr_ibi`** | Continuous heart rate derived from the median inter-beat interval: $$\text{IBI} = \frac{\Delta \text{peaks}}{f_s}, \quad \text{med\_ibi} = \text{median}(\text{IBI}), \quad \text{hr\_ibi} = \frac{60.0}{\text{med\_ibi}}$$ | Beats per minute (**bpm**) | Lines 120–124 |
| **`ibi_sd`** | Sample standard deviation of inter-beat intervals across detected beats in the segment: $$\text{ibi\_sd} = \text{std}(\text{IBI}, \text{ddof}=1)$$ | Seconds (**s**) | Lines 120–124 |
| **`hr`** | Coarse heart rate estimated from discrete beat count scaled to 60 seconds: $$\text{hr} = n_{\text{peaks}} \times 6.0$$ | Beats per minute (**bpm**, discrete steps of 6 bpm) | Line 117 |
| **`systolic_rise_time`** | Median elapsed time from systolic foot (minimum sample in the 300 ms preceding peak) to systolic peak: $$\text{rise\_time} = \frac{t_{\text{peak}} - t_{\text{foot}}}{f_s}, \quad \text{systolic\_rise\_time} = \text{median}(\text{rise\_time})$$ | Seconds (**s**) | Lines 130–144, 159, 198 |
| **`pulse_width`** | Full width at 50% pulse height (FWHM) evaluated via linear interpolation between ascending and descending 50% crossing times: $$y_{50} = y_{\text{foot}} + 0.5 \cdot (y_{\text{peak}} - y_{\text{foot}}), \quad \text{pw} = \frac{t_{\text{desc}} - t_{\text{asc}}}{f_s}, \quad \text{pulse\_width} = \text{median}(\text{pw})$$ | Seconds (**s**) | Lines 166–186, 199 |
| **`decay_ratio`** | Ratio of diastolic decay duration to systolic rise duration: $$r\_time = \frac{t_{\text{peak}} - t_{\text{foot}}}{f_s}, \quad d\_time = \frac{t_{\text{turn}} - t_{\text{peak}}}{f_s}, \quad d\_ratio = \frac{d\_time}{r\_time}, \quad \text{decay\_ratio} = \text{median}(d\_ratio)$$ where $t_{\text{turn}}$ is the diastolic minimum between the current peak and next peak. | **Dimensionless time ratio** ($\text{s}/\text{s}$) | Lines 151–160, 188–190, 200 |
| **`beat_count`** | Total number of valid systolic peaks detected by the Elgendi algorithm: $$\text{beat\_count} = \text{float}(n_{\text{peaks}})$$ | Count (**beats**, integer stored as float) | Lines 99–101 |
| **`is_clipped`** | Binary indicator flagging rail clipping or amplifier saturation ($>1\%$ samples within $10^{-4}$ of voltage rail $[0.0, 1.0]$): $$\text{is\_clipped} = 1.0 \quad \text{if } (\text{mean}(ppg \le 10^{-4}) > 0.01 \lor \text{mean}(ppg \ge 1.0 - 10^{-4}) > 0.01) \quad \text{else } 0.0$$ | Binary indicator (**0.0 or 1.0**) | Lines 85–87 |
| **`snr_ratio`** | Relative physiological band spectral power fraction ($[0.5, 8.0]\text{ Hz}$ relative to total non-DC spectral energy): $$\text{snr\_ratio} = \frac{\sum_{0.5 \le f \le 8.0} |X(f)|^2}{\sum_{f > 0} |X(f)|^2 + 10^{-12}}$$ | Dimensionless fraction in $[0, 1]$ | Lines 89–96 |

### Explicit Clarification on `decay_ratio`:
- **Ratio Type:** `decay_ratio` is strictly a **TIME RATIO** ($d\_time / r\_time$, seconds of diastolic decay divided by seconds of systolic rise). It is **NOT** an amplitude or voltage ratio.
- **Agreement with Study W:** It matches the definition in `src/features_w.py` (lines 113–125) **identically** (`d_ratio = (d_time / r_time)`).

---

## (b) State Verification, Amendment C1, and Atomic Commits

### 1. Verification of Pristine Data State
Command executed:
```powershell
git log --stat -- "data/features_c*" "results/dev_C*"
```
Output:
```
(empty - 0 commits touch data/features_c* or results/dev_C*)
```

Directory check:
```powershell
Get-ChildItem -Path "data\features_c*", "results\dev_C*" -ErrorAction SilentlyContinue
```
Output:
```
(empty - 0 files exist on disk)
```
**Conclusion:** Zero Study C feature extractions, data caches, or model results existed or exist prior to protocol amendment and pre-flight locking.

---

### 2. Amendment C1 Appended to `PLAN_C.md`
Amendment C1 was appended to `PLAN_C.md` without modifying any preceding text, and committed prior to any feature extraction or scoring:

- **Git Commit:** `d1d0846` (`PLAN_C amendment 1, before any Study C extraction or results`)
- **Recorded Plan SHA-256 Hash:** `c8bc3e7703c8a49de3f41e208a4f040932f27aa22a0d26ba4550732abfff2d34`

#### Summary of Amendment Provisions:
1. **$C1 = C0 + \text{`hr\_ibi`}$:** Uses continuous median IBI heart rate ($60.0 / \text{median}(\text{IBI})$ in bpm). Discrete $n_{\text{beats}} \times 6$ is retired to the quality field `beat_count`.
2. **`decay_ratio` correction:** Clarified as a time ratio ($d\_time / r\_time$) matching the code and Study W.
3. **Secondary morphology ablation step ($C2m$):**
   - $C0$: Age, Sex, BMI
   - $C1$: $C0 + \text{`hr\_ibi`}$
   - $C2m$: $C1 + \text{pulse morphology shape only}$ (`systolic_rise_time`, `pulse_width`, `decay_ratio`)
   - $C2$: $C2m + \text{signal quality fields}$ (`beat_count`, `is_clipped`, `snr_ratio`, `ibi_sd`)
   - Decomposes accuracy and AUROC improvements into:
     $$\Delta \text{Gain}(\text{Morphology}) = C2m - C1$$
     $$\Delta \text{Gain}(\text{Quality}) = C2 - C2m$$
     $$\Delta \text{Gain}(\text{Pipeline}) = C2 - C0$$

---

### 3. Atomic Git Commits (One Commit Per Component)
Each subsequent change was implemented and committed as an atomic unit:
- `d1d0846`: `PLAN_C amendment 1, before any Study C extraction or results`
- `4245dc9`: `PLAN_C amendment 1: add C2m morphology-only ladder step to features_c.py`
- `2f95b84`: `PLAN_C amendment 1: add C2m morphology ablation and stepwise gain decomposition to train_c.py`
- `147af1f`: `PLAN_C amendment 1: add C2m morphology ablation row to dev runner report`
- `ccceb61`: `PLAN_C amendment 1: add C2m ablation and 6-gate pre-flight verification to test_c.py`
- `3141b9d`: `PLAN_C amendment 1: add pre-flight security tests and C2m unit tests`

---

## (c) Pre-Flight Security Gating in `src/test_c.py` & Security Proof

### 1. `verify_preflight_gates` Implementation
```python
def verify_preflight_gates(
    dev_report_path: str = DEV_REPORT_PATH,
    plan_file_path: str = PLAN_FILE_PATH,
    test_report_path: str = REPORT_PATH,
    lock_file_path: str = LOCK_FILE_PATH,
) -> Dict[str, str]:
    """Enforce pre-flight security and protocol gates before any test execution."""
```

### 2. Strict Order of Operations
The single-pass test script (`src/test_c.py`) enforces strict chronological order:
1. **Gate 1 (Repository Tag):** Verifies git tag `plan-c-locked` exists (`git tag -l plan-c-locked`).
2. **Gate 2 (Development Report Exists):** Verifies `results/dev_C.md` exists on disk.
3. **Gate 3 (Plan Hash Integrity):** Computes SHA-256 of `PLAN_C.md` and verifies exact match against the plan hash recorded in `results/dev_C.md`.
4. **Gate 4 (Frozen Model Verification):** Parses model filenames and digests from `results/dev_C.md` and verifies that every model file exists and has an identical SHA-256 digest on disk.
5. **Gate 5 (No Prior Test Report):** Checks that `results/test_C.md` does **not** exist (prohibits repeated test runs).
6. **Gate 6 (No Prior Test Lock):** Checks that `results/.test_locked_C` does **not** exist.
7. **Pre-Scoring Lock Acquisition:** `check_and_acquire_lock_pre_scoring` writes `results/.test_locked_C` with git commit, plan hash, and timestamp.
8. **Test Data Access:** Only *after* all gates pass and the lock is created on disk is `load_aami_cal_metadata` called.

### 3. Verification Proof via Mock Unit Tests
In `tests/test_test_c.py`, mock-based unit tests systematically assert that when ANY gate (Gates 1–6) fails, `load_aami_cal_metadata` and `h5py.File` are **NEVER called**:
```python
@pytest.mark.parametrize("failing_gate", ["gate1", "gate2", "gate3", "gate4", "gate5", "gate6"])
def test_aami_file_never_touched_when_any_gate_fails(tmp_path, failing_gate):
    ...
    mock_h5py.assert_not_called()
    mock_load_meta.assert_not_called()
```

Verbatim pytest execution:
```
tests\test_test_c.py ...............                                     [100%]
============================= 15 passed in 2.05s ==============================
```

---

## (d) Explanation of the 1 `xfail` Test

Test location: [`tests/test_fiducials.py:155`](file:///D:/DS_RESEARCH/ppg-spark-vascular/tests/test_fiducials.py#L155)  
Test name: `test_synthetic_lag_minus_30`

### Reason for `xfail`:
- **Historical Implementation (pre-A3):** `compute_ptt_offset` originally searched an asymmetric lag window $[0, 60)$ samples, based on the assumption that PPG pulse arrival always lags the central arterial pressure waveform in a specific coordinate direction.
- **Failure Mode:** When synthetic signals had ABP pulse leading PPG pulse by 30 samples (physiological arterial-to-peripheral transit direction, lag = -30), the one-sided $[0, 60)$ window could not discover negative lags, causing peak alignment failure.
- **Resolution under Amendment A3:** Amendment A3 (documented in `results/ptt_sign_check.md`) introduced a symmetric search window $[-W, W)$ (default $W=37$). The revised implementation is validated by `test_synthetic_lag_recovery_A3_plus_30` and `test_synthetic_lag_recovery_A3_minus_30`, both of which pass cleanly.
- **Pre-Registration Testing Policy:** In accordance with protocol integrity rules, historical tests documenting known pre-amendment failure modes are marked:
  ```python
  @pytest.mark.xfail(strict=True, reason="superseded by A3; see results/ptt_sign_check.md")
  ```
  `strict=True` ensures that if this superseded test ever unexpectedly passes with the legacy one-sided window, pytest will raise an error.

---

## (e) Full Test Suite Execution Output

Command executed:
```powershell
python -m pytest tests/
```

Verbatim terminal output:
```
============================= test session starts =============================
platform win32 -- Python 3.13.1, pytest-9.1.1, pluggy-1.6.0
rootdir: D:\DS_RESEARCH\ppg-spark-vascular
plugins: anyio-4.12.1, langsmith-0.6.2
collected 56 items

tests\test_baselines.py ...                                              [  5%]
tests\test_controls_w.py .                                               [  7%]
tests\test_exploratory_dev_models.py ..                                  [ 10%]
tests\test_features_c.py .....                                           [ 19%]
tests\test_features_w.py ...                                             [ 25%]
tests\test_fiducials.py ....x......                                      [ 44%]
tests\test_gate.py ..                                                    [ 48%]
tests\test_gate_agreement.py ..                                          [ 51%]
tests\test_prospective_and_scale_w.py ....                               [ 58%]
tests\test_test_c.py ...............                                     [ 85%]
tests\test_test_w.py ....                                                [ 92%]
tests\test_train_c.py ...                                                [ 98%]
tests\test_train_w.py .                                                  [100%]

======================== 55 passed, 1 xfailed in 5.56s ========================
```
