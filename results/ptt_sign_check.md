# PTT Sign and Boundary Check

**Date:** 2026-10-29  
**Status:** Gate check completed. Synthetic test failed. Part 2 blocked per pre-registered gate rule.

---

## 1. Initial Confirmation

- **`tests/test_fiducials.py`:** Initially contained only `test_ptt_offset_and_agreement` with a single fixed positive lag (`known_lag = 35`). It did not contain the synthetic $\pm 30$-sample lag tests.
- **`results/ptt_sign_check.md`:** Did not exist prior to this check.

Both checks were conducted, the synthetic tests were added, and this report documents the exact mathematical sign convention, synthetic results, and gate failure.

---

## 2. Sign Convention in Source Code

### `src/ingest.py` (lines 64–70)
```python
        for lag in range(0, 60):
            a_sub = a[lag:, :]
            s0_sub = x_centered[:-lag or None, 0, :]
            s1_sub = x_centered[:-lag or None, 1, :]
            norm_a = np.sqrt((a_sub ** 2).sum(axis=0)) + 1e-12
            r0 = (s0_sub * a_sub).sum(axis=0) / (np.sqrt((s0_sub ** 2).sum(axis=0)) * norm_a + 1e-12)
            r1 = (s1_sub * a_sub).sum(axis=0) / (np.sqrt((s1_sub ** 2).sum(axis=0)) * norm_a + 1e-12)
```

### `src/fiducials.py` (lines 196–204)
```python
    corrs = []
    for lag in range(0, max_lag):
        a_sub = abp_c[lag:]
        p_sub = ppg_c[:-lag or None]
        norm = np.linalg.norm(a_sub) * np.linalg.norm(p_sub) + 1e-12
        r = float(np.dot(p_sub, a_sub) / norm)
        corrs.append(r)

    best_lag = int(np.argmax(corrs))
    return best_lag, float(corrs[best_lag])
```

### `src/fiducials.py` (lines 255–263)
```python
    tol_samples = tol_ms * fs / 1000.0  # 6.25 samples at 125 Hz
    # Under ingest.py convention (ppg[t] ~ abp[t + lag]),
    # the PTT-corrected ABP peak position is abp_peak - ptt_offset.
    abp_peaks_corrected = abp_peaks.astype(float) - float(ptt_offset)

    matched = 0
    for p_pk in ppg_peaks:
        min_dist = np.min(np.abs(p_pk - abp_peaks_corrected))
        if min_dist <= tol_samples:
            matched += 1
```

### Mathematical Sign Analysis
In both implementations:
- `p_sub = ppg[:-lag]` spans index $t \in [0, N - lag)$.
- `a_sub = abp[lag:]` spans index $t + lag \in [lag, N)$.
- The dot product $\sum_t ppg[t] \cdot abp[t + lag]$ is maximized when $ppg[t] \approx abp[t + lag]$.
- Therefore, a positive lag ($lag > 0$) strictly represents **PPG leading ABP** (i.e., the waveform feature appears in PPG at time $t$ and in ABP at time $t + lag$).
- Correspondingly, line 256 corrects ABP peak times by subtracting $ptt\_offset$: $abp\_peaks\_corrected = abp\_peaks - ptt\_offset$.

---

## 3. Physiological Context vs. Convention

In cardiovascular physiology:
- The ventricular pressure pulse propagates down the arterial tree to the radial artery (the site of the ABP catheter in VitalDB).
- The pulse subsequently propagates through the digital arteries into the peripheral microvascular capillary bed of the finger (the site of the photoplethysmogram, PPG).
- Therefore, the pressure wave arrives at the ABP catheter **before** the optical wave arrives at the PPG sensor:
  $$t_{\text{ABP}} < t_{\text{PPG}}$$
- Defining $\tau = t_{\text{PPG}} - t_{\text{ABP}} > 0$ as the true physiological pulse transit time:
  $$t_{\text{ABP}} = t_{\text{PPG}} - \tau$$
- Under the correlation formulation $abp[t + lag] \approx ppg[t]$, this physiological timing requires:
  $$t + lag = t - \tau \implies lag = -\tau < 0$$
- However, both `src/ingest.py` (line 64) and `src/fiducials.py` (line 196) restrict the search space strictly to non-negative lags:
  $$\text{lag} \in [0, 60)$$
- As a consequence, when the true physiological relationship holds ($t_{\text{ABP}} < t_{\text{PPG}}$, or $lag = -30$), the search loop cannot evaluate negative lags.

---

## 4. Synthetic $\pm 30$-Sample Lag Tests

Two unit tests were added to `tests/test_fiducials.py`:

```python
def test_synthetic_lag_plus_30():
    """Verify PTT offset and beat agreement when ABP is delayed by +30 samples (PPG leads ABP)."""
    fs = 125
    ppg, ppg_peaks, _ = generate_synthetic_pulse_train(
        fs=fs, duration_s=10.0, bpm=75.0, noise_std=0.0, seed=42
    )
    known_lag = 30
    abp = np.zeros_like(ppg)
    abp[known_lag:] = ppg[:-known_lag] * 40.0 + 80.0
    abp[:known_lag] = np.mean(abp[known_lag:])

    best_lag, max_corr = compute_ptt_offset(ppg, abp, max_lag=60)
    assert best_lag == known_lag, f"Expected lag {known_lag}, got {best_lag}"
    assert max_corr > 0.90, f"Expected high correlation, got {max_corr}"

    res = process_segment(ppg, abp, fs=fs)
    assert res["passes_rule"] is True
    assert res["matched_fraction"] >= 0.80


def test_synthetic_lag_minus_30():
    """Verify PTT offset and beat agreement when ABP leads PPG by 30 samples (lag = -30)."""
    fs = 125
    base, true_peaks, _ = generate_synthetic_pulse_train(
        fs=fs, duration_s=10.0, bpm=75.0, noise_std=0.0, seed=42
    )
    # Physiological direction: ABP pulse precedes PPG pulse by 30 samples
    known_lag = -30
    abp = base * 40.0 + 80.0
    ppg = np.zeros_like(base)
    ppg[30:] = base[:-30]
    ppg[:30] = np.mean(ppg[30:])

    best_lag, max_corr = compute_ptt_offset(ppg, abp, max_lag=60)
    assert best_lag == known_lag, f"Expected lag {known_lag}, got {best_lag}"
    assert max_corr > 0.90, f"Expected high correlation, got {max_corr}"

    res = process_segment(ppg, abp, fs=fs)
    assert res["passes_rule"] is True
    assert res["matched_fraction"] >= 0.80
```

---

## 5. Pytest Execution Output

Command executed: `python -m pytest tests/test_fiducials.py`

```text
============================= test session starts =============================
platform win32 -- Python 3.13.1, pytest-9.1.1, pluggy-1.6.0
rootdir: D:\DS_RESEARCH\ppg-spark-vascular
plugins: anyio-4.12.1, langsmith-0.6.2
collected 6 items

tests\test_fiducials.py ....F.                                           [100%]

================================== FAILURES ===================================
_________________________ test_synthetic_lag_minus_30 _________________________

    def test_synthetic_lag_minus_30():
        """Verify PTT offset and beat agreement when ABP leads PPG by 30 samples (lag = -30)."""
        fs = 125
        base, true_peaks, _ = generate_synthetic_pulse_train(
            fs=fs, duration_s=10.0, bpm=75.0, noise_std=0.0, seed=42
        )
        # Physiological direction: ABP pulse precedes PPG pulse by 30 samples
        known_lag = -30
        abp = base * 40.0 + 80.0
        ppg = np.zeros_like(base)
        ppg[30:] = base[:-30]
        ppg[:30] = np.mean(ppg[30:])
    
        best_lag, max_corr = compute_ptt_offset(ppg, abp, max_lag=60)
>       assert best_lag == known_lag, f"Expected lag {known_lag}, got {best_lag}"
E       AssertionError: Expected lag -30, got 59
E       assert 59 == -30

tests\test_fiducials.py:169: AssertionError
=========================== short test summary info ===========================
FAILED tests/test_fiducials.py::test_synthetic_lag_minus_30 - AssertionError: Expected lag -30, got 59
========================= 1 failed, 5 passed in 0.30s =========================
```

---

## 6. Diagnostic Finding & Boundary Hit Explanation

1. **Test Failure Line:** `tests/test_fiducials.py:169` (`assert best_lag == known_lag`).
2. **Observed Value:** `best_lag = 59`.
3. **Mechanistic Explanation:**
   - When ABP leads PPG by 30 samples, $ppg[t] \approx abp[t - 30]$.
   - Because the algorithm only searches $lag \in [0, 60)$ under the assumption $ppg[t] \approx abp[t + lag]$, it does not search negative offsets.
   - For a quasi-periodic cardiac waveform with period $T \approx 100$ samples (75 bpm), shifting $abp$ forward by $lag > 0$ eventually brings the *next* cardiac pulse into phase:
     $$\text{effective lag} \approx T - 30 = 100 - 30 = 70 \text{ samples}$$
   - Because the search window terminates at `max_lag - 1 = 59`, the cross-correlation monotonically increases toward sample 70, peaking exactly at the upper search boundary: `lag = 59`.
   - This reproduces and explains the finding in `results/detector_diagnostics.md`, where **23,806 validation segments (25.53%) hit lag 59**, accounting for **64.12% of all validation detector failures**.

---

## 7. Gate Status & Pre-Registration Constraint

- **Gate Rule 3:** *"If the synthetic test fails, stop and report the line. Do not proceed to Part 2."*
- **Outcome:** The synthetic test `test_synthetic_lag_minus_30` failed at `tests/test_fiducials.py:169`.
- **Action:** Execution is formally stopped at Part 1. Part 2 (validation baselines) and Part 3 (test evaluation) are not executed.
