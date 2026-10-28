# Dataset report

**Subject overlap train_file vs calfree_test: 0** (good)

## train_file
- PPG channel index used: 1 (corr with ABP: ch0 0.32, ch1 0.95)
- segments written: 465480 of 465480 (dropped: 0)
- subjects: 1293; segments per subject min/median/max: 360/360/360
- non-null %: age 100.0, sex 100.0, bmi 100.0
- subject-level age median [IQR]: 61.0 [51.0-70.0]; BMI: 22.8 [20.7-25.0]; female share: 42.3%
- subjects whose age / bmi varies across their segments: 11 / 17
- BMI vs weight/height^2: 99th pct abs diff = 0.05
- SBP quantiles 5/25/50/75/95: [87.4, 102.1, 114.0, 127.3, 148.6]
- DBP quantiles 5/25/50/75/95: [44.6, 54.4, 62.2, 70.5, 83.8]
- **share of SBP variance that is between subjects: 41.5%** (the rest is within-subject, invisible to age/sex/BMI)
- median within-subject SBP SD: 13.2 mmHg
- SBP >= 120: segments 37.7% | subjects (median SBP) 419 of 1293 (32.4%)
- SBP >= 130: segments 21.1% | subjects (median SBP) 161 of 1293 (12.5%)
- SBP >= 140: segments 10.3% | subjects (median SBP) 44 of 1293 (3.4%)

## calfree_test
- PPG channel index used: 1 (corr with ABP: ch0 0.29, ch1 0.94)
- segments written: 57600 of 57600 (dropped: 0)
- subjects: 144; segments per subject min/median/max: 400/400/400
- non-null %: age 100.0, sex 100.0, bmi 100.0
- subject-level age median [IQR]: 60.5 [48.0-71.0]; BMI: 23.1 [20.7-24.5]; female share: 40.3%
- subjects whose age / bmi varies across their segments: 0 / 1
- BMI vs weight/height^2: 99th pct abs diff = 0.05
- SBP quantiles 5/25/50/75/95: [87.5, 101.9, 113.9, 127.2, 148.4]
- DBP quantiles 5/25/50/75/95: [44.7, 54.5, 62.7, 70.3, 83.5]
- **share of SBP variance that is between subjects: 39.5%** (the rest is within-subject, invisible to age/sex/BMI)
- median within-subject SBP SD: 13.3 mmHg
- SBP >= 120: segments 37.7% | subjects (median SBP) 42 of 144 (29.2%)
- SBP >= 130: segments 21.1% | subjects (median SBP) 18 of 144 (12.5%)
- SBP >= 140: segments 10.3% | subjects (median SBP) 6 of 144 (4.2%)
