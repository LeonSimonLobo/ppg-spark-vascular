# Dataset report

**Subject overlap train_file vs calfree_test: 0** (good)

## train_file
- PPG channel index used: 1 (corr with ABP: ch0 0.41, ch1 0.92)
- segments written: 3000 of 3000 (dropped: 0)
- subjects: 9; segments per subject min/median/max: 120/360/360
- non-null %: age 100.0, sex 100.0, bmi 100.0
- subject-level age median [IQR]: 67.0 [45.0-75.0]; BMI: 20.7 [19.5-22.4]; female share: 44.4%
- subjects whose age / bmi varies across their segments: 0 / 0
- BMI vs weight/height^2: 99th pct abs diff = 0.05
- SBP quantiles 5/25/50/75/95: [95.5, 108.3, 116.2, 125.6, 140.7]
- DBP quantiles 5/25/50/75/95: [48.0, 55.3, 65.1, 75.6, 86.6]
- **share of SBP variance that is between subjects: 25.0%** (the rest is within-subject, invisible to age/sex/BMI)
- median within-subject SBP SD: 11.3 mmHg
- SBP >= 120: segments 39.1% | subjects (median SBP) 3 of 9 (33.3%)
- SBP >= 130: segments 16.7% | subjects (median SBP) 0 of 9 (0.0%)
- SBP >= 140: segments 5.4% | subjects (median SBP) 0 of 9 (0.0%)

## calfree_test
- PPG channel index used: 1 (corr with ABP: ch0 0.36, ch1 0.95)
- segments written: 3000 of 3000 (dropped: 0)
- subjects: 8; segments per subject min/median/max: 200/400/400
- non-null %: age 100.0, sex 100.0, bmi 100.0
- subject-level age median [IQR]: 58.5 [44.0-69.2]; BMI: 24.5 [23.5-25.2]; female share: 25.0%
- subjects whose age / bmi varies across their segments: 0 / 0
- BMI vs weight/height^2: 99th pct abs diff = 0.05
- SBP quantiles 5/25/50/75/95: [96.5, 112.0, 124.4, 136.7, 152.5]
- DBP quantiles 5/25/50/75/95: [50.8, 61.3, 66.8, 74.9, 95.1]
- **share of SBP variance that is between subjects: 33.3%** (the rest is within-subject, invisible to age/sex/BMI)
- median within-subject SBP SD: 14.7 mmHg
- SBP >= 120: segments 59.1% | subjects (median SBP) 5 of 8 (62.5%)
- SBP >= 130: segments 38.6% | subjects (median SBP) 2 of 8 (25.0%)
- SBP >= 140: segments 17.9% | subjects (median SBP) 1 of 8 (12.5%)
