# Demographic Consistency Report

## Training Subjects with Varying Demographics across Segments

### 1. Training Subjects with Varying Age (11 subjects)
| Subject ID | Segments | Min Age | Max Age | Delta Age | Min BMI | Max BMI |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| p000309_1 | 360 | 37.0 | 38.0 | 1.0 | 20.90 | 21.90 |
| p000591_1 | 360 | 72.0 | 73.0 | 1.0 | 21.00 | 21.40 |
| p001726_1 | 360 | 67.0 | 68.0 | 1.0 | 21.70 | 23.20 |
| p002242_1 | 360 | 80.0 | 81.0 | 1.0 | 19.20 | 21.80 |
| p002291_1 | 360 | 53.0 | 54.0 | 1.0 | 18.30 | 18.30 |
| p004791_1 | 360 | 69.0 | 70.0 | 1.0 | 21.20 | 21.60 |
| p005648_1 | 360 | 44.0 | 45.0 | 1.0 | 20.00 | 20.40 |
| p005778_1 | 360 | 80.0 | 81.0 | 1.0 | 23.70 | 23.80 |
| p005820_1 | 360 | 47.0 | 48.0 | 1.0 | 23.90 | 23.90 |
| p005878_1 | 360 | 56.0 | 57.0 | 1.0 | 22.20 | 23.10 |
| p006043_1 | 360 | 62.0 | 63.0 | 1.0 | 18.50 | 18.90 |

### 2. Training Subjects with Varying BMI (17 subjects)
| Subject ID | Segments | Min BMI | Max BMI | Delta BMI | Min Height (cm) | Max Height (cm) | Min Weight (kg) | Max Weight (kg) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| p000309_1 | 360 | 20.90 | 21.90 | 1.00 | 176.7 | 178.0 | 66.10 | 68.40 |
| p000591_1 | 360 | 21.00 | 21.40 | 0.40 | 167.0 | 167.2 | 58.50 | 59.70 |
| p000799_1 | 360 | 20.70 | 21.40 | 0.70 | 167.9 | 168.0 | 58.50 | 60.45 |
| p001726_1 | 360 | 21.70 | 23.20 | 1.50 | 161.1 | 161.4 | 56.50 | 60.15 |
| p001727_1 | 360 | 22.40 | 23.60 | 1.20 | 136.0 | 136.6 | 41.85 | 43.60 |
| p001828_1 | 360 | 25.40 | 25.80 | 0.40 | 164.8 | 166.0 | 69.90 | 70.10 |
| p002242_1 | 360 | 19.20 | 21.80 | 2.60 | 150.0 | 151.7 | 43.30 | 50.20 |
| p002596_1 | 360 | 18.60 | 19.40 | 0.80 | 151.9 | 152.6 | 42.90 | 45.25 |
| p002629_1 | 360 | 19.30 | 20.40 | 1.10 | 166.5 | 166.6 | 53.40 | 56.70 |
| p002821_1 | 360 | 21.10 | 21.90 | 0.80 | 157.1 | 157.1 | 52.10 | 54.10 |
| p003131_1 | 360 | 16.80 | 18.30 | 1.50 | 177.7 | 178.1 | 53.40 | 57.65 |
| p003938_1 | 360 | 15.00 | 17.00 | 2.00 | 158.3 | 158.5 | 37.60 | 42.80 |
| p004791_1 | 360 | 21.20 | 21.60 | 0.40 | 156.1 | 156.1 | 51.75 | 52.65 |
| p005648_1 | 360 | 20.00 | 20.40 | 0.40 | 174.3 | 174.8 | 60.90 | 62.30 |
| p005778_1 | 360 | 23.70 | 23.80 | 0.10 | 162.1 | 162.1 | 62.40 | 62.45 |
| p005878_1 | 360 | 22.20 | 23.10 | 0.90 | 169.9 | 170.5 | 64.60 | 66.70 |
| p006043_1 | 360 | 18.50 | 18.90 | 0.40 | 165.5 | 165.7 | 50.90 | 51.65 |

### Observations
- All 11 subjects with varying age change by exactly 1.0 year across their 360 recorded segments (likely reflecting recording sessions spanning a patient birthday or separate surgical encounters).
- The 17 subjects with varying BMI exhibit slight fluctuations in recorded height (0.1–1.2 cm) and weight (0.8–5.2 kg), resulting in BMI differences between 0.10 and 2.60 kg/m².


---

## 3. Demographic Rule Check: Subject `p003938_1`

Detailed breakdown across all 360 segments for subject `p003938_1`:

| Field | Min | Median | Max | Distinct Values | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Age** | 43.0 | 43.0 | 43.0 | 1 (`[43.0]`) | Completely invariant across all 360 segments |
| **Height (cm)** | 158.30 | 158.30 | 158.50 | 2 (`[158.3, 158.5]`) | Difference of 0.2 cm between recording episodes |
| **Weight (kg)** | 37.60 | 37.60 | 42.80 | 2 (`[37.60, 42.80]`) | Difference of 5.20 kg between recording episodes |
| **BMI ($kg/m^2$)** | 15.00 | 15.00 | 17.00 | 2 (`[15.0, 17.0]`) | Corresponds directly to the two (height, weight) tuples |

### Segment Distribution & Mathematical Consistency
Subject `p003938_1` has two distinct demographic states in the dataset:
1. **State 1 (299 segments)**:
   - Height: $158.3\text{ cm} = 1.583\text{ m}$
   - Weight: $37.60\text{ kg}$
   - Calculated $\text{BMI} = \frac{37.60}{(1.583)^2} = 15.005\text{ kg/m}^2 \approx 15.00$ (residual: $0.005$)
2. **State 2 (61 segments)**:
   - Height: $158.5\text{ cm} = 1.585\text{ m}$
   - Weight: $42.80\text{ kg}$
   - Calculated $\text{BMI} = \frac{42.80}{(1.585)^2} = 17.037\text{ kg/m}^2 \approx 17.00$ (residual: $0.037$)

### Clinical Plausibility
- **BMI Range**: Values between 15.0 and 17.0 indicate severe to moderate underweight ($\text{BMI} < 18.5$, with $< 16.0$ defined as severely underweight by WHO).
- **Clinical Context**: For an adult of height 158.3 cm (~5 ft 2 in), weights of 37.6 kg (82.9 lbs) and 42.8 kg (94.4 lbs) are clinically plausible for surgical/hospitalized patients suffering from conditions such as cachexia, chronic wasting, oncology, or nutritional deficit.
- **Consistency**: Height and weight are highly consistent with each other, and both states match the stored BMI within standard single-decimal rounding tolerance ($\le 0.037\text{ kg/m}^2$).
