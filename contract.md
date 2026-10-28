# Data contract

## data/parquet/{train_file,calfree_test}.parquet (src/ingest.py)

subject_id string, segment_idx int32, file_split string, split string (train|val|test),
age, sex (1=M,0=F), height (cm), weight (kg), bmi, sbp, dbp: float32,
ppg: list<float32> (1250 samples, 125 Hz, PPG only, per-segment normalized)

## data/features/\*.parquet (src/spark_features.py): one row per segment, NO ppg

meta: subject_id, segment_idx, file_split, split, age, sex, height, weight, bmi, sbp, dbp
B1 (double): hr_bpm, rise_time_s, crest_time_ratio, pw50_s, area_ratio, max_upslope,
beat_skew, beat_kurt, ibi_sd
B2 extra (double): sdppg_a, sdppg_b, sdppg_c, sdppg_d, sdppg_e, b_a, c_a, d_a, e_a,
aging_index, notch_time_s, notch_rel_amp, diast_peak_time_s,
diast_peak_rel_amp, reflection_index, delta_t_s
quality: sqi_template_corr (double), n_valid_beats (int), frac_valid_beats (double)
Failed extraction = NaN features, never a dropped row.
