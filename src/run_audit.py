"""Run prospective control and amplitude scale audit, then format the markdown report."""

import time
import pyarrow.parquet as pq
from src.prospective_and_scale_w import (
    run_prospective_prefix_control,
)
from src.generate_prospective_report import format_prospective_and_scale_report

def main():
    t0 = time.time()
    print("Loading features_w_train.parquet...", flush=True)
    df = pq.read_table("data/interim/features_w_train.parquet").to_pandas()
    print(f"Loaded {len(df):,} rows in {time.time()-t0:.2f}s.", flush=True)

    print("Running prospective prefix-centering control (prefix_len=60, 2000 bootstraps)...", flush=True)
    t1 = time.time()
    prefix_res = run_prospective_prefix_control(df, prefix_len=60)
    print(f"Prefix control finished in {time.time()-t1:.2f}s.", flush=True)

    for target in ["target_prefix_baseline", "target_session_median"]:
        t_res = prefix_res[target]
        print(f"\n=== {target} ===", flush=True)
        print(f"Prevalence: {t_res['prevalence']:.2f}%", flush=True)
        for m in ["hr", "morphology", "joint", "residualized"]:
            auc = t_res[m]["auroc"]
            ci = t_res[m]["ci_95"]
            print(f"  {m:15s}: AUROC = {auc:.4f} [95% CI: {ci[0]:.4f}, {ci[1]:.4f}]", flush=True)
        for pair, p_data in t_res["paired_differences"].items():
            diff = p_data["diff"]
            ci = p_data["ci_95"]
            print(f"  Delta {pair:22s}: {diff:+.4f} [95% CI: {ci[0]:+.4f}, {ci[1]:+.4f}]", flush=True)

    print("\nFormatting and saving report to results/dev_W_prospective_and_scale.md...", flush=True)
    format_prospective_and_scale_report(prefix_res, {}, output_path="results/dev_W_prospective_and_scale.md")
    print("Done!", flush=True)

if __name__ == "__main__":
    main()
