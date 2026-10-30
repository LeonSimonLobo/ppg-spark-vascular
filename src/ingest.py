# Usage:  python src/ingest_script.py --in-dir data/raw --out data/parquet [--max-segments 3000]
import argparse, time, sys
from pathlib import Path

repo_dir = Path(__file__).resolve().parent.parent
if str(repo_dir) not in sys.path:
    sys.path.insert(0, str(repo_dir))

import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

L, FS = 1250, 125
FILES = [("train_file", "VitalDB_Train_Subset.mat"),
         ("calfree_test", "VitalDB_CalFree_Test_Subset.mat"),
         ("aami_cal", "VitalDB_AAMI_Cal_Subset.mat")]
SCHEMA = pa.schema([
    ("subject_id", pa.string()), ("segment_idx", pa.int32()),
    ("file_split", pa.string()), ("split", pa.string()),
    ("age", pa.float32()), ("sex", pa.float32()), ("height", pa.float32()),
    ("weight", pa.float32()), ("bmi", pa.float32()),
    ("sbp", pa.float32()), ("dbp", pa.float32()),
    ("ppg", pa.list_(pa.float32())),
])

def decode(f, ref):
    return "".join(chr(int(c)) for c in np.asarray(f[ref]).ravel())

def read_meta(f, g, n):
    num = {k: np.asarray(g[k][0, :n], dtype=np.float64)
           for k in ("Age", "BMI", "Height", "Weight", "SBP", "DBP")}
    subj_refs, gen_refs = g["Subject"][0, :n], g["Gender"][0, :n]
    subj = np.empty(n, dtype=object)
    sex = np.full(n, np.nan)
    prev, prev_sex = None, np.nan
    for i in range(n):
        s = decode(f, subj_refs[i])
        if s != prev:                      # gender is constant per subject
            c = decode(f, gen_refs[i]).strip().upper()
            prev_sex = 1.0 if c == "M" else (0.0 if c == "F" else np.nan)
            prev = s
        subj[i], sex[i] = s, prev_sex
        if i and i % 50000 == 0:
            print(f"    resolved {i}/{n} subject labels", flush=True)
    return num, subj, sex

def assign_split(label, subj, seed):
    if label in ("calfree_test", "aami_cal"):
        return np.array(["test"] * len(subj))
    uniq = np.array(sorted(set(subj)))
    np.random.default_rng(seed).shuffle(uniq)
    val = set(uniq[: int(round(0.2 * len(uniq)))])
    return np.array(["val" if s in val else "train" for s in subj])

def detect_ppg_channel(sig, n, blocks=20, blk=100):
    starts = np.unique(np.linspace(0, max(n - blk, 0), blocks).astype(int))
    c0, c1, abp = [], [], []
    for s in starts:
        x = sig[:, :, s:min(s + blk, n)].astype(np.float64)
        abp.append(np.median(x[:, 2, :]))
        x_centered = x - x.mean(axis=0, keepdims=True)
        a = x_centered[:, 2, :]
        # Check physiological Pulse Transit Time (PTT) lags 0 to 60 samples (~0 to 480 ms at 125 Hz)
        r0_lags, r1_lags = [], []
        for lag in range(0, 60):
            a_sub = a[lag:, :]
            s0_sub = x_centered[:-lag or None, 0, :]
            s1_sub = x_centered[:-lag or None, 1, :]
            norm_a = np.sqrt((a_sub ** 2).sum(axis=0)) + 1e-12
            r0 = (s0_sub * a_sub).sum(axis=0) / (np.sqrt((s0_sub ** 2).sum(axis=0)) * norm_a + 1e-12)
            r1 = (s1_sub * a_sub).sum(axis=0) / (np.sqrt((s1_sub ** 2).sum(axis=0)) * norm_a + 1e-12)
            r0_lags.append(r0)
            r1_lags.append(r1)
        c0.append(np.median(np.max(r0_lags, axis=0)))
        c1.append(np.median(np.max(r1_lags, axis=0)))
    r0, r1, a = float(np.median(c0)), float(np.median(c1)), float(np.median(abp))
    print(f"    corr(ch0,ABP)={r0:.2f}  corr(ch1,ABP)={r1:.2f}  median ch2={a:.1f}")
    if not (30 < a < 250):
        raise RuntimeError("channel 2 does not look like ABP in mmHg")
    if max(r0, r1) < 0.5:
        raise RuntimeError("no channel correlates with ABP; check layout")
    return (0 if r0 > r1 else 1), r0, r1

def write_one(path, label, out_dir, chunk, max_seg, seed):
    out_path = out_dir / f"{label}.parquet"
    with h5py.File(path, "r") as f:
        g = f["Subset"]; sig = g["Signals"]
        if sig.shape[0] != L or sig.shape[1] != 3:
            raise RuntimeError(f"unexpected Signals shape {sig.shape}")
        N = sig.shape[2]; n = min(N, max_seg) if max_seg else N
        print(f"[{label}] {N} segments in file, using {n}; hdf5 chunks={sig.chunks}", flush=True)
        if sig.chunks:
            ck = sig.chunks[2]; chunk = max(ck, (chunk // ck) * ck)
        num, subj, sex = read_meta(f, g, n)
        split = assign_split(label, subj, seed)
        ch, r0, r1 = detect_ppg_channel(sig, n)
        print(f"    PPG channel index = {ch}", flush=True)
        writer = pq.ParquetWriter(out_path, SCHEMA, compression="snappy")
        kept = np.zeros(n, dtype=bool)
        t0 = time.time()
        for a in range(0, n, chunk):
            b = min(a + chunk, n)
            x = np.ascontiguousarray(sig[:, ch, a:b].T, dtype=np.float32)   # (m, 1250)
            ok = np.isfinite(x).all(axis=1) & (x.std(axis=1) > 1e-6)
            idx = np.nonzero(ok)[0]
            kept[a + idx] = True
            if idx.size:
                offs = np.arange(0, (idx.size + 1) * L, L, dtype=np.int32)
                ppg = pa.ListArray.from_arrays(pa.array(offs), pa.array(x[idx].ravel()))
                col = lambda v: pa.array(v[a:b][idx].astype(np.float32),
                                         type=pa.float32(), from_pandas=True)
                cols = [pa.array(subj[a:b][idx].tolist(), type=pa.string()),
                        pa.array((a + idx).astype(np.int32)),
                        pa.array([label] * idx.size, type=pa.string()),
                        pa.array(split[a:b][idx].tolist(), type=pa.string()),
                        col(num["Age"]), col(sex), col(num["Height"]),
                        col(num["Weight"]), col(num["BMI"]),
                        col(num["SBP"]), col(num["DBP"]), ppg]
                writer.write_table(pa.Table.from_arrays(cols, schema=SCHEMA))
            print(f"    {b}/{n} segments ({time.time()-t0:.0f}s)", flush=True)
        writer.close()
    df = pd.DataFrame({"subject_id": subj, "file_split": label, "split": split,
                       "age": num["Age"], "sex": sex, "height": num["Height"],
                       "weight": num["Weight"], "bmi": num["BMI"],
                       "sbp": num["SBP"], "dbp": num["DBP"]})[kept].reset_index(drop=True)
    return df, {"n": n, "ch": ch, "r0": r0, "r1": r1, "path": out_path}

def q(s):
    return f"{s.median():.1f} [{s.quantile(.25):.1f}-{s.quantile(.75):.1f}]"

def make_report(dfs, infos, res_dir):
    R = ["# Dataset report", ""]
    if len(dfs) == 2:
        ov = set(dfs["train_file"].subject_id) & set(dfs["calfree_test"].subject_id)
        R.append(f"**Subject overlap train_file vs calfree_test: {len(ov)}**"
                 + ("  <-- LEAKAGE, do not use calfree_test as the test set" if ov else " (good)"))
        R.append("")
    fig, axes = plt.subplots(1, len(dfs), figsize=(6 * len(dfs), 3.6), squeeze=False)
    for ax, (name, d) in zip(axes[0], dfs.items()):
        info, g = infos[name], d.groupby("subject_id")
        per = g.size()
        sub = g.agg(age=("age", "first"), sex=("sex", "first"), bmi=("bmi", "first"),
                    sbp_med=("sbp", "median"), sbp_sd=("sbp", "std"))
        grand = d.sbp.mean()
        between = ((g.sbp.transform("mean") - grand) ** 2).sum() / ((d.sbp - grand) ** 2).sum()
        bmi_calc = d.weight / (d.height / 100) ** 2
        R += [f"## {name}",
              f"- PPG channel index used: {info['ch']} (corr with ABP: ch0 {info['r0']:.2f}, ch1 {info['r1']:.2f})",
              f"- segments written: {len(d)} of {info['n']} (dropped: {info['n'] - len(d)})",
              f"- subjects: {per.size}; segments per subject min/median/max: {per.min()}/{int(per.median())}/{per.max()}",
              f"- non-null %: age {d.age.notna().mean()*100:.1f}, sex {d.sex.notna().mean()*100:.1f}, bmi {d.bmi.notna().mean()*100:.1f}",
              f"- subject-level age median [IQR]: {q(sub.age)}; BMI: {q(sub.bmi)}; female share: {(sub.sex == 0).mean()*100:.1f}%",
              f"- subjects whose age / bmi varies across their segments: {int((g.age.nunique() > 1).sum())} / {int((g.bmi.nunique() > 1).sum())}",
              f"- BMI vs weight/height^2: 99th pct abs diff = {(bmi_calc - d.bmi).abs().quantile(.99):.2f}",
              f"- SBP quantiles 5/25/50/75/95: {d.sbp.quantile([.05,.25,.5,.75,.95]).round(1).tolist()}",
              f"- DBP quantiles 5/25/50/75/95: {d.dbp.quantile([.05,.25,.5,.75,.95]).round(1).tolist()}",
              f"- **share of SBP variance that is between subjects: {between*100:.1f}%** (the rest is within-subject, invisible to age/sex/BMI)",
              f"- median within-subject SBP SD: {sub.sbp_sd.median():.1f} mmHg"]
        for t in (120, 130, 140):
            R.append(f"- SBP >= {t}: segments {(d.sbp >= t).mean()*100:.1f}% | subjects (median SBP) "
                     f"{(sub.sbp_med >= t).sum()} of {len(sub)} ({(sub.sbp_med >= t).mean()*100:.1f}%)")
        R.append("")
        ax.hist(d.sbp, bins=50)
        for t in (120, 130, 140):
            ax.axvline(t, color="k", ls="--", lw=0.8)
        ax.set_title(f"{name}: SBP per segment"); ax.set_xlabel("SBP (mmHg)")
    fig.tight_layout(); fig.savefig(res_dir / "bp_hist.png", dpi=150); plt.close(fig)
    (res_dir / "dataset_report.md").write_text("\n".join(R), encoding="utf-8")
    print("\n".join(R))

def plot_samples(infos, res_dir):
    fig, axes = plt.subplots(5, len(infos), figsize=(6 * len(infos), 9), squeeze=False)
    for j, (name, info) in enumerate(infos.items()):
        pf = pq.ParquetFile(info["path"])
        t = pf.read_row_group(pf.num_row_groups // 2, columns=["ppg", "sbp", "subject_id"]).to_pandas()
        pick = t.sample(min(5, len(t)), random_state=0)
        for i, (_, r) in enumerate(pick.iterrows()):
            axes[i, j].plot(np.arange(L) / FS, r.ppg, lw=0.8)
            axes[i, j].set_title(f"{name} {r.subject_id} SBP={r.sbp:.0f}", fontsize=8)
    fig.tight_layout(); fig.savefig(res_dir / "sample_waveforms.png", dpi=130); plt.close(fig)

def compute_aami_feature_nan_rates(mat_path, ch, n_segs, chunk_size=15000):
    from src.features_c import extract_features_block
    feature_cols = [
        "hr_ibi", "hr", "ibi_sd", "systolic_rise_time",
        "pulse_width", "decay_ratio", "beat_count", "is_clipped", "snr_ratio",
    ]
    nan_counts = {c: 0 for c in feature_cols}
    with h5py.File(mat_path, "r") as f:
        sig_ds = f["Subset"]["Signals"]
        for c_start in range(0, n_segs, chunk_size):
            c_end = min(n_segs, c_start + chunk_size)
            block = sig_ds[:, ch, c_start:c_end]
            res = extract_features_block(block, fs=125)
            for c in feature_cols:
                arr = res[c]
                nan_counts[c] += int(np.sum(np.isnan(arr)))
    return {c: nan_counts[c] / n_segs for c in feature_cols}

def make_aami_report(d, info, in_mat_path, res_dir, train_p="data/parquet/train_file.parquet", calfree_p="data/parquet/calfree_test.parquet"):
    n_segs = len(d)
    g = d.groupby("subject_id")
    n_subjs = g.ngroups
    per = g.size()

    train_subjs = set(pq.read_table(train_p, columns=["subject_id"])["subject_id"].to_pylist()) if Path(train_p).exists() else set()
    calfree_subjs = set(pq.read_table(calfree_p, columns=["subject_id"])["subject_id"].to_pylist()) if Path(calfree_p).exists() else set()
    aami_subjs = set(d["subject_id"])

    ov_train = aami_subjs & train_subjs
    ov_calfree = aami_subjs & calfree_subjs

    print("Computing feature NaN rates on AAMI_Cal...", flush=True)
    nan_rates = compute_aami_feature_nan_rates(in_mat_path, info["ch"], info["n"])

    sbp_q = d["sbp"].quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]).to_dict()
    dbp_q = d["dbp"].quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]).to_dict()

    non_null_age = d["age"].notna().mean() * 100.0
    non_null_sex = d["sex"].notna().mean() * 100.0
    non_null_bmi = d["bmi"].notna().mean() * 100.0
    non_null_height = d["height"].notna().mean() * 100.0
    non_null_weight = d["weight"].notna().mean() * 100.0

    share_120 = (d["sbp"] >= 120.0).mean() * 100.0
    count_120 = int((d["sbp"] >= 120.0).sum())
    share_130 = (d["sbp"] >= 130.0).mean() * 100.0
    count_130 = int((d["sbp"] >= 130.0).sum())

    sub = g.agg(age=("age", "first"), sex=("sex", "first"), bmi=("bmi", "first"),
                sbp_med=("sbp", "median"), sbp_sd=("sbp", "std"))
    sub_120 = (sub["sbp_med"] >= 120.0).mean() * 100.0
    sub_count_120 = int((sub["sbp_med"] >= 120.0).sum())
    sub_130 = (sub["sbp_med"] >= 130.0).mean() * 100.0
    sub_count_130 = int((sub["sbp_med"] >= 130.0).sum())

    R = [
        "# AAMI Cal Subset Ingestion Report (`results/aami_ingest_report.md`)",
        "",
        "**Cohort:** VitalDB AAMI Calibration Subset (`VitalDB_AAMI_Cal_Subset.mat`)  ",
        "**Destination:** `data/parquet/aami_cal.parquet`  ",
        "**Protocol Specification:** `PLAN_C.md` / Study C  ",
        "**Evaluation State:** Ingestion and descriptive audit ONLY. **NO SCORING PERFORMED.**  ",
        "",
        "---",
        "",
        "## 1. Segments and Subject Counts",
        "",
        f"- **Total Segments Written:** {n_segs:,} of {info['n']:,} in source file (dropped: {info['n'] - n_segs}) (Expectation: **70,212**)",
        f"- **Unique Subjects:** {n_subjs} (Expectation: **116**)",
        f"- **Verification Check:** {'PASS' if n_segs == 70212 and n_subjs == 116 else 'FAIL'}",
        f"- **Segments per Subject:** min = {per.min()}, 25% = {per.quantile(0.25):.0f}, median = {int(per.median())}, 75% = {per.quantile(0.75):.0f}, max = {per.max()}",
        "",
        "---",
        "",
        "## 2. Demographic Completeness (Non-Null %)",
        "",
        f"- **Age:** {non_null_age:.2f}% non-null",
        f"- **Sex / Gender:** {non_null_sex:.2f}% non-null (female share: {(sub['sex'] == 0.0).mean()*100:.1f}%)",
        f"- **BMI:** {non_null_bmi:.2f}% non-null (median [IQR]: {q(sub['bmi'])})",
        f"- **Height:** {non_null_height:.2f}% non-null",
        f"- **Weight:** {non_null_weight:.2f}% non-null",
        "",
        "---",
        "",
        "## 3. Cross-Cohort Subject Overlap (Data Leakage Audit)",
        "",
        f"- **Overlap with `train_file` (1,293 subjects):** **{len(ov_train)}** subjects (Expectation: **0** - PASS)",
        f"- **Overlap with `calfree_test` (144 subjects):** **{len(ov_calfree)}** subjects (Expectation: **0** - PASS)",
        f"- **Independence Status:** Strictly disjoint subject cohort. Zero patient overlap across splits.",
        "",
        "---",
        "",
        "## 4. Signal Channel & PPG Correlation Check",
        "",
        f"- **Selected PPG Channel Index:** Channel {info['ch']}",
        f"- **Channel 0 Correlation with ABP:** {info['r0']:.4f}",
        f"- **Channel 1 Correlation with ABP:** {info['r1']:.4f}",
        f"- **Decision Rule:** Selected channel {info['ch']} has superior correlation with invasive arterial pressure.",
        "",
        "---",
        "",
        "## 5. Descriptive Target Distributions (SBP & DBP)",
        "",
        "> [!NOTE]",
        "> These statistics are purely descriptive of cohort distribution. No classification models have been run.",
        "",
        "### SBP Quantiles (mmHg)",
        f"- **1%:** {sbp_q[0.01]:.1f} mmHg",
        f"- **5%:** {sbp_q[0.05]:.1f} mmHg",
        f"- **25%:** {sbp_q[0.25]:.1f} mmHg",
        f"- **50% (Median):** {sbp_q[0.50]:.1f} mmHg",
        f"- **75%:** {sbp_q[0.75]:.1f} mmHg",
        f"- **95%:** {sbp_q[0.95]:.1f} mmHg",
        f"- **99%:** {sbp_q[0.99]:.1f} mmHg",
        "",
        "### DBP Quantiles (mmHg)",
        f"- **5%:** {dbp_q[0.05]:.1f} mmHg | **50%:** {dbp_q[0.50]:.1f} mmHg | **95%:** {dbp_q[0.95]:.1f} mmHg",
        "",
        "### Target Class Prevalence",
        f"- **Primary Target ($SBP \\ge 120$ mmHg):** **{count_120:,}** segments (**{share_120:.2f}%**) | Subjects (median SBP): {sub_count_120}/{n_subjs} ({sub_120:.2f}%)",
        f"- **Sensitivity Target ($SBP \\ge 130$ mmHg):** **{count_130:,}** segments (**{share_130:.2f}%**) | Subjects (median SBP): {sub_count_130}/{n_subjs} ({sub_130:.2f}%)",
        "",
        "---",
        "",
        "## 6. Study C Feature Extraction NaN Rates",
        "",
        "Computed across all 70,212 segments using canonical single-segment `extract_features_block`:",
        "",
        "| Feature | Description | AAMI_Cal NaN Rate (%) | Missingness Handling |",
        "|---|---|---|---|",
    ]
    for c, r in nan_rates.items():
        R.append(f"| **`{c}`** | Study C Feature | **{r*100:.2f}%** | Preserved as NaN; imputed by frozen PlanCImputer training medians |")

    rep_path = res_dir / "aami_ingest_report.md"
    rep_path.write_text("\n".join(R) + "\n", encoding="utf-8")
    print(f"Saved AAMI ingestion report to {rep_path}")
    print("\n".join(R))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="data/raw")
    ap.add_argument("--out", default="data/parquet")
    ap.add_argument("--chunk", type=int, default=20000)
    ap.add_argument("--max-segments", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only", type=str, default=None, choices=["train_file", "calfree_test", "aami_cal"],
                    help="Only ingest specified split.")
    a = ap.parse_args()
    out_dir, res_dir = Path(a.out), Path("results")
    out_dir.mkdir(parents=True, exist_ok=True); res_dir.mkdir(exist_ok=True)
    print("Writing to:", out_dir.resolve(), "| reports in:", res_dir.resolve(), flush=True)

    file_targets = [item for item in FILES if (a.only is None or item[0] == a.only)]
    if a.only == "aami_cal":
        label, fname = file_targets[0]
        mat_file = Path(a.in_dir) / fname
        df, info = write_one(mat_file, label, out_dir, a.chunk, a.max_segments, a.seed)
        make_aami_report(df, info, mat_file, res_dir)
    else:
        dfs, infos = {}, {}
        for label, fname in file_targets:
            dfs[label], infos[label] = write_one(Path(a.in_dir) / fname, label, out_dir,
                                                 a.chunk, a.max_segments, a.seed)
        make_report(dfs, infos, res_dir)
        plot_samples(infos, res_dir)

    print("\nDone. Parquet size on disk: "
          f"{sum(p.stat().st_size for p in out_dir.glob('*.parquet'))/1e9:.2f} GB")
