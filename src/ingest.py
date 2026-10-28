# Usage:  python src/ingest_script.py --in-dir data/raw --out data/parquet [--max-segments 3000]
import argparse, time
from pathlib import Path
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
         ("calfree_test", "VitalDB_CalFree_Test_Subset.mat")]
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
    if label == "calfree_test":
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

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="data/raw")
    ap.add_argument("--out", default="data/parquet")
    ap.add_argument("--chunk", type=int, default=20000)
    ap.add_argument("--max-segments", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    out_dir, res_dir = Path(a.out), Path("results")
    out_dir.mkdir(parents=True, exist_ok=True); res_dir.mkdir(exist_ok=True)
    print("Writing to:", out_dir.resolve(), "| reports in:", res_dir.resolve(), flush=True)
    dfs, infos = {}, {}
    for label, fname in FILES:
        dfs[label], infos[label] = write_one(Path(a.in_dir) / fname, label, out_dir,
                                             a.chunk, a.max_segments, a.seed)
    make_report(dfs, infos, res_dir)
    plot_samples(infos, res_dir)
    print("\nDone. Parquet size on disk: "
          f"{sum(p.stat().st_size for p in out_dir.glob('*.parquet'))/1e9:.2f} GB")
