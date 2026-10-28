# src/inspect_subset.py   Usage: python src/inspect_subset.py data/raw
import sys, os
from pathlib import Path
import numpy as np
import h5py

def chars(x):
    a = np.asarray(x).ravel()
    return "".join(chr(int(v)) for v in a[:20]) if a.size else ""

def describe(f, name, ds):
    mc = ds.attrs.get("MATLAB_class", b"")
    mc = mc.decode() if isinstance(mc, bytes) else mc
    lines = [f"  {name}: shape={ds.shape} dtype={ds.dtype} matlab_class={mc}"]
    if ds.size == 0:
        return lines
    try:
        if ds.dtype.kind == "O":   # array of references (e.g. one char per segment)
            n = ds.size
            first = f[ds[tuple(0 for _ in ds.shape)]]
            last = f[ds[tuple(s - 1 for s in ds.shape)]]
            lines.append(f"    references: {n}; first -> shape={first.shape} dtype={first.dtype} "
                         f"value={chars(first[()])!r}; last -> {chars(last[()])!r}")
        else:
            idx = tuple(slice(0, min(s, 3)) for s in ds.shape)
            s = np.asarray(ds[idx])
            lines.append(f"    first values: {s.ravel()[:6]}")
            if s.dtype.kind in "fiu":
                lines.append(f"    min/max of that sample: {np.nanmin(s):.4g} / {np.nanmax(s):.4g}")
    except Exception as e:
        lines.append(f"    (could not sample: {e})")
    return lines

def main(folder):
    Path("results").mkdir(exist_ok=True)
    out = []
    for p in sorted(Path(folder).glob("*.mat")):
        out.append("=" * 60)
        out.append(f"{p.name}  ({os.path.getsize(p)/1e9:.2f} GB)")
        try:
            with h5py.File(p, "r") as f:
                out.append(f"top-level keys: {[k for k in f.keys() if k != '#refs#']}")
                g = f["Subset"]
                for k in g.keys():
                    out += describe(f, k, g[k])
        except Exception as e:
            out.append(f"FAILED: {e}")
    text = "\n".join(out)
    Path("results/schema_raw.txt").write_text(text, encoding="utf-8")
    print(text)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "D:/data/raw")