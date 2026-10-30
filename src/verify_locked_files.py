"""Verify locked files manifest against current file hashes."""

import hashlib
from pathlib import Path


def verify_locked_manifest(manifest_path="results/locked_files_manifest.csv"):
    repo_dir = Path(__file__).resolve().parent.parent
    mp = repo_dir / manifest_path
    if not mp.exists():
        raise FileNotFoundError(f"Manifest not found: {mp}")

    lines = mp.read_text(encoding="utf-8").strip().splitlines()
    header = lines[0]
    rows = lines[1:]

    mismatches = []
    checked = []

    for r in rows:
        if not r.strip():
            continue
        path_str, expected_sha, expected_sz = r.split(",")
        p = repo_dir / path_str
        if not p.exists():
            mismatches.append((path_str, "MISSING", expected_sha))
            continue

        data = p.read_bytes()
        actual_sha = hashlib.sha256(data).hexdigest()
        actual_sz = len(data)

        if actual_sha != expected_sha or actual_sz != int(expected_sz):
            mismatches.append((path_str, actual_sha, expected_sha))
        else:
            checked.append((path_str, actual_sha))

    print(f"Verified {len(checked)} locked files.")
    if mismatches:
        print(f"ERROR: {len(mismatches)} MISMATCHES FOUND:")
        for path_str, actual, expected in mismatches:
            print(f"  {path_str}: actual={actual} != expected={expected}")
        return False, checked, mismatches
    else:
        print("ALL LOCKED FILES MATCH MANIFEST BIT-FOR-BIT!")
        return True, checked, []


if __name__ == "__main__":
    success, checked, mismatches = verify_locked_manifest()
    assert success, "Locked files verification failed!"
