"""Package the large V7 data files as GitHub release assets.

Run from the repository root after `git lfs pull`:
    python make_release_assets.py
Creates release_assets/ with three zip files and RELEASE_ASSETS_SHA256.tsv.
"""
import glob
import hashlib
import os
import zipfile

EXPECTED_COUNTS = {"v7_inputs.zip": 4, "v7_trajectories.zip": 146, "v7_textgrids.zip": 1}
GROUPS = {
    "v7_inputs.zip": sorted(glob.glob("data/inputs/*.parquet"))
    + ["data/inputs/trajectory_eligible_tokens.tsv"],
    "v7_trajectories.zip": sorted(glob.glob("data/trajectories/*.parquet")),
    "v7_textgrids.zip": ["data/textgrids/v7_textgrids.zip"],
}

os.makedirs("release_assets", exist_ok=True)
rows = []
for name, files in GROUPS.items():
    if len(files) != EXPECTED_COUNTS[name] or len(set(files)) != len(files):
        raise SystemExit(f"{name}: expected {EXPECTED_COUNTS[name]} distinct source files, got {len(files)}")
    for f in files:
        with open(f, "rb") as fh:
            if fh.read(40).startswith(b"version https://git-lfs"):
                raise SystemExit(f"{f} is an LFS pointer; run `git lfs pull` first.")
    out = os.path.join("release_assets", name)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
        for f in files:
            z.write(f, arcname=f.replace(os.sep, "/"))
    size = os.path.getsize(out)
    if size >= 2 * 1024**3:
        raise SystemExit(f"{name} is over the 2 GiB release-asset limit.")
    h = hashlib.sha256()
    with open(out, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    rows.append(f"{name}\t{size}\t{h.hexdigest()}")
    print(f"{name}: {len(files)} files, {size / 1e6:.0f} MB")

with open(os.path.join("release_assets", "RELEASE_ASSETS_SHA256.tsv"), "w", newline="\n") as fh:
    fh.write("asset\tbytes\tsha256\n" + "\n".join(rows) + "\n")
