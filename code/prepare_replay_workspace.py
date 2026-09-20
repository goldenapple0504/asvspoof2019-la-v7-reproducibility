"""Lay out packaged derived inputs for the original V7 model code.

This prepares a separate working directory. It never edits the release data.
ASVspoof audio is not needed for the model-data-to-results replay.
"""

import argparse
import hashlib
import os
import shutil
from pathlib import Path


def link_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if dst.stat().st_size != src.stat().st_size:
            raise ValueError(f"Existing replay file differs: {dst}")
        def digest(path):
            h = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    h.update(chunk)
            return h.digest()
        if digest(src) != digest(dst):
            raise ValueError(f"Existing replay file differs: {dst}")
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--release", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--work", type=Path, required=True)
    a = p.parse_args()
    release, work = a.release.resolve(), a.work.resolve()
    if work == release or release in work.parents:
        raise ValueError("Working directory must be outside the release repository")
    data, code = release / "data", release / "code"
    v6, v7 = work / "rebuild_v6_trajectory", work / "rebuild_v7"
    mapping = {
        "trajectory_features_full.parquet": v6 / "extracted/trajectory_features_full.parquet",
        "flatness_cpp_trajectories.parquet": v6 / "cpp_flatness_extension/extracted/flatness_cpp_trajectories.parquet",
        "all6_scaling_parameters.tsv": v6 / "final_six_feature_trajectory/all6_scaling_parameters.tsv",
    }
    for name, target in mapping.items():
        link_or_copy(data / "inputs" / name, target)
    for src in (data / "trajectories").glob("*.parquet"):
        link_or_copy(src, v7 / "production/extraction/chunks" / src.name)
    for src in (data / "matching").glob("production_*matches.tsv"):
        link_or_copy(src, v7 / "matching" / src.name)
    for src in code.glob("*.py"):
        link_or_copy(src, v7 / "scripts" / src.name)
    for src in code.glob("*.R"):
        link_or_copy(src, v7 / "scripts" / src.name)
    marker = v7 / "FULL_V7_RUN_AUTHORIZED.txt"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("FULL_V7_RUN_AUTHORIZED\n", encoding="utf-8")
    print(f"project-root: {work}")
    print(f"output-root:  {v7}")
    print("Run the model-data assembly, model fitting, inference and meta-analysis commands from README.md.")


if __name__ == "__main__":
    main()
