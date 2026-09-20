"""Prepare the recorded RQ1 post-estimation script after V7 model fitting."""

import argparse
from pathlib import Path

from prepare_replay_workspace import link_or_copy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--release", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--work", type=Path, required=True)
    a = p.parse_args()
    release, work = a.release.resolve(), a.work.resolve()
    package = work / "rq1_postestimation"
    source_models = work / "rebuild_v7/production/models"
    files = sorted(source_models.glob("A??_*.rds"))
    if len(files) != 60:
        raise ValueError(f"Expected 60 fitted V7 model bundles, found {len(files)}")
    for src in files:
        link_or_copy(src, package / "models" / src.name)
    link_or_copy(release / "code/v7_model_functions.R", package / "code/v7_model_functions.R")
    link_or_copy(release / "data/results/rq1/MODEL_INDEX.tsv", package / "MODEL_INDEX.tsv")
    link_or_copy(work / "rebuild_v7/production/inference/time_resolved_specificity.tsv",
                 package / "reference_inference/time_resolved_specificity.tsv")
    print(package)


if __name__ == "__main__":
    main()
