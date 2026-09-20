"""Assemble checkpointed per-system/per-feature V7 model inputs after matching."""

from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from run_v7_interior_extraction import AUTHORIZATION_TEXT
from v7_common import FLUX_TIMES_MS, FRAME_TIMES_MS, SYSTEMS, atomic_tsv, sha256


FEATURES = ("energy", "centroid", "tilt", "flux", "flatness", "cpps")
SHORT_FEATURES = set(FEATURES) - {"cpps"}


def require_authorization(output: Path, execute_full: bool) -> None:
    marker = output / "FULL_V7_RUN_AUTHORIZED.txt"
    if not execute_full or not marker.is_file() or marker.read_text(encoding="utf-8").strip() != AUTHORIZATION_TEXT:
        raise RuntimeError("Model-data assembly requires full-run authorization")


def atomic_gzip_tsv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", newline="") as handle:
        frame.to_csv(handle, sep="\t", index=False, na_rep="")
    os.replace(temporary, path)


def read_filtered(dataset_path: Path, token_column: str, tokens: set[str], feature: str) -> pd.DataFrame:
    dataset = ds.dataset(str(dataset_path), format="parquet")
    expression = ds.field(token_column).isin(sorted(tokens)) & (ds.field("feature") == feature)
    columns = [token_column, "file_id", "utterance_id", "partition", "system", "phone_A", "phone_B",
               "boundary_class", "region", "relative_time_ms", "feature", "raw_value"]
    available = set(dataset.schema.names)
    columns = [column for column in columns if column in available]
    return dataset.to_table(filter=expression, columns=columns).to_pandas()


def expected_times(feature: str) -> list[int]:
    return (FLUX_TIMES_MS if feature == "flux" else FRAME_TIMES_MS).astype(int).tolist()


def validate_model_data(data: pd.DataFrame, system: str, feature: str) -> dict:
    times = expected_times(feature)
    counts = data.groupby(["match_id", "match_side", "region"]).relative_time_ms.agg(list)
    complete = counts.map(lambda values: sorted(map(int, values)) == times)
    matches = data[["match_id", "match_side", "source_token_id", "exact_phone_pair"]].drop_duplicates()
    match_gate = matches.groupby("match_id").agg(rows=("match_side", "size"), sides=("match_side", "nunique"),
                                                  pairs=("exact_phone_pair", "nunique"))
    return {
        "system": system, "feature": feature, "observations": int(len(data)),
        "matched_sets": int(data.match_id.nunique()), "token_occurrences": int(data.token_occurrence_id.nunique()),
        "regions": int(data.region.nunique()), "time_points": len(times),
        "all_grids_complete": bool(complete.all()), "all_values_finite": bool(np.isfinite(data.feature_z).all()),
        "pair_integrity": bool(((match_gate.rows == 2) & (match_gate.sides == 2) & (match_gate.pairs == 1)).all()),
        "no_duplicate_rows": bool(not data.duplicated(["match_id", "match_side", "region", "relative_time_ms"]).any()),
    }


def assemble_one(project: Path, output: Path, system: str, feature: str) -> tuple[pd.DataFrame, dict]:
    matching_file = output / "matching" / (
        "production_cpps_matches.tsv" if feature == "cpps" else
        "production_flatness_matches.tsv" if feature == "flatness" else
        "production_short_frame_matches.tsv"
    )
    matched = pd.read_csv(matching_file, sep="\t", low_memory=False)
    matched = matched[matched.target_system.eq(system)].copy()
    if matched.empty:
        raise RuntimeError(f"No matching rows for {system}/{feature}")
    tokens = set(matched.token_id.astype(str))
    boundary_path = project / "rebuild_v6_trajectory" / (
        "extracted/trajectory_features_full.parquet" if feature in {"energy", "centroid", "tilt", "flux"}
        else "cpp_flatness_extension/extracted/flatness_cpp_trajectories.parquet"
    )
    boundary = read_filtered(boundary_path, "token_id", tokens, feature).rename(columns={"token_id": "source_token_id"})
    boundary["region"] = "Boundary"
    interior_path = output / "production/extraction/chunks"
    interior = read_filtered(interior_path, "source_token_id", tokens, feature)
    interior = interior[interior.region.isin(["Interior_A", "Interior_B"])].copy()
    raw = pd.concat([boundary, interior], ignore_index=True, sort=False)
    scaling = pd.read_csv(
        project / "rebuild_v6_trajectory/final_six_feature_trajectory/all6_scaling_parameters.tsv", sep="\t"
    ).set_index("feature").loc[feature]
    raw["feature_z"] = (raw.raw_value - float(scaling.feature_mean)) / float(scaling.feature_sd)
    metadata = matched[["token_id", "target_system", "match_id", "match_side", "exact_phone_pair",
                        "A_duration_ms", "B_duration_ms"]].drop_duplicates()
    data = metadata.merge(raw, left_on="token_id", right_on="source_token_id", how="inner", validate="one_to_many")
    data["authenticity"] = pd.Categorical(np.where(data.match_side.eq("bonafide"), "BF", "TTS"), ["BF", "TTS"])
    data["region"] = pd.Categorical(data.region, ["Boundary", "Interior_A", "Interior_B"])
    data["token_occurrence_id"] = system + "::" + data.source_token_id.astype(str)
    data["utterance_id"] = data.get("utterance_id", data.file_id).fillna(data.file_id).astype(str)
    data["duration_A_c"] = data.A_duration_ms - data.A_duration_ms.mean()
    data["duration_B_c"] = data.B_duration_ms - data.B_duration_ms.mean()
    data["feature_mean"] = float(scaling.feature_mean)
    data["feature_sd"] = float(scaling.feature_sd)
    columns = ["target_system", "feature", "partition", "match_id", "match_side", "source_token_id",
               "token_occurrence_id", "utterance_id", "authenticity", "region", "relative_time_ms",
               "raw_value", "feature_z", "duration_A_c", "duration_B_c", "A_duration_ms", "B_duration_ms",
               "exact_phone_pair", "phone_A", "phone_B", "boundary_class", "feature_mean", "feature_sd"]
    data = data[columns].sort_values(["match_id", "match_side", "region", "relative_time_ms"]).reset_index(drop=True)
    gate = validate_model_data(data, system, feature)
    gate["all_gates_pass"] = all(gate[key] for key in (
        "all_grids_complete", "all_values_finite", "pair_integrity", "no_duplicate_rows"
    )) and gate["regions"] == 3
    return data, gate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--execute-full", action="store_true")
    args = parser.parse_args()
    project, output = args.project_root.resolve(), args.output_root.resolve()
    require_authorization(output, args.execute_full)
    gates = []
    for system in SYSTEMS:
        for feature in FEATURES:
            destination = output / "production/model_data" / f"{system}_{feature}.tsv.gz"
            checkpoint = destination.with_suffix(".checkpoint.json")
            if destination.is_file() and checkpoint.is_file():
                record = json.loads(checkpoint.read_text(encoding="utf-8"))
                if record.get("sha256") == sha256(destination) and record.get("all_gates_pass") is True:
                    gates.append(record)
                    continue
            data, gate = assemble_one(project, output, system, feature)
            if not gate["all_gates_pass"]:
                atomic_tsv(pd.DataFrame([gate]), output / "production/model_data" / f"{system}_{feature}_FAILED.tsv")
                raise RuntimeError(f"Model-data gate failed for {system}/{feature}")
            atomic_gzip_tsv(data, destination)
            gate["sha256"] = sha256(destination)
            checkpoint.write_text(json.dumps(gate, indent=2, sort_keys=True), encoding="utf-8")
            gates.append(gate)
    audit = pd.DataFrame(gates)
    atomic_tsv(audit, output / "production/model_data/model_data_gates.tsv")
    if len(audit) != 60 or not audit.all_gates_pass.all():
        raise RuntimeError("Expected 60 production model-data checkpoints passing every gate")


if __name__ == "__main__":
    main()
