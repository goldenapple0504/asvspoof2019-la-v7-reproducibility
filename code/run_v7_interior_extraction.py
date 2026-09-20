"""Checkpointed production Interior extraction derived from the validated benchmark.

Dry-run mode validates and freezes the plan but performs no acoustic extraction.
Full execution is doubly locked by a command-line flag and authorization file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from run_v7_benchmark import extract_audio_group
from v7_common import SYSTEMS, atomic_tsv, sha256


AUTHORIZATION_TEXT = "FULL_V7_RUN_AUTHORIZED"
EXPECTED_TOKENS = 482_509
DEFAULT_FILES_PER_CHUNK = 250


def require_authorization(root: Path, execute_full: bool) -> None:
    marker = root / "FULL_V7_RUN_AUTHORIZED.txt"
    if not execute_full:
        raise RuntimeError("Full extraction requires --execute-full; dry-run is the only unlocked mode")
    if not marker.is_file() or marker.read_text(encoding="utf-8").strip() != AUTHORIZATION_TEXT:
        raise RuntimeError("Full V7 authorization marker is absent or invalid")


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def load_population(project: Path, output: Path) -> pd.DataFrame:
    population = pd.read_csv(
        project / "rebuild_v6_trajectory/extracted/trajectory_eligible_tokens.tsv",
        sep="\t", low_memory=False,
    )
    eligibility = pd.read_csv(output / "diagnostics/interior_eligibility.tsv", sep="\t", low_memory=False)
    if len(population) != EXPECTED_TOKENS or population.token_id.nunique() != EXPECTED_TOKENS:
        raise RuntimeError("Official V6 population identity/count gate failed")
    if len(eligibility) != EXPECTED_TOKENS or not eligibility.both_interiors_eligible.all():
        raise RuntimeError("Full short-frame Interior eligibility gate failed")
    meta = eligibility[[
        "token_id", "midpoint_A_sec", "midpoint_B_sec", "phone_A_start_sec", "phone_A_end_sec",
        "phone_B_start_sec", "phone_B_end_sec", "audio_frames", "sample_rate", "audio_path",
    ]]
    data = population.merge(meta, on="token_id", how="inner", validate="one_to_one", suffixes=("", "_v7"))
    for name in meta.columns:
        alt = name + "_v7"
        if alt in data:
            data[name] = data[alt]
            data.drop(columns=alt, inplace=True)
    data["analysis_family"] = data["partition"]
    data["match_side"] = np.where(data.system.eq("bonafide"), "bonafide", "tts")
    data["exact_phone_pair"] = data.get("phone_pair", data.phone_A.astype(str) + "->" + data.phone_B.astype(str))
    cpps_eligibility = pd.read_parquet(
        project / "rebuild_v6_trajectory/cpp_flatness_extension/extracted/feature_token_eligibility.parquet",
        filters=[("feature", "==", "cpps")], columns=["token_id", "eligibility", "failure_reason"],
    ).rename(columns={"eligibility": "boundary_cpps_eligible", "failure_reason": "boundary_cpps_failure_reason"})
    data = data.merge(cpps_eligibility, on="token_id", how="left", validate="one_to_one")
    if data.boundary_cpps_eligible.isna().any():
        raise RuntimeError("Official Boundary CPPS eligibility is incomplete")
    # The validated benchmark extractor needs only a status and finite sentinel
    # to decide whether the official Boundary CPPS operator is valid. The raw
    # Boundary value itself is never copied into the Interior output.
    data["cpp_status"] = np.where(data.boundary_cpps_eligible, "ok", "ineligible")
    data["boundary_cpp_db"] = np.where(data.boundary_cpps_eligible, 0.0, np.nan)
    required = [
        "token_id", "file_id", "analysis_family", "system", "match_side", "phone_A", "phone_B",
        "exact_phone_pair", "boundary_class", "boundary_time", "A_duration_ms", "B_duration_ms",
        "audio_path", "audio_frames", "sample_rate", "phone_A_start_sec", "phone_A_end_sec",
        "phone_B_start_sec", "phone_B_end_sec", "midpoint_A_sec", "midpoint_B_sec", "cpp_status",
        "boundary_cpp_db",
    ]
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise RuntimeError(f"Population lacks production extractor columns: {missing}")
    return data[required].sort_values(["audio_path", "token_id"]).reset_index(drop=True)


def canonical_membership_hash(frame: pd.DataFrame) -> str:
    canonical = frame[["token_id", "audio_path"]].drop_duplicates().sort_values(["token_id", "audio_path"])
    payload = canonical.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def assign_chunk_membership(population: pd.DataFrame, files_per_chunk: int) -> pd.DataFrame:
    paths = sorted(population.audio_path.unique())
    mapping = {path: index // files_per_chunk for index, path in enumerate(paths)}
    plan = population[["token_id", "audio_path"]].copy()
    plan["chunk_id"] = plan.audio_path.map(mapping).astype(int)
    return plan


def make_plan(membership: pd.DataFrame) -> pd.DataFrame:
    summary = membership.groupby("chunk_id", as_index=False).agg(
        audio_files=("audio_path", "nunique"), tokens=("token_id", "nunique"),
        first_audio_path=("audio_path", "min"), last_audio_path=("audio_path", "max"),
    )
    hashes = membership.groupby("chunk_id").apply(canonical_membership_hash, include_groups=False).rename("membership_sha256")
    return summary.merge(hashes, on="chunk_id", validate="one_to_one")


def classify_exclusions(failures: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if failures.empty:
        columns = ["token_id", "region", "feature", "exclusion_reason"]
        return pd.DataFrame(columns=columns), pd.DataFrame(columns=["token_id", "region", "feature", "error"])
    expected_reason = failures.failure_reason.astype(str).str.startswith((
        "official_v6_boundary_cpps_invalid_or_incomplete_support",
        "incomplete_strict_phone_support_for_nine_point_cpps_trajectory",
        "nonfinite_or_incomplete_cpps_trajectory",
        "cpps:ValueError:local_native_geometry:",
    ))
    expected = failures[failures.feature.eq("cpps") & expected_reason].copy()
    expected = expected.rename(columns={"failure_reason": "exclusion_reason"})
    expected["classification"] = "expected_cpps_complete_three_region_validity_exclusion"
    unexpected = failures.drop(expected.index).copy().rename(columns={"failure_reason": "error"})
    return expected, unexpected


def checkpoint_valid(checkpoint: Path, paths: list[Path]) -> bool:
    if not checkpoint.is_file() or not all(path.is_file() for path in paths):
        return False
    record = json.loads(checkpoint.read_text(encoding="utf-8"))
    return all(record.get("sha256", {}).get(path.name) == sha256(path) for path in paths)


def write_checkpoint(path: Path, record: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def execute_chunk(project: Path, output: Path, population: pd.DataFrame, chunk_id: int, workers: int,
                  planned_membership_sha256: str) -> dict:
    base = output / "production/extraction"
    feature_path = base / "chunks" / f"interior_features_{chunk_id:05d}.parquet"
    validity_path = base / "validity" / f"region_validity_{chunk_id:05d}.tsv"
    exclusion_path = base / "cpps_validity_exclusions" / f"cpps_validity_exclusions_{chunk_id:05d}.tsv"
    error_path = base / "unexpected_errors" / f"unexpected_errors_{chunk_id:05d}.tsv"
    checkpoint = base / "checkpoints" / f"chunk_{chunk_id:05d}.json"
    targets = [feature_path, validity_path, exclusion_path, error_path]
    if checkpoint_valid(checkpoint, targets):
        return {"chunk_id": chunk_id, "status": "CHECKPOINT_SKIP"}
    for directory in {path.parent for path in targets + [checkpoint]}:
        directory.mkdir(parents=True, exist_ok=True)
    groups = [(str(project), group.to_dict(orient="records")) for _, group in population.groupby("audio_path", sort=True)]
    rows: list[dict] = []
    failures: list[dict] = []
    qc: list[dict] = []
    began = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for extracted, failed, region_qc in executor.map(extract_audio_group, groups, chunksize=1):
            rows.extend(extracted)
            failures.extend(failed)
            qc.extend(region_qc)
    data = pd.DataFrame(rows)
    data = data[data.region.isin(["Interior_A", "Interior_B"])].copy()
    qc_frame = pd.DataFrame(qc)
    cpps = data[data.feature.eq("cpps")]
    cpps_gate = cpps.groupby(["source_token_id", "region"]).raw_value.agg(
        rows="size", all_finite=lambda values: bool(np.isfinite(values).all())
    )
    invalid_cpps = cpps_gate[(cpps_gate.rows.ne(9)) | (~cpps_gate.all_finite)]
    for token_id, region in invalid_cpps.index:
        failures.append({"token_id": token_id, "region": region, "feature": "cpps",
                         "failure_reason": "nonfinite_or_incomplete_cpps_trajectory"})
        data = data[~(data.source_token_id.eq(token_id) & data.region.eq(region) & data.feature.eq("cpps"))]
        mask = qc_frame.token_id.eq(token_id) & qc_frame.region.eq(region)
        qc_frame.loc[mask, "cpps_trajectory_valid"] = False
        qc_frame.loc[mask, "cpps_failure_reason"] = "nonfinite_or_incomplete_cpps_trajectory"
    scaling = pd.read_csv(
        project / "rebuild_v6_trajectory/final_six_feature_trajectory/all6_scaling_parameters.tsv", sep="\t"
    )
    data = data.merge(scaling[["feature", "feature_mean", "feature_sd"]], on="feature", validate="many_to_one")
    data["feature_z"] = (data.raw_value - data.feature_mean) / data.feature_sd
    if data.raw_value.isna().any() or not np.isfinite(data.raw_value).all():
        raise RuntimeError(f"Chunk {chunk_id} contains non-finite extracted values")
    expected, unexpected = classify_exclusions(pd.DataFrame(failures))
    if not unexpected.empty:
        atomic_tsv(unexpected, error_path)
        raise RuntimeError(f"Chunk {chunk_id} has {len(unexpected)} unexpected extraction errors")
    atomic_parquet(data, feature_path)
    atomic_tsv(qc_frame, validity_path)
    atomic_tsv(expected, exclusion_path)
    atomic_tsv(unexpected, error_path)
    record = {
        "chunk_id": chunk_id, "status": "PASS", "tokens": int(population.token_id.nunique()),
        "audio_files": int(population.audio_path.nunique()), "rows": int(len(data)),
        "cpps_validity_exclusions": int(len(expected)), "unexpected_errors": 0,
        "elapsed_seconds": time.perf_counter() - began,
        "membership_sha256": canonical_membership_hash(population),
        "planned_membership_sha256": planned_membership_sha256,
        "sha256": {path.name: sha256(path) for path in targets},
    }
    if record["membership_sha256"] != planned_membership_sha256:
        raise RuntimeError(f"Chunk {chunk_id} membership differs from frozen plan")
    write_checkpoint(checkpoint, record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute-full", action="store_true")
    parser.add_argument("--workers", type=int, default=max(1, min(12, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--files-per-chunk", type=int, default=DEFAULT_FILES_PER_CHUNK)
    args = parser.parse_args()
    project, output = args.project_root.resolve(), args.output_root.resolve()
    if args.dry_run == args.execute_full:
        raise RuntimeError("Select exactly one of --dry-run or --execute-full")
    population = load_population(project, output)
    membership = assign_chunk_membership(population, args.files_per_chunk)
    plan = make_plan(membership)
    dry = output / "diagnostics/production_dry_run"
    dry.mkdir(parents=True, exist_ok=True)
    atomic_tsv(plan, dry / "extraction_chunk_plan.tsv")
    plan_record = {
        "status": "DRY_RUN_PLAN_ONLY" if args.dry_run else "AUTHORIZED_FULL_EXECUTION",
        "tokens": int(population.token_id.nunique()), "audio_files": int(population.audio_path.nunique()),
        "chunks": int(len(plan)), "files_per_chunk": args.files_per_chunk, "workers": args.workers,
        "planned_chunk_ids": plan.chunk_id.astype(int).tolist(),
        "membership_sha256": canonical_membership_hash(membership),
        "full_extraction_executed": False,
    }
    (dry / "extraction_plan.json").write_text(json.dumps(plan_record, indent=2), encoding="utf-8")
    if args.dry_run:
        print(json.dumps(plan_record, indent=2))
        return
    require_authorization(output, args.execute_full)
    for chunk_id in plan.chunk_id.astype(int):
        ids = set(membership.loc[membership.chunk_id.eq(chunk_id), "token_id"])
        planned_hash = plan.loc[plan.chunk_id.eq(chunk_id), "membership_sha256"].iloc[0]
        record = execute_chunk(project, output, population[population.token_id.isin(ids)], chunk_id, args.workers, planned_hash)
        print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()
