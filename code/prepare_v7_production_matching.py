"""Prepare separate short-frame and CPPS production matching with hard gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from run_v7_interior_extraction import AUTHORIZATION_TEXT
from v7_common import SYSTEMS, atomic_tsv


SMD_GATE = 0.25
SHORT_FEATURES = ("energy", "centroid", "tilt", "flux", "flatness")


def canonical_pairs(frame: pd.DataFrame, pair_column: str) -> pd.DataFrame:
    required = {"target_system", "match_id", "match_side", "token_id", pair_column, "A_duration_ms", "B_duration_ms"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"Matching table missing canonical columns: {sorted(missing)}")
    keys = ["target_system", "match_id"]
    counts = frame.groupby(keys, sort=False).agg(rows=("match_side", "size"), sides=("match_side", "nunique"))
    if not ((counts.rows.eq(2)) & (counts.sides.eq(2))).all() or set(frame.match_side) != {"bonafide", "tts"}:
        raise RuntimeError("Incomplete or duplicate matching pair")
    columns = keys + ["token_id", pair_column, "A_duration_ms", "B_duration_ms"]
    bf = frame.loc[frame.match_side.eq("bonafide"), columns].copy()
    tt = frame.loc[frame.match_side.eq("tts"), columns].copy()
    pairs = bf.merge(tt, on=keys, how="inner", validate="one_to_one", suffixes=("_bf", "_tts"))
    if not pairs[f"{pair_column}_bf"].equals(pairs[f"{pair_column}_tts"]):
        raise RuntimeError("Exact-pair mismatch within match")
    result = pd.DataFrame({
        "target_system": pairs.target_system,
        "bonafide_token_id": pairs.token_id_bf,
        "tts_token_id": pairs.token_id_tts,
        "exact_phone_pair": pairs[f"{pair_column}_bf"],
        "bonafide_duration_A_ms": pairs.A_duration_ms_bf.astype(float),
        "bonafide_duration_B_ms": pairs.B_duration_ms_bf.astype(float),
        "tts_duration_A_ms": pairs.A_duration_ms_tts.astype(float),
        "tts_duration_B_ms": pairs.B_duration_ms_tts.astype(float),
    })
    return result.sort_values(["target_system", "bonafide_token_id", "tts_token_id"]).reset_index(drop=True)


def canonical_frame_hash(frame: pd.DataFrame) -> str:
    payload = frame.to_csv(index=False, lineterminator="\n", float_format="%.17g").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def flatness_matching_identity(project: Path) -> dict:
    short = pd.read_csv(project / "rebuild_v6_trajectory/matching/trajectory_matched_tokens.tsv", sep="\t", low_memory=False)
    short = short[(short.window_ms_per_side.eq(20)) & short.matching_set.eq("non_cpp")].copy()
    extension = pd.read_csv(
        project / "rebuild_v6_trajectory/cpp_flatness_extension/matching/extension_matched_tokens.tsv",
        sep="\t", low_memory=False,
    )
    flatness = extension[extension.feature.eq("flatness")].copy()
    short_pairs = canonical_pairs(short, "exact_phone_pair")
    flatness_pairs = canonical_pairs(flatness, "phone_pair")
    short_hash, flatness_hash = canonical_frame_hash(short_pairs), canonical_frame_hash(flatness_pairs)
    identity = short_pairs.equals(flatness_pairs)
    return {
        "gate": "official_V6_flatness_matching_lineage_identity",
        "pass": bool(identity and short_hash == flatness_hash),
        "tables_identical": bool(identity and short_hash == flatness_hash),
        "lineage_resolution_pass": True,
        "comparison_key": "target_system + paired token IDs; match-side assignment; exact ordered pair; four durations",
        "short_frame_rows": int(len(short)), "flatness_extension_rows": int(len(flatness)),
        "short_frame_pairs": int(len(short_pairs)), "flatness_extension_pairs": int(len(flatness_pairs)),
        "short_frame_canonical_sha256": short_hash, "flatness_extension_canonical_sha256": flatness_hash,
        "token_counts_identical": bool(short.groupby("target_system").size().equals(flatness.groupby("target_system").size())),
        "paired_rows_identical": bool(identity),
        "production_flatness_action": "reuse identity-proven short-frame table" if identity else "use official V6 flatness-extension matching table",
    }


def load_official_flatness_matching(project: Path) -> pd.DataFrame:
    extension = pd.read_csv(
        project / "rebuild_v6_trajectory/cpp_flatness_extension/matching/extension_matched_tokens.tsv",
        sep="\t", low_memory=False,
    )
    flatness = extension[extension.feature.eq("flatness")].copy()
    flatness["exact_phone_pair"] = flatness["phone_pair"]
    return flatness


def smd(bona_fide: pd.Series | np.ndarray, tts: pd.Series | np.ndarray) -> float:
    a, b = np.asarray(bona_fide, dtype=float), np.asarray(tts, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return math.nan
    denominator = math.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    if denominator == 0:
        return 0.0 if b.mean() == a.mean() else math.copysign(math.inf, b.mean() - a.mean())
    return float((b.mean() - a.mean()) / denominator)


def hungarian_within_exact_pair(pool: pd.DataFrame, target_system: str) -> pd.DataFrame:
    """Deterministic official matching definition for one target system."""
    output = []
    for pair, group in pool.groupby("exact_phone_pair", sort=True):
        bf = group[group.system.eq("bonafide")].sort_values("token_id").reset_index(drop=True)
        tt = group[group.system.eq(target_system)].sort_values("token_id").reset_index(drop=True)
        if bf.empty or tt.empty:
            continue
        combined = pd.concat([bf[["A_duration_ms", "B_duration_ms"]], tt[["A_duration_ms", "B_duration_ms"]]], ignore_index=True)
        mean = combined.mean(axis=0)
        scale = combined.std(axis=0, ddof=0).replace(0, 1.0)
        bf_z = (bf[["A_duration_ms", "B_duration_ms"]] - mean) / scale
        tt_z = (tt[["A_duration_ms", "B_duration_ms"]] - mean) / scale
        rows, columns = linear_sum_assignment(cdist(bf_z.to_numpy(), tt_z.to_numpy(), metric="euclidean"))
        for sequence, (i, j) in enumerate(zip(rows, columns, strict=True)):
            match_id = f"V7_CPPS_{target_system}_{len(output):07d}"
            distance = float(np.linalg.norm(bf_z.iloc[i].to_numpy() - tt_z.iloc[j].to_numpy()))
            for side, record in (("bonafide", bf.iloc[i]), ("tts", tt.iloc[j])):
                item = record.to_dict()
                item.update({
                    "target_system": target_system, "match_id": match_id, "match_side": side,
                    "standardized_match_distance": distance, "matching_sequence_within_pair": sequence,
                    "matching_algorithm": "exact ordered pair; pooled duration z ddof=0; Euclidean; scipy linear_sum_assignment; token-ID sorted",
                })
                output.append(item)
    return pd.DataFrame(output)


def balance_and_integrity(matched: pd.DataFrame, system: str, feature_family: str) -> dict:
    group = matched[matched.target_system.eq(system)].copy()
    bf, tt = group[group.match_side.eq("bonafide")], group[group.match_side.eq("tts")]
    counts = group.groupby("match_id").agg(rows=("match_side", "size"), sides=("match_side", "nunique"))
    pair_consistency = group.groupby("match_id").exact_phone_pair.nunique().eq(1).all()
    token_side_unique = not group.duplicated(["match_id", "match_side"]).any()
    result = {
        "feature_family": feature_family, "target_system": system,
        "matched_pairs": int(group.match_id.nunique()), "matched_rows": int(len(group)),
        "duration_A_SMD": smd(bf.A_duration_ms, tt.A_duration_ms),
        "duration_B_SMD": smd(bf.B_duration_ms, tt.B_duration_ms),
        "every_match_two_sides": bool(((counts.rows == 2) & (counts.sides == 2)).all()),
        "exact_pair_within_match": bool(pair_consistency),
        "exact_pair_frequencies_identical": bool(
            bf.exact_phone_pair.value_counts().sort_index().equals(tt.exact_phone_pair.value_counts().sort_index())
        ),
        "unique_side_per_match": bool(token_side_unique),
    }
    smds = [result["duration_A_SMD"], result["duration_B_SMD"]]
    result["duration_SMD_0_25_pass"] = bool(all(np.isfinite(smds)) and max(map(abs, smds)) <= SMD_GATE)
    result["all_gates_pass"] = bool(
        result["matched_pairs"] > 0 and result["every_match_two_sides"] and result["exact_pair_within_match"]
        and result["exact_pair_frequencies_identical"] and result["unique_side_per_match"]
        and result["duration_SMD_0_25_pass"]
    )
    return result


def load_population(project: Path) -> pd.DataFrame:
    tokens = pd.read_csv(
        project / "rebuild_v6_trajectory/extracted/trajectory_eligible_tokens.tsv", sep="\t", low_memory=False
    )
    tokens["exact_phone_pair"] = tokens.get("phone_pair", tokens.phone_A.astype(str) + "->" + tokens.phone_B.astype(str))
    return tokens


def complete_cpps_tokens(project: Path, output: Path) -> set[str]:
    validity_files = sorted((output / "production/extraction/validity").glob("region_validity_*.tsv"))
    if not validity_files:
        raise RuntimeError("No completed production region-validity checkpoints")
    validity = pd.concat((pd.read_csv(path, sep="\t") for path in validity_files), ignore_index=True)
    good = validity[validity.cpps_trajectory_valid].groupby("token_id").region.nunique()
    complete = set(good[good.eq(3)].index.astype(str))
    official_dataset = ds.dataset(
        project / "rebuild_v6_trajectory/cpp_flatness_extension/extracted/flatness_cpp_trajectories.parquet",
        format="parquet",
    )
    official = official_dataset.to_table(
        filter=ds.field("feature") == "cpps", columns=["token_id", "relative_time_ms"]
    ).to_pandas()
    boundary = official.groupby("token_id").relative_time_ms.nunique()
    boundary_complete = set(boundary[boundary.eq(9)].index.astype(str))
    return complete & boundary_complete


def reuse_short_matching(project: Path, population: pd.DataFrame) -> pd.DataFrame:
    matched = pd.read_csv(
        project / "rebuild_v6_trajectory/matching/trajectory_matched_tokens.tsv", sep="\t", low_memory=False
    )
    matched = matched[(matched.window_ms_per_side.eq(20)) & matched.matching_set.eq("non_cpp")].copy()
    if not set(matched.token_id).issubset(set(population.token_id)):
        raise RuntimeError("Official short-frame matching contains tokens outside the retained V7 population")
    return matched


def require_authorization(output: Path, execute_full: bool) -> None:
    marker = output / "FULL_V7_RUN_AUTHORIZED.txt"
    if not execute_full or not marker.is_file() or marker.read_text(encoding="utf-8").strip() != AUTHORIZATION_TEXT:
        raise RuntimeError("Production matching requires --execute-full and exact full-run authorization")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--execute-full", action="store_true")
    args = parser.parse_args()
    project, output = args.project_root.resolve(), args.output_root.resolve()
    if args.validate_only == args.execute_full:
        raise RuntimeError("Select exactly one of --validate-only or --execute-full")
    population = load_population(project)
    short = reuse_short_matching(project, population)
    dry = output / "diagnostics/production_dry_run"
    dry.mkdir(parents=True, exist_ok=True)
    short_audit = pd.DataFrame([balance_and_integrity(short, system, "short_frame") for system in SYSTEMS])
    atomic_tsv(short_audit, dry / "short_matching_reuse_gates.tsv")
    flatness_identity = pd.DataFrame([flatness_matching_identity(project)])
    flatness = load_official_flatness_matching(project)
    flatness_audit = pd.DataFrame([balance_and_integrity(flatness, system, "flatness_official_extension") for system in SYSTEMS])
    atomic_tsv(flatness_identity, dry / "flatness_matching_identity_gate.tsv")
    atomic_tsv(flatness_audit, dry / "flatness_matching_lineage_gates.tsv")
    if not short_audit.all_gates_pass.all():
        raise RuntimeError("Short-frame V6 matching reuse failed production gates")
    if not flatness_identity.lineage_resolution_pass.all():
        raise RuntimeError("Official V6 flatness matching lineage was not resolved")
    if not flatness_audit.all_gates_pass.all():
        raise RuntimeError("Official V6 flatness-extension matching failed production gates")
    if args.validate_only:
        plan = {
            "status": "VALIDATE_ONLY_NO_CPPS_REMATCH", "short_matching_rows": int(len(short)),
            "short_matching_systems": int(short.target_system.nunique()), "short_all_gates_pass": True,
            "flatness_matching_identity_pass": bool(flatness_identity.tables_identical.iloc[0]),
            "flatness_lineage_resolution_pass": True,
            "flatness_production_lineage": "official V6 flatness-extension matching table",
            "cpps_action_after_extraction": "require complete three-region trajectories, then independent exact-pair Hungarian per system",
        }
        (dry / "matching_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
        print(json.dumps(plan, indent=2))
        return
    require_authorization(output, args.execute_full)
    complete = complete_cpps_tokens(project, output)
    cpps_pool = population[population.token_id.astype(str).isin(complete)].copy()
    matches = []
    for system in SYSTEMS:
        partition = "TRAIN" if system in {"A01", "A02", "A03", "A04"} else "EVAL"
        eligible = cpps_pool[(cpps_pool.partition.eq(partition)) & cpps_pool.system.isin(["bonafide", system])]
        matches.append(hungarian_within_exact_pair(eligible, system))
    cpps = pd.concat(matches, ignore_index=True)
    audit = pd.DataFrame([balance_and_integrity(cpps, system, "cpps_complete_three_region") for system in SYSTEMS])
    if not audit.all_gates_pass.all():
        atomic_tsv(audit, output / "matching/production_cpps_matching_gates.tsv")
        raise RuntimeError("CPPS rematching failed pair-integrity or duration-SMD gates")
    atomic_tsv(short, output / "matching/production_short_frame_matches.tsv")
    atomic_tsv(flatness, output / "matching/production_flatness_matches.tsv")
    atomic_tsv(cpps, output / "matching/production_cpps_matches.tsv")
    atomic_tsv(audit, output / "matching/production_cpps_matching_gates.tsv")
    atomic_tsv(pd.DataFrame({"token_id": sorted(complete), "complete_three_region_cpps": True}),
               output / "matching/production_cpps_complete_three_region_tokens.tsv")


if __name__ == "__main__":
    main()
