"""Hard final quality gates for an authorized, completed V7 production run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_v7_interior_extraction import AUTHORIZATION_TEXT
from run_v7_interior_extraction import assign_chunk_membership, canonical_membership_hash
from v7_common import SYSTEMS, atomic_tsv, sha256


FEATURES = ("energy", "centroid", "tilt", "flux", "flatness", "cpps")


def cr2_outcome_gate(models: pd.DataFrame) -> dict[str, bool]:
    required = {
        "CR2_status", "CR2_evidence_complete", "CR2_utterances_multiple_matches",
        "CR2_match_clusters_independent_partitions", "CR2_estimates_produced", "CR2_success",
        "CR2_covariance_finite", "CR2_covariance_min_eigen", "CR2_rows", "CR2_columns", "fixed_columns",
    }
    if not required.issubset(models.columns):
        return {"schema_complete": False, "authorized_status_only": False, "all_rows_valid": False}
    status = models.CR2_status.astype(str)
    authorized = status.isin(["VALID_FINITE_PSD_CR2", "NOT_APPLICABLE_CROSSED_CLUSTERING"])
    valid = status.eq("VALID_FINITE_PSD_CR2")
    not_applicable = status.eq("NOT_APPLICABLE_CROSSED_CLUSTERING")
    valid_ok = (~valid) | (
        models.CR2_success.eq(True)
        & models.CR2_covariance_finite.eq(True)
        & models.CR2_estimates_produced.eq(True)
        & models.CR2_match_clusters_independent_partitions.eq(True)
        & models.CR2_rows.eq(models.fixed_columns) & models.CR2_columns.eq(models.fixed_columns)
        & pd.to_numeric(models.CR2_covariance_min_eigen, errors="coerce").ge(-1e-8)
    )
    na_ok = (~not_applicable) | (
        models.CR2_evidence_complete.eq(True)
        & pd.to_numeric(models.CR2_utterances_multiple_matches, errors="coerce").gt(0)
        & models.CR2_estimates_produced.eq(False)
        & models.CR2_match_clusters_independent_partitions.eq(False)
        & models.CR2_success.eq(False)
    )
    return {"schema_complete": True, "authorized_status_only": bool(authorized.all()),
            "all_rows_valid": bool((authorized & valid_ok & na_ok).all())}


def checkpoint_inventory_gates(planned_ids: list[int], records: list[dict],
                               expected_by_chunk: dict[int, set[str]], actual_by_chunk: dict[int, set[str]]) -> dict[str, bool]:
    record_ids = [int(record["chunk_id"]) for record in records]
    expected_tokens = [token for chunk in planned_ids for token in expected_by_chunk.get(chunk, set())]
    actual_tokens = [token for chunk in planned_ids for token in actual_by_chunk.get(chunk, set())]
    return {
        "exact_checkpoint_count": len(records) == len(planned_ids),
        "planned_chunk_ids_exactly_once": sorted(record_ids) == sorted(planned_ids) and len(record_ids) == len(set(record_ids)),
        "all_chunk_membership_exact": all(actual_by_chunk.get(chunk, set()) == expected_by_chunk.get(chunk, set()) for chunk in planned_ids),
        "no_duplicate_tokens_across_chunks": len(actual_tokens) == len(set(actual_tokens)),
        "no_missing_or_extra_tokens": set(actual_tokens) == set(expected_tokens),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--execute-full", action="store_true")
    args = parser.parse_args()
    root = args.output_root.resolve()
    marker = root / "FULL_V7_RUN_AUTHORIZED.txt"
    if not args.execute_full or not marker.is_file() or marker.read_text(encoding="utf-8").strip() != AUTHORIZATION_TEXT:
        raise RuntimeError("Production final validation is authorization-locked")
    gates = []
    def add(gate: str, passed: bool, observed: object, expected: object) -> None:
        gates.append({"gate": gate, "pass": bool(passed), "observed": str(observed), "expected": str(expected)})

    plan_record = json.loads((root / "diagnostics/production_dry_run/extraction_plan.json").read_text(encoding="utf-8"))
    plan = pd.read_csv(root / "diagnostics/production_dry_run/extraction_chunk_plan.tsv", sep="\t")
    planned_ids = plan.chunk_id.astype(int).tolist()
    checkpoints = sorted((root / "production/extraction/checkpoints").glob("chunk_*.json"))
    records = [json.loads(path.read_text(encoding="utf-8")) for path in checkpoints]
    eligibility = pd.read_csv(root / "diagnostics/interior_eligibility.tsv", sep="\t", usecols=["token_id", "audio_path"])
    membership = assign_chunk_membership(eligibility, int(plan_record["files_per_chunk"]))
    expected_by_chunk = {
        int(chunk): set(group.token_id.astype(str)) for chunk, group in membership.groupby("chunk_id", sort=True)
    }
    actual_by_chunk = {}
    hash_valid = True
    checkpoint_membership_hash_valid = True
    checkpoint_filename_id_valid = True
    audio_counts_valid = True
    for path, record in zip(checkpoints, records, strict=True):
        chunk = int(record["chunk_id"])
        checkpoint_filename_id_valid &= path.stem == f"chunk_{chunk:05d}"
        feature_path = root / "production/extraction/chunks" / f"interior_features_{chunk:05d}.parquet"
        validity_path = root / "production/extraction/validity" / f"region_validity_{chunk:05d}.tsv"
        exclusion_path = root / "production/extraction/cpps_validity_exclusions" / f"cpps_validity_exclusions_{chunk:05d}.tsv"
        error_path = root / "production/extraction/unexpected_errors" / f"unexpected_errors_{chunk:05d}.tsv"
        targets = [feature_path, validity_path, exclusion_path, error_path]
        hash_valid &= all(target.is_file() and record.get("sha256", {}).get(target.name) == sha256(target) for target in targets)
        if feature_path.is_file():
            actual = pd.read_parquet(feature_path, columns=["source_token_id"])
            actual_by_chunk[chunk] = set(actual.source_token_id.astype(str))
        checkpoint_membership_hash_valid &= (
            record.get("membership_sha256") == record.get("planned_membership_sha256")
            == plan.loc[plan.chunk_id.eq(chunk), "membership_sha256"].iloc[0]
        ) if chunk in set(planned_ids) else False
        audio_counts_valid &= record.get("audio_files") == int(plan.loc[plan.chunk_id.eq(chunk), "audio_files"].iloc[0]) if chunk in set(planned_ids) else False
    inventory = checkpoint_inventory_gates(planned_ids, records, expected_by_chunk, actual_by_chunk)
    add("exactly_146_extraction_checkpoints", inventory["exact_checkpoint_count"] and len(planned_ids) == 146, len(records), 146)
    add("planned_chunk_ids_exactly_once", inventory["planned_chunk_ids_exactly_once"] and checkpoint_filename_id_valid, sorted(int(r["chunk_id"]) for r in records), planned_ids)
    add("all_checkpoint_hashes_valid", hash_valid, hash_valid, True)
    add("checkpoint_membership_hashes_match_frozen_plan", checkpoint_membership_hash_valid, checkpoint_membership_hash_valid, True)
    add("summed_unique_token_coverage_482509", inventory["no_missing_or_extra_tokens"] and len(set().union(*actual_by_chunk.values())) == 482_509,
        len(set().union(*actual_by_chunk.values())) if actual_by_chunk else 0, 482_509)
    add("no_duplicate_or_missing_chunk_tokens", inventory["all_chunk_membership_exact"] and inventory["no_duplicate_tokens_across_chunks"], inventory, "all True")
    add("audio_file_coverage_matches_frozen_plan", audio_counts_valid and membership.audio_path.nunique() == plan_record["audio_files"],
        membership.audio_path.nunique(), plan_record["audio_files"])
    add("all_extraction_checkpoints_pass", len(records) == 146 and all(r["status"] == "PASS" for r in records), len(records), "146 and all PASS")
    add("zero_unexpected_extraction_errors", sum(r.get("unexpected_errors", -1) for r in records) == 0,
        sum(r.get("unexpected_errors", -1) for r in records), 0)
    short = pd.read_csv(root / "diagnostics/production_dry_run/short_matching_reuse_gates.tsv", sep="\t")
    flatness = pd.read_csv(root / "diagnostics/production_dry_run/flatness_matching_lineage_gates.tsv", sep="\t")
    cpps = pd.read_csv(root / "matching/production_cpps_matching_gates.tsv", sep="\t")
    add("short_matching_all_systems", len(short) == 10 and short.all_gates_pass.all(), len(short), 10)
    add("flatness_official_extension_lineage_all_systems", len(flatness) == 10 and flatness.all_gates_pass.all(), len(flatness), 10)
    add("cpps_complete_region_rematching_all_systems", len(cpps) == 10 and cpps.all_gates_pass.all(), len(cpps), 10)
    add("cpps_duration_SMD_gate", (cpps[["duration_A_SMD", "duration_B_SMD"]].abs() <= .25).all().all(),
        cpps[["duration_A_SMD", "duration_B_SMD"]].abs().max().max(), "<=0.25")
    models = pd.read_csv(root / "production/inference/model_acceptance_gates.tsv", sep="\t")
    cr2_gate = cr2_outcome_gate(models)
    add("sixty_rank_safe_nonsingular_models_with_authorized_CR2_outcomes", len(models) == 60
        and models.all_acceptance_gates_pass.all() and all(cr2_gate.values()), {"rows": len(models), **cr2_gate}, "60; authorized CR2 outcome per row")
    average = pd.read_csv(root / "production/inference/average_specificity.tsv", sep="\t")
    shape = pd.read_csv(root / "production/inference/joint_shape_specificity.tsv", sep="\t")
    time = pd.read_csv(root / "production/inference/time_resolved_specificity.tsv", sep="\t")
    add("sixty_average_targets", len(average) == 60 and np.isfinite(average.estimate).all(), len(average), 60)
    add("sixty_shape_targets", len(shape) == 60 and np.isfinite(shape.chisq).all(), len(shape), 60)
    expected_time = 10 * (5 * 9 + 8)
    add("all_predefined_time_targets", len(time) == expected_time and np.isfinite(time.estimate).all(), len(time), expected_time)
    meta = pd.read_csv(root / "production/meta_analysis/multivariate_REML_model_gates.tsv", sep="\t")
    scalar_meta = pd.read_csv(root / "production/meta_analysis/average_specificity_REML.tsv", sep="\t")
    trajectory_meta = pd.read_csv(root / "production/meta_analysis/projected_specificity_trajectories_and_prediction_intervals.tsv", sep="\t")
    sensitivity = pd.read_csv(root / "production/inference/covariance_sensitivity_classification_comparison.tsv", sep="\t")
    add("six_contrast_meta_analyses", len(meta) == 6 and meta.converged.all()
        and meta.fixed_covariance_pass.all() and meta.between_covariance_pass.all(), len(meta), 6)
    add("scalar_meta_heterogeneity_and_predictions", len(scalar_meta) == 18
        and np.isfinite(scalar_meta[["estimate", "SE", "tau", "tau_squared", "I_squared_percent", "Q", "prediction_lower", "prediction_upper"]]).all().all(), len(scalar_meta), 18)
    expected_meta_times = 5 * 2 * 9 + 2 * 8
    add("trajectory_meta_predictions_all_official_times", len(trajectory_meta) == expected_meta_times
        and np.isfinite(trajectory_meta[["projected_mean", "projected_between_system_variance", "prediction_lower", "prediction_upper"]]).all().all(), len(trajectory_meta), expected_meta_times)
    valid_sensitivity = sensitivity.CR2_status.eq("VALID_FINITE_PSD_CR2")
    not_applicable_sensitivity = sensitivity.CR2_status.eq("NOT_APPLICABLE_CROSSED_CLUSTERING")
    classification_columns = ["average_classification_changed", "shape_classification_changed", "SESOI_classification_changed"]
    add("CR2_classification_sensitivity_export", len(sensitivity) == 60
        and sensitivity.loc[valid_sensitivity, classification_columns].notna().all().all()
        and sensitivity.loc[not_applicable_sensitivity, classification_columns].isna().all().all(), len(sensitivity), 60)
    frame = pd.DataFrame(gates)
    destination = root / "production/quality_gates.tsv"
    atomic_tsv(frame, destination)
    if not frame["pass"].all():
        raise RuntimeError("One or more final V7 production quality gates failed")


if __name__ == "__main__":
    main()
