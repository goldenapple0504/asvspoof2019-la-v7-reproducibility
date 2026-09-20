"""Export the ten certified V2 candidates without promoting production files."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import run_v7_cpps_duration_balance_v2 as v2


SYSTEMS = ("A01", "A02", "A03", "A04", "A07", "A08", "A09", "A10", "A11", "A12")
ATTEMPTS = {"A01": "aggregate_moment_stage3_checkpointed_v4"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    project = Path(r"E:\speech tech\asv2019\LA\LA")
    root = project / "rebuild_v7"
    amendment = root / "matching" / v2.AMENDMENT
    output = amendment / "final_candidate_export"
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite final export: {output}")
    output.mkdir(parents=True)

    previous = pd.read_csv(root / "matching/production_cpps_matching_gates.tsv", sep="\t")
    all_matches: list[pd.DataFrame] = []
    summaries: list[dict] = []
    production_gates: list[dict] = []
    source_hashes: list[dict] = []
    reference_columns: list[str] | None = None
    for system in SYSTEMS:
        attempt = ATTEMPTS.get(system, "aggregate_moment_checkpointed_v2")
        directory = amendment / system / attempt
        report_name = "A01_pilot_report.json" if system == "A01" else f"{system}_candidate_report.json"
        report_path = directory / report_name
        matches_path = directory / "candidate_cpps_matches.tsv"
        validation_path = directory / "independent_validation.tsv"
        if not all(path.is_file() for path in (report_path, matches_path, validation_path)):
            raise RuntimeError(f"Incomplete certified source for {system}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        audit = report["independent_validation"]
        if (
            report["status"] != f"CERTIFIED_{system}_CANDIDATE_NOT_PROMOTED"
            or report["maximum_certified_cardinality"] != report["upper_cardinality"]
            or report["stage2"]["status"] != "optimal"
            or report["stage3"]["status"] != "optimal"
            or report["stage2"]["gap"] != 0
            or report["stage3"]["gap"] != 0
            or not all((audit["target_0_20_pass"], audit["final_0_25_pass"],
                        audit["exact_phone_pair_integrity"], audit["without_replacement"]))
        ):
            raise RuntimeError(f"Certification gate failed while exporting {system}")
        matches = pd.read_csv(matches_path, sep="\t", low_memory=False)
        if reference_columns is None:
            reference_columns = list(matches.columns)
        elif list(matches.columns) != reference_columns:
            raise RuntimeError(f"Candidate schema differs for {system}")
        if set(matches.target_system.astype(str)) != {system} or len(matches) != 2 * int(audit["pair_count"]):
            raise RuntimeError(f"Candidate row/system integrity failed for {system}")
        grouped = matches.groupby("match_id", sort=False)
        every_two = bool(grouped.size().eq(2).all())
        unique_side = bool(grouped.match_side.nunique().eq(2).all())
        exact_pair = bool(grouped.exact_phone_pair.nunique().eq(1).all())
        if not all((every_two, unique_side, exact_pair)):
            raise RuntimeError(f"Independent pair reconstruction failed for {system}")
        old = previous.loc[previous.target_system.eq(system)].iloc[0]
        summaries.append({
            "target_system": system,
            "partition": report.get("partition", "TRAIN"),
            "eligible_tts": json.loads((directory / "input_and_solver_settings.json").read_text(encoding="utf-8"))["eligible_tts"]
                if (directory / "input_and_solver_settings.json").is_file() else None,
            "eligible_bonafide": json.loads((directory / "input_and_solver_settings.json").read_text(encoding="utf-8"))["eligible_bonafide"]
                if (directory / "input_and_solver_settings.json").is_file() else None,
            "previous_pairs": int(old.matched_pairs),
            "revised_pairs": int(audit["pair_count"]),
            "pair_retention": float(audit["pair_retention"]),
            "previous_duration_A_SMD": float(old.duration_A_SMD),
            "revised_duration_A_SMD": float(audit["duration_A_SMD"]),
            "previous_duration_B_SMD": float(old.duration_B_SMD),
            "revised_duration_B_SMD": float(audit["duration_B_SMD"]),
            "maximum_cardinality_certified": True,
            "stage2_optimal": True,
            "stage3_optimal": True,
            "target_0_20_pass": True,
            "final_0_25_pass": True,
            "exact_ordered_phone_pair_integrity": exact_pair,
            "one_to_one_no_replacement": bool(audit["without_replacement"]),
            "deterministic_candidate_hash": audit["candidate_sha256"],
        })
        production_gates.append({
            "feature_family": "cpps_complete_three_region",
            "target_system": system,
            "matched_pairs": int(audit["pair_count"]),
            "matched_rows": len(matches),
            "duration_A_SMD": float(audit["duration_A_SMD"]),
            "duration_B_SMD": float(audit["duration_B_SMD"]),
            "every_match_two_sides": every_two,
            "exact_pair_within_match": exact_pair,
            "exact_pair_frequencies_identical": exact_pair,
            "unique_side_per_match": unique_side,
            "duration_SMD_0_25_pass": bool(audit["final_0_25_pass"]),
            "all_gates_pass": True,
        })
        source_hashes.extend([
            {"system": system, "source_file": str(report_path.relative_to(root)).replace("\\", "/"), "sha256": sha256(report_path)},
            {"system": system, "source_file": str(matches_path.relative_to(root)).replace("\\", "/"), "sha256": sha256(matches_path)},
            {"system": system, "source_file": str(validation_path.relative_to(root)).replace("\\", "/"), "sha256": sha256(validation_path)},
        ])
        all_matches.append(matches)

    combined = pd.concat(all_matches, ignore_index=True)
    if len(combined) != 2 * sum(row["revised_pairs"] for row in summaries):
        raise RuntimeError("Combined candidate row count failed")
    v2.atomic_tsv(combined, output / "candidate_cpps_matches.tsv")
    v2.atomic_tsv(pd.DataFrame(summaries), output / "candidate_previous_vs_revised_summary.tsv")
    v2.atomic_tsv(pd.DataFrame(production_gates), output / "candidate_production_cpps_matching_gates.tsv")
    v2.atomic_tsv(pd.DataFrame(source_hashes), output / "certified_source_hashes.tsv")

    complete = sorted(v2.complete_three_region_tokens(project, root))
    if len(complete) != 38_483:
        raise RuntimeError("Complete-three-region eligibility changed during export")
    v2.atomic_tsv(pd.DataFrame({"token_id": complete, "complete_three_region_cpps": True}),
                  output / "candidate_cpps_complete_three_region_tokens.tsv")

    frozen_before = pd.read_csv(root / "matching/cpps_duration_balance_amendment_v1/frozen_extraction_hashes_before.tsv", sep="\t")
    frozen_after = []
    for row in frozen_before.itertuples(index=False):
        path = root / str(row.relative_path)
        if not path.is_file():
            raise RuntimeError(f"Frozen extraction file missing: {row.relative_path}")
        frozen_after.append({"relative_path": row.relative_path, "bytes": path.stat().st_size, "sha256": sha256(path)})
    frozen_after_frame = pd.DataFrame(frozen_after)
    v2.atomic_tsv(frozen_after_frame, output / "frozen_extraction_hashes_after.tsv")
    if not frozen_before.equals(frozen_after_frame):
        raise RuntimeError("Frozen extraction hashes changed during V2 export")

    v2.atomic_tsv(pd.DataFrame([
        {"candidate_file": "candidate_cpps_matches.tsv", "future_production_destination": "matching/production_cpps_matches.tsv"},
        {"candidate_file": "candidate_production_cpps_matching_gates.tsv", "future_production_destination": "matching/production_cpps_matching_gates.tsv"},
        {"candidate_file": "candidate_cpps_complete_three_region_tokens.tsv", "future_production_destination": "matching/production_cpps_complete_three_region_tokens.tsv"},
    ]), output / "promotion_plan_not_executed.tsv")
    manifest = {
        "status": "ALL_TEN_CERTIFIED_EXPORTED_NOT_PROMOTED",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "systems": list(SYSTEMS),
        "systems_certified": 10,
        "total_pairs": int(sum(row["revised_pairs"] for row in summaries)),
        "total_rows": len(combined),
        "complete_three_region_tokens": len(complete),
        "all_target_0_20_pass": True,
        "all_final_0_25_pass": True,
        "frozen_extraction_files": len(frozen_after_frame),
        "frozen_extraction_hashes_unchanged": True,
        "production_promoted": False,
        "production_models_run": False,
        "inference_run": False,
    }
    v2.atomic_text(output / "design_amendment_manifest.json", json.dumps(manifest, indent=2))
    v2.atomic_text(output / "README.md", (
        "# V7 CPPS duration-balance amendment V2 final candidate export\n\n"
        "All ten system-specific exact SCIP candidates are certified and independently validated. "
        "This directory is an export for review; production files have not been promoted and no models or inference were run.\n"
    ))
    files = sorted(path for path in output.iterdir() if path.is_file() and path.name != "candidate_file_hashes.tsv")
    v2.atomic_tsv(pd.DataFrame([{"file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)} for path in files]),
                  output / "candidate_file_hashes.tsv")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
