"""Materialize already-frozen V6 short-frame/flatness matches for V7 production.

No matching algorithm is executed here.  Rows are filtered directly from the two
official V6 lineage tables, validated against the previously completed dry-run
audits, and installed atomically without modifying either V6 source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


AUTHORIZATION_TEXT = "FULL_V7_RUN_AUTHORIZED"
SYSTEMS = ("A01", "A02", "A03", "A04", "A07", "A08", "A09", "A10", "A11", "A12")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def smd(bf: np.ndarray, tts: np.ndarray) -> float:
    pooled = math.sqrt(((len(bf) - 1) * bf.var(ddof=1) + (len(tts) - 1) * tts.var(ddof=1)) /
                       (len(bf) + len(tts) - 2))
    if pooled:
        return float((tts.mean() - bf.mean()) / pooled)
    return 0.0 if bf.mean() == tts.mean() else math.inf


def audit(frame: pd.DataFrame, family: str) -> pd.DataFrame:
    rows = []
    for system in SYSTEMS:
        group = frame.loc[frame.target_system.eq(system)]
        bf, tts = group.loc[group.match_side.eq("bonafide")], group.loc[group.match_side.eq("tts")]
        grouped = group.groupby("match_id", sort=False)
        a_smd = smd(bf.A_duration_ms.to_numpy(float), tts.A_duration_ms.to_numpy(float))
        b_smd = smd(bf.B_duration_ms.to_numpy(float), tts.B_duration_ms.to_numpy(float))
        rows.append({
            "feature_family": family, "target_system": system,
            "matched_pairs": group.match_id.nunique(), "matched_rows": len(group),
            "duration_A_SMD": a_smd, "duration_B_SMD": b_smd,
            "every_match_two_sides": bool(grouped.size().eq(2).all()),
            "exact_pair_within_match": bool(grouped.exact_phone_pair.nunique().eq(1).all()),
            "exact_pair_frequencies_identical": bool(
                bf.exact_phone_pair.value_counts().sort_index().equals(
                    tts.exact_phone_pair.value_counts().sort_index())),
            "unique_side_per_match": bool(not group.duplicated(["match_id", "match_side"]).any()),
            "duration_SMD_0_25_pass": bool(abs(a_smd) <= .25 and abs(b_smd) <= .25),
        })
    result = pd.DataFrame(rows)
    result["all_gates_pass"] = result[[
        "every_match_two_sides", "exact_pair_within_match", "exact_pair_frequencies_identical",
        "unique_side_per_match", "duration_SMD_0_25_pass",
    ]].all(axis=1)
    return result


def compare_to_frozen_audit(observed: pd.DataFrame, expected: pd.DataFrame, label: str) -> None:
    if list(observed.target_system) != list(expected.target_system):
        raise RuntimeError(f"{label} system ordering differs from frozen audit")
    integer_columns = ["matched_pairs", "matched_rows"]
    boolean_columns = ["every_match_two_sides", "exact_pair_within_match", "exact_pair_frequencies_identical",
                       "unique_side_per_match", "duration_SMD_0_25_pass", "all_gates_pass"]
    if not all(observed[column].astype(int).equals(expected[column].astype(int)) for column in integer_columns):
        raise RuntimeError(f"{label} counts differ from frozen dry-run audit")
    if not all(observed[column].astype(bool).equals(expected[column].astype(bool)) for column in boolean_columns):
        raise RuntimeError(f"{label} integrity result differs from frozen dry-run audit")
    for column in ("duration_A_SMD", "duration_B_SMD"):
        if not np.allclose(observed[column], expected[column], rtol=0, atol=1e-12):
            raise RuntimeError(f"{label} {column} differs from frozen dry-run audit")


def atomic_tsv(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".lineage.tmp")
    frame.to_csv(temporary, sep="\t", index=False)
    os.replace(temporary, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--execute-full", action="store_true")
    args = parser.parse_args()
    project, root = args.project_root.resolve(), args.output_root.resolve()
    marker = root / "FULL_V7_RUN_AUTHORIZED.txt"
    if (not args.execute_full or not marker.is_file()
            or marker.read_text(encoding="utf-8").strip() != AUTHORIZATION_TEXT):
        raise RuntimeError("Frozen lineage materialization requires full-run authorization")

    short_source = project / "rebuild_v6_trajectory/matching/trajectory_matched_tokens.tsv"
    flat_source = project / "rebuild_v6_trajectory/cpp_flatness_extension/matching/extension_matched_tokens.tsv"
    short_destination = root / "matching/production_short_frame_matches.tsv"
    flat_destination = root / "matching/production_flatness_matches.tsv"
    if short_destination.exists() or flat_destination.exists():
        raise RuntimeError("Refusing to overwrite an existing production V6-lineage matching table")

    short = pd.read_csv(short_source, sep="\t", low_memory=False)
    short = short.loc[short.window_ms_per_side.eq(20) & short.matching_set.eq("non_cpp")].copy()
    flat = pd.read_csv(flat_source, sep="\t", low_memory=False)
    flat = flat.loc[flat.feature.eq("flatness")].copy()
    flat["exact_phone_pair"] = flat.phone_pair
    short_audit = audit(short, "short_frame")
    flat_audit = audit(flat, "flatness_official_extension")
    dry = root / "diagnostics/production_dry_run"
    compare_to_frozen_audit(short_audit, pd.read_csv(dry / "short_matching_reuse_gates.tsv", sep="\t"), "short")
    compare_to_frozen_audit(flat_audit, pd.read_csv(dry / "flatness_matching_lineage_gates.tsv", sep="\t"), "flatness")
    if not short_audit.all_gates_pass.all() or not flat_audit.all_gates_pass.all():
        raise RuntimeError("Frozen V6 lineage failed production matching gates")

    atomic_tsv(short, short_destination)
    atomic_tsv(flat, flat_destination)
    record_dir = root / "matching/frozen_v6_lineage_materialization"
    record_dir.mkdir(parents=True, exist_ok=False)
    short_audit.to_csv(record_dir / "short_frame_validation.tsv", sep="\t", index=False)
    flat_audit.to_csv(record_dir / "flatness_validation.tsv", sep="\t", index=False)
    record = {
        "status": "MATERIALIZED_FROZEN_V6_MATCHING_LINEAGES",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "matching_algorithm_run": False, "extraction_run": False,
        "sources": [
            {"path": str(short_source), "bytes": short_source.stat().st_size, "sha256": sha256(short_source)},
            {"path": str(flat_source), "bytes": flat_source.stat().st_size, "sha256": sha256(flat_source)},
        ],
        "outputs": [
            {"path": str(short_destination.relative_to(root)).replace("\\", "/"),
             "rows": len(short), "sha256": sha256(short_destination)},
            {"path": str(flat_destination.relative_to(root)).replace("\\", "/"),
             "rows": len(flat), "sha256": sha256(flat_destination)},
        ],
        "systems": list(SYSTEMS), "short_all_gates_pass": True, "flatness_all_gates_pass": True,
    }
    (record_dir / "lineage_materialization_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
