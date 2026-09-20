"""Atomically promote the fully certified CPPS V2 candidate into production matching.

This script never runs extraction or matching.  It validates the immutable final
candidate export, independently reconstructs every matching gate, backs up any
pre-existing destination, then atomically installs the three authorized files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

AUTHORIZATION_TEXT = "FULL_V7_RUN_AUTHORIZED"
SYSTEMS = ("A01", "A02", "A03", "A04", "A07", "A08", "A09", "A10", "A11", "A12")
EXPORT_REL = Path("matching/cpps_duration_balance_amendment_v2/final_candidate_export")
PROMOTION_REL = Path("matching/cpps_duration_balance_amendment_v2/promotion_v2")
FILES = {
    "candidate_cpps_matches.tsv": "matching/production_cpps_matches.tsv",
    "candidate_production_cpps_matching_gates.tsv": "matching/production_cpps_matching_gates.tsv",
    "candidate_cpps_complete_three_region_tokens.tsv": "matching/production_cpps_complete_three_region_tokens.tsv",
}


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


def require_authorization(root: Path, execute_full: bool) -> None:
    marker = root / "FULL_V7_RUN_AUTHORIZED.txt"
    if (not execute_full or not marker.is_file()
            or marker.read_text(encoding="utf-8").strip() != AUTHORIZATION_TEXT):
        raise RuntimeError("CPPS V2 promotion requires the valid full-run authorization marker")


def validate_export(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict]]:
    export = root / EXPORT_REL
    recorded = pd.read_csv(export / "candidate_file_hashes.tsv", sep="\t")
    for row in recorded.itertuples(index=False):
        path = export / str(row.file)
        if not path.is_file() or path.stat().st_size != int(row.bytes) or sha256(path) != str(row.sha256):
            raise RuntimeError(f"Candidate export hash/size gate failed: {path}")

    manifest = json.loads((export / "design_amendment_manifest.json").read_text(encoding="utf-8"))
    if not (
        manifest.get("status") == "ALL_TEN_CERTIFIED_EXPORTED_NOT_PROMOTED"
        and manifest.get("systems") == list(SYSTEMS)
        and manifest.get("systems_certified") == 10
        and manifest.get("total_pairs") == 9382
        and manifest.get("total_rows") == 18764
        and manifest.get("complete_three_region_tokens") == 38483
        and manifest.get("all_target_0_20_pass") is True
        and manifest.get("all_final_0_25_pass") is True
        and manifest.get("frozen_extraction_files") == 730
        and manifest.get("frozen_extraction_hashes_unchanged") is True
    ):
        raise RuntimeError("Immutable final candidate manifest gate failed")

    frozen = pd.read_csv(export / "frozen_extraction_hashes_after.tsv", sep="\t")
    if len(frozen) != 730:
        raise RuntimeError(f"Expected 730 frozen extraction files, found {len(frozen)}")
    for row in frozen.itertuples(index=False):
        path = root / str(row.relative_path)
        if not path.is_file() or path.stat().st_size != int(row.bytes) or sha256(path) != str(row.sha256):
            raise RuntimeError(f"Frozen extraction integrity gate failed: {path}")

    matches = pd.read_csv(export / "candidate_cpps_matches.tsv", sep="\t", low_memory=False)
    gates = pd.read_csv(export / "candidate_production_cpps_matching_gates.tsv", sep="\t")
    eligible = pd.read_csv(export / "candidate_cpps_complete_three_region_tokens.tsv", sep="\t")
    if len(matches) != 18764 or matches.match_id.nunique() != 9382:
        raise RuntimeError("Candidate match row/pair count gate failed")
    if len(eligible) != 38483 or not eligible.token_id.is_unique or not eligible.complete_three_region_cpps.eq(True).all():
        raise RuntimeError("Complete-three-region eligibility gate failed")
    if list(gates.target_system.astype(str)) != list(SYSTEMS) or len(gates) != 10:
        raise RuntimeError("Candidate system gate failed")
    if not gates.all_gates_pass.eq(True).all():
        raise RuntimeError("A candidate production matching gate is not PASS")

    independent: list[dict] = []
    for system in SYSTEMS:
        frame = matches.loc[matches.target_system.eq(system)].copy()
        grouped = frame.groupby("match_id", sort=False)
        bf = frame.loc[frame.match_side.eq("bonafide")]
        tts = frame.loc[frame.match_side.eq("tts")]
        a_smd = smd(bf.A_duration_ms.to_numpy(float), tts.A_duration_ms.to_numpy(float))
        b_smd = smd(bf.B_duration_ms.to_numpy(float), tts.B_duration_ms.to_numpy(float))
        gate = gates.loc[gates.target_system.eq(system)].iloc[0]
        checks = {
            "every_match_two_sides": bool(grouped.size().eq(2).all()),
            "unique_side_per_match": bool(grouped.match_side.nunique().eq(2).all()),
            "exact_pair_within_match": bool(grouped.exact_phone_pair.nunique().eq(1).all()),
            "bonafide_without_replacement": bool(bf.token_id.is_unique),
            "tts_without_replacement": bool(tts.token_id.is_unique),
            "correct_tts_system": bool(tts.system.eq(system).all()),
            "correct_bonafide_system": bool(bf.system.eq("bonafide").all()),
            "target_0_20_pass": bool(abs(a_smd) <= 0.20 + 1e-10 and abs(b_smd) <= 0.20 + 1e-10),
            "final_0_25_pass": bool(abs(a_smd) <= 0.25 and abs(b_smd) <= 0.25),
            "reported_SMD_reproduced": bool(
                np.isclose(a_smd, float(gate.duration_A_SMD), rtol=0, atol=1e-12)
                and np.isclose(b_smd, float(gate.duration_B_SMD), rtol=0, atol=1e-12)
            ),
            "reported_counts_reproduced": bool(len(frame) == int(gate.matched_rows)
                                                and frame.match_id.nunique() == int(gate.matched_pairs)),
        }
        if not all(checks.values()):
            raise RuntimeError(f"Independent promotion gate failed for {system}: {checks}")
        independent.append({"target_system": system, "matched_pairs": frame.match_id.nunique(),
                            "matched_rows": len(frame), "duration_A_SMD": a_smd,
                            "duration_B_SMD": b_smd, **checks})
    return matches, gates, eligible, independent


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".promotion.tmp")
    if temporary.exists():
        temporary.unlink()
    shutil.copyfile(source, temporary)
    if sha256(temporary) != sha256(source):
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Temporary promotion copy hash mismatch: {destination}")
    os.replace(temporary, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--execute-full", action="store_true")
    args = parser.parse_args()
    root = args.output_root.resolve()
    require_authorization(root, args.execute_full)
    _, _, _, independent = validate_export(root)

    export = root / EXPORT_REL
    promotion = root / PROMOTION_REL
    if promotion.exists():
        raise RuntimeError(f"Refusing to overwrite existing promotion record: {promotion}")
    promotion.mkdir(parents=True)

    pre_rows: list[dict] = []
    for candidate_name, destination_rel in FILES.items():
        destination = root / destination_rel
        if destination.is_file():
            backup = promotion / (destination.name + ".pre_v2_backup")
            shutil.copyfile(destination, backup)
            pre_rows.append({"destination": destination_rel, "existed": True,
                             "bytes": destination.stat().st_size, "sha256": sha256(destination),
                             "backup": str(backup.relative_to(root)).replace("\\", "/")})
        else:
            pre_rows.append({"destination": destination_rel, "existed": False,
                             "bytes": 0, "sha256": "", "backup": ""})

    for candidate_name, destination_rel in FILES.items():
        atomic_copy(export / candidate_name, root / destination_rel)

    installed: list[dict] = []
    for candidate_name, destination_rel in FILES.items():
        source, destination = export / candidate_name, root / destination_rel
        if sha256(source) != sha256(destination) or source.stat().st_size != destination.stat().st_size:
            raise RuntimeError(f"Installed production file differs from certified candidate: {destination}")
        installed.append({"candidate": candidate_name, "destination": destination_rel,
                          "bytes": destination.stat().st_size, "sha256": sha256(destination)})

    pd.DataFrame(pre_rows).to_csv(promotion / "pre_promotion_inventory.tsv", sep="\t", index=False)
    pd.DataFrame(independent).to_csv(promotion / "independent_promotion_validation.tsv", sep="\t", index=False)
    pd.DataFrame(installed).to_csv(promotion / "installed_file_hashes.tsv", sep="\t", index=False)
    record = {
        "status": "PROMOTED_CERTIFIED_CPPS_V2",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "systems": list(SYSTEMS), "matched_pairs": 9382, "matched_rows": 18764,
        "complete_three_region_tokens": 38483, "scientific_design_changed": False,
        "extraction_rerun": False, "matching_rerun": False,
        "source_export": str(EXPORT_REL).replace("\\", "/"), "installed": installed,
    }
    (promotion / "promotion_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
