"""Audit V6 and calculate deterministic phone-interior sample eligibility.

This stage does not extract interior acoustic values and does not modify V6.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from v7_common import (
    BASE_HALF_SUPPORT_MS,
    CPPS_TRAJECTORY_HALF_SUPPORT_MS,
    EPS,
    FRAME_MS,
    FRAME_TIMES_MS,
    N_FFT,
    REPRODUCTION_TOLERANCE,
    SEED,
    SYSTEMS,
    atomic_tsv,
    required_directories,
    sha256,
    trajectory_support,
)


def package_version(name: str) -> str:
    try:
        module = __import__(name)
        return str(getattr(module, "__version__", "unknown"))
    except Exception as exc:  # audit must record absence rather than silently omit it
        return f"unavailable:{type(exc).__name__}"


def build_diagnostics(tokens: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    eligibility_rows: list[dict] = []
    sample_rows: list[dict] = []
    for row in tokens.itertuples(index=False):
        boundary = float(row.boundary_time)
        start_a = boundary - float(row.A_duration_ms) / 1000
        end_a = boundary
        start_b = boundary
        end_b = boundary + float(row.B_duration_ms) / 1000
        midpoint_a = (start_a + end_a) / 2
        midpoint_b = (start_b + end_b) / 2
        regions = {}
        for region, phone, start, end, midpoint in (
            ("Interior_A", row.phone_A, start_a, end_a, midpoint_a),
            ("Interior_B", row.phone_B, start_b, end_b, midpoint_b),
        ):
            support = trajectory_support(start, end, midpoint, int(row.sample_rate), int(row.audio_frames))
            regions[region] = support
            reason = "" if support["eligible"] else (
                "audio_support" if not support["audio_support_ok"] else "touches_or_crosses_phone_boundary"
            )
            sample_rows.append({
                "token_id": row.token_id,
                "file_id": row.file_id,
                "partition": row.partition,
                "system": row.system,
                "region": region,
                "phone": phone,
                "phone_start_sec": start,
                "phone_end_sec": end,
                "phone_duration_ms": (end - start) * 1000,
                "midpoint_sec": midpoint,
                "sample_rate": int(row.sample_rate),
                "audio_frames": int(row.audio_frames),
                "rounding_convention": "Python built-in round (ties-to-even), identical to official V6",
                **support,
                "failure_reason": reason,
            })
        eligible_a = bool(regions["Interior_A"]["eligible"])
        eligible_b = bool(regions["Interior_B"]["eligible"])
        eligibility_rows.append({
            "token_id": row.token_id,
            "file_id": row.file_id,
            "partition": row.partition,
            "system": row.system,
            "label": row.label,
            "speaker_id": row.speaker_id,
            "phone_A": row.phone_A,
            "phone_B": row.phone_B,
            "exact_phone_pair": row.phone_pair,
            "boundary_class": row.boundary_class,
            "boundary_time": boundary,
            "phone_A_start_sec": start_a,
            "phone_A_end_sec": end_a,
            "phone_B_start_sec": start_b,
            "phone_B_end_sec": end_b,
            "midpoint_A_sec": midpoint_a,
            "midpoint_B_sec": midpoint_b,
            "A_duration_ms": float(row.A_duration_ms),
            "B_duration_ms": float(row.B_duration_ms),
            "sample_rate": int(row.sample_rate),
            "audio_frames": int(row.audio_frames),
            "audio_path": row.audio_path,
            "v6_boundary_eligible": bool(row.eligibility),
            "interior_A_eligible": eligible_a,
            "interior_B_eligible": eligible_b,
            "both_interiors_eligible": bool(eligible_a and eligible_b and row.eligibility),
            "eligibility_category": (
                "pass_both" if eligible_a and eligible_b else
                "fail_both" if not eligible_a and not eligible_b else
                "fail_A" if not eligible_a else "fail_B"
            ),
        })
    return pd.DataFrame(eligibility_rows), pd.DataFrame(sample_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    project = args.project_root.resolve()
    out = args.output_root.resolve()
    for directory in required_directories(out):
        directory.mkdir(parents=True, exist_ok=True)

    v6 = project / "rebuild_v6_trajectory"
    sources = {
        "population": v6 / "extracted/trajectory_eligible_tokens.tsv",
        "four_feature_raw": v6 / "extracted/trajectory_features_full.parquet",
        "flatness_cpps_raw": v6 / "cpp_flatness_extension/extracted/flatness_cpp_trajectories.parquet",
        "four_feature_matching": v6 / "matching/trajectory_matched_tokens.tsv",
        "flatness_cpps_matching": v6 / "cpp_flatness_extension/matching/extension_matched_tokens.tsv",
        "four_feature_scaling": v6 / "audit/production_scaling_constants.tsv",
        "all6_scaling": v6 / "final_six_feature_trajectory/all6_scaling_parameters.tsv",
        "paper_rq1_summary": v6 / "final_six_feature_trajectory/rq1_all6_trajectory_effect_summaries.tsv",
        "paper_rq1_sesoi": v6 / "final_six_feature_trajectory/rq1_all6_functional_sesoi.tsv",
        "paper_final_report": v6 / "final_six_feature_trajectory/final_report.md",
        "extract_four_script": v6 / "scripts/extract_trajectory_features_full.py",
        "extract_flatness_cpps_script": v6 / "scripts/extract_flatness_cpp_trajectories.py",
        "matching_script": v6 / "scripts/prepare_production_matching.py",
        "model_script": v6 / "scripts/fit_absorbed_fe_cr2.R",
    }
    missing = [str(path) for path in sources.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing official V6 sources: {missing}")
    source_table = pd.DataFrame([
        {"role": role, "path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
        for role, path in sources.items()
    ])
    atomic_tsv(source_table, out / "audit/v6_source_manifest.tsv")

    tokens = pd.read_csv(sources["population"], sep="\t", low_memory=False)
    if len(tokens) != 482_509 or tokens.token_id.nunique() != 482_509 or not tokens.eligibility.all():
        raise RuntimeError("Official V6 population gate failed")
    eligibility, samples = build_diagnostics(tokens)
    if len(eligibility) != len(tokens) or len(samples) != 2 * len(tokens):
        raise RuntimeError("Interior eligibility diagnostics are incomplete")
    atomic_tsv(eligibility, out / "diagnostics/interior_eligibility.tsv")
    atomic_tsv(samples, out / "diagnostics/sample_index_diagnostics.tsv")

    summary = eligibility.groupby(["partition", "system", "eligibility_category"], dropna=False).size().rename("tokens").reset_index()
    atomic_tsv(summary, out / "diagnostics/interior_eligibility_summary.tsv")
    relevant = eligibility.loc[eligibility.both_interiors_eligible, ["token_id"]]
    atomic_tsv(relevant, out / "audit/v6_boundary_reproduction_token_ids.tsv")

    versions = pd.DataFrame([
        ("Python", platform.python_version()),
        ("numpy", package_version("numpy")),
        ("pandas", package_version("pandas")),
        ("scipy", package_version("scipy")),
        ("soundfile", package_version("soundfile")),
        ("pyarrow", package_version("pyarrow")),
        ("parselmouth", package_version("parselmouth")),
    ], columns=["software", "version"])
    atomic_tsv(versions, out / "audit/runtime_versions.tsv")

    manifest = {
        "pipeline_version": "rebuild_v7",
        "parent_pipeline": "rebuild_v6_trajectory",
        "analysis": "boundary_interior_specificity",
        "boundary_reference": "official V6 final six-feature trajectory outputs",
        "status": "V6_AUDIT_AND_INTERIOR_GEOMETRY_COMPLETE_BOUNDARY_REPRODUCTION_PENDING",
        "source_files": source_table.to_dict(orient="records"),
        "parameters": {
            "random_seed": SEED,
            "epsilon": EPS,
            "fft_points": N_FFT,
            "frame_ms": FRAME_MS,
            "frame_centres_ms": FRAME_TIMES_MS.astype(int).tolist(),
            "base_trajectory_half_support_ms": BASE_HALF_SUPPORT_MS,
            "cpps_trajectory_half_support_ms": CPPS_TRAJECTORY_HALF_SUPPORT_MS,
            "reproduction_absolute_tolerance": REPRODUCTION_TOLERANCE,
            "interior_weight_A": 0.5,
            "interior_weight_B": 0.5,
            "strict_no_boundary_touch": True,
        },
        "feature_definitions": {
            "energy_centroid_tilt": "official V6 15-ms Hann, 512-point real FFT trajectory",
            "flux": "official V6 eight successive normalized-magnitude differences",
            "flatness": "official V6 nine-point trajectory; stale scalar statement superseded by user clarification",
            "CPPS": "official V6 nine-point translated 40-ms Praat operator trajectory; stale scalar statement superseded by user clarification",
        },
        "model_formulas": {
            "official_v6": "feature_z ~ group * ns(relative_time_ms, df=4) + duration_A_c + duration_B_c + factor(exact_phone_pair)",
            "v7_benchmark_smoke": "feature_z ~ authenticity * region * ns(relative_time_ms,df=4) + duration controls + exact phone-pair FE + target-system FE + random intercepts for token occurrence, match, and utterance; engineering validation only",
        },
        "quality_gates": {
            "v6_boundary_max_abs_difference": REPRODUCTION_TOLERANCE,
            "matching_duration_smd": 0.25,
            "both_adjacent_interiors_required": True,
        },
        "exclusions": [],
        "timestamps": {"audit_created_utc": datetime.now(timezone.utc).isoformat()},
        "software_versions": versions.to_dict(orient="records"),
        "counts": {
            "official_v6_tokens": int(len(tokens)),
            "both_interiors_eligible": int(eligibility.both_interiors_eligible.sum()),
        },
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest["counts"], indent=2))


if __name__ == "__main__":
    main()
