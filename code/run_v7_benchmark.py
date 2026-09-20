"""Deterministic ~2,000-token V7 extraction benchmark; no scientific inference."""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import platform
import sys
import threading
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from v7_common import (
    CPPS_TRAJECTORY_HALF_SUPPORT_MS,
    FLUX_TIMES_MS,
    FRAME_TIMES_MS,
    SEED,
    SYSTEMS,
    atomic_tsv,
    mono_float64,
    spectral_flux,
    spectral_frame_features,
    v6_frame_indices,
)


BENCHMARK_MATCHES_PER_SYSTEM = 100
EXPECTED_OCCURRENCES = 2 * BENCHMARK_MATCHES_PER_SYSTEM * len(SYSTEMS)


def import_cpps(project: Path):
    scripts = str(project / "rebuild_v6_trajectory/scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import extract_flatness_cpp_trajectories as extension  # type: ignore
    return extension


def working_set_bytes() -> int:
    """Current Windows working set without an external psutil dependency."""
    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
        ]
    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(counters)
    handle = ctypes.windll.kernel32.GetCurrentProcess()
    ok = ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
    return int(counters.WorkingSetSize) if ok else 0


class MemoryMonitor:
    def __init__(self, interval: float = 0.05):
        self.interval = interval
        self.peak = 0
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop_event.is_set():
            try:
                import psutil
                process = psutil.Process(os.getpid())
                processes = [process] + process.children(recursive=True)
                current = sum(item.memory_info().rss for item in processes if item.is_running())
            except Exception:
                current = working_set_bytes()
            self.peak = max(self.peak, current)
            self.stop_event.wait(self.interval)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop_event.set()
        self.thread.join()
        self.peak = max(self.peak, working_set_bytes())


def select_manifest(project: Path, output: Path) -> pd.DataFrame:
    path = output / "audit/benchmark_token_ids.tsv"
    if path.exists():
        manifest = pd.read_csv(path, sep="\t", low_memory=False)
        if len(manifest) != EXPECTED_OCCURRENCES:
            raise RuntimeError("Persistent benchmark manifest has unexpected size")
        return manifest
    matched = pd.read_csv(project / "rebuild_v6_trajectory/matching/trajectory_matched_tokens.tsv", sep="\t", low_memory=False)
    matched = matched[(matched.window_ms_per_side == 20) & (matched.matching_set == "non_cpp")].copy()
    eligibility = pd.read_csv(output / "diagnostics/interior_eligibility.tsv", sep="\t", low_memory=False)
    eligibility = eligibility[eligibility.both_interiors_eligible].copy()
    meta_columns = [
        "token_id", "label", "audio_path", "audio_frames", "sample_rate", "boundary_time",
        "phone_A_start_sec", "phone_A_end_sec", "phone_B_start_sec", "phone_B_end_sec",
        "midpoint_A_sec", "midpoint_B_sec", "v6_boundary_eligible",
    ]
    matched = matched.merge(eligibility[meta_columns], on="token_id", how="inner", validate="many_to_one", suffixes=("", "_v7"))
    rng = np.random.default_rng(SEED)
    pieces = []
    for system in SYSTEMS:
        group = matched[matched.target_system == system]
        ids = np.array(sorted(group.match_id.unique()), dtype=object)
        if len(ids) < BENCHMARK_MATCHES_PER_SYSTEM:
            raise RuntimeError(f"Insufficient benchmark matches for {system}")
        chosen = set(rng.choice(ids, size=BENCHMARK_MATCHES_PER_SYSTEM, replace=False).tolist())
        part = group[group.match_id.isin(chosen)].copy()
        if len(part) != 2 * BENCHMARK_MATCHES_PER_SYSTEM or not part.groupby("match_id").size().eq(2).all():
            raise RuntimeError(f"Incomplete benchmark pairs for {system}")
        pieces.append(part)
    manifest = pd.concat(pieces, ignore_index=True).sort_values(["target_system", "match_id", "match_side"])
    manifest["benchmark_seed"] = SEED
    manifest["token_occurrence_id"] = manifest.target_system.astype(str) + "::" + manifest.token_id.astype(str)
    atomic_tsv(manifest, path)
    return manifest


def strict_cpps_phone_support(start: float, end: float, midpoint: float, sample_rate: int) -> tuple[bool, float, float]:
    left = (midpoint - CPPS_TRAJECTORY_HALF_SUPPORT_MS / 1000 - start) * sample_rate
    right = (end - midpoint - CPPS_TRAJECTORY_HALF_SUPPORT_MS / 1000) * sample_rate
    return bool(left > 0 and right > 0), float(left), float(right)


def extract_audio_group(payload):
    project_text, records = payload
    project = Path(project_text)
    extension = import_cpps(project)
    frame = pd.DataFrame.from_records(records)
    rows: list[dict] = []
    failures: list[dict] = []
    path = str(frame.audio_path.iloc[0])
    try:
        audio_raw, sample_rate = sf.read(path, dtype="float64", always_2d=False)
        audio = mono_float64(audio_raw)
        sample_rate = int(sample_rate)
    except Exception as exc:
        return [], [{"token_id": t, "region": "ALL", "feature": "ALL", "failure_reason": f"audio:{type(exc).__name__}:{exc}"} for t in frame.token_id], []
    try:
        import parselmouth
        sound = parselmouth.Sound(path)
    except Exception as exc:
        sound = None
        sound_error = f"cpps_audio:{type(exc).__name__}:{exc}"
    else:
        sound_error = ""

    region_qc: list[dict] = []
    for token in frame.drop_duplicates("token_id").itertuples(index=False):
        base = {
            "source_token_id": token.token_id,
            "file_id": token.file_id,
            "utterance_id": token.file_id,
            "partition": token.analysis_family,
            "system": token.system,
            "authenticity": "bona_fide" if token.match_side == "bonafide" else "TTS",
            "phone_A": token.phone_A,
            "phone_B": token.phone_B,
            "exact_phone_pair": token.exact_phone_pair,
            "boundary_class": token.boundary_class,
            "boundary_time": float(token.boundary_time),
            "duration_A_ms": float(token.A_duration_ms),
            "duration_B_ms": float(token.B_duration_ms),
            "sample_rate": sample_rate,
        }
        regions = (
            ("Boundary", "", float(token.boundary_time), None, None),
            ("Interior_A", "A", float(token.midpoint_A_sec), float(token.phone_A_start_sec), float(token.phone_A_end_sec)),
            ("Interior_B", "B", float(token.midpoint_B_sec), float(token.phone_B_start_sec), float(token.phone_B_end_sec)),
        )
        for region, side, center, phone_start, phone_end in regions:
            magnitudes = []
            spectral_ok = True
            for relative_ms in FRAME_TIMES_MS.astype(int):
                start, end, center_error = v6_frame_indices(center, int(relative_ms), sample_rate)
                if start < 0 or end > len(audio):
                    failures.append({"token_id": token.token_id, "region": region, "feature": "spectral", "failure_reason": f"audio_support:{relative_ms}"})
                    spectral_ok = False
                    break
                if phone_start is not None and not (start - phone_start * sample_rate > 0 and phone_end * sample_rate - end > 0):
                    failures.append({"token_id": token.token_id, "region": region, "feature": "spectral", "failure_reason": f"phone_support:{relative_ms}"})
                    spectral_ok = False
                    break
                magnitude, values = spectral_frame_features(audio[start:end], sample_rate)
                magnitudes.append(magnitude)
                for feature in ("energy", "centroid", "tilt", "flatness"):
                    rows.append({
                        **base, "region": region, "interior_side": side, "relative_time_ms": int(relative_ms),
                        "feature": feature, "raw_value": values[feature], "frame_start_sample": start,
                        "frame_end_sample": end, "requested_center_sec": center + relative_ms / 1000,
                        "center_error_samples": center_error, "valid": True, "failure_reason": "",
                    })
            if spectral_ok:
                for index, relative_ms in enumerate(FLUX_TIMES_MS.astype(int)):
                    rows.append({
                        **base, "region": region, "interior_side": side, "relative_time_ms": int(relative_ms),
                        "feature": "flux", "raw_value": spectral_flux(magnitudes[index], magnitudes[index + 1]),
                        "frame_start_sample": np.nan, "frame_end_sample": np.nan,
                        "requested_center_sec": center + relative_ms / 1000, "center_error_samples": np.nan,
                        "valid": True, "failure_reason": "",
                    })

            cpps_support = True
            cpps_left = cpps_right = np.nan
            cpps_reason = ""
            if sound is None:
                cpps_support = False
                cpps_reason = sound_error
            elif region == "Boundary":
                duration = float(sound.get_total_duration())
                cpps_support = bool(center - 0.040 >= -1e-12 and center + 0.040 <= duration + 1e-12 and token.cpp_status == "ok" and np.isfinite(token.boundary_cpp_db))
                cpps_reason = "" if cpps_support else "official_v6_boundary_cpps_invalid_or_incomplete_support"
            else:
                cpps_support, cpps_left, cpps_right = strict_cpps_phone_support(phone_start, phone_end, center, sample_rate)
                cpps_reason = "" if cpps_support else "incomplete_strict_phone_support_for_nine_point_cpps_trajectory"
            if cpps_support:
                try:
                    values, _ = extension.trajectory_cpps(sound, center)
                    for relative_ms, value, _, _, error in values:
                        rows.append({
                            **base, "region": region, "interior_side": side, "relative_time_ms": int(relative_ms),
                            "feature": "cpps", "raw_value": float(value), "frame_start_sample": np.nan,
                            "frame_end_sample": np.nan, "requested_center_sec": center + relative_ms / 1000,
                            "center_error_samples": float(error), "valid": True, "failure_reason": "",
                        })
                except Exception as exc:
                    cpps_support = False
                    cpps_reason = f"cpps:{type(exc).__name__}:{exc}"
            if not cpps_support:
                failures.append({"token_id": token.token_id, "region": region, "feature": "cpps", "failure_reason": cpps_reason})
            region_qc.append({
                "token_id": token.token_id, "region": region, "spectral_trajectory_valid": spectral_ok,
                "cpps_trajectory_valid": cpps_support, "cpps_left_phone_margin_samples": cpps_left,
                "cpps_right_phone_margin_samples": cpps_right, "cpps_failure_reason": cpps_reason,
            })
    return rows, failures, region_qc


def validate_output(data: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    checks = []
    def add(name, passed, observed, expected):
        checks.append({"check": name, "pass": bool(passed), "observed": str(observed), "expected": str(expected)})
    add("manifest_occurrences", len(manifest) == EXPECTED_OCCURRENCES, len(manifest), EXPECTED_OCCURRENCES)
    add("all_target_systems", set(manifest.target_system) == set(SYSTEMS), sorted(manifest.target_system.unique()), list(SYSTEMS))
    add("both_match_sides", set(manifest.match_side) == {"bonafide", "tts"}, sorted(manifest.match_side.unique()), ["bonafide", "tts"])
    add("regions", set(data.region) == {"Boundary", "Interior_A", "Interior_B"}, sorted(data.region.unique()), ["Boundary", "Interior_A", "Interior_B"])
    add("features", set(data.feature) == {"energy", "centroid", "tilt", "flux", "flatness", "cpps"}, sorted(data.feature.unique()), ["energy", "centroid", "tilt", "flux", "flatness", "cpps"])
    add("no_nan_raw", not data.raw_value.isna().any(), int(data.raw_value.isna().sum()), 0)
    add("all_finite_raw", np.isfinite(data.raw_value).all(), int((~np.isfinite(data.raw_value)).sum()), 0)
    duplicated = data.duplicated(["source_token_id", "region", "feature", "relative_time_ms"]).sum()
    add("no_duplicate_unique_token_rows", duplicated == 0, int(duplicated), 0)
    expected_grids = {
        "energy": FRAME_TIMES_MS.astype(int).tolist(), "centroid": FRAME_TIMES_MS.astype(int).tolist(),
        "tilt": FRAME_TIMES_MS.astype(int).tolist(), "flatness": FRAME_TIMES_MS.astype(int).tolist(),
        "cpps": FRAME_TIMES_MS.astype(int).tolist(), "flux": FLUX_TIMES_MS.astype(int).tolist(),
    }
    bad_grids = 0
    for (_, _, feature), group in data.groupby(["source_token_id", "region", "feature"]):
        if sorted(group.relative_time_ms.astype(int).tolist()) != expected_grids[feature]:
            bad_grids += 1
    add("complete_valid_feature_grids", bad_grids == 0, bad_grids, 0)
    result = pd.DataFrame(checks)
    return result


def compare_sampled_boundaries(project: Path, data: pd.DataFrame) -> pd.DataFrame:
    expected_four = pd.read_parquet(project / "rebuild_v6_trajectory/extracted/trajectory_features_full.parquet", filters=[("token_id", "in", data.source_token_id.unique().tolist())], columns=["token_id", "feature", "relative_time_ms", "raw_value"])
    expected_ext = pd.read_parquet(project / "rebuild_v6_trajectory/cpp_flatness_extension/extracted/flatness_cpp_trajectories.parquet", filters=[("token_id", "in", data.source_token_id.unique().tolist())], columns=["token_id", "feature", "relative_time_ms", "raw_value"])
    expected = pd.concat([expected_four, expected_ext], ignore_index=True)
    actual = data[data.region == "Boundary"][["source_token_id", "feature", "relative_time_ms", "raw_value"]].rename(columns={"source_token_id": "token_id", "raw_value": "reproduced"})
    merged = expected.rename(columns={"raw_value": "official"}).merge(actual, on=["token_id", "feature", "relative_time_ms"], how="outer", indicator=True)
    merged["absolute_difference"] = (merged.official - merged.reproduced).abs()
    return merged.groupby("feature", dropna=False).agg(
        official_rows=("official", "count"), reproduced_rows=("reproduced", "count"),
        maximum_absolute_difference=("absolute_difference", "max"), mean_absolute_difference=("absolute_difference", "mean"),
        unmatched_rows=("_merge", lambda x: int((x != "both").sum())),
    ).reset_index()


def write_smoke_model_input(data: pd.DataFrame, manifest: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Expand unique acoustics to matched occurrences and retain complete tiny sets."""
    expanded = manifest[[
        "token_id", "target_system", "match_id", "match_side", "token_occurrence_id",
    ]].merge(data, left_on="token_id", right_on="source_token_id", how="inner", validate="many_to_many")
    selected = []
    expected = {"flux": 8, "energy": 9, "centroid": 9, "tilt": 9, "flatness": 9, "cpps": 9}
    for feature in expected:
        feature_rows = expanded[expanded.feature == feature]
        counts = feature_rows.groupby(["target_system", "match_id", "token_occurrence_id", "region"]).size().rename("rows").reset_index()
        complete_token = counts.groupby(["target_system", "match_id", "token_occurrence_id"]).agg(
            regions=("region", "nunique"), complete_grids=("rows", lambda values: bool((values == expected[feature]).all())),
        )
        complete_token["complete"] = (complete_token.regions == 3) & complete_token.complete_grids
        complete_match = complete_token.loc[complete_token.complete].groupby(level=[0, 1]).size().eq(2)
        valid_ids = set(complete_match[complete_match].index)
        for target_system in SYSTEMS:
            ids = sorted(match_id for system, match_id in valid_ids if system == target_system)[:20]
            if ids:
                selected.append(feature_rows[(feature_rows.target_system == target_system) & feature_rows.match_id.isin(ids)])
    smoke = pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()
    if smoke.empty:
        raise RuntimeError("No complete matched sets for modelling smoke test")
    atomic_tsv(smoke, path)
    return smoke


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=max(1, min(12, (os.cpu_count() or 2) - 1)))
    args = parser.parse_args()
    project = args.project_root.resolve()
    output = args.output_root.resolve()
    dependency_path = output / "python_libs"
    if dependency_path.exists():
        sys.path.insert(0, str(dependency_path))
    benchmark = output / "diagnostics/benchmark"
    benchmark.mkdir(parents=True, exist_ok=True)
    manifest = select_manifest(project, output)
    # The frozen V6 matched-token manifest already carries the inherited V5
    # CPPS status/value.  Deduplicate repeated bona-fide occurrences only for
    # acoustic extraction; matching occurrences are restored for smoke models.
    unique = manifest.drop_duplicates("token_id").copy()
    groups = [(str(project), group.to_dict(orient="records")) for _, group in unique.groupby("audio_path", sort=True)]
    all_rows: list[dict] = []
    failures: list[dict] = []
    region_qc: list[dict] = []
    warning_rows = []
    began = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught, MemoryMonitor() as memory:
        warnings.simplefilter("always")
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for rows, failed, qc in executor.map(extract_audio_group, groups, chunksize=1):
                all_rows.extend(rows)
                failures.extend(failed)
                region_qc.extend(qc)
        warning_rows = [{"category": item.category.__name__, "message": str(item.message)} for item in caught]
    elapsed = time.perf_counter() - began
    data = pd.DataFrame(all_rows)
    scaling = pd.read_csv(project / "rebuild_v6_trajectory/final_six_feature_trajectory/all6_scaling_parameters.tsv", sep="\t")
    data = data.merge(scaling[["feature", "feature_mean", "feature_sd"]], on="feature", how="left", validate="many_to_one")
    data["feature_z"] = (data.raw_value - data.feature_mean) / data.feature_sd
    data.to_parquet(output / "extracted_features/v7_benchmark_boundary_interior_features_long.parquet", index=False, compression="zstd")
    failure_frame = pd.DataFrame(failures)
    expected_mask = failure_frame.feature.eq("cpps") & failure_frame.failure_reason.astype(str).str.startswith((
        "official_v6_boundary_cpps_invalid_or_incomplete_support",
        "incomplete_strict_phone_support_for_nine_point_cpps_trajectory",
    ))
    cpps_exclusions = failure_frame[expected_mask].copy().rename(columns={"failure_reason": "exclusion_reason"})
    cpps_exclusions["classification"] = "expected_cpps_complete_three_region_validity_exclusion"
    unexpected_errors = failure_frame[~expected_mask].copy().rename(columns={"failure_reason": "error"})
    atomic_tsv(cpps_exclusions, benchmark / "cpps_validity_exclusions.tsv")
    atomic_tsv(unexpected_errors, benchmark / "unexpected_extraction_errors.tsv")
    atomic_tsv(pd.DataFrame(region_qc), benchmark / "region_validity.tsv")
    atomic_tsv(pd.DataFrame(warning_rows, columns=["category", "message"]), benchmark / "warnings.tsv")
    schema = validate_output(data, manifest)
    atomic_tsv(schema, benchmark / "schema_validation.tsv")
    boundary = compare_sampled_boundaries(project, data)
    boundary["pass_1e_6"] = (boundary.unmatched_rows == 0) & (boundary.maximum_absolute_difference <= 1e-6)
    atomic_tsv(boundary, benchmark / "sampled_boundary_reproduction.tsv")
    smoke = write_smoke_model_input(data, manifest, output / "models/benchmark_smoke_input.tsv")
    unique_tokens = int(unique.token_id.nunique())
    metrics = pd.DataFrame([{
        "benchmark_seed": SEED,
        "matched_token_occurrences": int(len(manifest)),
        "unique_source_tokens": unique_tokens,
        "target_systems": int(manifest.target_system.nunique()),
        "workers": args.workers,
        "elapsed_seconds": elapsed,
        "unique_tokens_per_second": unique_tokens / elapsed,
        "token_occurrences_per_second": len(manifest) / elapsed,
        "peak_process_tree_working_set_mb": memory.peak / 1024**2,
        "output_rows": int(len(data)),
        "smoke_model_input_rows": int(len(smoke)),
        "cpps_validity_exclusions": int(len(cpps_exclusions)),
        "unexpected_extraction_errors": int(len(unexpected_errors)),
        "warnings": int(len(warning_rows)),
        "projected_full_extraction_hours_linear_unique_token_rate": (482_509 / (unique_tokens / elapsed)) / 3600,
        "scientific_inference_allowed": False,
    }])
    atomic_tsv(metrics, benchmark / "extraction_benchmark_metrics.tsv")
    print(json.dumps(metrics.iloc[0].to_dict(), indent=2))
    if not schema["pass"].all() or not boundary.pass_1e_6.all() or not unexpected_errors.empty:
        raise RuntimeError("V7 benchmark extraction/schema Boundary gate failed")


if __name__ == "__main__":
    main()
