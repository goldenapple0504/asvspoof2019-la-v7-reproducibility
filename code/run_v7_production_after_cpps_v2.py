"""Resume authorized V7 production after certified CPPS V2 promotion.

Extraction and matching are intentionally absent from this controller.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

AUTHORIZATION_TEXT = "FULL_V7_RUN_AUTHORIZED"
R_SCRIPT = Path(r"C:\Program Files\R\R-4.6.1\bin\Rscript.exe")
SYSTEMS = ("A01", "A02", "A03", "A04", "A07", "A08", "A09", "A10", "A11", "A12")
STAGES = ("assemble_model_data", "fit_models", "finalize_inference", "meta_analysis", "final_validation")


def prevent_idle_sleep() -> None:
    """Keep Windows awake while this controller owns the production run."""
    if sys.platform == "win32":
        es_continuous, es_system_required = 0x80000000, 0x00000001
        result = ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | es_system_required)
        if not result:
            raise RuntimeError("SetThreadExecutionState failed; refusing an unprotected long run")


def restore_sleep_policy() -> None:
    if sys.platform == "win32":
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_authorization(root: Path, execute_full: bool) -> None:
    marker = root / "FULL_V7_RUN_AUTHORIZED.txt"
    if (not execute_full or not marker.is_file()
            or marker.read_text(encoding="utf-8").strip() != AUTHORIZATION_TEXT):
        raise RuntimeError("Continuation controller requires full-run authorization")


def preflight(root: Path) -> dict:
    promotion = root / "matching/cpps_duration_balance_amendment_v2/promotion_v2"
    record_path = promotion / "promotion_record.json"
    installed_path = promotion / "installed_file_hashes.tsv"
    if not record_path.is_file() or not installed_path.is_file():
        raise RuntimeError("Certified CPPS V2 promotion record is missing")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record.get("status") != "PROMOTED_CERTIFIED_CPPS_V2":
        raise RuntimeError("Certified CPPS V2 promotion status is invalid")
    installed = pd.read_csv(installed_path, sep="\t")
    for row in installed.itertuples(index=False):
        path = root / str(row.destination)
        if not path.is_file() or path.stat().st_size != int(row.bytes) or sha256(path) != str(row.sha256):
            raise RuntimeError(f"Promoted CPPS file integrity failed: {path}")
    gates = pd.read_csv(root / "matching/production_cpps_matching_gates.tsv", sep="\t")
    if (list(gates.target_system.astype(str)) != list(SYSTEMS) or len(gates) != 10
            or not gates.all_gates_pass.eq(True).all()
            or not (gates[["duration_A_SMD", "duration_B_SMD"]].abs() <= 0.25).all().all()):
        raise RuntimeError("Promoted CPPS matching gates failed continuation preflight")
    checkpoints = sorted((root / "production/extraction/checkpoints").glob("chunk_*.json"))
    if len(checkpoints) != 146:
        raise RuntimeError(f"Extraction checkpoint count changed: {len(checkpoints)}")
    frozen = pd.read_csv(
        root / "matching/cpps_duration_balance_amendment_v2/final_candidate_export/frozen_extraction_hashes_after.tsv",
        sep="\t",
    )
    if len(frozen) != 730:
        raise RuntimeError("Frozen extraction manifest no longer contains 730 files")
    for row in frozen.itertuples(index=False):
        path = root / str(row.relative_path)
        if not path.is_file() or path.stat().st_size != int(row.bytes) or sha256(path) != str(row.sha256):
            raise RuntimeError(f"Frozen extraction integrity failed before continuation: {path}")
    return {"extraction_checkpoints": 146, "frozen_extraction_files": 730,
            "cpps_systems": 10, "cpps_matching_gate": "PASS"}


def run(command: list[str], stage: str, log) -> None:
    start = datetime.now(timezone.utc)
    event = {"event": "stage_start", "stage": stage, "timestamp_utc": start.isoformat(), "command": command}
    log.write(json.dumps(event) + "\n"); log.flush()
    print(f"=== START {stage} {start.isoformat()} ===", flush=True)
    completed = subprocess.run(command, check=False)
    end = datetime.now(timezone.utc)
    result = {"event": "stage_end", "stage": stage, "timestamp_utc": end.isoformat(),
              "returncode": completed.returncode, "elapsed_seconds": (end - start).total_seconds()}
    log.write(json.dumps(result) + "\n"); log.flush()
    print(f"=== END {stage} code={completed.returncode} elapsed={result['elapsed_seconds']:.1f}s ===", flush=True)
    if completed.returncode != 0:
        raise RuntimeError(f"Production continuation stopped at {stage}; exit code {completed.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--execute-full", action="store_true")
    parser.add_argument("--start-stage", choices=STAGES, default="assemble_model_data")
    args = parser.parse_args()
    project, root = args.project_root.resolve(), args.output_root.resolve()
    require_authorization(root, args.execute_full)
    if not R_SCRIPT.is_file():
        raise RuntimeError(f"Required explicit R executable missing: {R_SCRIPT}")
    evidence = preflight(root)
    print("CONTINUATION PREFLIGHT " + json.dumps(evidence, sort_keys=True), flush=True)
    scripts = root / "scripts"
    commands = [
        ("assemble_model_data", [sys.executable, str(scripts / "assemble_v7_production_model_data.py"),
                                 "--project-root", str(project), "--output-root", str(root), "--execute-full"]),
        ("fit_models", [str(R_SCRIPT), str(scripts / "fit_v7_production_models.R"), str(root), "--execute-full"]),
        ("finalize_inference", [str(R_SCRIPT), str(scripts / "finalize_v7_production_inference.R"), str(root), "--execute-full"]),
        ("meta_analysis", [str(R_SCRIPT), str(scripts / "run_v7_specificity_meta_analysis.R"), str(root), "--execute-full"]),
        ("final_validation", [sys.executable, str(scripts / "validate_v7_production_outputs.py"),
                              "--output-root", str(root), "--execute-full"]),
    ]
    commands = commands[STAGES.index(args.start_stage):]
    if args.start_stage != "assemble_model_data":
        gates_path = root / "production/model_data/model_data_gates.tsv"
        if not gates_path.is_file():
            raise RuntimeError("Cannot resume after assembly: model_data_gates.tsv is missing")
        gates = pd.read_csv(gates_path, sep="\t")
        if len(gates) != 60 or not gates.all_gates_pass.eq(True).all():
            raise RuntimeError("Cannot resume after assembly: expected 60 passing model-data gates")
    log_path = root / "logs/production_after_cpps_v2.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    prevent_idle_sleep()
    try:
        with log_path.open("a", encoding="utf-8") as log:
            log.write(json.dumps({"event": "continuation_preflight", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                                  "evidence": evidence, "start_stage": args.start_stage,
                                  "idle_sleep_prevention": "ES_CONTINUOUS|ES_SYSTEM_REQUIRED",
                                  "explicitly_omitted_stages": ["extraction", "matching"]}) + "\n")
            log.flush()
            for stage, command in commands:
                run(command, stage, log)
        print("V7 PRODUCTION CONTINUATION COMPLETE", flush=True)
    finally:
        restore_sleep_policy()


if __name__ == "__main__":
    main()
