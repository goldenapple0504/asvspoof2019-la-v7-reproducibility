"""SCIP-certified, outcome-blind CPPS duration-balance amendment V2 (A01 pilot)."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyscipopt
import scipy
from pyscipopt import Model, quicksum


SYSTEM = "A01"
AUTHORIZED_SYSTEMS = ("A01", "A02", "A03", "A04", "A07", "A08", "A09", "A10", "A11", "A12")
AMENDMENT = "cpps_duration_balance_amendment_v2"
TARGET_SMD = 0.20
FINAL_GATE = 0.25
STAGE_TIME_LIMIT_SECONDS = 86400.0
TIE_OBJECTIVE_TOLERANCE = 1e-8
ALLOWED_COLUMNS = (
    "token_id", "file_id", "partition", "system", "speaker_id", "phone_A", "phone_B",
    "exact_phone_pair", "boundary_time", "A_duration_ms", "B_duration_ms", "audio_path",
)
FORBIDDEN_OUTCOME_FRAGMENTS = ("raw_value", "feature_z", "cpps_value", "energy_value", "centroid_value",
                               "tilt_value", "flux_value", "flatness_value", "effect", "estimate", "outcome")


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_tsv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, sep="\t", index=False, lineterminator="\n")
    os.replace(temporary, path)


def stable_hash(frame: pd.DataFrame, columns: list[str]) -> str:
    payload = frame.loc[:, columns].sort_values(columns, kind="mergesort").to_csv(
        index=False, lineterminator="\n", float_format="%.17g"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_matching_input(frame: pd.DataFrame) -> pd.DataFrame:
    forbidden = [column for column in frame.columns if any(x in column.lower() for x in FORBIDDEN_OUTCOME_FRAGMENTS)]
    if forbidden:
        raise RuntimeError(f"Outcome/acoustic columns are prohibited from V2 matching: {sorted(forbidden)}")
    missing = set(ALLOWED_COLUMNS) - set(frame.columns)
    if missing:
        raise RuntimeError(f"Missing CPPS matching metadata: {sorted(missing)}")
    result = frame.loc[:, ALLOWED_COLUMNS].copy()
    result.token_id = result.token_id.astype(str)
    if result.token_id.duplicated().any() or not np.isfinite(result[["A_duration_ms", "B_duration_ms"]].to_numpy(float)).all():
        raise RuntimeError("Invalid V2 matching metadata")
    return result


def complete_three_region_tokens(project: Path, root: Path) -> set[str]:
    validity_files = sorted((root / "production/extraction/validity").glob("region_validity_*.tsv"))
    if len(validity_files) != 146:
        raise RuntimeError(f"Expected 146 validity checkpoints, found {len(validity_files)}")
    validity = pd.concat((pd.read_csv(path, sep="\t", usecols=["token_id", "region", "cpps_trajectory_valid"])
                          for path in validity_files), ignore_index=True)
    interior_complete = validity[validity.cpps_trajectory_valid].groupby("token_id").region.nunique()
    interior_tokens = set(interior_complete[interior_complete.eq(3)].index.astype(str))
    boundary = ds.dataset(project / "rebuild_v6_trajectory/cpp_flatness_extension/extracted/flatness_cpp_trajectories.parquet",
                          format="parquet").to_table(filter=ds.field("feature") == "cpps",
                                                     columns=["token_id", "relative_time_ms"]).to_pandas()
    boundary_complete = boundary.groupby("token_id").relative_time_ms.nunique()
    return interior_tokens & set(boundary_complete[boundary_complete.eq(9)].index.astype(str))


def build_edges(pool: pd.DataFrame) -> tuple[pd.DataFrame, int, dict[str, int]]:
    records: list[dict] = []
    upper = 0
    support = {"common_strata": 0, "bonafide_common_support": 0, "tts_common_support": 0}
    bf_support, tt_support = set(), set()
    bf_all, tt_all = pool[pool.system.eq("bonafide")], pool[pool.system.eq(SYSTEM)]
    for pair in sorted(set(bf_all.exact_phone_pair) & set(tt_all.exact_phone_pair)):
        bf = bf_all[bf_all.exact_phone_pair.eq(pair)].sort_values("token_id", kind="mergesort")
        tt = tt_all[tt_all.exact_phone_pair.eq(pair)].sort_values("token_id", kind="mergesort")
        upper += min(len(bf), len(tt)); support["common_strata"] += 1
        bf_support.update(bf.token_id); tt_support.update(tt.token_id)
        combined = pd.concat([bf[["A_duration_ms", "B_duration_ms"]], tt[["A_duration_ms", "B_duration_ms"]]])
        means, scales = combined.mean(), combined.std(ddof=0).replace(0, 1.0)
        bfz, ttz = (bf[["A_duration_ms", "B_duration_ms"]] - means) / scales, (tt[["A_duration_ms", "B_duration_ms"]] - means) / scales
        for b in bf.itertuples(index=False):
            for t in tt.itertuples(index=False):
                distance = float(math.hypot(float(bfz.loc[bfz.index[bf.token_id.eq(b.token_id)].item(), "A_duration_ms"] -
                                                   ttz.loc[ttz.index[tt.token_id.eq(t.token_id)].item(), "A_duration_ms"]),
                                            float(bfz.loc[bfz.index[bf.token_id.eq(b.token_id)].item(), "B_duration_ms"] -
                                                   ttz.loc[ttz.index[tt.token_id.eq(t.token_id)].item(), "B_duration_ms"])))
                records.append({"exact_phone_pair": pair, "bf_token_id": str(b.token_id), "tts_token_id": str(t.token_id),
                                "bf_A": float(b.A_duration_ms), "bf_B": float(b.B_duration_ms),
                                "tts_A": float(t.A_duration_ms), "tts_B": float(t.B_duration_ms), "distance": distance})
    edges = pd.DataFrame(records).sort_values(["exact_phone_pair", "bf_token_id", "tts_token_id"], kind="mergesort").reset_index(drop=True)
    edges["canonical_rank"] = np.arange(1, len(edges) + 1, dtype=np.int64)
    support["bonafide_common_support"] = len(bf_support); support["tts_common_support"] = len(tt_support)
    return edges, upper, support


def normalized_edges(edges: pd.DataFrame, dimension: str) -> tuple[np.ndarray, np.ndarray]:
    bf, tt = edges[f"bf_{dimension}"].to_numpy(float), edges[f"tts_{dimension}"].to_numpy(float)
    values = np.concatenate((bf, tt)); return (bf - values.mean()) / values.std(ddof=0), (tt - values.mean()) / values.std(ddof=0)


def configured_model(log_path: Path) -> Model:
    model = Model(f"V2_{SYSTEM}_CPPS_duration_balance")
    model.setIntParam("parallel/maxnthreads", 1)
    model.setIntParam("randomization/randomseedshift", 0)
    model.setIntParam("randomization/permutationseed", 0)
    model.setRealParam("limits/time", STAGE_TIME_LIMIT_SECONDS)
    # Keep native file logging disabled because SCIP logfile handles remain
    # open on Windows, but retain console output so long exact runs expose
    # live nodes, primal/dual bounds, and gap in the visible solver window.
    return model


def build_model(edges: pd.DataFrame, n: int, log_path: Path, objective: str,
                distance_limit: float | None = None, warm_start: list[int] | None = None) -> tuple[Model, list]:
    model = configured_model(log_path)
    aggregate_warm_values = []
    variables = [model.addVar(name=f"x_{i:05d}_{edge.bf_token_id}_{edge.tts_token_id}", vtype="B")
                 for i, edge in edges.iterrows()]
    for _, positions in edges.groupby("bf_token_id", sort=True).groups.items():
        model.addCons(quicksum(variables[i] for i in positions) <= 1)
    for _, positions in edges.groupby("tts_token_id", sort=True).groups.items():
        model.addCons(quicksum(variables[i] for i in positions) <= 1)
    model.addCons(quicksum(variables) == n, name="fixed_cardinality")
    for dimension in ("A", "B"):
        bf, tt = normalized_edges(edges, dimension)
        # Exact aggregate-moment reformulation: these linear linking equations
        # are algebraically identical to the original dense squared sums, but
        # keep the quadratic constraint four-dimensional rather than expanding
        # O(E^2) edge products in SCIP's expression graph.
        sb = model.addVar(name=f"sum_bonafide_{dimension}", lb=-1e20, ub=1e20, vtype="C")
        st = model.addVar(name=f"sum_tts_{dimension}", lb=-1e20, ub=1e20, vtype="C")
        qb = model.addVar(name=f"sumsq_bonafide_{dimension}", lb=-1e20, ub=1e20, vtype="C")
        qt = model.addVar(name=f"sumsq_tts_{dimension}", lb=-1e20, ub=1e20, vtype="C")
        model.addCons(sb == quicksum(float(bf[i]) * variables[i] for i in range(len(variables))))
        model.addCons(st == quicksum(float(tt[i]) * variables[i] for i in range(len(variables))))
        model.addCons(qb == quicksum(float(bf[i] * bf[i]) * variables[i] for i in range(len(variables))))
        model.addCons(qt == quicksum(float(tt[i] * tt[i]) * variables[i] for i in range(len(variables))))
        aggregate_warm_values.extend(((sb, bf), (st, tt), (qb, bf * bf), (qt, tt * tt)))
        c2 = TARGET_SMD ** 2
        expression = ((2 * n - 2) * (st - sb) * (st - sb) + c2 * n * (st * st + sb * sb)
                      - c2 * n * n * (qt + qb))
        model.addCons(expression <= 0, name=f"selected_sample_SMD_{dimension}_le_0_20")
    distance = quicksum(float(edges.distance.iloc[i]) * variables[i] for i in range(len(variables)))
    if distance_limit is not None:
        model.addCons(distance <= float(distance_limit), name="stage2_distance_optimality_fix")
    if objective == "distance":
        model.setObjective(distance, "minimize")
    elif objective == "canonical_tie":
        # Secondary objective after fixing the primary distance to its certified optimum.
        model.setObjective(quicksum(float(edges.canonical_rank.iloc[i]) * variables[i] for i in range(len(variables))), "minimize")
    elif objective == "feasibility":
        model.setObjective(0.0, "minimize")
    else:
        raise ValueError(objective)
    if warm_start is not None:
        solution = model.createSol()
        selected = set(warm_start)
        for i, variable in enumerate(variables):
            model.setSolVal(solution, variable, 1.0 if i in selected else 0.0)
        for variable, coefficients in aggregate_warm_values:
            model.setSolVal(solution, variable, float(sum(coefficients[i] for i in selected)))
        if not model.addSol(solution, free=True):
            raise RuntimeError("SCIP rejected the prespecified warm-start solution")
    return model, variables


def solve(model: Model, variables: list, stage: str, n: int, began: float) -> dict:
    model.optimize(); status = str(model.getStatus())
    result = {"stage": stage, "cardinality": n, "status": status, "elapsed_seconds": time.perf_counter() - began,
              "nodes": int(model.getNNodes()), "gap": float(model.getGap()) if status == "optimal" else math.nan,
              "primal_bound": float(model.getPrimalbound()), "dual_bound": float(model.getDualbound()),
              "solver_version": str(model.version()), "certified": status in {"optimal", "infeasible"}}
    if status == "optimal":
        result["objective"] = float(model.getObjVal())
        result["selected_indices"] = [i for i, variable in enumerate(variables) if model.getVal(variable) > .5]
    # Explicitly close SCIP's logfile handle. This is required on Windows before
    # a completed stage checkpoint/log can be atomically replaced or cleaned up.
    model.freeProb()
    return result


def smd(a: np.ndarray, b: np.ndarray) -> float:
    pooled = math.sqrt(((len(a)-1)*a.var(ddof=1)+(len(b)-1)*b.var(ddof=1))/(len(a)+len(b)-2))
    return float((b.mean()-a.mean())/pooled) if pooled else (0.0 if a.mean()==b.mean() else math.inf)


def validation(matches: pd.DataFrame, prior_pairs: int) -> dict:
    bf, tt = matches[matches.match_side.eq("bonafide")], matches[matches.match_side.eq("tts")]
    pairs = bf.merge(tt, on="match_id", suffixes=("_bf", "_tts"), validate="one_to_one")
    result = {"pair_count": len(pairs), "bonafide_count": len(bf), "tts_count": len(tt),
              "exact_phone_pair_integrity": bool((pairs.exact_phone_pair_bf == pairs.exact_phone_pair_tts).all()),
              "bonafide_one_to_one": bool(bf.token_id.is_unique), "tts_one_to_one": bool(tt.token_id.is_unique),
              "without_replacement": bool(bf.token_id.is_unique and tt.token_id.is_unique),
              "duration_A_bonafide_mean": float(bf.A_duration_ms.mean()), "duration_A_bonafide_sd": float(bf.A_duration_ms.std(ddof=1)),
              "duration_A_tts_mean": float(tt.A_duration_ms.mean()), "duration_A_tts_sd": float(tt.A_duration_ms.std(ddof=1)),
              "duration_B_bonafide_mean": float(bf.B_duration_ms.mean()), "duration_B_bonafide_sd": float(bf.B_duration_ms.std(ddof=1)),
              "duration_B_tts_mean": float(tt.B_duration_ms.mean()), "duration_B_tts_sd": float(tt.B_duration_ms.std(ddof=1)),
              "duration_A_SMD": smd(bf.A_duration_ms.to_numpy(float), tt.A_duration_ms.to_numpy(float)),
              "duration_B_SMD": smd(bf.B_duration_ms.to_numpy(float), tt.B_duration_ms.to_numpy(float)),
              "original_V1_pair_count": prior_pairs, "pair_retention": len(pairs) / prior_pairs}
    result["target_0_20_pass"] = abs(result["duration_A_SMD"]) <= TARGET_SMD + 1e-10 and abs(result["duration_B_SMD"]) <= TARGET_SMD + 1e-10
    result["final_0_25_pass"] = abs(result["duration_A_SMD"]) <= FINAL_GATE and abs(result["duration_B_SMD"]) <= FINAL_GATE
    result["candidate_sha256"] = stable_hash(matches, ["target_system", "match_id", "match_side", "token_id", "exact_phone_pair", "A_duration_ms", "B_duration_ms"])
    return result


def make_matches(pool: pd.DataFrame, edges: pd.DataFrame, selected: list[int]) -> pd.DataFrame:
    lookup = pool.set_index("token_id", drop=False); rows = []
    chosen = edges.iloc[selected].sort_values(["exact_phone_pair", "bf_token_id", "tts_token_id"], kind="mergesort").reset_index(drop=True)
    for sequence, edge in chosen.iterrows():
        match_id = f"V7_CPPS_V2_{SYSTEM}_{sequence:07d}"
        for side, token_id in (("bonafide", edge.bf_token_id), ("tts", edge.tts_token_id)):
            row = lookup.loc[token_id].to_dict(); row.update({"target_system": SYSTEM, "match_id": match_id, "match_side": side,
                "standardized_match_distance": float(edge.distance), "matching_algorithm": "SCIP 10.0 direct MIQCP; exact ordered-pair, one-to-one, outcome-blind; selected-sample SMD <=0.20; V2"})
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    global SYSTEM
    parser = argparse.ArgumentParser(); parser.add_argument("--project-root", type=Path, required=True); parser.add_argument("--v7-root", type=Path, required=True); parser.add_argument("--system", required=True); parser.add_argument("--attempt", default="aggregate_moment_reformulation"); parser.add_argument("--resume-stage2", action="store_true")
    args = parser.parse_args()
    if args.system not in AUTHORIZED_SYSTEMS: raise RuntimeError(f"Unauthorized V2 system: {args.system}")
    SYSTEM = args.system
    project, root = args.project_root.resolve(), args.v7_root.resolve(); out = root / "matching" / AMENDMENT / SYSTEM / args.attempt
    if out.exists() and not args.resume_stage2: raise RuntimeError("Refusing to overwrite an existing V2 A01 attempt directory")
    complete = complete_three_region_tokens(project, root)
    if len(complete) != 38483: raise RuntimeError(f"Frozen CPPS eligibility changed: {len(complete)}")
    tokens = pd.read_csv(project / "rebuild_v6_trajectory/extracted/trajectory_eligible_tokens.tsv", sep="\t", low_memory=False)
    tokens["exact_phone_pair"] = tokens.get("phone_pair", tokens.phone_A.astype(str) + "->" + tokens.phone_B.astype(str))
    pool = validate_matching_input(tokens[tokens.token_id.astype(str).isin(complete) & tokens.partition.eq("TRAIN") & tokens.system.isin(["bonafide", SYSTEM])].copy())
    edges, upper, support = build_edges(pool)
    prior = pd.read_csv(root / "matching/production_cpps_matching_gates.tsv", sep="\t"); prior_pairs = int(prior.loc[prior.target_system.eq(SYSTEM), "matched_pairs"].iloc[0])
    # An approved Stage-2 resume must reuse, rather than recreate, its certified Stage-1 directory.
    out.mkdir(parents=True, exist_ok=args.resume_stage2); input_record = {"system": SYSTEM, "attempt": args.attempt, "eligible_complete_three_region_tokens": len(complete), "eligible_bonafide": int((pool.system=="bonafide").sum()), "eligible_tts": int((pool.system==SYSTEM).sum()), "upper_cardinality": upper, "edge_count": len(edges), "support": support, "input_hash": stable_hash(pool, ["token_id", "system", "exact_phone_pair", "A_duration_ms", "B_duration_ms"]), "solver": {"pyscipopt": pyscipopt.__version__, "scip": str(Model().version()), "python": sys.executable, "scipy": scipy.__version__, "architecture": platform.architecture()[0]}, "settings": {"threads":1,"randomseedshift":0,"permutationseed":0,"stage_time_limit_seconds":STAGE_TIME_LIMIT_SECONDS,"tie_objective_tolerance":TIE_OBJECTIVE_TOLERANCE}, "formulation": "exact aggregate-moment MIQCP; linear links from edge selections to duration sums and sums-of-squares; direct selected-sample SMD quadratic constraints"}
    if not args.resume_stage2 or not (out / "input_and_solver_settings.json").is_file():
        atomic_text(out / "input_and_solver_settings.json", json.dumps(input_record, indent=2))
    attempts=[]; maximum=None
    if args.resume_stage2:
        attempt_path = out / "stage1_cardinality_attempts.tsv"; selected_path = out / "stage1_selected_indices.json"
        if not attempt_path.is_file() or not selected_path.is_file(): raise RuntimeError("Cannot resume Stage 2 without a certified Stage 1 checkpoint")
        attempts = pd.read_csv(attempt_path, sep="\t").to_dict("records")
        if len(attempts) != 1 or attempts[0]["status"] != "optimal" or not bool(attempts[0]["certified"]): raise RuntimeError("Stage 1 checkpoint is not certified optimal")
        maximum = int(attempts[0]["cardinality"])
    for n in ([] if maximum is not None else range(upper, 1, -1)):
        began=time.perf_counter(); model, variables=build_model(edges,n,out/f"stage1_N{n:04d}.scip.log","feasibility"); result=solve(model,variables,"stage1_maximum_cardinality",n,began); attempts.append({k:v for k,v in result.items() if k!="selected_indices"}); atomic_tsv(pd.DataFrame(attempts),out/"stage1_cardinality_attempts.tsv"); atomic_text(out/"stage1_checkpoint.json",json.dumps({"latest":attempts[-1],"attempts":len(attempts)},indent=2))
        if result["status"]=="optimal":
            maximum=n
            atomic_text(out / "stage1_selected_indices.json", json.dumps(result["selected_indices"]))
            break
        if result["status"]!="infeasible": raise RuntimeError(f"Stage 1 did not certify N={n}: {result['status']}")
    if maximum is None: raise RuntimeError("No feasible CPPS matching cardinality >=2 was certified")
    stage1_selected=json.loads((out / "stage1_selected_indices.json").read_text(encoding="utf-8"))
    began=time.perf_counter(); model,variables=build_model(edges,maximum,out/"stage2_distance.scip.log","distance",warm_start=stage1_selected); stage2=solve(model,variables,"stage2_minimum_distance",maximum,began)
    atomic_text(out / "stage2_checkpoint.json", json.dumps({k:v for k,v in stage2.items() if k != "selected_indices"}, indent=2))
    if stage2["status"]!="optimal": raise RuntimeError(f"Stage 2 did not certify distance optimum: {stage2['status']}")
    atomic_text(out / "stage2_selected_indices.json", json.dumps(stage2["selected_indices"]))
    distance_optimum=float(stage2["objective"]); began=time.perf_counter(); model,variables=build_model(edges,maximum,out/"stage3_tie.scip.log","canonical_tie",distance_optimum+TIE_OBJECTIVE_TOLERANCE,warm_start=stage2["selected_indices"]); stage3=solve(model,variables,"stage3_canonical_tie",maximum,began)
    atomic_text(out / "stage3_checkpoint.json", json.dumps({k:v for k,v in stage3.items() if k != "selected_indices"}, indent=2))
    if stage3["status"]!="optimal": raise RuntimeError(f"Stage 3 did not certify canonical tie solution: {stage3['status']}")
    matches=make_matches(pool,edges,stage3["selected_indices"]); audit=validation(matches,prior_pairs)
    if not audit["target_0_20_pass"] or not audit["final_0_25_pass"] or not audit["exact_phone_pair_integrity"] or not audit["without_replacement"]: raise RuntimeError("Independent V2 validation failed")
    atomic_tsv(matches,out/"candidate_cpps_matches.tsv"); atomic_tsv(pd.DataFrame([audit]),out/"independent_validation.tsv")
    report={"status":"CERTIFIED_A01_CANDIDATE_NOT_PROMOTED","maximum_certified_cardinality":maximum,"upper_cardinality":upper,"stage1_attempts":attempts,"stage2":{k:v for k,v in stage2.items() if k!="selected_indices"},"stage3":{k:v for k,v in stage3.items() if k!="selected_indices"},"independent_validation":audit,"production_promoted":False,"other_systems_run":False}
    atomic_text(out/"A01_pilot_report.json",json.dumps(report,indent=2)); print(json.dumps(report,indent=2))

if __name__=="__main__": main()
