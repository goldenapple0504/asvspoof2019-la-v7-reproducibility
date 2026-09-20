"""Rebuild the manuscript's numeric tables from finalized V7 TSV files."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SYSTEMS = ["A01", "A02", "A03", "A04", "A07", "A08", "A09", "A10", "A11", "A12"]
FEATURES = ["energy", "centroid", "tilt", "flatness", "flux", "cpps"]
SUPPLEMENTARY_FEATURES = ["energy", "centroid", "tilt", "flatness", "flux", "cpps"]
SUPPLEMENTARY_FEATURE_LABELS = {
    "energy": "Energy",
    "centroid": "Spectral centroid",
    "tilt": "Spectral tilt",
    "flatness": "Spectral flatness",
    "flux": "Spectral flux",
    "cpps": "CPPS",
}
ARCHITECTURES = {
    "A01": "VAE + AR LSTM-RNN → WaveNet",
    "A02": "VAE + AR LSTM-RNN → WORLD",
    "A03": "Feedforward NN → WORLD",
    "A04": "CART → waveform concatenation",
    "A07": "LSTM-RNN → WORLD + GAN",
    "A08": "AR LSTM-RNN → neural source-filter",
    "A09": "LSTM-RNN → Vocaine",
    "A10": "Attention seq2seq → WaveRNN",
    "A11": "Attention seq2seq → Griffin-Lim",
    "A12": "RNN → WaveNet",
}


def read(path):
    return pd.read_csv(path, sep="\t", low_memory=False)


def write(frame, out, name):
    frame.to_csv(out / name, sep="\t", index=False)


def make_tables(data: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    used = read(data / "utterances/v7_included_utterances.tsv")
    counts = used.groupby("system").file_id.nunique()
    assert counts.reindex(SYSTEMS).sum() == 34364 and counts["bonafide"] == 1984
    t1 = pd.DataFrame({
        "system": SYSTEMS + ["All TTS", "Bona fide"],
        "partition": ["TRAIN"] * 4 + ["EVAL"] * 6 + ["—", "TRAIN + EVAL"],
        "architecture": [ARCHITECTURES[s] for s in SYSTEMS] + ["—", "Natural speech"],
        "analysed_utterances": [int(counts[s]) for s in SYSTEMS] + [34364, 1984],
    })
    write(t1, out, "table1.tsv")

    avg = read(data / "results/rq1/average_authenticity.tsv")
    tests = read(data / "results/rq1/trajectory_tests.tsv")
    records = []
    for feature in FEATURES:
        a = avg.loc[avg.feature.eq(feature)]
        tt = tests.loc[tests.feature.eq(feature)]
        assert len(a) == 30 and len(tt) == 60
        records.append({
            "feature": feature,
            "significant_average": int(a.p_BH.lt(0.05).sum()),
            "significant_global": int(tt.loc[tt.test_type.eq("global_trajectory"), "p_BH"].lt(0.05).sum()),
            "significant_shape": int(tt.loc[tt.test_type.eq("shape"), "p_BH"].lt(0.05).sum()),
            "negative": int(a.estimate_tts_minus_bf.lt(0).sum()),
            "positive": int(a.estimate_tts_minus_bf.gt(0).sum()),
            "median_sd": float(a.estimate_tts_minus_bf.median()),
            "minimum_sd": float(a.estimate_tts_minus_bf.min()),
            "maximum_sd": float(a.estimate_tts_minus_bf.max()),
        })
    t2 = pd.DataFrame(records)
    assert list(t2[["significant_average", "significant_global", "significant_shape"]].sum()) == [148, 148, 91]
    write(t2, out, "table2.tsv")

    rows = []
    for level, source, sig_col, eq_col in [
        ("Average", data / "results/inference/average_specificity.tsv", "p_BH_60", "SESOI_classification"),
        ("Time-resolved", data / "results/inference/time_resolved_specificity.tsv", "predefined_followup_significant", "SESOI_equivalent"),
    ]:
        f = read(source)
        significant = f[sig_col].lt(0.05) if level == "Average" else f[sig_col].astype(bool)
        equivalent = f[eq_col].eq("EQUIVALENT") if level == "Average" else f[eq_col].astype(bool)
        for sig in (True, False):
            group = equivalent[significant.eq(sig)]
            rows.append({"level": level, "significant": sig, "equivalent": int(group.sum()),
                         "not_equivalent": int((~group).sum()), "total": int(len(group))})
    t3 = pd.DataFrame(rows)
    assert t3.groupby("level").total.sum().to_dict() == {"Average": 60, "Time-resolved": 530}
    write(t3, out, "table3.tsv")

    meta = read(data / "results/meta_analysis/average_specificity_REML.tsv")
    t4 = meta.loc[meta.estimand.eq("POOLED"), ["feature", "estimate", "CI_lower", "CI_upper",
                                               "tau", "I_squared_percent", "prediction_lower", "prediction_upper"]]
    assert len(t4) == 6
    write(t4.set_index("feature").loc[FEATURES].reset_index(), out, "table4.tsv")

    from prepare_v7_production_matching import smd
    sets = [("short_frame", "production_short_frame_matches.tsv"),
            ("flatness_official_extension", "production_flatness_matches.tsv"),
            ("cpps_complete_three_region", "production_cpps_matches.tsv")]
    balance = []
    for family, filename in sets:
        matches = read(data / "matching" / filename)
        for system in SYSTEMS:
            f = matches.loc[matches.target_system.eq(system)]
            bf = f.loc[f.match_side.eq("bonafide")]
            tt = f.loc[f.match_side.eq("tts")]
            assert len(bf) == len(tt) == f.match_id.nunique()
            da = smd(bf.A_duration_ms, tt.A_duration_ms)
            db = smd(bf.B_duration_ms, tt.B_duration_ms)
            balance.append({"feature_family": family, "target_system": system,
                            "matched_pairs": len(bf), "duration_A_SMD": da,
                            "duration_B_SMD": db,
                            "duration_SMD_0_25_pass": bool(max(abs(da), abs(db)) <= 0.25)})
    t_s1 = pd.DataFrame(balance)
    assert len(t_s1) == 30 and t_s1.duration_SMD_0_25_pass.all()
    write(t_s1, out, "table_s1_matching_balance.tsv")

    gates = read(data / "results/inference/model_acceptance_gates.tsv")
    assert len(gates) == 60 and gates.model_id.nunique() == 60
    removed = []
    for gate in gates.itertuples(index=False):
        attempts = read(data / "results/model_diagnostics" / f"{gate.model_id}_random_hierarchy.tsv")
        assert attempts.attempt.tolist() == list(range(1, len(attempts) + 1))
        assert attempts.iloc[-1].random_structure == gate.final_random_structure
        assert not bool(attempts.iloc[-1].singular)
        if len(attempts) == 1:
            assert "match_id" in gate.final_random_structure
            continue
        assert len(attempts) == 2
        first = attempts.iloc[0]
        assert bool(first.singular) and "match_id" in first.zero_variance_components.split(";")
        assert gate.final_random_structure == "token_occurrence_id + utterance_id"
        removed.append({
            "feature": SUPPLEMENTARY_FEATURE_LABELS[gate.feature],
            "system": gate.system,
            "initial_fit": "singular; match variance at zero",
            "final_random_structure": "token occurrence + utterance",
        })
    assert len(removed) == 19
    t_s2 = pd.DataFrame(removed)
    t_s2["feature_order"] = t_s2.feature.map(
        {SUPPLEMENTARY_FEATURE_LABELS[f]: i for i, f in enumerate(SUPPLEMENTARY_FEATURES)}
    )
    t_s2["system_order"] = t_s2.system.map({s: i for i, s in enumerate(SYSTEMS)})
    t_s2 = t_s2.sort_values(["feature_order", "system_order"]).drop(
        columns=["feature_order", "system_order"]
    )
    write(t_s2, out, "table_s2_random_structure.tsv")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--out", type=Path, default=Path("data/results/tables"))
    args = p.parse_args()
    make_tables(args.data, args.out)
