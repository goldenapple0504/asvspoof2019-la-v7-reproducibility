"""Convert official finalized V7 result TSVs to the plotting CSV schema."""

import argparse
from pathlib import Path

import pandas as pd


def read(path):
    return pd.read_csv(path, sep="\t", low_memory=False)


def save(frame, out, name):
    frame.to_csv(out / name, index=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--out", type=Path, default=Path("data/figures_inputs"))
    a = p.parse_args()
    data, out = a.data, a.out
    out.mkdir(parents=True, exist_ok=True)
    rq1 = data / "results/rq1"
    inf = data / "results/inference"
    meta = data / "results/meta_analysis"

    avg = read(rq1 / "average_authenticity.tsv")
    save(avg.rename(columns={"estimate_tts_minus_bf": "estimate", "p_BH": "p_bh"})[
        ["system", "feature", "region", "estimate", "p_bh"]], out, "rq1_region_average.csv")
    time = read(rq1 / "time_resolved_authenticity.tsv")
    save(time.rename(columns={"relative_time_ms": "time_ms", "estimate_tts_minus_bf": "estimate",
                              "CI95_low": "ci_lo", "CI95_high": "ci_hi", "p_BH_global": "p_bh"})[
        ["system", "feature", "region", "time_ms", "estimate", "SE", "ci_lo", "ci_hi",
         "wald_z", "p_raw", "p_bonferroni_within_trajectory", "p_bh"]], out, "rq1_time_resolved.csv")

    a2 = read(inf / "average_specificity.tsv")
    a2["classification"] = a2.SESOI_classification.map(
        lambda s: "meaningful" if s.startswith("MEANINGFUL") else s.lower())
    save(a2.rename(columns={"CI95_low": "ci_lo", "CI95_high": "ci_hi", "p_BH_60": "p_bh"})[
        ["system", "feature", "estimate", "ci_lo", "ci_hi", "p_bh", "classification"]], out,
         "rq2_average.csv")
    t2 = read(inf / "time_resolved_specificity.tsv")
    save(t2.rename(columns={"relative_time_ms": "time_ms",
                            "predefined_followup_significant": "significant",
                            "SESOI_equivalent": "equivalent"})[
        ["system", "feature", "time_ms", "estimate", "significant", "equivalent"]], out, "rq2_time.csv")
    sh = read(inf / "joint_shape_specificity.tsv")
    save(sh.rename(columns={"p_BH_60": "p_bh"})[["system", "feature", "p_bh"]], out, "rq2_shape.csv")

    univariate = read(meta / "average_specificity_REML.tsv")
    univariate = univariate.loc[univariate.estimand.eq("POOLED")]
    save(univariate.rename(columns={"estimate": "pooled", "CI_lower": "ci_lo", "CI_upper": "ci_hi",
                                    "prediction_lower": "pi_lo", "prediction_upper": "pi_hi",
                                    "I_squared_percent": "I2"})[
        ["feature", "pooled", "ci_lo", "ci_hi", "pi_lo", "pi_hi", "I2", "tau"]], out,
         "meta_univariate.csv")
    multivariate = read(meta / "projected_specificity_trajectories_and_prediction_intervals.tsv")
    save(multivariate.rename(columns={"relative_time_ms": "time_ms", "projected_mean": "estimate",
                                         "mean_CI_lower": "ci_lo", "mean_CI_upper": "ci_hi",
                                         "prediction_lower": "pi_lo", "prediction_upper": "pi_hi"})[
        ["feature", "partition", "time_ms", "estimate", "ci_lo", "ci_hi", "pi_lo", "pi_hi"]],
         out, "meta_multivariate.csv")


if __name__ == "__main__":
    main()
