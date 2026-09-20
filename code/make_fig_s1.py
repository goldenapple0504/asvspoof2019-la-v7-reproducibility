"""Plot all 60 finalized direct-effect trajectories for supplementary Fig. S1."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


SYSTEMS = ["A01", "A02", "A03", "A04", "A07", "A08", "A09", "A10", "A11", "A12"]
FEATURES = ["energy", "centroid", "tilt", "flatness", "flux", "cpps"]
LABELS = {"energy": "Energy", "centroid": "Spectral centroid", "tilt": "Spectral tilt",
          "flatness": "Spectral flatness", "flux": "Spectral flux", "cpps": "CPPS"}
REGIONS = [("Boundary", "#1f77b4"), ("Interior_A", "#ff7f0e"), ("Interior_B", "#2ca02c")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=Path("data/figures_inputs/rq1_time_resolved.csv"))
    p.add_argument("--out", type=Path, default=Path("data/figures"))
    a = p.parse_args()
    df = pd.read_csv(a.input)
    assert len(df) == 1590
    fig, axes = plt.subplots(6, 10, figsize=(24, 15), sharex=True, sharey="row")
    for i, feature in enumerate(FEATURES):
        f = df.loc[df.feature.eq(feature)]
        low, high = f.ci_lo.min(), f.ci_hi.max()
        pad = 0.06 * (high - low)
        for j, system in enumerate(SYSTEMS):
            ax = axes[i, j]
            sub = f.loc[f.system.eq(system)]
            for region, color in REGIONS:
                g = sub.loc[sub.region.eq(region)].sort_values("time_ms")
                assert len(g) in (8, 9)
                ax.plot(g.time_ms, g.estimate, color=color, linewidth=1.15,
                        label=region.replace("_", " "))
                ax.fill_between(g.time_ms, g.ci_lo, g.ci_hi, color=color, alpha=0.12,
                                linewidth=0)
            ax.axhline(0, color="#9abfe0", linewidth=0.65)
            ax.axvline(0, color="#9abfe0", linewidth=0.6, linestyle=":")
            ax.set_ylim(low - pad, high + pad)
            if i == 0:
                ax.set_title(system, fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{LABELS[feature]}\nTTS − BF (SD)", fontsize=8)
            if i == len(FEATURES) - 1:
                ax.set_xlabel("Time (ms)", fontsize=8)
            ax.tick_params(labelsize=7)
            if j > 0:
                ax.tick_params(labelleft=False)
    fig.suptitle("Fig. S1. Direct TTS–bona fide effect trajectories for all 60 system–feature models",
                 fontsize=14)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="lower center", frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.005))
    fig.text(0.5, 0.028,
             "Lines show Aᵣ(t) = TTS − bona fide; shaded bands are unadjusted pointwise 95% confidence intervals.",
             ha="center", fontsize=9)
    fig.subplots_adjust(left=0.045, right=0.995, top=0.93, bottom=0.09, wspace=0.07, hspace=0.12)
    a.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out / "figS1_all_direct_effect_trajectories.pdf", bbox_inches="tight")
    fig.savefig(a.out / "figS1_all_direct_effect_trajectories.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
