"""
Figures for the V7 Results section (phoneme-boundary TTS paper).

All figures are drawn with matplotlib from the frozen V7 outputs.
Nothing is simulated: every data figure reads a CSV exported from the
pipeline. Fig. 1 (Methods schematic) needs no data.

Usage
-----
    python make_figures.py --data data/ --out figures/

Expected CSV files in --data (column names must match):

  rq1_time_resolved.csv  Fig. 3
      system, feature, region, time_ms, estimate, ci_lo, ci_hi
      (direct TTS-minus-bona fide trajectory A_r(t); region = Boundary / Interior_A / Interior_B)

  rq1_region_average.csv  Fig. 2
      system, feature, region, estimate, p_bh

  rq2_average.csv         Fig. 4, Fig. 6
      system, feature, estimate, ci_lo, ci_hi, p_bh, classification
      (classification = equivalent / meaningful / inconclusive;
       ci_lo/ci_hi = reported 95% confidence interval; classification is
       read from the adjusted TOST output and is not inferred from that CI)

  rq2_time.csv            Fig. 5
      system, feature, time_ms, estimate, significant, equivalent
      (significant / equivalent = True/False after the corrections in 2.5)

  rq2_shape.csv           Fig. 5 (row markers)
      system, feature, p_bh

  meta_univariate.csv     Fig. 6
      feature, pooled, ci_lo, ci_hi, pi_lo, pi_hi, I2, tau

  meta_multivariate.csv   Fig. 7
      feature, partition, time_ms, estimate, ci_lo, ci_hi, pi_lo, pi_hi
      (partition = TRAIN or EVAL; rows are the partition-moderated projected
       trajectories from the primary multivariate REML output)

Feature labels in the CSVs must match FEATURES below (edit the dict if
the pipeline uses other names). Any figure whose CSV is missing is skipped.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

# ------------------------------------------------------------------ settings
SESOI = 0.10
ALPHA = 0.05

# pipeline label -> label printed on figures (order = order in Methods 2.3.2)
FEATURES = {
    "energy": "Energy",
    "centroid": "Spectral centroid",
    "tilt": "Spectral tilt",
    "flatness": "Spectral flatness",
    "flux": "Spectral flux",
    "cpps": "CPPS",
}
SYSTEMS = ["A01", "A02", "A03", "A04", "A07", "A08", "A09", "A10", "A11", "A12"]
TRAIN_SYSTEMS = {"A01", "A02", "A03", "A04"}
REGIONS = ["Boundary", "Interior_A", "Interior_B"]
REGION_LABEL = {"Boundary": "Boundary", "Interior_A": "Interior A",
                "Interior_B": "Interior B"}

GRID_9 = [-20, -15, -10, -5, 0, 5, 10, 15, 20]
GRID_FLUX = [-15, -10, -5, 0, 5, 10, 15, 20]

# Okabe-Ito categorical colours (colour-blind safe).
# Diverging heatmaps use matplotlib RdBu_r, not the Okabe-Ito palette.
C_TTS = "#D55E00"
C_BF = "#0072B2"
C_EQ = "#009E73"
C_INC = "#E69F00"
C_MEAN = "#CC79A7"
C_GREY = "0.35"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,  # editable text in PDF
})

# Elsevier widths: single column 90 mm, double column 190 mm
MM = 1 / 25.4
W1, W2 = 90 * MM, 190 * MM


def grid_for(feature):
    return GRID_FLUX if feature == "flux" else GRID_9


def save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def load(data, fname):
    path = data / fname
    if not path.exists():
        print(f"skip: {fname} not found")
        return None
    df = pd.read_csv(path)
    if "feature" in df.columns:
        unknown = set(df["feature"]) - set(FEATURES)
        if unknown:
            raise ValueError(f"{fname}: unknown feature labels {unknown}")
    return df


def sym_limit(values, floor=0.05):
    v = np.nanmax(np.abs(values))
    return max(floor, float(np.ceil(v * 20) / 20))


# ------------------------------------------------ Fig. 1 (Methods schematic)
def fig1_schematic(out, dur_a=110, dur_b=110):
    """Three regions on one A->B phone pair.

    Interior anchors are drawn at the phone midpoints, matching the V7
    region definition. Shaded spans show only the support of the 15-ms
    spectral frames; CPPS uses a separate 40-ms local operator.
    """
    fig, ax = plt.subplots(figsize=(W2, 2.5))
    y_phone, y_win = 3.0, 1.9

    ax.add_patch(mpatches.Rectangle((-dur_a, y_phone), dur_a, 0.8,
                                    fc="0.92", ec="0.4", lw=0.6))
    ax.add_patch(mpatches.Rectangle((0, y_phone), dur_b, 0.8,
                                    fc="0.85", ec="0.4", lw=0.6))
    ax.text(-dur_a / 2, y_phone + 0.4, "Phone A", ha="center", va="center")
    ax.text(dur_b / 2, y_phone + 0.4, "Phone B", ha="center", va="center")
    ax.plot([0, 0], [y_win - 0.45, 4.0], color="k", lw=0.9, ls="--")
    ax.text(0, 4.05, "MFA boundary", ha="center", va="bottom", fontsize=7)

    anchors = {"Interior_A": -dur_a / 2, "Boundary": 0.0,
               "Interior_B": dur_b / 2}
    colours = {"Interior_A": C_BF, "Boundary": C_TTS, "Interior_B": C_BF}
    for reg, t0 in anchors.items():
        # Outer support of the 15-ms spectral frames
        # (centre +/- 20 ms +/- 7.5 ms). This shading does NOT represent CPPS.
        ax.add_patch(mpatches.Rectangle((t0 - 27.5, y_win - 0.35), 55, 0.7,
                                        fc=colours[reg], alpha=0.12,
                                        ec=colours[reg], lw=0.6))
        for t in GRID_9:
            ax.plot(t0 + t, y_win, "o", ms=3.2, color=colours[reg],
                    mec="white", mew=0.3)
        ax.text(t0, y_win - 0.7, REGION_LABEL[reg], ha="center",
                va="top", color=colours[reg], fontweight="bold")
        ax.plot([t0, t0], [y_win + 0.35, y_phone], color=colours[reg],
                lw=0.5, ls=":")

    # zoomed time grid for the boundary region
    y_z = 0.2
    for t in GRID_9:
        ax.plot(t * 2.2, y_z, "|", color="k", ms=6)
        ax.text(t * 2.2, y_z - 0.3, f"{t:+d}" if t else "0", ha="center",
                va="top", fontsize=6)
    for t in GRID_FLUX:
        ax.plot(t * 2.2, y_z + 0.35, "v", color=C_GREY, ms=3)
    ax.plot([-20 * 2.2, 20 * 2.2], [y_z, y_z], color="k", lw=0.5)
    ax.text(-20 * 2.2 - 6, y_z, "Time grid per region (ms)", ha="right",
            va="center", fontsize=6.5)
    ax.text(-20 * 2.2 - 6, y_z + 0.35, "Spectral flux (8 values)", ha="right",
            va="center", fontsize=6.5, color=C_GREY)

    ax.set_xlim(-dur_a - 45, dur_b + 45)
    ax.set_ylim(-0.6, 4.4)
    ax.axis("off")
    save(fig, out, "fig1_regions_schematic")


# ------------------------------------------ Fig. 3 example direct-effect trajectories
def fig3_trajectories(df, out, system, feature):
    """Illustrative direct TTS-minus-bona fide trajectory A_r(t)."""
    d = df[(df.system == system) & (df.feature == feature)]
    if d.empty:
        print(f"skip fig3: no rows for {system} / {feature}")
        return
    fig, axes = plt.subplots(1, 3, figsize=(W2, 2.2), sharey=True)
    for ax, reg in zip(axes, REGIONS):
        s = d[d.region == reg].sort_values("time_ms")
        ax.fill_between(s.time_ms, s.ci_lo, s.ci_hi, color=C_TTS,
                        alpha=0.2, lw=0)
        ax.plot(s.time_ms, s.estimate, color=C_TTS, lw=1.2)
        ax.axhline(0, color="0.5", lw=0.6)
        ax.axvline(0, color="0.7", lw=0.5, ls="--")
        ax.set_title(REGION_LABEL[reg])
        ax.set_xticks(grid_for(feature)[::2] if feature != "flux"
                      else [-15, -5, 5, 15])
        ax.set_xlabel("Relative time (ms)")
    axes[0].set_ylabel("TTS − bona fide (SD)")
    save(fig, out, f"fig3_trajectories_{system}_{feature}")


# ------------------------------------------------ Fig. 2 RQ1 heatmap
def fig2_rq1_heatmap(df, out, clip=None):
    feats = list(FEATURES)
    cols = [(f, r) for f in feats for r in REGIONS]
    mat = np.full((len(SYSTEMS), len(cols)), np.nan)
    sig = np.zeros_like(mat, dtype=bool)
    for i, s in enumerate(SYSTEMS):
        for j, (f, r) in enumerate(cols):
            row = df[(df.system == s) & (df.feature == f) & (df.region == r)]
            if len(row) == 1:
                mat[i, j] = row.estimate.iloc[0]
                sig[i, j] = row.p_bh.iloc[0] < ALPHA

    lim = clip if clip else sym_limit(mat)
    fig, ax = plt.subplots(figsize=(W2, 3.4))
    im = ax.imshow(mat, cmap="RdBu_r",
                   norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    yy, xx = np.where(~sig & ~np.isnan(mat))
    dark = np.abs(mat[yy, xx]) > 0.6 * lim
    ax.scatter(xx, yy, marker="x", s=10, lw=0.6,
               c=np.where(dark, "white", "0.2"))

    ax.set_yticks(range(len(SYSTEMS)), SYSTEMS)
    ax.set_xticks(range(len(cols)), ["B", "IA", "IB"] * len(feats))
    ax.tick_params(length=0)
    for k in range(1, len(feats)):
        ax.axvline(k * 3 - 0.5, color="white", lw=2.5)
    ax.axhline(len(TRAIN_SYSTEMS) - 0.5, color="white", lw=2.5)
    for k, f in enumerate(feats):
        ax.text(k * 3 + 1, -0.9, FEATURES[f], ha="center", va="bottom",
                fontsize=7.5)
    for side in ax.spines.values():
        side.set_visible(False)

    extend = "both" if clip and np.nanmax(np.abs(mat)) > clip else "neither"
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015, extend=extend)
    cb.set_label("TTS − bona fide (SD)")
    cb.outline.set_linewidth(0.4)
    save(fig, out, "fig2_rq1_region_average")


# ------------------------------------------ Fig. 4 average specificity
def _class_style(c):
    return {"equivalent": (C_EQ, "o"), "inconclusive": (C_INC, "D"),
            "meaningful": ("k", "s")}.get(c, (C_GREY, "^"))


def fig4_average_specificity(df, out):
    feats = list(FEATURES)
    fig, axes = plt.subplots(2, 3, figsize=(W2, 4.2), sharex=True)
    lim = max(0.15, sym_limit(df[["ci_lo", "ci_hi"]].values))
    y = np.arange(len(SYSTEMS))[::-1]
    for ax, f in zip(axes.flat, feats):
        ax.axvspan(-SESOI, SESOI, color="0.9", lw=0)
        ax.axvline(0, color="0.5", lw=0.5)
        d = df[df.feature == f].set_index("system")
        for yi, s in zip(y, SYSTEMS):
            if s not in d.index:
                continue
            r = d.loc[s]
            col, mk = _class_style(r.classification)
            ax.plot([r.ci_lo, r.ci_hi], [yi, yi], color=col, lw=1)
            ax.plot(r.estimate, yi, mk, color=col, ms=3.5,
                    mfc=col if r.p_bh < ALPHA else "white", mew=0.8)
        ax.set_yticks(y, SYSTEMS)
        ax.set_xlim(-lim, lim)
        ax.set_title(FEATURES[f])
        ax.tick_params(axis="y", length=0)
    for ax in axes[-1]:
        ax.set_xlabel("Average specificity, D̄ (SD)")
    handles = [
        mpatches.Patch(color="0.9", label=f"SESOI ±{SESOI:.2f}"),
        Line2D([], [], color=C_EQ, marker="o", ls="", label="Equivalent"),
        Line2D([], [], color=C_INC, marker="D", ls="", label="Inconclusive"),
        Line2D([], [], color="k", marker="s", ls="", label="Meaningful"),
        Line2D([], [], color=C_GREY, marker="o", mfc="white", ls="",
               label="Not significant (open)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, out, "fig4_rq2_average_specificity")


# ------------------------------------------ Fig. 5 time-resolved D(t)
def fig5_time_resolved(df, shape, out):
    feats = list(FEATURES)
    vals = df.estimate.values
    lim = sym_limit(vals)
    norm = TwoSlopeNorm(0, -lim, lim)
    fig, axes = plt.subplots(2, 3, figsize=(W2, 4.6))
    for ax, f in zip(axes.flat, feats):
        grid = GRID_9  # common x axis; flux starts at -15
        mat = np.full((len(SYSTEMS), len(grid)), np.nan)
        sig = np.zeros_like(mat, bool)
        eq = np.zeros_like(mat, bool)
        d = df[df.feature == f]
        for i, s in enumerate(SYSTEMS):
            for j, t in enumerate(grid):
                r = d[(d.system == s) & (d.time_ms == t)]
                if len(r) == 1:
                    mat[i, j] = r.estimate.iloc[0]
                    sig[i, j] = bool(r.significant.iloc[0])
                    eq[i, j] = bool(r.equivalent.iloc[0])
        cmap = plt.get_cmap("RdBu_r").copy()
        cmap.set_bad("0.97")
        im = ax.imshow(np.ma.masked_invalid(mat), cmap=cmap, norm=norm,
                       aspect="auto")
        yy, xx = np.where(sig)
        dark = np.abs(mat[yy, xx]) > 0.6 * lim
        ax.scatter(xx, yy, marker="o", s=5, lw=0,
                   c=np.where(dark, "white", "k"))
        yy, xx = np.where(~eq & ~np.isnan(mat))
        ax.scatter(xx, yy, marker="s", s=22, facecolors="none",
                   edgecolors="k", lw=0.5)
        ax.set_xticks(range(len(grid)), [str(t) for t in grid],
                      fontsize=6)
        labels = []
        for s in SYSTEMS:
            r = shape[(shape.system == s) & (shape.feature == f)]
            star = "*" if len(r) == 1 and r.p_bh.iloc[0] < ALPHA else ""
            labels.append(f"{s}{star}")
        ax.set_yticks(range(len(SYSTEMS)), labels)
        ax.tick_params(length=0)
        ax.set_title(FEATURES[f])
        for side in ax.spines.values():
            side.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("Relative time (ms)")
    fig.subplots_adjust(hspace=0.35, wspace=0.3)
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    cb.set_label("D(t) (SD)")
    cb.outline.set_linewidth(0.4)
    handles = [
        Line2D([], [], marker="o", color="k", ls="", ms=3,
               label="Significant (Bonferroni + BH-FDR)"),
        Line2D([], [], marker="s", mfc="none", mec="k", ls="", ms=5,
               label="Not equivalent (TOST, ±0.10 SD)"),
        Line2D([], [], ls="", label="* shape specificity significant"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.45, -0.05))
    save(fig, out, "fig5_rq2_time_resolved")


# ------------------------------------------ Fig. 6 forest plots
def fig6_forest(avg, meta, out):
    feats = list(FEATURES)
    fig, axes = plt.subplots(2, 3, figsize=(W2, 4.6), sharex=True)
    lim = max(0.15, sym_limit(
        np.r_[avg[["ci_lo", "ci_hi"]].values.ravel(),
              meta[["pi_lo", "pi_hi"]].values.ravel()]))
    for ax, f in zip(axes.flat, feats):
        d = avg[avg.feature == f].set_index("system")
        m = meta[meta.feature == f].iloc[0]
        n = len(SYSTEMS)
        y = np.arange(n + 2)[::-1]  # systems, gap, pooled
        ax.axvspan(-SESOI, SESOI, color="0.9", lw=0)
        ax.axvline(0, color="0.5", lw=0.5)
        for yi, s in zip(y[:n], SYSTEMS):
            if s in d.index:
                r = d.loc[s]
                ax.plot([r.ci_lo, r.ci_hi], [yi, yi], color=C_GREY, lw=0.8)
                ax.plot(r.estimate, yi, "s", color=C_GREY, ms=3)
        yp = y[-1]
        ax.plot([m.pi_lo, m.pi_hi], [yp, yp], color=C_MEAN, lw=1.2)
        if pd.notna(m.ci_lo):
            ax.add_patch(mpatches.Polygon(
                [[m.ci_lo, yp], [m.pooled, yp + 0.4], [m.ci_hi, yp],
                 [m.pooled, yp - 0.4]], color="k"))
        else:
            ax.plot(m.pooled, yp, "D", color="k", ms=4)
        ax.set_yticks(list(y[:n]) + [yp], SYSTEMS + ["Pooled"])
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(-lim, lim)
        tau = f", τ = {m.tau:.3f}" if pd.notna(m.get("tau", np.nan)) else ""
        ax.set_title(f"{FEATURES[f]}\nI² = {m.I2:.1f}%{tau}")
    for ax in axes[-1]:
        ax.set_xlabel("Average specificity (SD)")
    handles = [
        mpatches.Patch(color="0.9", label=f"SESOI ±{SESOI:.2f}"),
        Line2D([], [], color=C_MEAN, lw=1.2, label="Prediction interval"),
        Line2D([], [], color="k", marker="D", ls="",
               label="Pooled (REML) with CI"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, out, "fig6_meta_forest")


# ------------------------------------------ Fig. 7 partition-specific projected trajectories
def fig7_projected_trajectory(meta, out):
    required = {"feature", "partition", "time_ms", "estimate",
                "ci_lo", "ci_hi", "pi_lo", "pi_hi"}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(f"meta_multivariate.csv: missing columns {sorted(missing)}")

    partitions = ["TRAIN", "EVAL"]
    unknown = set(meta["partition"].astype(str).str.upper()) - set(partitions)
    if unknown:
        raise ValueError(f"meta_multivariate.csv: unknown partition labels {unknown}")
    meta = meta.copy()
    meta["partition"] = meta["partition"].astype(str).str.upper()

    feats = list(FEATURES)
    fig, axes = plt.subplots(2, 3, figsize=(W2, 3.8), sharex=True,
                             sharey=True)
    part_col = {"TRAIN": "#000000", "EVAL": "#7F7F7F"}
    part_ls = {"TRAIN": "-", "EVAL": "--"}
    for ax, f in zip(axes.flat, feats):
        ax.axhspan(-SESOI, SESOI, color="0.92", lw=0)
        ax.axhline(0, color="0.5", lw=0.5)
        for part in partitions:
            d = meta[(meta.feature == f) & (meta.partition == part)] \
                .sort_values("time_ms")
            if d.empty:
                continue
            col = part_col[part]
            ax.fill_between(d.time_ms, d.pi_lo, d.pi_hi, facecolor="none",
                            edgecolor=col, lw=0.5, ls=":", alpha=0.7)
            ax.fill_between(d.time_ms, d.ci_lo, d.ci_hi, color=col,
                            alpha=0.18, lw=0)
            ax.plot(d.time_ms, d.estimate, color=col, lw=1.1,
                    ls=part_ls[part], marker="o", ms=2.3, label=part)
        ax.set_title(FEATURES[f])
        ax.set_xticks([-20, -10, 0, 10, 20])
    for ax in axes[:, 0]:
        ax.set_ylabel("D(t) (SD)")
    for ax in axes[-1]:
        ax.set_xlabel("Relative time (ms)")
    handles = [
        Line2D([], [], color="#000000", lw=1.2, label="TRAIN projected mean"),
        Line2D([], [], color="#7F7F7F", lw=1.2, ls="--",
               label="EVAL projected mean"),
        mpatches.Patch(color="#7F7F7F", alpha=0.18, label="95% CI (filled band)"),
        mpatches.Patch(facecolor="none", edgecolor="#7F7F7F", ls=":",
                       lw=0.5, label="95% prediction interval (dotted outline)"),
        mpatches.Patch(color="0.92", label="SESOI ±0.10 SD"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.04), fontsize=6)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, out, "fig7_meta_projected_trajectory")


# ----------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--out", type=Path, default=Path("figures"))
    p.add_argument("--example-system", default="A01",
                   help="system shown in Fig. 3")
    p.add_argument("--example-feature", default="energy")
    p.add_argument("--rq1-clip", type=float, default=0.3,
                   help="cap the colour scale of Fig. 2 (default: 0.3 SD)")
    a = p.parse_args()

    fig1_schematic(a.out)

    rq1_time = load(a.data, "rq1_time_resolved.csv")
    if rq1_time is not None:
        fig3_trajectories(rq1_time, a.out, a.example_system, a.example_feature)

    rq1 = load(a.data, "rq1_region_average.csv")
    if rq1 is not None:
        fig2_rq1_heatmap(rq1, a.out, a.rq1_clip)

    avg = load(a.data, "rq2_average.csv")
    if avg is not None:
        fig4_average_specificity(avg, a.out)

    tr = load(a.data, "rq2_time.csv")
    sh = load(a.data, "rq2_shape.csv")
    if tr is not None and sh is not None:
        fig5_time_resolved(tr, sh, a.out)

    mu = load(a.data, "meta_univariate.csv")
    if avg is not None and mu is not None:
        fig6_forest(avg, mu, a.out)

    mm = load(a.data, "meta_multivariate.csv")
    if mm is not None:
        fig7_projected_trajectory(mm, a.out)


if __name__ == "__main__":
    main()
