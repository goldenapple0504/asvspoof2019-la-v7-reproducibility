# ASVspoof 2019 LA V7 reproducibility repository

This repository contains the **final V7** boundary/interior analysis inputs,
code, and reported outputs. Canonical finalization completed on 2026-09-10:
146 extraction checkpoints, 482,509 unique eligible phone-pair tokens,
36,348 utterances, 60 accepted mixed models, 530 prespecified time-point
tests, six multivariate meta-analyses, and 21/21 final quality gates. The
seven main figures were exported from finalized outputs on 2026-09-18.
ASVspoof audio is not distributed here.

The primary specificity estimand is
`(TTS−BF)_Boundary − 0.5×[(TTS−BF)_Interior_A + (TTS−BF)_Interior_B]`.
There are ten TTS systems and six trajectories: energy, centroid, tilt,
flux, flatness, and CPPS. CPPS uses the frozen Praat 6.1.38 operator through
the inherited `extract_flatness_cpp_trajectories.py` helper.

## Contents

| Path | Purpose |
| --- | --- |
| `code/` | Final V7 extraction, matching, model, inference, meta-analysis, and figure/table scripts, plus small replay adapters. Original production filenames are preserved so imports resolve. |
| `data/utterances/` | Frozen V7 utterance IDs by system and partition, joined transcripts, and known source-stage exclusion reasons. |
| `data/textgrids/` | Per-file SHA-256 manifest for the MFA TextGrids of all 36,348 used utterances; the TextGrid archive itself is a release asset. |
| `data/inputs/` | Frozen ten-system Boundary trajectories, eligible tokens, scaling constants, and inherited matching lineages needed by V7. These are selected inputs, not an archive of earlier projects. |
| `data/trajectories/` (release asset) | 146 Parquet chunks of Interior_A and Interior_B trajectories for all six features. Combine these with Boundary tables in `data/inputs/` for all three regions. |
| `data/validity/` | Per-token, per-region validity and CPPS exclusion reasons. |
| `data/matching/` | Final exact ordered-phone-pair matched rows for short-frame, flatness, and complete-three-region CPPS sets. |
| `data/results/` | Official final RQ1, RQ2, meta-analysis, quality-gate, and table TSVs, plus 60 final model random-structure diagnostics used by Table S2. |
| `data/figures_inputs/`, `data/figures/` | Seven exact plotting CSVs, exported Fig. 1–7, and exported Fig. S1. |
| `FILE_MANIFEST.tsv` | Relative path, byte count, and SHA-256 for every repository file and every file restored from release assets. |

Large data files are not part of the repository tree. They are attached as
assets to the [v1.0 release](https://github.com/goldenapple0504/asvspoof2019-la-v7-reproducibility/releases/tag/v1.0):

| Asset | Restores |
| --- | --- |
| `v7_inputs.zip` | `data/inputs/*.parquet`, `data/inputs/trajectory_eligible_tokens.tsv` |
| `v7_trajectories.zip` | `data/trajectories/` (146 Parquet chunks) |
| `v7_textgrids.zip` | `data/textgrids/v7_textgrids.zip` |
| `RELEASE_ASSETS_SHA256.tsv` | SHA-256 of the three assets |

Download the three archives and unzip them in the repository root; each
archive stores repository-relative paths. Every restored file can then be
checked against `FILE_MANIFEST.tsv`. `make_release_assets.py` rebuilds these
archives from a fully hydrated V7 checkout.

## Software and upstream data

- Python 3.13.14 for the frozen analysis; audited package versions are in
  `requirements.txt` and `PYTHON_PRAAT_VERSIONS.tsv`. Matplotlib 3.11.1 was
  used to verify figure regeneration in this release.
- R 4.6.1; `R_sessionInfo.txt` records lme4, Matrix, mixmeta, and
  clubSandwich loaded. Additional versions are in `R_PACKAGE_VERSIONS.tsv`.
  Set `V7_R_LIB` to a library with these versions or install them normally.
- MFA 3.4.1. The successful command form, pronunciation dictionary
  `english_us_arpa.dict`, and acoustic model `english_us_arpa.zip` are in
  `code/02_run_mfa.txt`. The selected TextGrids are supplied in the release, so rerunning
  MFA is optional for a derived-data replay.
- Download ASVspoof 2019 LA audio and the ASVspoof/VCTK metadata from the
  [official ASVspoof database page](https://www.asvspoof.org/database) and
  [University of Edinburgh dataset record](https://doi.org/10.7488/ds/2555).
  The transcript join uses `ASVspoof2019_LA_VCTK_MetaInfo.tsv`.
  `code/01_join_transcripts.py` recreates
  `data/utterances/v7_transcripts.tsv` byte-for-byte from that file and the
  frozen ID list. Upstream ASVspoof and VCTK terms still apply to source
  material; `LICENSE` does not relicense them.

## Run order

For an acoustic rerun, download the audio and metadata, join transcripts,
run MFA using `code/02_run_mfa.txt`, then execute the V7 eligibility,
extraction, and matching scripts. Some historical acoustic scripts retain
absolute paths from the source machine. The frozen TextGrids and feature
tables are supplied so the **reported statistics** can be replayed without
audio.

From the repository root, after installing `requirements.txt` and the R
versions above:

```text
python code/prepare_replay_workspace.py --work ../v7_replay
python code/assemble_v7_production_model_data.py --project-root ../v7_replay --output-root ../v7_replay/rebuild_v7 --execute-full
Rscript ../v7_replay/rebuild_v7/scripts/fit_v7_production_models.R ../v7_replay/rebuild_v7 --execute-full
Rscript ../v7_replay/rebuild_v7/scripts/finalize_v7_production_inference.R ../v7_replay/rebuild_v7 --execute-full
Rscript ../v7_replay/rebuild_v7/scripts/run_v7_specificity_meta_analysis.R ../v7_replay/rebuild_v7 --execute-full
python code/prepare_rq1_replay.py --work ../v7_replay
Rscript code/run_rq1_direct_authenticity.R ../v7_replay/rq1_postestimation
python code/export_tables.py --data data --out ../v7_replay/tables
python code/export_figure_inputs.py --data data --out ../v7_replay/figure_inputs
python code/make_figures.py --data ../v7_replay/figure_inputs --out ../v7_replay/figures --rq1-clip 0.3
python code/make_fig_s1.py --input ../v7_replay/figure_inputs/rq1_time_resolved.csv --out ../v7_replay/figures
```

The replay adapter builds a working directory outside this repository and
links or copies the derived inputs into the original V7 layout. Its local
authorization marker only satisfies a guard in the historical production
scripts; the marker is excluded from this repository. For a quick check
without refitting, run the final four Python commands after unpacking the
release assets and compare their outputs with `data/results/tables/` and
`data/figures/`.

The final matching is one-to-one within each **ordered** phone pair. CPPS
matching uses Euclidean distance between the pooled-standardized two phone
durations and `scipy.optimize.linear_sum_assignment`; token IDs are sorted
before assignment. The V7 seed is `20260819` (`v7_common.py`).

## Manuscript traceability

This mapping was checked against the current Results manuscript and
`Supplementary_material.docx` supplied with it. The table exporter
reproduces their reported counts and rounded estimates. The supplied figure
images came from the completed figure pipeline; pixels can differ when
rerendered with different fonts.

| Item | Script | Direct output and input |
| --- | --- | --- |
| Table 1 | `code/export_tables.py` | `data/results/tables/table1.tsv` ← `data/utterances/v7_included_utterances.tsv` |
| Table 2 | `code/export_tables.py` | `data/results/tables/table2.tsv` ← `data/results/rq1/average_authenticity.tsv`, `trajectory_tests.tsv` |
| Table 3 | `code/export_tables.py` | `data/results/tables/table3.tsv` ← `data/results/inference/average_specificity.tsv`, `time_resolved_specificity.tsv` |
| Table 4 | `code/export_tables.py` | `data/results/tables/table4.tsv` ← `data/results/meta_analysis/average_specificity_REML.tsv` |
| Table S1 | `code/export_tables.py` | `data/results/tables/table_s1_matching_balance.tsv` ← three final `data/matching/production_*matches.tsv` files |
| Table S2 | `code/export_tables.py` | `data/results/tables/table_s2_random_structure.tsv` ← `data/results/model_diagnostics/*_random_hierarchy.tsv`, `data/results/inference/model_acceptance_gates.tsv` |
| Fig. 1 | `code/make_figures.py` | `data/figures/fig1_regions_schematic.pdf` (schematic) |
| Fig. 2 | `code/export_figure_inputs.py`, `code/make_figures.py` | `data/figures_inputs/rq1_region_average.csv` → `data/figures/fig2_rq1_region_average.pdf` |
| Fig. 3 | same | `data/figures_inputs/rq1_time_resolved.csv` → `data/figures/fig3_trajectories_A01_energy.pdf` |
| Fig. 4 | same | `data/figures_inputs/rq2_average.csv` → `data/figures/fig4_rq2_average_specificity.pdf` |
| Fig. 5 | same | `data/figures_inputs/rq2_time.csv`, `rq2_shape.csv` → `data/figures/fig5_rq2_time_resolved.pdf` |
| Fig. 6 | same | `data/figures_inputs/rq2_average.csv`, `meta_univariate.csv` → `data/figures/fig6_meta_forest.pdf` |
| Fig. 7 | same | `data/figures_inputs/meta_multivariate.csv` → `data/figures/fig7_meta_projected_trajectory.pdf` |
| Fig. S1 | `code/make_fig_s1.py` | `data/figures_inputs/rq1_time_resolved.csv` → `data/figures/figS1_all_direct_effect_trajectories.pdf` |

Table S2 is in `Supplementary_material.docx` and lists the 19 models whose
initial fit had zero match-intercept variance. The 60 included random-structure
diagnostics document both fit attempts and the 41 models that retained the
full structure. The exported Table S2 matches all 19 supplementary rows
exactly. The original Fig. S1 generator was not found; `make_fig_s1.py` was reconstructed from the
finalized RQ1 pointwise table, while its original exported PDF/PNG are
included.

## Validation and limits

- `data/results/quality_gates.tsv` records all 21 final gates as passed.
- `data/results/inference/model_acceptance_gates.tsv` records
  `NOT_APPLICABLE_CROSSED_CLUSTERING` for all 60 models. CR2 produced no
  sensitivity estimates because match and utterance clusters were crossed;
  inference uses the model-based covariance.
- All 36,348 eligible utterances have TextGrids. The 7,919 duplicate source
  copies were byte-identical; the archive contains one copy per ID.
- `code/export_figure_inputs.py` reproduced all seven plotting CSVs exactly
  from official TSVs: 180 RQ1 averages, 1,590 RQ1 time points, 60 RQ2
  averages, 530 RQ2 time points, 60 shape tests, six univariate meta rows,
  and 106 multivariate projections.
- `code/export_tables.py` reproduced Table S1 pair counts exactly; SMDs
  agreed with the contemporaneous review table to `3.9e-16` or less.
- `code/export_tables.py` reproduced all 19 Table S2 rows exactly from the
  included final model diagnostics and acceptance gates.
- Reassembling A01/energy from packaged trajectories yielded 497,070 model
  rows and matched official model-data values to absolute tolerance `1e-12`.
- `data/utterances/v7_known_excluded_utterances.tsv` records source-stage
  missing/corrupt audio and alignment failures. It is a known-exclusion
  list, not a claim that every utterance absent from the frozen subset has
  a diagnosed failure. Token-level region and CPPS exclusions are in
  `data/validity/`.

The `.rds` model objects and 600 raw contrast TSVs are omitted because they
can be regenerated from the included matched trajectories and code; the
final inference and meta-analysis TSVs used by the manuscript are included.
