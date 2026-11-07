# Manuscript map: camera-ready-v2 to camera-ready-v3/v4/v5/v6/v7/v8/v9/v10/v11/v12

The current PDF has 45 pages: main text 1–10, references 11–12, scientific appendices 13–37, and declarations/checklist 38–45. The section structure and ten-page main-text budget are retained. The v3 compression reduced the scientific appendices from 41 to 23 pages; v8 adds two pages for verified post hoc comparisons and operational guidance, preserving proofs, controls and uncertainty.

Camera-ready-v4 applies seven wording clarifications to v3. Table and equation numbering, numerical values, the page budget, and the crosswalk below are unchanged. In v4, the revised caption of Table E.2 appears on page 31.

Camera-ready-v5 refines citation placement and attribution, with all 29 printed references retained. The two bibliography pages are unchanged. Table/equation numbering and the 43-page budget remain unchanged.

Camera-ready-v6 consolidates Mansouri–Arai attribution to six manuscript locations (five main-text citations and one in Appendix A.2), replacing seven repeated citations with internal references. The bibliography, table/equation numbering, results and 43-page budget are unchanged.

Camera-ready-v7 clarifies definitions, loss aggregation, matrix and calibration conventions, gate weighting, uncertainty statistics, and training/acquisition procedures. Numerical results, bibliography entries, table/equation numbering and figure artwork remain unchanged. The overlap and fold-stability definition is in Appendix C.3; Table C.2 now keeps its support-correction note together.

Camera-ready-v8 adds Table D.6 for the retained-record four-summary comparison and two numerical sentence comparisons, with reproducible code and input hashes. Later D tables increase by one; existing numbers, figures and bibliography entries are unchanged. Proposed policies and untested extensions remain explicitly distinguished from completed checks.

Camera-ready-v9 rebalances Sections 8 and 9 and clarifies three statements about information summaries. The PDF remains 45 pages with ten main-text pages; table/equation numbering, numerical results, figures, references, and reproducibility artifacts are unchanged from v8.

Camera-ready-v10 clarifies operational matrices, experimental predictors, slice membership and rediscovery, flip-prediction observations, and acquisition-surrogate scope. It fixes algorithm/table reading order while retaining 45 pages and ten main-text pages. Table/equation numbers, numerical results, figures, references, and reproducibility artifacts are unchanged; Table D.9 control labels now name feature moments and centered covariance.

Camera-ready-v11 replaces the abstract and keeps the introduction framework citation unbroken. Pagination remains 45 pages with ten main-text pages; table/equation numbers, results, figures, references, and reproducibility artifacts are unchanged.

Camera-ready-v12 replaces the introduction and contribution bullets, preserves Figure 1, keeps the framework paragraph together, and places the intervention figure before the synthetic subsection. Pagination remains 45 pages with ten main-text pages; the abstract, table/equation numbers, results, artwork, references, and reproducibility artifacts are unchanged.

## Table crosswalk

Multiple semantic labels now identify a single table with complementary panels. Historical reports and preserved source snapshots keep their original numbering; the current guides use v8–v12 numbers.

| Stable LaTeX label | v2 table | v3–v7 table | v8–v12 table |
|---|---|---|---|
| `tab:appB-conditioning` | B.1 | B.1 | B.1 |
| `tab:appB-certificates` | B.2 | B.1 | B.1 |
| `tab:appB-training` | B.3 | B.2 | B.2 |
| `tab:d-experiment-overview` | D.1 | D.1 | D.1 |
| `tab:d-validation-performance` | D.2 | D.2 | D.2 |
| `tab:d-validation-information` | D.3 | D.2 | D.2 |
| `tab:d-intervention-outcomes` | D.4 | D.3 | D.3 |
| `tab:d-confident-flips` | D.5 | D.3 | D.3 |
| `tab:d-selection-comparison` | D.6 | D.4 | D.4 |
| `tab:d-matrix-sensitivity` | D.7 | D.4 | D.4 |
| `tab:d-matrix-per-seed` | D.8 | D.5 | D.5 |
| `tab:four-way-swap` | — | — | D.6 |
| `tab:d-flip-prediction` | D.9 | D.6 | D.7 |
| `tab:d-synthetic-summary` | D.10 | D.7 | D.8 |
| `tab:d-structural-comparisons` | D.11 | D.7 | D.8 |
| `tab:d-hidden-selectors` | D.12 | D.8 | D.9 |
| `tab:d-hidden-contrasts` | D.13 | D.8 | D.9 |
| `tab:d-predefined-diagnostics` | D.14 | D.9 | D.10 |
| `tab:d-nuisance-evidence` | D.15 | Prose in D.5; expanded source retained | Prose in D.5; expanded source retained |
| `tab:d-nuisance-sensitivity` | D.16 | D.10 | D.11 |
| `tab:e-frozen-performance` | E.1 | E.1 | E.1 |
| `tab:e-frozen-information` | E.2 | E.1 | E.1 |
| `tab:e-frontier-performance` | E.3 | E.2 | E.2 |
| `tab:e-frontier-information` | E.4 | E.2 | E.2 |
| `tab:e-repair-performance` | E.5 | E.3 | E.3 |
| `tab:e-repair-slices` | E.6 | E.3 | E.3 |
| `tab:e-groupfree-performance` | E.7 | E.4 | E.4 |
| `tab:e-groupfree-information` | E.8 | E.4 | E.4 |
| `tab:acquisition-budget` | E.9 | E.5 | E.5 |
| `tab:acquisition-products` | E.10 | E.5 | E.5 |
| `tab:acquisition-strength` | E.11 | E.5 | E.5 |

## Removed duplicate figure inclusions

All five figure source/PDF assets remain in `tex/camera_ready_v2/figures/`; only their inclusion in the appendix was removed. Main figures, including the conceptual bird artwork, are unchanged.

| Removed appendix figure | Evidence retained in the current PDF | Editable asset |
|---|---|---|
| Figure D.1 (`fig:prospective-controls`) | Main Figure 3; Tables D.3–D.4 and the E.3 prospective training discussion | `figures/figA01_prospective.pdf` (PDF and same-stem TeX) |
| Figure D.2 (`fig:structural-checks`) | Table D.8; Appendix E.1 reserve proof | `figures/figA02_structural.pdf` (PDF and same-stem TeX) |
| Figure D.3 (`fig:discovery-controls`) | Table D.9, including random-direction and unweighted paired contrasts | `figures/figA03_discovery_controls.pdf` (PDF and same-stem TeX) |
| Figure E.1 (`fig:training-frontier`) | Tables E.1–E.2 | `figures/figA04_training.pdf` (PDF and same-stem TeX) |
| Figure E.2 (`fig:acquisition`) | Table E.5 | `figures/figA05_acquisition.pdf` (PDF and same-stem TeX) |

## Expanded records and execution details

The files in [extended_appendix_details](extended_appendix_details/README.md) are byte-identical copies of the expanded appendix sources from v2. They retain the exact prompt strings and declared CLIP revision, full prospective per-seed means/covariances, intermediate acquisition score/loss quantities, and expanded derivations. These source records use historical numbering. The [v2 compiled PDF](https://github.com/E-zClap/fisher-glass-learning/releases/download/camera-ready-v2/main.pdf) remains accessible as the corresponding expanded rendering. Interpretations and claims in the current paper take precedence.

| Detail | Preserved source | Executable/numeric counterpart |
|---|---|---|
| Estimation and certification steps | `extended_appendix_details/transferred_estimation.tex` | `../scripts/reproduce_w2_audit.py`, `../artifacts/w2_audit/` |
| Discovery gate, prompts, declared model revision and support correction | `extended_appendix_details/transferred_discovery.tex` | `../experiments/waterbirds_hidden_slice/`, `../artifacts/hidden_slice_corrected_v2/` |
| Full per-seed prospective means/covariances and matrix construction | `extended_appendix_details/prospective_reanalysis.tex` | `../reanalysis/matrix_constructions/` |
| Frozen/end-to-end/repair training and detailed objective exposition | `extended_appendix_details/e02_training.tex`, `e03_extended_training.tex` | [README cohort map](../README.md#manuscript-to-artifact-map); historical and fresh cohorts remain distinct |
| Influence derivatives and acquisition score/loss breakdown | `extended_appendix_details/e04_influence.tex`, `e05_acquisition.tex` | `../reports/waterbirds_tail_transverse_influence_report.md`, `../reports/waterbirds_counterfactual_antidote_report.md` |

The original numerical records and their historical limitations are unchanged: no missing checkpoints, seed outputs or acquisition intervals have been reconstructed from aggregates. See the [reproduction guide](REPRODUCING_EXPERIMENTS.md) and [manifest](../ARTIFACT_MANIFEST.json).
