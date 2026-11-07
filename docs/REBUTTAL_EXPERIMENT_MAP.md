# Rebuttal experiments and retained follow-up analyses

This map documents the evidence included in the final camera-ready-v22 release and standalone supplement. It is an experiment inventory, not a transcript of the submitted response. Paths below are relative to the package root. Section/table labels follow the current manuscript; the earlier expanded appendix sources retain historical numbering.

## Completed rebuttal experiments

| Experiment | Retained files and implementation | Protocol and interpretation | Replay |
|---|---|---|---|
| Held-out matched background intervention, primary prospective v2 | `artifacts/prospective_v2/`: frozen selection, per-run selection/training/shift results, shift summary and paired-generation metadata; `experiments/prospective_nuisance_shift/`; `artifacts/checkpoint_identifiers.csv` | Primary seeds 22–26; validation selection is frozen before held-out evaluation. Same/opposite arms retain bird foreground/matte/label with matched construction. All 20 candidate ERM/FG run records are retained. Main Figure 2 and Appendix D.2 report this rerun, not the historical July values. Full image set and checkpoints are excluded. | `make_report.py` with the v2 command below. |
| Development intervention cohort, prospective v1 | `artifacts/prospective_v1/` and the same implementation/checkpoint index | Seeds 17–21; absolute projected-information selection, distinct from v2's projected/raw criterion. Both cohorts use the same benchmark image cohort. Keep these records separate; v1 is not an independent additional primary confirmation. | `make_report.py` with the v1 command below. |
| Four validation summaries and intervention associations | `reanalysis/four_way_swap/`: script, README, per-model/group records, panels and results; both prospective artifact cohorts | Raw Fisher, projected information, mean inverse information and full inverse-CVaR are compared on retained candidates. Checks cover 40 candidate models, 20 selected models and 80 intervention groups. Table D.6 is a post hoc diagnostic comparison; it is not four separately trained objective arms. | `python reanalysis/four_way_swap/reanalyse.py` |
| Ridge-to-exact audit and later training replications | `artifacts/w2_audit/`: diagnostic JSONs, frozen-training result, ridge-sensitivity records, end-to-end records and original report; `experiments/ridge_exact_audit.py`, `make_ridge_exact_audit_report.py`, `run_w2_ridge_exact_audit.sh` | Nuisance-ridge bias, conditioning and nominal lower certificates; 75 held-out audits and associated replications are described in Appendix B.4. The replay reconstructs 260 rows and 37 aggregate rows from 19 retained input files. These fresh models are a different cohort from the earlier training summaries. | `python scripts/reproduce_w2_audit.py` |
| Metadata-blind hidden-slice discovery and controls | `artifacts/hidden_slice_corrected_v2/`: blind protocol, aggregate/main table, paired comparisons, sensitivity and five per-seed discovery/blind-test/posthoc records; `experiments/waterbirds_hidden_slice/` | Corrected v2, seeds 17–21: cross-fitting and gate selection precede blind certification and metadata reveal. Conditional on the declared representation/direction; CLIP-versus-random evidence is inconclusive and does not establish nuisance semantics. Never substitute the older uncorrected archive. | `python scripts/reproduce_retained_tables.py` |
| Constrained hidden-slice repair | Same corrected cohort, including `runs/seed_17` through `seed_21/repair.json`, selected arms and seed-vector aggregates; hidden-slice repair implementation | Selection uses the declared accuracy budgets. The Fisher-based repair selects the unchanged model in all five seeds; this does not establish Fisher-based repair efficacy. Numeric selected-arm records are included, but fitted coefficient arrays are omitted. | Same retained-table command. |
| Group-free coefficient audit | `artifacts/group_free_v2/`: per-seed selection, blind and posthoc records, aggregate/main/frontier tables; `experiments/waterbirds_group_free_lambda/` | Schema-v2 post hoc audit, five seeds. Its reported worst-group loss of 12.6 percentage points must remain visible. It is not independent held-out confirmation or a validated general threshold policy. | Same retained-table command. |

## Additional retained analyses in the current paper

| Analysis | Included evidence and replay | Scope |
|---|---|---|
| Matched eligible selectors, Table D.4(a) | `reanalysis/matched_selectors/`; per-seed prospective v2 selections/outcomes; `scripts/reproduce_retained_tables.py` verifies 60 decisions. | All selectors use the same seed-specific eligible pool. Accuracy and loss select waterbird/water in all five seeds, like the frozen rule. The 52.6-point intervention drop alone does not establish incremental selector value. |
| Matrix construction sensitivity, Tables D.4(b)–D.5 | `reanalysis/matrix_constructions/validation_inputs/*.npz`, full-precision outputs, source hashes; the retained-table script replays 40 matrix/seed/group records. | Uncentered, centered, unnormalized predictive-Fisher and normalized-weight calculations use the same retained validation coordinates/logits. No classifier refit or new intervention. |
| Within-group failure prediction, Table D.7 and supplementary Table F.2 | `reanalysis/flip_prediction/`: original donor-row NPZs, archived results, original source snapshots, paired changes, protocol, hashes and refit dependencies; `aggregate.py` verifies all 15 seed/variant records. Table F.2 directly renders all nine retained paired metric intervals from `paired_summary.json` via `supplement_flip_intervals.tex`. | Five training seeds; initially correct source images, two opposite-type donors, shared five-fold species-grouped auxiliary logistic validation. Baseline uses NLL, true-class confidence, absolute raw logit and task coordinate; augmentation adds projected background and residual-magnitude coordinates. Selected waterbird/water gains are about 0.0257 AUPRC and 0.0150 AUROC; this is a joint-coordinate comparison, with no background-score-alone comparator. The supplemental interval table introduces no new analysis or refit. |
| Paired supporting contrasts | `reanalysis/supporting_contrasts/recompute.py`, result JSON/CSV and 20 source hashes. | Replays hidden-minus-predefined-minimum Student-t contrasts and CivilComments ridge/unregularized means from retained records. |
| Selected classes and interval conventions | `reanalysis/statistical_reporting/`; retained-table script verifies 25 selected class choices. | Distinguishes class-selection from within-class overlap. Prospective/matrix summaries use the declared nominal normal intervals; corrected hidden-slice/group-free summaries use Student-t intervals. |
| Classification mechanism, Section 3.2 / Appendix A.4 | Active proof in `tex/camera_ready_v2/appendix/transferred_theory.tex`; `reanalysis/classification_mechanism/verify.py` and `verification.json`. | Deterministic quadrature and finite-sum checks cover 28 parameter pairs and 525 exact-risk cases. The uniform minimax result is analytic; this is not a new empirical experiment. |
| Figure 1 real matched illustration | Two JPEGs in `tex/camera_ready_v2/figures/assets/fig01_w28/`, editable TikZ/PDF and `fig01_image_provenance.json`. | Byte-preserved draw-0 same/opposite composites for source row 4888. Accuracy annotations are group means across five training seeds; no individual prediction or flip is claimed. Panel B is schematic. |

## Historical results with incomplete raw-output retention

The previous `fisherglass_repro_repo` contained 14 experiment/launcher files and four setup/documentation files, with no manuscript, figures, result records or reports. All 14 implementations are represented here. The synthetic and frontier-plot modules include the later public-package corrections; the other 12 match the earlier bundle byte for byte.

| Historical result | What is included | What cannot be recovered from these files |
|---|---|---|
| Original frozen-feature training and end-to-end coefficient sweep | `reports/fisher_glass_results_report.md`, manuscript tables, `waterbirds_fisher_glass_training.py`, `end_to_end_deep_suite.py`, launchers and plotting code. | Original per-seed JSON/CSV, model checkpoints and the historical environment lockfile. Later W2 runs cannot replace the original cohort. |
| Acquisition and counterfactual augmentation | `reports/waterbirds_tail_transverse_influence_report.md`, `waterbirds_counterfactual_antidote_report.md`, other retained descriptive reports, `waterbirds_antidote_acquisition.py` and `waterbirds_counterfactual_antidote.py`. | Original per-seed outputs, selected indices, fitted coefficients and seed-level historical uncertainty. Retained aggregate means do not supply these missing records. |
| Original synthetic and predefined-group diagnostics | Historical results report and `synthetic_fisher_glass.py`, `multiseed_fisher_glass_diagnostics.py`, `waterbirds_fisher_glass.py`, `civilcomments_fisher_glass.py`. | Original multiseed raw outputs. Fresh W2 diagnostics and figure regeneration are separate evidence. |

Fresh independent reruns are supported by [REPRODUCING_EXPERIMENTS.md](REPRODUCING_EXPERIMENTS.md) and [REPRODUCING_PRIMARY_EXPERIMENTS.md](REPRODUCING_PRIMARY_EXPERIMENTS.md). They require separately obtained datasets/model assets and create new run records. The retained-table commands do not perform those runs.

## Replay commands

From the package root, with `requirements-replay.txt` installed:

```shell
python scripts/reproduce_retained_tables.py --verify-only
python scripts/reproduce_retained_tables.py
python scripts/reproduce_w2_audit.py
python reanalysis/flip_prediction/aggregate.py
python reanalysis/four_way_swap/reanalyse.py
python reanalysis/supporting_contrasts/recompute.py
python reanalysis/classification_mechanism/verify.py
python experiments/prospective_nuisance_shift/make_report.py --out-dir artifacts/prospective_v2 --report build/reproduced_tables/prospective_v2_report.md
python experiments/prospective_nuisance_shift/make_report.py --out-dir artifacts/prospective_v1 --report build/reproduced_tables/prospective_v1_report.md
python scripts/reproduce_retained_tables.py --verify-only
```

The original numeric donor-row auxiliary refit is separately documented in `reanalysis/flip_prediction/README.md`; it requires the pinned `requirements-refit.txt`. The archived metrics were previously reproduced to maximum absolute difference `2.73e-12`. It does not retrain the image classifier or score new images.

## Provenance and scope

`ARTIFACT_MANIFEST.json` records the original five source archive hashes, original-member hashes, published hashes and any transformations. The original tarballs are excluded because they mix numerical outputs with images, fitted coefficients, caches or logs. Most curated members preserve the original bytes; six W2 files omit private dataset-root paths, and hidden-slice/group-free records omit fitted weight/bias arrays. Use published hashes for distributed-file integrity. Historical canonical artifact hashes remain provenance identifiers.

The historical W2 report preserves its original build-directory links. Use the included audit exports in [Markdown](../artifacts/w2_audit/report/ridge_exact_audit.md), [JSON](../artifacts/w2_audit/report/ridge_exact_audit.json), and [CSV](../artifacts/w2_audit/report/ridge_exact_audit.csv); the original `run_config.txt` and `system_info.txt` build logs are excluded.

The full benchmark dataset, full intervention-image set, checkpoints, feature caches, and historical preparation manifest/resolved CLIP revision are not included. The two Figure 1 examples are the image exception. Curated records are sufficient for the listed numerical replays, but not for resuming the original pipelines. No missing historical artifact has been replaced with a newly run experiment.

Computation/scaling guidance, coordinate-choice guidance, nuisance admission screens, and threshold-based coefficient selection are described as operational proposals where applicable; they are not additional completed benchmark evaluations.
