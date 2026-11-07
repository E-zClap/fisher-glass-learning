# Fisher-Glass supplemental material

This is the **camera-ready-v15** reproducibility release for the final paper, [available on GitHub](https://github.com/E-zClap/fisher-glass-learning/releases/tag/camera-ready-v15). It includes the final manuscript, standalone scientific supplement, experiment implementations, retained results, and numerical replay inputs. Preparing this release introduced no new empirical experiments.

Start with [supplementary_material.pdf](supplementary_material.pdf) for the scientific supplement and its reproduction guide, or [main.pdf](main.pdf) for the complete final paper. The [experiment map](docs/REBUTTAL_EXPERIMENT_MAP.md) identifies each retained cohort, its protocol, reproduction command, and limitations.

The final paper has 47 pages: main text and the funding/competing-interest statement on pages 1–10, references on pages 11–12, scientific Appendices A–E on pages 13–39, and artifact availability/checklist on pages 40–47. The main-paper SHA256 is:

```text
9c568df6006e5dc4a448b8e16d36165f3f62a6cedb895e5f4b3b2465aa8d1db7
```

This release synchronizes the final source and PDF, restores the official checklist guidance, clarifies the matched eligible-selector comparison and ridge limitation, and includes the standalone supplementary document and retained paired flip-prediction intervals. Scientific Appendices A–E and experimental values are unchanged. [ARTIFACT_MANIFEST.json](ARTIFACT_MANIFEST.json) records the files included here, their hashes, and original archive provenance.

## Package contents

| Directory or file | Contents |
|---|---|
| `tex/camera_ready_v2/` | Current full-paper source/PDF, standalone supplement source, editable figure sources/PDFs, and the two Figure 1 image examples with sanitized provenance. |
| `artifacts/` | Prospective v1/v2, corrected hidden-slice/repair, group-free coefficient selection, and W2 ridge/exact audit records; checkpoint identifiers. |
| `reanalysis/` | Matched selectors, matrix constructions, four-summary swap comparison, flip prediction, supporting contrasts, statistical reporting, and deterministic classification-example checks. |
| `experiments/` | Full implementation coverage of the earlier code-only bundle, plus prospective, hidden-slice and group-free pipelines. |
| `scripts/` | Retained-result replay, W2 replay, dataset preparation, and a fresh synthetic-run wrapper. |
| `docs/` | Rebuttal coverage map, full rerun instructions, manuscript map, and historical expanded-appendix source records. |
| `reports/` | Historical aggregate reports, including results whose original per-seed outputs were not recovered. |

The scientific supplement retains Appendices A–E and the current main-paper labels, followed by a reproduction guide. Supplementary Table F.2 displays all nine retained paired flip-prediction metric intervals directly from `reanalysis/flip_prediction/paired_summary.json`; no new analysis or refit is introduced. The supplement's `main_reference_labels.aux` is an intentional, filtered cross-reference input; compiler logs and temporary build products are excluded.

## Replay retained results without training

Run these commands from the package root. They require the numeric dependencies in `requirements-replay.txt`; no datasets, model weights, GPU, downloads, or remote jobs are required.

```shell
python -m pip install -r requirements-replay.txt
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

Most exports are written under `build/reproduced_tables/`; the flip, four-summary and supporting-contrast commands regenerate their retained exports in place. The final hash check detects unintended changes to packaged files. The classification check verifies analytic identities by quadrature and finite sums; it is not an empirical experiment.

An optional, separately documented [auxiliary logistic refit](reanalysis/flip_prediction/README.md) uses the original numeric donor rows and `requirements-refit.txt`. Complete new dataset/model runs are described in [primary experiment reproduction](docs/REPRODUCING_PRIMARY_EXPERIMENTS.md) and [other experiment reproduction](docs/REPRODUCING_EXPERIMENTS.md). Those are different operations from replaying retained results.

## Compile the paper or supplement

With a LaTeX installation and `latexmk`, run from the package root:

```shell
cd tex/camera_ready_v2
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error supplementary_material.tex
```

Keep the supplied `main_reference_labels.aux`: it is a required static source input for references from the supplement to main-paper results. The supplement also uses its four `supplement*.tex` source files, the included bibliography files, and the shared appendix/figure sources. Generated build files are not part of the curated manifest.

## Provenance and limits

Use corrected hidden-slice v2 only. Prospective v1 is a development cohort; prospective v2 is the primary frozen evaluation. Group-free selection is post hoc, and the four-summary comparison is a post hoc diagnostic comparison rather than independently trained objective arms. The fresh W2 replications are separate from the original training cohort.

The Fisher-based corrected repair selects the unchanged model in all five seeds. The matched eligible accuracy and loss selectors select the same waterbird/water slice as the frozen rule in all five primary seeds. Neither result supports an additional efficacy or selection-uniqueness claim. The flip-prediction analysis compares a baseline against two added coordinates; it has no retained background-score-alone comparator.

Original per-seed outputs for the legacy frozen-feature/end-to-end training, acquisition, synthetic and predefined-group summaries were not recovered. Their source and aggregate reports are included; exact historical replay is not possible for those missing outputs. Full datasets, checkpoints, feature caches, fitted hidden-slice/group-free coefficient arrays, and the full generated intervention-image set are excluded. The two retained Figure 1 composites are the image exception. See [DATA_LICENSES.md](DATA_LICENSES.md) for dataset/model/asset terms and scope.

Original archive/member hashes identify historical provenance. Published-file hashes identify the curated files actually distributed; sanitization and omitted arrays are recorded in the manifest. Do not substitute the older uncorrected hidden-slice archive or treat curated result directories as resumable training runs.
