# Reproduce the experiments from data and source

The release supports two operations: **independent reruns** of the declared experiments, and **numerical replay** of retained historical outputs. Missing historical output files do not prevent a new run of the implemented experiment. The commands below train or simulate afresh, retain their new per-seed outputs, and aggregate them. They do not relabel new results as recovered historical runs or promise identical floating-point results across environments.

For the principal prospective intervention and corrected hidden-slice experiments, use [Reproducing the primary experiments](REPRODUCING_PRIMARY_EXPERIMENTS.md). This guide covers the synthetic main-text experiment, Table 1's frozen-feature training, the end-to-end penalty frontier, predefined-group diagnostics, and feature-space acquisition. No experiment implementation from the earlier `fisherglass_repro_repo` package is missing: the public `experiments/` tree includes the necessary modules and their local imports.

## Environment and data

Run from the repository root. Use Python 3.10 or newer; Python 3.12.4 was used for the source/CLI and small computational checks described below. Create a separate environment, then install:

```shell
python -m venv .venv
```

Activate it using `. .venv/bin/activate` on Linux/macOS/WSL, or `.venv\Scripts\Activate.ps1` in PowerShell. Install a matching `torch`/`torchvision` pair for the desired CPU/CUDA runtime, then:

```shell
python -m pip install -r requirements.txt
python -c "from pathlib import Path; Path('build/fresh_reproduction').mkdir(parents=True,exist_ok=True)"
python -m pip freeze > build/fresh_reproduction/environment.txt
python -c "import sys,torch; print(sys.version); print(torch.__version__,torch.version.cuda,torch.cuda.is_available())"
```

The local validation environment used NumPy 1.26.4, SciPy 1.13.1, scikit-learn 1.7.1, PyTorch 2.7.1, torchvision 0.22.1, WILDS 2.0.0, matplotlib 3.10.0 and Pillow 10.4.0. These identify a tested source/smoke environment, **not a recovered lockfile for the historical frozen-training/acquisition runs**. The separately pinned Table D.9 environment applies only to that auxiliary refit. Save the actual environment, exact command, device, new JSON outputs and caches for every full rerun.

The synthetic experiment needs only NumPy and matplotlib and runs on CPU. Frozen-feature training and acquisition also run on CPU; `--device cuda` in those scripts accelerates feature extraction, while their small linear-head fitting remains as implemented. The end-to-end and primary image pipelines should use the declared GPU configuration.

Waterbirds is the standard Waterbirds v1.0 dataset with the official WILDS split. Prepare it explicitly before running any preflight/launcher:

```shell
python scripts/prepare_waterbirds.py --root data/wilds
```

This helper uses the original Stanford-hosted `waterbird_complete95_forest2water2.tar.gz`, verifies all 11,788 referenced images and official split/group counts, and writes the WILDS release marker after validation. The legacy CodaLab endpoints used by WILDS 2.0 and the older prospective download script return HTTP 404; do not use `get_dataset(..., download=True)` as the Waterbirds preparation route. For an existing offline copy, use `python scripts/prepare_waterbirds.py --root data/wilds --archive /path/to/waterbird_complete95_forest2water2.tar.gz` instead. Subsequent experiment commands load the prepared data with downloads disabled.

For the CivilComments diagnostic, additionally run:

```shell
python -c "from wilds import get_dataset; get_dataset(dataset='civilcomments', root_dir='data/wilds', download=True)"
```

Use the original dataset terms linked in [DATA_LICENSES.md](../DATA_LICENSES.md). Waterbirds should contain `data/wilds/waterbirds_v1.0/metadata.csv` and its image paths, with official split counts 4,795 training, 1,199 validation and 5,794 test. The loaders preserve those splits. The prospective pixel intervention additionally requires CUB segmentation masks; the primary-experiment guide describes that separate preparation. Feature-space acquisition does **not** need those masks, a separate CUB image download, or a new Places download.

The frozen-feature scripts use `torchvision.models.MobileNet_V3_Small_Weights.DEFAULT`, which resolves to `IMAGENET1K_V1` in the tested torchvision version. If not cached, torchvision obtains the pretrained weights. In that version, its standard transform resizes to 256 with bilinear interpolation, center-crops to 224, converts to tensor, and normalizes with ImageNet means `(0.485,0.456,0.406)` and standard deviations `(0.229,0.224,0.225)`. The backbone runs in evaluation mode; features are the pooled, flattened convolutional representation, without augmentation or fine-tuning.

Use a new cache location for the rerun, as in the commands below. The historical frozen-feature loader trusts an existing cache without revalidating its model/dataset provenance. Reusing an unrelated cache would change the experiment. A correctly produced cache may be shared among the frozen-training and acquisition commands that use the same data/representation.

## Synthetic tail, reserve and portfolio experiment

Reproduce Section 5.2 and Appendix D.3 with:

```shell
python scripts/run_synthetic_reproduction.py
```

This wrapper invokes the unchanged `experiments/synthetic_fisher_glass.py` for seeds **11,12,13,14,15**, using **300,000 environments**, **256 observations per environment**, `gamma=1`, `delta_max=0.35`, `alpha=0.95`, `eps=1e-12`, `logit_scale=4`, `reserve_r=0.05`, and `anchor_delta=0.5`. It then calls the original `aggregate_synthetic` function in `multiseed_fisher_glass_diagnostics.py`. No external data or weights are used.

The simulation draws `delta=0.35*U^(1/gamma)`, independent normal `s,r`, task coordinate `z=s+delta*r`, label `1{z>0}`, and prediction logit `4s`. It evaluates exact population feature-information blocks, Monte Carlo prediction accuracy/loss, the upper-tail inverse-information exponent, a 5% reserve with anchor information 0.25, and portfolios with `K=1,2,3`. Each seed also evaluates the `gamma=1,2,4` tail sweep. The largest 5% of inverse-information observations supply the log-survival slope fit.

Outputs are:

- `build/fresh_reproduction/synthetic/seed_11/` through `seed_15/`: complete JSON, figures and execution log for each seed.
- `build/fresh_reproduction/synthetic/summary.json`: the native five-seed summary, including means and nominal normal-approximation half-widths.

The single-seed mass-sweep values in the historical report are distinct from five-seed mean estimates; inspect each JSON's `tail_boundary` separately. A short installation check is:

```shell
python scripts/run_synthetic_reproduction.py --smoke --out-dir build/reproduction_smoke/synthetic
```

Smoke mode uses 2,000 environments and 32 observations per environment, marks its outputs as smoke results, and is not a substitute for the paper-scale invocation.

## Frozen-feature training: main Table 1 and Appendix E.2

After preparing Waterbirds, run this single-line command in any supported shell:

```shell
python experiments/waterbirds_fisher_glass_training.py --root-dir data/wilds --cache build/fresh_reproduction/features/waterbirds_mobilenetv3_small.npz --device cpu --batch-size 64 --seeds 17 18 19 20 21 --epochs 1200 --lr 0.01 --weight-decay 0.0001 --alpha 0.95 --fg-alpha 0.75 --ridge 0.001 --eps 1e-8 --fg-lambdas 0.01 0.03 0.1 0.3 --gdro-fg-lambdas 0.01 0.03 0.1 0.3 --selection-worst-tolerance 0.02 --jtt-upweight 20 --audit-bootstrap-replicates 100000 --audit-confidence 0.95 --audit-seed 20260729 --output build/fresh_reproduction/frozen_training.json
```

The command explicitly overrides the source's three-seed default with the paper's **five seeds 17–21**. Both FG families search the declared four-value grid. Do not restrict Group DRO+FG to the two coefficients selected in the historical runs: the selection procedure must see all four candidates.

The shared loader standardizes every feature using training-set means and population standard deviations, replacing feature standard deviations below `1e-6` with one. It fits a balanced L-BFGS logistic background probe on training features (`max_iter=1000`, seed 18 for this invocation) and standardizes its scalar scores using training moments. The linear task head uses full-batch AdamW, 1,200 updates, `lr=.01`, weight decay `.0001`, weights `N(0,.01^2)` and zero intercept. Current task logits are standardized on the full training set during the Fisher penalty; evaluation uses the fitted training-logit moments.

The script runs ERM, Group DRO, Loss-CVaR, JTT, standalone FG and Group DRO+FG. Group DRO uses the maximum group mean loss. Loss-CVaR uses the largest `ceil(.1*n)` training losses. JTT refits after upweighting initial ERM mistakes by 20. Training Fisher tail level `.75` and audit level `.95` both select the worst of four equally weighted groups. For each FG family/seed, candidates within `.02` of the best validation worst-group accuracy are eligible; the largest validation certificate wins, with higher validation worst-group accuracy breaking certificate ties. No test outcome selects the coefficient.

`frozen_training.json` contains every candidate and selected run under `runs`, and the table-ready six-method aggregation under `summary`. For main Table 1, use `erm`, `fisher_glass`, `group_dro`, and `group_dro_fisher_glass`; Appendix E.2 adds the other baselines. Multiply accuracy means/half-widths by 100 for percentages, while certificate values keep their original units. The `*_mean_ci` pairs are `[mean, 1.96*sample_SD/sqrt(5)]`.

The current source additionally computes the later ridge/exact audit fields; those do not change the declared fitting objective or selection criterion. The historical raw training JSON/checkpoints are unavailable, but the source and fitting/selection specification needed for this independent run are present. The later retained W2 replication is explicitly a different run cohort, not a replacement for Table 1's original results.

## Feature-space acquisition: Section 6 and Appendix E.5

Run the main strength `gamma=1` with all tabulated products and controls:

```shell
python experiments/waterbirds_counterfactual_antidote.py --root-dir data/wilds --cache build/fresh_reproduction/features/waterbirds_mobilenetv3_small.npz --device cpu --batch-size 64 --seeds 17 18 19 20 21 --trainer sklearn --objective erm --max-iter 1000 --alpha 0.95 --ridge 0.001 --eps 1e-8 --gamma 1.0 --budgets 0.01 0.02 0.05 --selectors fg_tail_x_loss_swap fg_tail_rank_x_loss_rank_swap fg_tail_x_uncertainty_swap fg_tail_transverse_swap low_loss_tail_transverse_swap high_loss_swap uncertainty_swap raw_fisher_swap group_balanced_swap random_swap --output build/fresh_reproduction/acquisition_gamma1.json --summary-csv build/fresh_reproduction/acquisition_gamma1.csv
```

The first five named selectors plus high loss cover the paper's main and alternative-product comparisons; the extra controls reproduce the wider retained report. The source first fits a balanced `LogisticRegression(solver='lbfgs', max_iter=1000)` baseline, using the installed version's defaults for `C`, tolerance, intercept and L2 penalty. In the tested version these are `C=1`, `tol=1e-4`, `fit_intercept=True`, `penalty='l2'`. The `--epochs`, `--lr` and `--weight-decay` options govern the alternative Torch backend; they do not govern the reported sklearn fits.

Training feature means define same-label opposite-background displacements. Each official-validation candidate is transformed as `x_cf=x+gamma*(mean_train[y,1-bg]-mean_train[y,bg])`, assigned the same label and the opposite background, and rescored by the fixed training-fitted background probe. The pool's examples are ranked by the named selector; selected features augment the training data, and the balanced logistic head is fitted from scratch. The official test split supplies evaluation. These are feature-space interventions, not pixel swaps or assertions of semantic label preservation.

Budgets are fractions of the original 4,795 training examples: the implemented `round`/cap rule gives **48,96,240** acquired candidates for 1%,2%,5%. The selector RNG uses each seed plus the budget and deterministic selector-name offsets; all selectors and all five seeds are retained. Training/probe normalization, residual-score construction, augmentation, and post-refit normalization are implemented in `waterbirds_counterfactual_antidote.py`, `waterbirds_antidote_acquisition.py`, and `waterbirds_fisher_glass_training.py`.

For the strength-by-budget comparison, repeat the same invocation with `--gamma 0.25`, `--gamma 0.5`, and `--gamma 1.5`, giving each a distinct `--output` and `--summary-csv` such as `acquisition_gamma025.json`/`.csv`. Retaining all selectors is valid; for the paper's strength table only, `--selectors fg_tail_x_loss_swap high_loss_swap` produces its two required arms. This restriction does not change those arms' selector-specific RNG streams.

Both `summary` and every newly produced per-seed result are retained in each JSON. The CSV contains aggregate means. Match paper labels as follows:

| Paper label | CLI selector | Output quantity |
|---|---|---|
| TTI surrogate / geometric score × loss | `fg_tail_x_loss_swap` | `delta_worst_group_accuracy`, `delta_fisher_glass_certificate` |
| High loss | `high_loss_swap` | Same quantities |
| Pure tail | `fg_tail_transverse_swap` | Same quantities |
| Rank product | `fg_tail_rank_x_loss_rank_swap` | Same quantities |
| Geometric score × uncertainty | `fg_tail_x_uncertainty_swap` | Same quantities |

Multiply the accuracy differences by 100 to obtain percentage points; leave certificate changes unscaled. The original report's per-seed acquisition JSON/CSV was not recovered, so original seed intervals cannot be replayed. A new invocation creates its own seed records and intervals and enables an independent comparison to the retained descriptive means.

The earlier unpaired reserve-acquisition control can also be rerun:

```shell
python experiments/waterbirds_antidote_acquisition.py --root-dir data/wilds --cache build/fresh_reproduction/features/waterbirds_mobilenetv3_small.npz --seeds 17 18 19 20 21 --trainer sklearn --objective erm --max-iter 1000 --budgets 0.01 0.02 0.05 --selectors fg_tail_transverse low_loss_tail_transverse raw_fisher high_loss uncertainty group_balanced random --alpha 0.95 --ridge 0.001 --eps 1e-8 --output build/fresh_reproduction/unpaired_acquisition.json --summary-csv build/fresh_reproduction/unpaired_acquisition.csv
```

## End-to-end penalty frontier: Appendix E.3

The following shell loop is for Bash, WSL or Git Bash. Data are already staged. It reproduces the six reported coefficients, five seeds, 10 epochs, batch size 128 and evaluation batch size 512, with the launcher defaults for AdamW (`lr=1e-4`, weight decay `1e-4`), bf16 autocast, TF32, channels-last layout, eight image workers and prefetch factor four:

```bash
for coefficient in 0.00005 0.0001 0.0003 0.001 0.003 0.01; do
  BENCHMARKS=waterbirds METHODS="group_dro group_dro_fisher_glass" \
  SEEDS="17 18 19 20 21" EPOCHS=10 BATCH_SIZE=128 EVAL_BATCH_SIZE=512 \
  DEVICE=cuda DOWNLOAD=0 ALLOW_MODEL_DOWNLOAD=1 RESUME=0 \
  FG_LAMBDA="$coefficient" \
  OUTPUT="build/fresh_reproduction/frontier/waterbirds_fast_lambda${coefficient}.json" \
  LOG_PATH="build/fresh_reproduction/frontier/lambda${coefficient}.log" \
  bash experiments/run_full_end_to_end_suite.sh
done
```

`RESUME=0` ensures these are fresh fits; select a new output directory rather than reusing earlier files. The trainer reports the final checkpoint after all epochs, not a restored best-validation checkpoint. To generate the frontier summary and plots entirely within the new-output directory:

```shell
python experiments/plot_lambda_frontier.py --input-dir build/fresh_reproduction/frontier --pattern "waterbirds_fast_lambda*.json" --summary-json build/fresh_reproduction/frontier/summary.json --summary-csv build/fresh_reproduction/frontier/summary.csv --frontier-pdf build/fresh_reproduction/frontier/frontier.pdf --frontier-png build/fresh_reproduction/frontier/frontier.png --deltas-pdf build/fresh_reproduction/frontier/deltas.pdf --deltas-png build/fresh_reproduction/frontier/deltas.png
```

The plotter's spread columns are sample standard deviations where named `*_std`; do not describe those as the nominal mean-interval half-widths used in other tables. Original frontier output files were not recovered. The command creates a new fully retained frontier with the specified protocol; retained W2 files provide an additional, separately identified replication.

## Predefined-group diagnostics and complete diagnostic aggregation

With both datasets staged, the existing multiseed driver runs and aggregates all three diagnostics:

```shell
python experiments/multiseed_fisher_glass_diagnostics.py --root-dir data/wilds --seeds 11 12 13 14 15 --civilcomments-seeds 13 14 15 16 17 --waterbirds-seeds 17 18 19 20 21 --synthetic-n-env 300000 --out-dir build/fresh_reproduction/diagnostics --summary-output build/fresh_reproduction/diagnostic_summary.json
```

It uses the same synthetic settings, CivilComments TF-IDF with 50,000 maximum features, and frozen MobileNet Waterbirds diagnostics. CivilComments uses training-fitted unigram/bigram TF-IDF with sublinear term frequency, `min_df=5`, balanced SGD logistic task/probe fits (`alpha=1e-5`, 50 iterations, tolerance `.001`), and minimum evaluated group size 100. Waterbirds uses the official four groups, balanced L-BFGS heads (`max_iter=1000`), and the training-normalized features/scores specified above. The current scripts include nominal ridge/exact audit records in addition to the feature-Schur diagnostic.

The diagnostic driver uses the Waterbirds diagnostic's default feature-cache path `build/features/waterbirds_mobilenetv3_small.npz`; ensure that path is unused or belongs to this exact data/representation. To aggregate already completed files, repeat the command with `--skip-run`. This command is separate from the source-only synthetic wrapper because it requires both real datasets.

## Checks completed for this guide, and remaining limits

The synthetic generator completed a small CPU run, and the standalone wrapper completed all five smoke seeds plus the existing aggregation function. All seven documented experiment/aggregation entry points accepted `--help` with their imports available: synthetic, frozen training, both acquisition scripts, multiseed diagnostics, end-to-end training and frontier plotting. A separate synthetic-feature fixture exercised the existing frozen-training pipeline's six methods, validation selection, evaluation and aggregation (14 candidate/selected rows), and all ten counterfactual acquisition selectors, their head refits and aggregations. The fixture used 80 generated training rows and 40 generated validation/test rows; it did not load Waterbirds, download weights or establish any reported real-data result. The frozen fixture used 12,000 audit simulations, the implementation's minimum for that simultaneous-tail calculation; the paper-scale command above retains 100,000.

The full dataset-dependent commands were inspected against their parsers, data loaders, objectives and aggregators. No full real-data training run was performed to prepare this guide. Historical package lockfiles and some historical output files remain unavailable. Accordingly, the declared protocol can be rerun independently, while exact historical byte identity and complete cross-version numerical equality are not asserted. Keep new results in the designated `build/fresh_reproduction/` directories and compare their measured behavior with the manuscript, rather than replacing the archival evidence silently.
