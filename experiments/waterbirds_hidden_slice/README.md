# Metadata-blind hidden-slice discovery and repair on Waterbirds

This experiment is the end-to-end evaluation of the hidden-slice adversary. It
does not begin with the four Waterbirds groups. Instead, it learns a
class-conditional, transferable slice gate without background or group labels,
freezes that gate, certifies it on disjoint examples, repairs the discovered
slice, and then restarts the adversary against each repaired model. Waterbirds
metadata is revealed afterward for interpretation and for explicitly labelled
oracle controls.

The primary result uses a frozen CLIP text direction as the audited nuisance
coordinate. A random direction is the negative control. A metadata-trained
background probe is reported only as an oracle-nuisance sensitivity analysis.

## Information boundary

The group-blind path may use only:

- images and landbird/waterbird task labels;
- official split membership and deterministic row identifiers;
- frozen image representations and classifier outputs;
- the predeclared CLIP background direction; and
- task-label-only overall, balanced, and per-class validation accuracy for the
  predeclared repair safety envelope, plus held-out hidden-slice Fisher and
  gate stability for diagnostics.

It may not use background labels, four-way group labels, group counts,
worst-group accuracy, metadata purity, or metadata-group Fisher for discovery,
gate selection, hyperparameter selection, repair selection, seed selection, or
the headline result. Every seed is retained. The headline slice mass is fixed
at 10%, shrinkage is fixed at 0.10, and the sensitivity grids are declared
before metadata is revealed.

`prepare_data.py` is the metadata firewall. It obtains images, task labels,
row identifiers, and official splits for the blind path; separately, it copies
background/group fields into `sealed_metadata.npz` without using them to
construct `h`, `b_vl`, or any blind-stage decision. It writes
`train_blind.npz`, `val_blind.npz`, and `test_blind.npz` for the main pipeline.
The blind files contain exactly
`row_id`, `h`, `y`, and the CLIP text nuisance coordinate `b_vl`. The
discovery, non-oracle repair, and certification paths consume the blind
artifact only. The sealed artifact has exactly one stage-level consumer:
`posthoc.py`, which runs only after every requested seed has completed blind
certification. Within that stage, metadata may be used only to interpret
already-frozen slices and to produce the explicitly labelled oracle
background-probe and oracle Group DRO controls.

Oracle outputs must never feed back into the primary configuration or select a
reported run. They are reference columns, not part of the hidden method.

## Frozen protocol

- Seeds: `17 18 19 20 21`.
- Cross-fitting: the head is fit on a label-only official-training partition;
  on validation, 5 folds rotate disjoint slice-discovery and certification
  roles. The official test split remains untouched until final evaluation.
- Class-conditional linear gate: one gate per task class, 10 restarts.
- Gate optimization: exact target-mass sigmoid-bias calibration, a
  capped-simplex projection with its correct implicit Jacobian, consistent
  constrained checkpoint/restart selection, and a constant feasible-gate
  objective floor.
- Headline within-class slice mass: `0.10`; sensitivity: `0.05 0.10 0.20`.
- Headline Fisher shrinkage: `0.10`; sensitivity: `0.05 0.10 0.20`.
- Admissibility: effective sample size at least `0.8` of nominal slice size and
  maximum normalized weight at most `2/m`.
- Selectors: hidden Fisher, equal-mass high loss, and equal-mass random.
- Nuisance bases: CLIP text direction (primary), random axis (negative
  control), and metadata-trained probe (oracle sensitivity).
- Certificates: temperature-calibrated predictive Fisher (primary) and
  unweighted feature-Schur Fisher (robustness check), together with loss,
  accuracy, confidence/entropy, margin, mass, and effective sample size.
- Repairs: loss-only pseudo-group, Fisher-only pseudo-group, and combined
  repair, plus a matched λ=0 ERM continuation with the same warm start and
  additional optimizer steps. Every repair family includes an exact no-op
  candidate; Fisher penalties use the dimensionless ratio
  `F_initial / (F_current + epsilon)`. Oracle Group DRO is an explicitly
  metadata-using comparator.
- Repair selection: overall, balanced, and both per-class task-label accuracy
  must remain within their fixed budgets. Eligible candidates are selected by
  frozen-slice loss, Fisher, or the declared joint Pareto criterion. A
  task-label-only calibration partition supplies step-zero-eligible early
  stopping.
- Every active repair is evaluated both on the original frozen gate and after
  the adversary is initialized from scratch and the worst slice is
  rediscovered. An exact no-op reuses the identical ERM adversary. The original
  ERM and every active repair/control use matched adversary seeds.

Hyperparameters are selected without group metadata, under the predeclared
task-label safety envelope. Post-hoc group composition and oracle
worst-group accuracy are evaluation quantities only. Fresh-adversary
cross-fitting for a selected repair is a validation diagnostic because the
same validation split enforces the accuracy budget; its untouched-test
certificate is the confirmatory repair result.

## Environment and data

Python 3.8 or newer is supported. The launchers use the executable named by
the `PYTHON` environment variable, or `python` when it is unset.

Install a CUDA build of PyTorch appropriate for the target NVIDIA driver first,
then install the experiment requirements from the repository root:

```bash
pip install -r experiments/waterbirds_hidden_slice/requirements.txt
```

By default the scripts expect:

```text
data/wilds/waterbirds_v1.0/metadata.csv
data/wilds/waterbirds_v1.0/<CUB image paths>
```

Model loading is local-only unless `ALLOW_MODEL_DOWNLOAD=1` is set. The CLIP
checkpoint may therefore need to be cached in advance. `DOWNLOAD=1` separately
allows WILDS to obtain Waterbirds; it does not authorize model downloads.

## Run the full experiment

Linux, WSL, or Git Bash:

```bash
. .venv/bin/activate
PYTHON="$PWD/.venv/bin/python" DEVICE=cuda ALLOW_MODEL_DOWNLOAD=1 \
  bash experiments/waterbirds_hidden_slice/run_all.sh
```

PowerShell:

```powershell
$env:DEVICE = "cuda"
& experiments\waterbirds_hidden_slice\run_all.ps1
```

`DEVICE=auto` is the default and selects CUDA when PyTorch can see it. For a
paper run, set `DEVICE=cuda` explicitly so a missing CUDA installation fails
early instead of silently committing to a very slow CPU run.

The corrected launcher deliberately writes to `waterbirds_hidden_slice_v2`
and an attested MobileNet cache. Do not point it at the old
`build/results/waterbirds_hidden_slice` directory: those artifacts were
produced by the invalid pre-v2 gate gradient/selection path and are not
scientifically reusable. Every downstream stage now verifies the recorded
algorithm version and exact scientific-source digest before accepting an
artifact.

Discovery performs many small class-conditional optimizations, so GPU use is
bursty and `nvidia-smi` can display low instantaneous utilization even when
CUDA is active. The corrected mass solver uses warm-started Newton updates with
periodic bracketed verification to reduce this overhead; the banner and
preflight still fail explicitly if `DEVICE=cuda` is unavailable. The discovery
stage prints each objective/mass/shrinkage/fold before optimizing it, so a long
paper run no longer appears silent between completed seeds.

Both launchers first run the read-only preflight and then run these stages in
order:

1. `prepare_data.py` extracts the frozen MobileNet representation and CLIP text
   nuisance coordinate, and creates the blind/sealed artifact boundary.
2. `discover.py` fits Fisher, high-loss, and random class-conditional gates on
   discovery folds for the predeclared mass/shrinkage grid.
3. `repair.py` trains the loss-only, Fisher-only, and combined pseudo-group
   repairs from the frozen protocol.
4. `certify.py` applies every already-frozen gate once to the untouched test
   data and computes held-out bootstrap intervals.
5. `posthoc.py` reveals metadata for composition, the oracle nuisance-probe and
   Group DRO controls, oracle-group comparisons, and image selection only after
   the hidden results are frozen.
6. `aggregate.py` computes paired seed summaries, confidence intervals,
   sensitivity tables, and the machine-readable result table.
7. `make_report.py` writes the reviewer-ready Markdown report, LaTeX table,
   and rebuttal text directly from the aggregate.
8. `make_figure.py` creates the four-panel reviewer figure.

The launchers centralize all shared settings as environment variables. Common
overrides include:

| Variable | Default | Meaning |
| --- | --- | --- |
| `ROOT_DIR` | `data/wilds` | WILDS dataset root |
| `OUT_DIR` | `build/results/waterbirds_hidden_slice_v2` | Corrected-v2 generated artifacts |
| `DATA_DIR` | `$OUT_DIR/prepared` | Blind and sealed prepared bundles |
| `IMAGE_ROOT` | `$ROOT_DIR/waterbirds_v1.0` | Images used only for feature extraction and the final qualitative panel |
| `FEATURE_CACHE` | `build/features/waterbirds_mobilenetv3_small_attested.npz` | Frozen image features with extractor identity |
| `NUISANCE_CACHE` | `build/features/waterbirds_clip_vit_base_patch32_nuisance.npz` | Frozen CLIP nuisance scores |
| `DEVICE` | `auto` | `auto`, `cuda`, `cuda:0`, or `cpu` |
| `DETERMINISTIC` | `1` | Enable deterministic Torch/CUDA algorithms; the scripts set `CUBLAS_WORKSPACE_CONFIG=:4096:8` before CUDA BLAS use |
| `SEEDS` | `17 18 19 20 21` | Paired experiment seeds |
| `FOLDS` | `5` | Cross-fitting folds |
| `GATE_RESTARTS` | `10` | Gate restarts per class/configuration |
| `MAIN_MASS` | `0.10` | Headline within-class slice mass |
| `MASSES` | `0.05 0.10 0.20` | Slice-mass sensitivity grid |
| `MAIN_SHRINKAGE` | `0.10` | Headline shrinkage |
| `SHRINKAGES` | `0.05 0.10 0.20` | Shrinkage sensitivity grid |
| `CLIP_MODEL` | `openai/clip-vit-base-patch32` | Frozen VL encoder |
| `NUM_WORKERS` | `8` | Data-loader workers |
| `CALIBRATION_FRACTION` | `0.15` | Predeclared calibration split fraction |
| `HEAD_STEPS`, `HEAD_LR` | `1200`, `0.01` | Shared linear-head optimizer settings, including the matched ERM-continuation control |
| `GATE_STEPS`, `GATE_LR` | `1000`, `0.03` | Shared gate optimizer settings |
| `LOSS_LAMBDAS` | `0.1 0.3 1.0` | Loss-repair grid |
| `FISHER_LAMBDAS` | `0.01 0.03 0.1` | Dimensionless Fisher-repair grid; combined repair uses the Cartesian product |
| `ACCURACY_BUDGET` | `0.02` | Allowed overall validation-accuracy drop |
| `BALANCED_ACCURACY_BUDGET` | `0.02` | Allowed balanced-validation-accuracy drop |
| `PER_CLASS_ACCURACY_BUDGET` | `0.02` | Allowed drop for each task class |
| `EARLY_STOPPING_PATIENCE`, `EARLY_STOPPING_EVAL_EVERY` | `12`, `25` | Task-label checkpoint patience and evaluation frequency |
| `EARLY_STOPPING_MIN_DELTA` | `1e-5` | Minimum checkpoint-NLL improvement |
| `BOOTSTRAP_REPLICATES` | `2000` | Certificate bootstrap replicates |
| `RANDOM_SLICES` | `200` | Equal-mass random comparator draws |
| `DOWNLOAD` | `0` | Set to `1` to let WILDS download Waterbirds |
| `ALLOW_MODEL_DOWNLOAD` | `0` | Set to `1` to allow pretrained-model downloads |

For example:

```bash
DEVICE=cuda:1 SEEDS="17 18 19" NUM_WORKERS=12 \
  bash experiments/waterbirds_hidden_slice/run_all.sh
```

```powershell
$env:DEVICE = "cuda:1"
$env:SEEDS = "17 18 19"
$env:NUM_WORKERS = "12"
& experiments\waterbirds_hidden_slice\run_all.ps1
```

## Smoke mode

Use `--smoke` only to verify plumbing on the target machine:

```bash
bash experiments/waterbirds_hidden_slice/run_all.sh --smoke
```

```powershell
& experiments\waterbirds_hidden_slice\run_all.ps1 -Smoke
```

Smoke mode uses one seed, two folds, one gate restart, short downstream
optimization, and a small bootstrap. Data preparation still validates or
builds the complete frozen feature/nuisance caches, so a first smoke run can be
substantial. Its slices, certificates, repair comparisons, confidence
intervals, and diagnostic tables are statistically meaningless. The launcher
skips the paper figure, and the LaTeX/rebuttal outputs are explicit incomplete-
run placeholders. Do not use smoke output in the paper or reviewer response.

## Outputs and audit trail

All generated data stays under `OUT_DIR`. Each stage records its parsed
arguments, seed/fold identifiers, input hashes, protocol hash, algorithm
version, and exact scientific-source digest. The expected
handoff bundle contains the frozen protocol, per-seed/fold discovery and
certification records, repair records, `aggregate.json`, `main_table.csv`,
`paired_comparisons.csv`, `sensitivity.csv`, reviewer/LaTeX/rebuttal text,
post-hoc composition, and PDF/PNG figures. Checkpoints and feature caches are larger and
can be omitted from a small results bundle, but their hashes must remain in the
manifests.

Archive the exact repository commit (or a source tarball) with the paper
artifacts. The recorded hashes prevent mixed data/model/result stages, but they
are integrity checks rather than a substitute for source-version archival.

Interpret the main table using the `clip_text`/mass `0.10`/shrinkage `0.10`
rows. The mass and shrinkage sweeps, random-axis results, and oracle-probe
results are sensitivity analyses; none may replace the predeclared headline
row after looking at metadata.
