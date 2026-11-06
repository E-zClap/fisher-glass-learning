# Prospective Waterbirds nuisance-shift experiment

Closes the diagnostic-to-failure loop that reviewer W1 asked for:

> high source accuracy + benign source loss + low Fisher-Glass
> ⟹ failure under a held-out nuisance shift

The slice is chosen **from validation data alone**, everything is frozen and hashed,
and only then is a label-preserving background intervention applied to held-out images.

---

## Quick start (Linux + NVIDIA GPU)

Run from the **repository root**:

```bash
bash experiments/prospective_nuisance_shift/run_all.sh --smoke
```

Smoke mode is 1 seed / 1 epoch / 40 birds per group. It exercises every step in
~10 minutes (after the download) and its numbers are meaningless — it exists to
prove the pipeline runs on your machine. If it finishes cleanly:

```bash
bash experiments/prospective_nuisance_shift/run_all.sh
```

Data downloads once and is shared by both; add `--skip-download` on later runs.

```bash
bash experiments/prospective_nuisance_shift/run_all.sh --skip-download --epochs 30
```

### Requirements

```bash
pip install -r experiments/prospective_nuisance_shift/requirements.txt
```

Python **3.9 or newer** and a CUDA build of PyTorch. Everything else is standard.
`wilds` is optional — it is only used as a convenience downloader for Waterbirds;
none of the experiment code imports it.

To check an environment without running anything expensive:

```bash
python experiments/prospective_nuisance_shift/preflight.py
```

It compiles and imports every module, prints library versions and GPU info,
verifies the data layout, and builds one paired intervention image end to end.
`run_all.sh` runs it as step 1.

### Downloads (~1.14 GB total, done by `download_data.sh`)

| What | Size | Source |
| --- | --- | --- |
| Waterbirds v1.0 (composited images + `metadata.csv`) | ~1.1 GB | WILDS / CodaLab |
| CUB-200-2011 `segmentations.tgz` | 39.3 MB | `data.caltech.edu` |

**Places365 is not needed.** Donor backgrounds are recovered from training-split
Waterbirds images by inpainting their own (small) bird hole.

### Expected runtime and disk

| Step | GPU time | Disk |
| --- | --- | --- |
| Training (20 runs: ERM×5 seeds + FG×3 λ×5 seeds, 20 epochs) | ~1.5 h | ~250 MB checkpoints |
| Paired image generation (23k images, CPU-bound, multiprocess) | ~10 min | ~1 GB |
| Shift evaluation (10 models) + bootstraps | ~15 min | small |

Cut the cost with `--seeds "17 18 19"`, `--fg-lambdas 0.03`, `--draws 1`,
`--max-per-group 800`, or `--groups 0` (only the selected slice).

---

## What the experiment does

### 1. A real fine-tuned model, not a linear probe

`train_end_to_end.py` fully fine-tunes MobileNetV3-small (ImageNet init) on
Waterbirds over five seeds. `--arch resnet50` is available if a reviewer asks for
the standard architecture. Methods: `erm` (headline), `fisher_glass` (the
intervention arm), and optionally `group_dro` / `group_dro_fisher_glass`.

### 2. Validation-only selection, then freeze

`select_slice.py` uses **only** the validation split to:

- fit a temperature,
- fit a **cross-fitted** logistic background probe on the trained pooled
  representation — its score is the nuisance coordinate `b` (out-of-fold on
  validation, full-fit applied to test),
- score every candidate slice, and apply

  ```
  G_benign = { g : Acc_val(g) ≥ Acc_val(overall)  and  NLL_val(g) ≤ median_h NLL_val(h) }
  g*       = argmin_{g ∈ G_benign} λ_min(F^rid_{T|N,g})
  ```

  so every eligible slice already looks safe by accuracy *and* loss, and
  Fisher-Glass only chooses among the apparently-safe ones;

  > **Which statistic.** `--selection-statistic` defaults to `projected_over_raw`,
  > the *fraction* of task Fisher surviving nuisance projection, not the absolute
  > `projected_fisher`. The absolute value carries `J_TT`, the within-slice spread
  > of the standardised logit, which varies several-fold across Waterbirds groups
  > (0.71 to 2.10 in one run), so it mostly measures how confidently spread the
  > logits are rather than how much of the signal is nuisance-explained. The ratio
  > is scale-free. Both are always recorded, and the rule reports which slice the
  > other variant would have chosen (`absolute_fisher_rule_would_select`).
  >
  > Note how many slices survive the benign filter — the statistic only *ranks*
  > within that set, so if the filter leaves two candidates, the statistic's
  > contribution is one binary choice, and the report prints `benign_names` to
  > make that visible.
- choose the Fisher-Glass λ as the largest validation certificate subject to
  validation accuracy within 1 pp of ERM — never using shifted test accuracy.

Two slice families are scored: the four canonical `(label, background)` groups,
and — for the finer-grained version — quantile bins of the probe score *within*
each group.

> Bin arithmetic: the validation split has 467 / 466 / 133 / 133 examples per
> group, so 5 bins would give 93 per bin. The default is `--n-bins 4
> --min-bin 100` (117 per bin). Use `--n-bins 5 --min-bin 90` for the 5-quantile
> version.

Everything is written to `frozen_selection.json` with a `frozen_hash` over the
rule, the slice, every checkpoint SHA-256 and every temperature.
`evaluate_shift.py` aborts if a checkpoint no longer matches.

Also recorded, as controls: which slice a **loss**-based rule would have picked,
which one a **worst-accuracy** rule would have picked, and which one the
**calibrated predictive-Fisher** variant picks (the confidence-saturation check).

### 3. A pixel-exact, label-preserving background intervention

Waterbirds was generated by the official `generate_waterbirds.py` as

```
composite = round(cub · seg) + round(crop_and_resize(place, size) · (1 − seg))
```

with `seg = segmentation_png / 255` a **soft** matte (the CUB mattes average up
to five annotators, so ~5% of pixels lie strictly between 0 and 1).
`compose.py` inverts that exactly:

1. recover the source image's own background — verbatim outside the matte
   support, Telea-inpainted inside;
2. `bird = clip(composite − round(bg_src · (1 − seg)), 0, 255) · [seg > 0]`.
   Where `seg == 1` this is exactly the CUB pixels; where `seg == 0` it is
   exactly zero; only the thin `0 < seg < 1` rim depends on the estimate, and
   there the inpainted value is read one to three pixels from real background;
3. `new = bird + round(crop_and_resize(bg_donor, size) · (1 − seg))`.

Each held-out bird gets two kinds of variant, with the **same** bird pixels,
matte and label:

- **`same`** — a *different* background of the same type. This is the artifact
  control: it pays exactly the same recomposition cost as the intervention.
- **`opposite`** — a background of the opposite type. This is the intervention.

so the meaningful quantity is `Δ_nuisance = Acc(same) − Acc(opposite)`.

Donors are training-split images with a small bird (`fg ≤ 0.08`), and up to four
donors are tried so the visibly-inpainted background area stays under 3%. The
rule depends only on matte geometry and is applied identically to both arms, so
it cannot favour one background type. The realised inpaint fraction is reported
per arm in `paired_generation.json` — check that the two arms match.

Look at `build/prospective_shift/previews/paired_group0.jpg`
(columns: original | same-type | opposite-type) before trusting any number.

### 4. The measurement that source loss cannot explain

```
ConfidentFlip = P[ ŷ(T_opposite(x)) ≠ y  |  ŷ(x) = y,  p_y(x) ≥ 0.9,  x ∈ g* ]
```

Restricted to predictions that were *correct* on the original image with
*calibrated* confidence ≥ 0.9 — so neither source accuracy nor source loss can
account for the failures. `confident_flip_same` is the identical quantity for the
same-type swap and is the artifact floor the intervention has to beat.

Uncertainty is a cluster bootstrap: draws of the same bird move together
(source-level), and a stricter species-level bootstrap is reported alongside.

### 5. Two independent cross-checks

- **All four groups** are generated and evaluated, so "the certificate picked the
  slice that fails" can be checked against the slices it did *not* pick.
- **`natural_shift_fallback.py`** repeats the comparison with no image generation
  at all: the frozen slice vs the same-label / opposite-background slice on the
  untouched official test split. Causally weaker (different birds), but free of
  any compositing, so it independently rules out a recomposition artifact.

---

## Files

| File | Role |
| --- | --- |
| `run_all.sh` | driver for the whole pipeline |
| `download_data.sh` | Waterbirds + CUB mattes |
| `fetch_cub_masks.py` | downloads/extracts `segmentations.tgz`, verifies all 11 788 images have a matte |
| `preflight.py` | compiles + imports every module, checks versions, data layout and the compositor before any GPU time is spent |
| `common.py` | Fisher-Schur geometry, CVaR, temperature fit, cluster bootstrap, provenance hashing |
| `mask_stats.py` | caches per-image matte area (drives donor eligibility) |
| `compose.py` | exact inversion of the Waterbirds composition + donor bank |
| `models.py` | architecture factory, transform, AMP/device helpers |
| `data.py` | datasets and the intervention manifest |
| `train_end_to_end.py` | step 1 — fine-tuning |
| `select_slice.py` | step 2 — validation-only selection, then freeze |
| `build_paired_images.py` | step 3 — paired intervention images |
| `evaluate_shift.py` | step 4 — prospective evaluation on the frozen images |
| `natural_shift_fallback.py` | step 5 — unpaired natural-image cross-check |
| `make_figure.py` | step 6 — two-panel figure + bin-monotonicity figure |
| `make_report.py` | step 7 — markdown report with the rebuttal paragraph filled in |

> `select_slice.py` is deliberately **not** named `select.py`: that would shadow
> Python's stdlib `select`, which `multiprocessing` and the DataLoader workers
> import.

---

## What to send back

```bash
tar czf prospective_shift_results.tar.gz \
  reports/prospective_nuisance_shift_report.md \
  tex/figures/waterbirds_prospective_shift*.pdf \
  build/prospective_shift/previews \
  build/prospective_shift/*.json build/prospective_shift/*.csv \
  build/prospective_shift/runs/*/run.json \
  build/prospective_shift/runs/*/selection.json \
  build/prospective_shift/runs/*/shift_results.json \
  build/prospective_shift/runs/*/natural_shift.json
```

That excludes the checkpoints and the generated images, so it should be a few MB.
`run_all.sh` prints the same command when it finishes.

---

## Caveats worth stating in the paper

- **The training penalty needs a nuisance that varies inside a slice.** The
  obvious choice — condition on the four `(label, background)` groups and use the
  binary background annotation as `b` — is *degenerate*: background is constant
  within a group by construction, so `J_TN²/J_NN` collapses to either `E[z]²` or
  `0` depending on where that constant falls relative to the batch mean, and the
  penalty silently becomes "maximise the within-group logit variance", which
  fights the classification loss and destabilises training. The default here is
  therefore `--fg-nuisance probe`: an online linear background probe fit on
  *detached* pooled features, whose continuous score varies inside every slice
  and mirrors the geometry the diagnostic uses. A slice whose nuisance barely
  varies is skipped outright (`--fg-min-nuisance-std`), so the degenerate case
  cannot recur. `--fg-nuisance background` reproduces the old behaviour and
  should be paired with `--fg-slices labels`.
- **Batch composition still limits which slices contribute.** With the
  95%-confounded training split and batch 32, the minority groups (184 and 56
  training images) rarely reach `--fg-min-slice`, so in practice the penalty acts
  on `landbird/land` and `waterbird/water`. That includes the selected slice, but
  the CVaR is effectively over two groups rather than four — raise
  `--batch-size` if you want the minority groups to contribute.
- **The projected Fisher also shrinks as the logits concentrate.** `J_TT` is the
  within-slice spread of the standardised logit, so a very confident, converged
  model has a small projected Fisher regardless of nuisance alignment. That is
  why the `projected/raw` ratio is reported alongside the absolute value: the
  ratio isolates the part attributable to the nuisance direction.
- **Rim pixels.** ~5% of matte pixels are soft. The bird layer there is
  reconstructed from an inpainted background estimate rather than the original
  Places pixels. `rim_fraction` is recorded per image, and the same-type arm is
  affected identically.
- **Donor backgrounds are recovered, not original Places files.** Each has a
  small inpainted hole where its own bird was. Mean visible inpainted area is
  reported and is matched across arms.
- **The selection is prospective, the image generation is not conditioned on it.**
  Images are built for all four groups regardless of which one the rule selected,
  so generation cannot have been tuned to the selected slice.
