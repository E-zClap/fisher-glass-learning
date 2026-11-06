# Waterbirds Fisher-Antidote Acquisition Report

## Purpose

This experiment tests whether Fisher-Glass can do more than certify a weak
slice. The goal is to use the weakest nuisance-projected Fisher mode as an
acquisition rule:

1. Train a base Waterbirds classifier.
2. Find the weakest validation slice under the Fisher-Glass certificate.
3. Compute the efficient-score residual
   \[
   \phi = z - \beta b,\qquad \beta = J_{TN}/(J_{NN}+\lambda).
   \]
4. Select a small candidate reserve from the validation pool using
   \(\phi^2\).
5. Add the selected examples to the training set, retrain the linear head, and
   evaluate on the official Waterbirds test split.

The experiment asks whether Fisher-antidote acquisition lifts the
Fisher-Glass certificate more efficiently than random, group-balanced,
high-loss, uncertainty, or weak-slice selection.

## Implementation

Script:

```bash
python experiments/waterbirds_antidote_acquisition.py \
  --root-dir data/wilds \
  --seeds 17 18 19 20 21 \
  --budgets 0.01 0.02 0.05 \
  --selectors fisher_antidote paired_antidote paired_random low_loss_antidote \
              weak_slice_antidote weak_slice_random high_loss uncertainty \
              group_balanced random \
  --trainer sklearn \
  --output build/results/waterbirds_antidote_acquisition.json \
  --summary-csv build/results/waterbirds_antidote_acquisition_summary.csv
```

Outputs:

- `build/results/waterbirds_antidote_acquisition.json`
- `build/results/waterbirds_antidote_acquisition_summary.csv`

The model uses frozen ImageNet MobileNetV3-small features and a balanced
logistic classifier. The candidate pool is the Waterbirds validation split;
evaluation is on the official test split.

## Selectors

| Selector | Meaning |
|---|---|
| `fisher_antidote` | Highest efficient residual \(\phi^2\) over all candidates |
| `paired_antidote` | Highest \(\phi^2\) among same-label, opposite-background candidates for the weakest slice |
| `paired_random` | Random same-label, opposite-background candidates |
| `low_loss_antidote` | Highest \(\phi^2\) among below-median-loss candidates |
| `weak_slice_antidote` | Highest \(\phi^2\) inside the weakest slice |
| `weak_slice_random` | Random examples inside the weakest slice |
| `high_loss` | Highest current loss |
| `uncertainty` | Highest predictive uncertainty |
| `group_balanced` | Equal examples from the four Waterbirds groups |
| `random` | Uniform random validation examples |

## Baseline

Across seeds, the balanced logistic head is deterministic in this setup:

| Metric | Value |
|---|---:|
| Average test accuracy | 0.782 |
| Worst-group test accuracy | 0.525 |
| \(I^{FG}_{0.95}\) | 0.209 |
| Weakest Fisher-Glass slice | landbird/land |

The weakest slice is again not the lowest-accuracy slice; it is a
high-accuracy but low-identifiability slice.

## Main Results

### 1% acquisition budget

| Selector | \(\Delta I^{FG}_{0.95}\) | \(\Delta\) worst acc | selected \(\phi^2\) | selected loss | low-loss fraction |
|---|---:|---:|---:|---:|---:|
| high_loss | +0.0545 | +0.0685 | 0.165 | 15.131 | 0.00 |
| paired_random | +0.0146 | -0.0028 | 0.437 | 2.118 | 0.24 |
| random | +0.0115 | +0.0056 | 0.337 | 1.408 | 0.56 |
| paired_antidote | +0.0098 | +0.0031 | 1.876 | 0.280 | 0.60 |
| uncertainty | +0.0096 | +0.0016 | 0.375 | 0.689 | 0.00 |
| fisher_antidote | +0.0015 | -0.0016 | 2.154 | 0.145 | 0.79 |

### 2% acquisition budget

| Selector | \(\Delta I^{FG}_{0.95}\) | \(\Delta\) worst acc | selected \(\phi^2\) | selected loss | low-loss fraction |
|---|---:|---:|---:|---:|---:|
| high_loss | +0.0643 | +0.1075 | 0.159 | 11.785 | 0.00 |
| paired_random | +0.0266 | -0.0031 | 0.428 | 2.248 | 0.26 |
| paired_antidote | +0.0212 | +0.0000 | 1.379 | 0.536 | 0.54 |
| group_balanced | +0.0202 | +0.0389 | 0.380 | 1.725 | 0.46 |
| uncertainty | +0.0189 | +0.0234 | 0.383 | 0.778 | 0.00 |
| fisher_antidote | +0.0053 | +0.0031 | 1.701 | 0.413 | 0.67 |

### 5% acquisition budget

| Selector | \(\Delta I^{FG}_{0.95}\) | \(\Delta\) worst acc | selected \(\phi^2\) | selected loss | low-loss fraction |
|---|---:|---:|---:|---:|---:|
| high_loss | +0.0926 | +0.1215 | 0.207 | 7.025 | 0.00 |
| paired_random | +0.0477 | +0.0065 | 0.440 | 2.340 | 0.26 |
| random | +0.0420 | +0.0458 | 0.346 | 1.375 | 0.52 |
| group_balanced | +0.0373 | +0.0899 | 0.364 | 1.755 | 0.47 |
| paired_antidote | +0.0341 | +0.0000 | 0.775 | 1.319 | 0.38 |
| fisher_antidote | +0.0317 | +0.0296 | 1.136 | 0.693 | 0.53 |

## Interpretation

The experiment cleanly separates two facts.

First, Fisher-Glass spectroscopy works as intended. The Fisher-antidote
selectors pick examples with far larger efficient residual \(\phi^2\) than
random, uncertainty, or high-loss selection. At 1%, `fisher_antidote` selects
examples with mean \(\phi^2=2.154\), compared with 0.337 for random and 0.165
for high-loss. These examples are mostly low-loss: the low-loss fraction is
0.79 for `fisher_antidote` and 0.60 for `paired_antidote`.

Second, residual-only acquisition is not enough to beat high-loss acquisition
on downstream Waterbirds performance in this unpaired validation-pool setup.
High-loss selection dominates both certificate lift and worst-group accuracy.
This is because the Waterbirds validation pool contains many obvious hard
examples whose labels directly correct the decision boundary. In this setting,
high-loss data are still available, so the loss signal is very strong.

## What This Means for the Paper

This should not be presented as “Fisher-antidote acquisition beats active
learning baselines.” It does not.

The stronger and more honest claim is:

> Fisher-Glass spectroscopy identifies a distinct class of low-loss,
> high-efficient-residual examples: examples where task evidence is screened by
> nuisance variation. In the current unpaired Waterbirds reserve experiment,
> those examples are diagnostically sharp but not sufficient to outperform
> high-loss acquisition. A true interventional antidote requires a physically
> paired or counterfactual candidate pool, not merely unpaired held-out images.

This result actually sharpens the next experiment. The missing ingredient is
not the Fisher-Glass score; it is the physical intervention pool. The paired
selectors already move in this direction, but they still use unpaired examples.
The next version should use segmentation/background swaps or curated same-bird
counterfactuals, where the task label is fixed and only the nuisance actuator is
moved.

## Paper-Safe Conclusion

Use this experiment as a stress test, not as the main positive empirical claim:

> Fisher-antidote acquisition reveals low-loss soft modes that high-loss
> selection ignores, but unpaired acquisition does not by itself solve
> Waterbirds. This motivates the interventional version of Fisher-Glass:
> physically valid task-preserving nuisance motions are needed to turn
> spectroscopy into an antidote.

