# Waterbirds Counterfactual Fisher-Antidote Report

## Purpose

This experiment tests the interventional version of tail-transversality design.
The previous unpaired acquisition experiment showed that Fisher-Glass can find
low-loss, high-efficient-residual examples, but that simply adding those
ordinary validation examples does not outperform high-loss acquisition.

Here the candidate actions are paired same-label, opposite-background
counterfactuals in frozen-feature space:

\[
x_{\mathrm{cf}}
=
x+\gamma
\left(
\bar x_{y,1-b}-\bar x_{y,b}
\right),
\qquad
y_{\mathrm{cf}}=y,\quad b_{\mathrm{cf}}=1-b.
\]

This is not a pixel-level segmentation/background-swap benchmark. It is a
controlled feature-space proxy for the next claim:

> Fisher-Glass should select task-preserving nuisance interventions, not
> generic unpaired examples.

## Implementation

Script:

```bash
python experiments/waterbirds_counterfactual_antidote.py \
  --root-dir data/wilds \
  --seeds 17 18 19 20 21 \
  --budgets 0.01 0.02 0.05 \
  --selectors fg_tail_transverse_swap low_loss_tail_transverse_swap \
              raw_fisher_swap high_loss_swap uncertainty_swap \
              group_balanced_swap random_swap \
  --trainer sklearn \
  --gamma 1.0 \
  --output build/results/waterbirds_counterfactual_antidote.json \
  --summary-csv build/results/waterbirds_counterfactual_antidote_summary.csv
```

Outputs:

- `build/results/waterbirds_counterfactual_antidote.json`
- `build/results/waterbirds_counterfactual_antidote_summary.csv`

The base model is a balanced logistic classifier on frozen ImageNet
MobileNetV3-small Waterbirds features. The nuisance coordinate is a learned
background probe. The counterfactual action uses label-conditional feature
means to shift each validation example toward the opposite background while
preserving the bird label. The selected counterfactual features are added to
the training set, the linear head is retrained, and evaluation is on the
official Waterbirds test split.

Counterfactual metadata:

| Quantity | Value |
|---|---:|
| \(\gamma\) | 1.0 |
| mean feature-shift norm | 9.297 |
| p95 feature-shift norm | 9.463 |
| label-conditional shift fraction | 1.000 |

## Baseline

The frozen-feature balanced logistic baseline is deterministic across seeds in
this setup.

| Metric | Value |
|---|---:|
| Average test accuracy | 0.782 |
| Worst-group test accuracy | 0.525 |
| \(I^{FG}_{0.95}\) | 0.209 |
| inverse-Fisher p95 | 4.683 |
| Weakest Fisher-Glass slice | landbird/land |

## Selectors

| Selector | Meaning |
|---|---|
| `fg_tail_transverse_swap` | Counterfactual swaps with largest tail-weighted efficient residual |
| `low_loss_tail_transverse_swap` | Same, restricted to low-loss counterfactuals |
| `raw_fisher_swap` | Counterfactual swaps with largest raw task-score energy |
| `high_loss_swap` | Counterfactual swaps with largest current loss |
| `uncertainty_swap` | Counterfactual swaps with largest predictive uncertainty |
| `group_balanced_swap` | Equal counterfactual swaps from the four resulting groups |
| `random_swap` | Uniform random counterfactual swaps |

## Main Results

### 1% counterfactual budget

| Selector | \(\Delta I^{FG}_{0.95}\) | \(\Delta\) worst acc | selected tail gain | selected loss | low-loss fraction |
|---|---:|---:|---:|---:|---:|
| high_loss_swap | +0.0603 | +0.0327 | 0.471 | 11.913 | 0.00 |
| raw_fisher_swap | +0.0090 | +0.0047 | 0.946 | 0.893 | 0.96 |
| group_balanced_swap | +0.0068 | +0.0140 | 0.685 | 1.340 | 0.47 |
| fg_tail_transverse_swap | +0.0045 | +0.0062 | 2.946 | 0.176 | 0.77 |
| random_swap | +0.0040 | +0.0059 | 0.529 | 0.737 | 0.50 |
| uncertainty_swap | +0.0034 | +0.0218 | 0.351 | 0.688 | 0.00 |
| low_loss_tail_transverse_swap | -0.0016 | +0.0016 | 2.816 | 0.000 | 1.00 |

### 2% counterfactual budget

| Selector | \(\Delta I^{FG}_{0.95}\) | \(\Delta\) worst acc | selected tail gain | selected loss | low-loss fraction |
|---|---:|---:|---:|---:|---:|
| high_loss_swap | +0.0581 | +0.0576 | 0.450 | 9.038 | 0.00 |
| raw_fisher_swap | +0.0207 | +0.0078 | 0.943 | 1.712 | 0.88 |
| random_swap | +0.0147 | +0.0084 | 0.513 | 0.972 | 0.54 |
| group_balanced_swap | +0.0123 | +0.0196 | 0.711 | 1.091 | 0.51 |
| fg_tail_transverse_swap | +0.0098 | +0.0280 | 2.402 | 0.303 | 0.72 |
| uncertainty_swap | +0.0061 | +0.0218 | 0.370 | 0.721 | 0.00 |
| low_loss_tail_transverse_swap | -0.0018 | +0.0047 | 2.211 | 0.000 | 1.00 |

### 5% counterfactual budget

| Selector | \(\Delta I^{FG}_{0.95}\) | \(\Delta\) worst acc | selected tail gain | selected loss | low-loss fraction |
|---|---:|---:|---:|---:|---:|
| high_loss_swap | +0.0630 | +0.0701 | 0.403 | 4.794 | 0.00 |
| raw_fisher_swap | +0.0399 | -0.0031 | 0.711 | 1.783 | 0.74 |
| fg_tail_transverse_swap | +0.0340 | +0.0358 | 1.691 | 1.038 | 0.61 |
| random_swap | +0.0309 | +0.0184 | 0.576 | 1.023 | 0.49 |
| group_balanced_swap | +0.0268 | +0.0464 | 0.658 | 1.316 | 0.46 |
| uncertainty_swap | +0.0210 | +0.0234 | 0.415 | 0.871 | 0.00 |
| low_loss_tail_transverse_swap | +0.0015 | +0.0000 | 1.380 | 0.000 | 1.00 |

## Interpretation

The experiment gives a better result than the unpaired reserve test, but it
still does not validate the strongest optimistic version.

First, the interventional proxy helps. The `fg_tail_transverse_swap` selector
now consistently improves both the Fisher-Glass certificate and worst-group
accuracy. At 5% budget it raises \(I^{FG}_{0.95}\) by +0.0340 and worst-group
accuracy by +0.0358. This is materially better than the earlier unpaired
tail-transverse selector, where 1% selection barely moved the certificate and
could reduce worst-group accuracy.

Second, the Fisher-Glass score is still measuring a distinct intervention
direction. At 1% budget, `fg_tail_transverse_swap` selects actions with mean
tail gain 2.946, far above random (0.529), uncertainty (0.351), group-balanced
(0.685), high-loss (0.471), and raw-Fisher (0.946). Its selected actions are
mostly low-loss, with low-loss fraction 0.77. This supports the spectroscopy
claim: Fisher-Glass finds nuisance-transverse task information that ordinary
loss and uncertainty do not target.

Third, high-loss counterfactual swaps still dominate downstream repair in this
feature-space proxy. At all budgets, `high_loss_swap` gives the largest
certificate lift and worst-group accuracy improvement. The reason is direct:
the baseline Waterbirds classifier still has obvious high-loss errors, and
counterfactuals of those errors are highly corrective for the decision
boundary.

Fourth, the low-loss-only tail-transverse condition is too restrictive. It
selects very high tail gain and exactly low-loss actions, but barely repairs
the model. This tells us that the missing intervention is not merely "low loss
plus high efficient residual"; it needs enough decision-boundary leverage after
the nuisance is actuated.

## Paper Meaning

This experiment should not be sold as:

> Fisher-Glass counterfactual acquisition beats active-learning baselines.

It does not.

The paper-safe positive claim is:

> Counterfactual nuisance interventions make tail-transversality selection
> useful: unlike unpaired selection, FG-tail-transverse swaps consistently lift
> the certificate and worst-group accuracy. However, in this Waterbirds
> frozen-feature proxy, high-loss counterfactual swaps remain the strongest
> downstream repair because the base classifier still has large correctable
> errors.

The strongest conceptual message is:

> Fisher-Glass is best understood as an identifiability spectroscope. It tells
> us which task signal remains after nuisance response is screened. Turning
> that spectrum into an antidote requires an intervention pool that physically
> actuates the nuisance factor; feature-space counterfactuals are a first proxy,
> but a definitive test needs pixel-level same-bird background swaps or curated
> paired interventions.

## Recommended Use in the Paper

Use this as an appendix or a short main-text stress test. It strengthens the
paper if framed as an honest refinement:

1. Unpaired observational acquisition is not enough.
2. Counterfactual nuisance action is the right object.
3. Tail-transversality selection improves once the candidate pool is
   interventional.
4. High-loss remains the best repair when high-loss errors are abundant.

This turns a potential weakness into a sharper claim: Fisher-Glass is not a
generic active learner. It is a way to identify and target missing
nuisance-transverse task information.
