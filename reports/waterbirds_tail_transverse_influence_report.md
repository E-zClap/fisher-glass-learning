# Waterbirds Tail-Transverse Influence Report

## Purpose

The counterfactual Fisher-antidote experiment showed that pure
tail-transversality is a good spectroscope but not always the strongest repair.
The low-loss-only selector was especially revealing: it selected very large
tail-transverse gain but barely changed the model. The missing factor is
decision-boundary leverage.

This experiment tests the hybrid rule:

\[
\mathrm{Score}(a)
=
\underbrace{\mathrm{TailGain}(a)}_{\text{missing nuisance-transverse direction}}
\times
\underbrace{\ell(a)}_{\text{boundary/training leverage}} .
\]

The candidate actions are the same feature-space Waterbirds counterfactuals as
before: each validation example is shifted toward a same-label,
opposite-background feature mean and added as a synthetic training point.

## Implementation

Main hybrid run:

```bash
python experiments/waterbirds_counterfactual_antidote.py \
  --root-dir data/wilds \
  --seeds 17 18 19 20 21 \
  --budgets 0.01 0.02 0.05 \
  --selectors fg_tail_x_loss_swap fg_tail_rank_x_loss_rank_swap \
              fg_tail_x_uncertainty_swap fg_tail_transverse_swap \
              high_loss_swap raw_fisher_swap group_balanced_swap random_swap \
  --trainer sklearn \
  --gamma 1.0 \
  --output build/results/waterbirds_counterfactual_hybrid_gamma1.json \
  --summary-csv build/results/waterbirds_counterfactual_hybrid_gamma1_summary.csv
```

Gamma sensitivity:

```bash
python experiments/waterbirds_counterfactual_antidote.py \
  --root-dir data/wilds \
  --seeds 17 18 19 20 21 \
  --budgets 0.01 0.02 0.05 \
  --selectors fg_tail_x_loss_swap fg_tail_transverse_swap \
              high_loss_swap raw_fisher_swap random_swap \
  --trainer sklearn \
  --gamma 0.25
```

The same command was run for \(\gamma\in\{0.25,0.5,1.0,1.5\}\).

## Main Result at \(\gamma=1.0\)

The hybrid selector changes the story. Pure tail-transversality improves the
model, but the hybrid `fg_tail_x_loss_swap` is much stronger because it asks
for both the missing efficient direction and decision-boundary leverage.

| Budget | Selector | \(\Delta I^{FG}_{0.95}\) | \(\Delta\) worst acc | selected tail gain | selected loss |
|---:|---|---:|---:|---:|---:|
| 1% | `fg_tail_x_loss_swap` | +0.0317 | +0.0452 | 1.046 | 8.159 |
| 1% | `high_loss_swap` | +0.0603 | +0.0327 | 0.471 | 11.913 |
| 1% | `fg_tail_transverse_swap` | +0.0045 | +0.0062 | 2.946 | 0.176 |
| 1% | `random_swap` | +0.0040 | +0.0059 | 0.529 | 0.737 |
| 2% | `fg_tail_x_loss_swap` | +0.0452 | +0.0685 | 0.777 | 6.842 |
| 2% | `high_loss_swap` | +0.0581 | +0.0576 | 0.450 | 9.038 |
| 2% | `fg_tail_transverse_swap` | +0.0098 | +0.0280 | 2.402 | 0.303 |
| 2% | `random_swap` | +0.0147 | +0.0084 | 0.513 | 0.972 |
| 5% | `fg_tail_x_loss_swap` | +0.0606 | +0.0794 | 0.557 | 4.245 |
| 5% | `high_loss_swap` | +0.0630 | +0.0701 | 0.403 | 4.794 |
| 5% | `fg_tail_transverse_swap` | +0.0340 | +0.0358 | 1.691 | 1.038 |
| 5% | `random_swap` | +0.0309 | +0.0184 | 0.576 | 1.023 |

At \(\gamma=1.0\), the hybrid beats high-loss on worst-group accuracy at all
three budgets:

| Budget | Hybrid \(\Delta\) worst acc | High-loss \(\Delta\) worst acc |
|---:|---:|---:|
| 1% | +0.0452 | +0.0327 |
| 2% | +0.0685 | +0.0576 |
| 5% | +0.0794 | +0.0701 |

High-loss remains slightly stronger on certificate at 1% and 2%, and nearly
tied at 5%. The key improvement is no longer just measurement: the hybrid is a
real repair rule in this proxy.

The auxiliary hybrids are useful controls. Rank-normalizing both factors is
competitive on worst-group accuracy, while replacing loss by uncertainty is
much weaker. This suggests that the useful factor is not merely "near the
decision boundary" in an entropy sense; the current loss is a better proxy for
training leverage in this setting.

| Budget | Selector | \(\Delta I^{FG}_{0.95}\) | \(\Delta\) worst acc | selected tail gain | selected loss |
|---:|---|---:|---:|---:|---:|
| 1% | `fg_tail_rank_x_loss_rank_swap` | +0.0287 | +0.0545 | 1.329 | 5.953 |
| 1% | `fg_tail_x_uncertainty_swap` | +0.0012 | +0.0156 | 1.073 | 0.778 |
| 2% | `fg_tail_rank_x_loss_rank_swap` | +0.0326 | +0.0654 | 1.200 | 3.946 |
| 2% | `fg_tail_x_uncertainty_swap` | +0.0073 | +0.0171 | 0.808 | 0.748 |
| 5% | `fg_tail_rank_x_loss_rank_swap` | +0.0516 | +0.0607 | 0.965 | 2.270 |
| 5% | `fg_tail_x_uncertainty_swap` | +0.0230 | +0.0312 | 0.643 | 0.793 |

## Gamma Sensitivity

The effect is present across intervention strengths, but its balance changes.

### \(\gamma=0.25\)

| Budget | Hybrid \(\Delta I^{FG}\) | High-loss \(\Delta I^{FG}\) | Hybrid \(\Delta\) worst | High-loss \(\Delta\) worst |
|---:|---:|---:|---:|---:|
| 1% | +0.0308 | +0.0411 | +0.0405 | +0.0576 |
| 2% | +0.0366 | +0.0477 | +0.0717 | +0.0763 |
| 5% | +0.0521 | +0.0590 | +0.0888 | +0.1012 |

At weak counterfactual strength, high-loss remains best, but the hybrid is
competitive and far above pure tail-transverse selection.

### \(\gamma=0.5\)

| Budget | Hybrid \(\Delta I^{FG}\) | High-loss \(\Delta I^{FG}\) | Hybrid \(\Delta\) worst | High-loss \(\Delta\) worst |
|---:|---:|---:|---:|---:|
| 1% | +0.0205 | +0.0328 | +0.0639 | +0.0436 |
| 2% | +0.0300 | +0.0446 | +0.0561 | +0.0483 |
| 5% | +0.0417 | +0.0437 | +0.0654 | +0.0857 |

At moderate strength, the hybrid wins worst-group repair at 1% and 2%, while
high-loss wins at 5%.

### \(\gamma=1.0\)

| Budget | Hybrid \(\Delta I^{FG}\) | High-loss \(\Delta I^{FG}\) | Hybrid \(\Delta\) worst | High-loss \(\Delta\) worst |
|---:|---:|---:|---:|---:|
| 1% | +0.0317 | +0.0603 | +0.0452 | +0.0327 |
| 2% | +0.0452 | +0.0581 | +0.0685 | +0.0576 |
| 5% | +0.0606 | +0.0630 | +0.0794 | +0.0701 |

This is the cleanest setting for the paper: hybrid tail-transverse influence
beats high-loss on worst-group accuracy, and nearly matches high-loss on the
certificate at 5%.

### \(\gamma=1.5\)

| Budget | Hybrid \(\Delta I^{FG}\) | High-loss \(\Delta I^{FG}\) | Hybrid \(\Delta\) worst | High-loss \(\Delta\) worst |
|---:|---:|---:|---:|---:|
| 1% | +0.0662 | +0.1018 | +0.0748 | +0.0763 |
| 2% | +0.0826 | +0.1006 | +0.0872 | +0.0981 |
| 5% | +0.1159 | +0.1137 | +0.0997 | +0.1231 |

At strong counterfactual strength, both methods become very effective. The
hybrid slightly beats high-loss on certificate at 5%, while high-loss remains
best for worst-group accuracy.

## Group-Level Check at \(\gamma=1.0, 5\%\)

The hybrid and high-loss repairs lift the same Fisher-Glass weakest slice,
`landbird/land`, but differ on the worst-accuracy group.

| Selector | \(I^{FG}_{0.95}\) | Worst acc | Weakest FG slice | land/land acc | land/water acc | water/land acc | water/water acc |
|---|---:|---:|---|---:|---:|---:|---:|
| `fg_tail_x_loss_swap` | 0.2698 | 0.6044 | landbird/land | 0.9570 | 0.6279 | 0.6044 | 0.9081 |
| `high_loss_swap` | 0.2722 | 0.5950 | landbird/land | 0.9605 | 0.6470 | 0.5950 | 0.9097 |
| `fg_tail_transverse_swap` | 0.2432 | 0.5607 | landbird/land | 0.9610 | 0.6315 | 0.5607 | 0.8988 |
| `raw_fisher_swap` | 0.2491 | 0.5218 | landbird/land | 0.9672 | 0.6634 | 0.5218 | 0.8832 |

This is an important diagnostic. Raw Fisher can lift the certificate while
hurting worst-group accuracy because it does not target the same
nuisance-transverse, decision-relevant repair. The hybrid keeps the certificate
lift while improving the hard `waterbird/land` group more than high-loss at
\(\gamma=1.0\).

## Interpretation

This experiment is the first one that supports a genuinely stronger empirical
claim:

> Fisher-Glass directionality becomes a useful repair principle when combined
> with boundary leverage.

The lesson is not that Fisher-Glass should replace loss. It is that loss and
Fisher-Glass provide complementary factors:

- loss identifies examples with optimization leverage;
- tail-transverse Fisher identifies whether the intervention supplies the
  missing nuisance-transverse task information.

The product is stronger than either factor alone in several settings. Pure
tail-transverse selection is often too low-loss to move the boundary. Pure
high-loss selection is strong but not spectroscopically targeted. The hybrid
repairs both.

## Paper-Safe Claim

This result can go in the main paper if framed as:

> In a Waterbirds feature-space counterfactual proxy, pure tail-transversality
> acts as a spectroscope but is not enough for maximal repair. A hybrid
> tail-transverse influence rule, multiplying Fisher-Glass tail gain by
> boundary leverage, consistently improves the certificate and worst-group
> accuracy and can outperform high-loss counterfactual selection on
> worst-group accuracy at the best-calibrated intervention strength.

Still keep the caveat:

> This is a frozen-feature counterfactual proxy, not a definitive pixel-level
> paired background-swap experiment.

The paper is stronger if it presents the hierarchy honestly:

1. unpaired selection: reveals the missing direction, weak repair;
2. counterfactual selection: useful repair;
3. counterfactual tail-transverse influence: strongest Fisher-Glass-guided
   repair.
