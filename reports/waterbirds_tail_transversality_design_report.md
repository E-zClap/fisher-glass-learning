# Waterbirds Tail-Transversality Design Report

## Purpose

This experiment tests the proposed **tail-transversality design** rule:

\[
\mathrm{Gain}_\alpha(a)
=
\frac{1}{\mathrm{cost}(a)}
\mathbb E_{e\in \mathrm{FG\ tail}}
\left[
\frac{p_e^\top H_{e,a}p_e}{(\lambda_e+\varepsilon)^2}
\right].
\]

In the scalar Waterbirds diagnostic, the lifted efficient direction reduces to
an efficient-score residual

\[
\phi_e(a)=z_a-\beta_e b_a,
\qquad
\beta_e=\frac{J_{TN,e}}{J_{NN,e}+\lambda_{\rm ridge}},
\]

so the tail-transversality score is the tail-weighted residual energy
\(\phi_e(a)^2/(\lambda_e+\varepsilon)^2\). With four Waterbirds groups and
\(\alpha=0.95\), the FG tail is the weakest group, so this coincides with the
weakest-tail residual score.

The experiment asks whether selecting a tiny reserve by tail-transversality
lifts \(I^{FG}_{0.95}\) and reduces inverse-Fisher p95 more efficiently than
loss, uncertainty, raw Fisher, group-balanced reserve, or random reserve.

## Command

```bash
python experiments/waterbirds_antidote_acquisition.py \
  --root-dir data/wilds \
  --seeds 17 18 19 20 21 \
  --budgets 0.01 0.02 0.05 \
  --selectors fg_tail_transverse low_loss_tail_transverse raw_fisher \
              high_loss uncertainty group_balanced random \
  --trainer sklearn \
  --output build/results/waterbirds_tail_transversality_design.json \
  --summary-csv build/results/waterbirds_tail_transversality_design_summary.csv
```

Outputs:

- `build/results/waterbirds_tail_transversality_design.json`
- `build/results/waterbirds_tail_transversality_design_summary.csv`

## Baseline

The base model is a balanced logistic classifier on frozen ImageNet
MobileNetV3-small Waterbirds features.

| Metric | Value |
|---|---:|
| Average test accuracy | 0.782 |
| Worst-group test accuracy | 0.525 |
| \(I^{FG}_{0.95}\) | 0.209 |
| Weakest Fisher-Glass slice | landbird/land |

## Policies

| Policy | Criterion |
|---|---|
| `fg_tail_transverse` | Maximize tail-weighted efficient residual gain |
| `low_loss_tail_transverse` | Maximize tail-transverse gain among low-loss examples |
| `raw_fisher` | Maximize raw task score energy \(z^2\) |
| `high_loss` | Maximize current loss |
| `uncertainty` | Maximize predictive uncertainty |
| `group_balanced` | Equal examples from each known Waterbirds group |
| `random` | Uniform random validation examples |

## Main Results

### 1% reserve budget

| Policy | \(\Delta I^{FG}_{0.95}\) | \(\Delta\) worst acc | selected tail gain | raw Fisher | selected loss | low-loss fraction |
|---|---:|---:|---:|---:|---:|---:|
| high_loss | +0.0545 | +0.0685 | 0.165 | 1.888 | 15.131 | 0.00 |
| uncertainty | +0.0096 | +0.0016 | 0.375 | 0.365 | 0.689 | 0.00 |
| random | +0.0059 | +0.0022 | 0.330 | 0.871 | 1.052 | 0.53 |
| group_balanced | +0.0049 | +0.0115 | 0.466 | 1.236 | 1.598 | 0.51 |
| fg_tail_transverse | +0.0015 | -0.0016 | 2.154 | 1.659 | 0.145 | 0.79 |
| raw_fisher | -0.0048 | -0.0109 | 0.589 | 5.025 | 2.878 | 0.90 |
| low_loss_tail_transverse | -0.0027 | +0.0016 | 1.991 | 1.883 | 0.000 | 1.00 |

### 2% reserve budget

| Policy | \(\Delta I^{FG}_{0.95}\) | \(\Delta\) worst acc | selected tail gain | raw Fisher | selected loss | low-loss fraction |
|---|---:|---:|---:|---:|---:|---:|
| high_loss | +0.0643 | +0.1075 | 0.159 | 1.456 | 11.785 | 0.00 |
| group_balanced | +0.0199 | +0.0455 | 0.378 | 1.128 | 1.989 | 0.48 |
| uncertainty | +0.0189 | +0.0234 | 0.383 | 0.371 | 0.778 | 0.00 |
| random | +0.0177 | +0.0146 | 0.377 | 0.850 | 1.418 | 0.51 |
| raw_fisher | +0.0143 | +0.0031 | 0.594 | 3.994 | 3.511 | 0.80 |
| fg_tail_transverse | +0.0053 | +0.0031 | 1.701 | 1.332 | 0.413 | 0.67 |
| low_loss_tail_transverse | -0.0027 | +0.0000 | 1.487 | 1.927 | 0.000 | 1.00 |

### 5% reserve budget

| Policy | \(\Delta I^{FG}_{0.95}\) | \(\Delta\) worst acc | selected tail gain | raw Fisher | selected loss | low-loss fraction |
|---|---:|---:|---:|---:|---:|---:|
| high_loss | +0.0926 | +0.1215 | 0.207 | 0.961 | 7.025 | 0.00 |
| group_balanced | +0.0353 | +0.0941 | 0.385 | 1.123 | 1.856 | 0.47 |
| raw_fisher | +0.0348 | +0.0156 | 0.490 | 2.649 | 3.185 | 0.68 |
| random | +0.0340 | +0.0455 | 0.345 | 0.854 | 1.369 | 0.53 |
| uncertainty | +0.0331 | +0.0452 | 0.327 | 0.391 | 0.989 | 0.00 |
| fg_tail_transverse | +0.0317 | +0.0296 | 1.136 | 1.170 | 0.693 | 0.53 |
| low_loss_tail_transverse | -0.0053 | +0.0000 | 0.866 | 1.582 | 0.000 | 1.00 |

## Interpretation

The experiment gives a clean, useful, but not fully positive result.

First, the tail-transversality score works as a **measurement**. It selects
examples with much larger tail efficient-residual gain than all ordinary
selection policies. At a 1% budget, `fg_tail_transverse` selects mean tail gain
2.154, compared with 0.330 for random, 0.375 for uncertainty, and 0.165 for
high-loss. The low-loss constrained version selects only low-loss examples and
still has very high tail gain.

Second, raw Fisher is a good negative control. At 1%, `raw_fisher` selects very
large raw task score energy, 5.025, but it **reduces** the Fisher-Glass
certificate by -0.0048 and reduces worst-group accuracy by -0.0109. This
supports the paper's central distinction: high raw task information is not the
same as high nuisance-transverse task information.

Third, in this unpaired Waterbirds validation-pool experiment,
tail-transversality does not beat high-loss selection downstream. High-loss
examples directly correct the most obvious decision-boundary errors, so they
produce the largest improvements in both worst-group accuracy and
\(I^{FG}_{0.95}\). Tail-transversality examples are often low-loss soft-mode
examples; they are diagnostically distinctive, but unpaired reserve addition is
too weak an intervention to dominate label-error correction.

## Scientific Takeaway

This experiment is valuable, but it should not be sold as "Fisher-Glass active
learning beats active learning." It does not.

The correct claim is:

> Tail-transversality design computes a distinct missing-information direction:
> it identifies low-loss examples with high nuisance-transverse task residual
> that raw Fisher, uncertainty, and loss do not target. In unpaired Waterbirds
> acquisition, this measurement alone is not enough to outperform high-loss
> selection, showing that a true Fisher antidote requires physically valid
> paired or counterfactual nuisance interventions.

This strengthens the conceptual paper if used carefully. It demonstrates that:

1. Fisher-Glass can compute what information is missing.
2. Raw Fisher can choose the wrong data.
3. Low-loss high-transversality examples exist.
4. Turning spectroscopy into repair requires the intervention pool to actually
   move the nuisance actuator while preserving the task.

## Recommendation for the Main Paper

Do **not** put this as a main positive experiment.

Use it as an appendix stress test or one-sentence main-text motivation for
future interventional Fisher-Glass:

> A Waterbirds acquisition stress test confirms that tail-transversality
> selects low-loss, high-efficient-residual examples and that raw Fisher can
> select harmful examples. However, unpaired acquisition does not outperform
> high-loss selection, indicating that Fisher-antidote repair requires
> physically paired or counterfactual nuisance interventions.

The next decisive experiment should use segmentation/background swaps,
counterfactual Waterbirds, or curated same-object nuisance interventions. That
would test the actual tail-transversality design claim: choose the smallest
physical intervention whose score is maximally transverse to the weak
Fisher-Glass direction.

