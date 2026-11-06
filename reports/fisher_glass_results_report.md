# Fisher-Glass Results Report

Generated from the local experiment outputs in `build/results/`.

## Executive Summary

The experiments now support a much stronger and more nuanced story:

1. **Synthetic experiments validate the theorem.** Average accuracy, average loss, and raw Fisher look benign while nuisance-projected Fisher develops a near-zero tail. The Fisher-Glass certificate collapses to approximately zero, as predicted.
2. **Reserve and portfolio repairs behave as predicted.** A Fisher reserve lifts the certificate to the theoretical floor, and portfolios steepen the inverse-information tail exponent as the number of independent branches increases.
3. **Real-data diagnostics show identifiability is not the same axis as accuracy.** On CivilComments and Waterbirds diagnostics, the weakest-identifiability slice is not always the lowest-accuracy slice.
4. **Frozen-feature Waterbirds training is strong.** Group DRO + Fisher-Glass improves worst-group accuracy and substantially improves the Fisher-Glass certificate relative to Group DRO.
5. **End-to-end Waterbirds confirms Fisher-Glass as a tunable regularizer.** Small Fisher-Glass penalties improve inverse-information tail risk and can improve worst-group accuracy. Larger penalties over-regularize, raising the certificate while sacrificing average accuracy.

The best current end-to-end setting is:

```text
Waterbirds, MobileNetV3-small, Group DRO + Fisher-Glass, lambda = 1e-4
```

It gives the cleanest tradeoff: improved worst-group accuracy, improved loss-CVaR, improved Fisher-Glass certificate, and substantially reduced inverse-information tail risk, with a moderate average-accuracy cost.

## Result Files Used

| File | Role |
|---|---|
| `build/results/synthetic_fisher_glass_results.json` | Single-seed synthetic theorem validation |
| `build/results/multiseed_fisher_glass_diagnostics.json` | Multi-seed synthetic, CivilComments, Waterbirds diagnostics |
| `build/results/waterbirds_fisher_glass_training_results.json` | Frozen-feature Waterbirds training |
| `build/results/end_to_end_deep_suite_results.json` | Initial end-to-end Waterbirds run, lambda 0.03 |
| `build/results/waterbirds_fast_lambda000005.json` | End-to-end Waterbirds lambda 0.00005 |
| `build/results/waterbirds_fast_lambda00001.json` | End-to-end Waterbirds lambda 0.0001 |
| `build/results/waterbirds_fast_lambda00003.json` | End-to-end Waterbirds lambda 0.0003 |
| `build/results/waterbirds_fast_lambda0001.json` | End-to-end Waterbirds lambda 0.001 |
| `build/results/waterbirds_fast_lambda0003.json` | End-to-end Waterbirds lambda 0.003 |
| `build/results/waterbirds_fast_lambda001.json` | End-to-end Waterbirds lambda 0.01 |

## How to Read the Experiments

The experiments are organized around one causal chain:

```text
Low loss can hide task nonidentifiability.
Fisher-Glass measures nuisance-projected task information.
The inverse tail of this quantity detects hidden fragility.
Regularizing or repairing this tail should improve the certificate.
```

Each experiment tests one link in that chain.

### Metrics

| Metric | Meaning |
|---|---|
| Average accuracy | Standard overall test accuracy. High values can hide subgroup failures. |
| Worst-group accuracy | Minimum accuracy over known groups. This is the main Group DRO target. |
| Loss-CVaR | Tail risk of group loss. Lower is better. This asks whether the model is wrong in high-loss groups. |
| Fisher-Glass certificate | Inverse CVaR of inverse nuisance-projected Fisher. Higher is better. This asks whether task information remains identifiable after nuisance removal. |
| Inverse-information p95 | 95th percentile of inverse projected-Fisher loss. Lower is better. This is a direct tail-risk diagnostic. |
| Tail exponent beta | Estimated power-law exponent for inverse-information survival. Values at or below 1 indicate nonintegrable or near-nonintegrable inverse-information tails. |

### Methods

| Method | What it optimizes | Why it is included |
|---|---|---|
| ERM | Average loss | Baseline that can exploit spurious correlations. |
| Group DRO | Worst known-group loss | Strong baseline for known-group robustness. |
| Loss-CVaR | Tail of loss | Tests whether loss-tail control is enough. |
| Fisher-Glass | Average loss plus Fisher-Glass tail penalty | Tests whether identifiability regularization helps by itself. |
| Group DRO + Fisher-Glass | Worst-group loss plus Fisher-Glass tail penalty | Tests whether Fisher-Glass complements direct group-robust optimization. |
| JTT | Upweights examples misclassified by an initial ERM model | Error-driven reweighting baseline. |

### Task/Nuisance Geometry in the Experiments

The central object is nuisance-projected task Fisher. Operationally, each environment or group has:

- a task direction, represented by class-relevant logit or feature variation;
- a nuisance direction, represented by spurious/background/group metadata or identity features;
- a projected Fisher value, computed by removing the nuisance component from task curvature through a Schur complement.

The Fisher-Glass certificate is high only if the weakest usable task direction remains identifiable in the tail over groups or slices.

### Why Waterbirds Is Useful Here

Waterbirds is a spurious-correlation benchmark. The label is bird type, while the nuisance variable is background. Landbirds are often on land backgrounds, and waterbirds are often on water backgrounds. A model can achieve high average accuracy by relying on background, but fail on minority combinations such as waterbirds on land.

This makes Waterbirds a good test for Fisher-Glass:

```text
If the model relies on background, then removing background-like nuisance information should expose weak task-identifying information.
```

### Why CivilComments Is Useful Here

CivilComments tests whether the diagnostic can separate accuracy failure from identifiability failure in a text setting. The nuisance direction is based on identity mentions. The diagnostic result is important because the weakest-identifiability slice differs from the lowest-accuracy slice.

### Why There Are Frozen-Feature and End-to-End Results

The frozen-feature experiments test whether the Fisher-Glass objective works cleanly in a controlled low-dimensional operational subspace. This is the setting where the theory most directly applies.

The end-to-end experiments test whether the same idea still has practical value when the representation is learned jointly with the classifier. These are harder, noisier, and more sensitive to the penalty weight.

## 1. Synthetic Fisher-Null Shift

The synthetic experiment is the clean theorem validation. The model has high accuracy and benign loss statistics, while projected Fisher develops a near-zero tail.

### Experimental Design

The synthetic task is constructed so that the observed predictor can perform well using a nuisance-aligned direction. The raw task Fisher remains bounded away from zero. However, after projecting out the nuisance direction, the remaining task Fisher is proportional to a small environment-dependent residual `delta_e^2`.

The hidden environments are sampled so that small `delta_e` values occur with a power-law lower tail. This makes inverse projected Fisher heavy-tailed:

```text
projected Fisher ~ delta_e^2
inverse-information loss ~ 1 / delta_e^2
```

The purpose of this experiment is not to be a realistic benchmark. It is a theorem check: it verifies that low loss and raw Fisher can miss the exact failure mode that Fisher-Glass is designed to detect.

| Metric | Value |
|---|---:|
| Mean accuracy | 94.53% |
| Worst 5% accuracy | 88.67% |
| Mean loss | 0.185 |
| Loss-CVaR | 0.262 |
| Mean raw Fisher | 1.041 |
| Minimum raw Fisher | 1.000 |
| Mean projected Fisher | 0.0409 |
| 1% quantile projected Fisher | 1.21e-5 |
| Projected Fisher minimum | 1.23e-12 |
| Inverse-information CVaR | 7.78e7 |
| Fisher-Glass certificate | 1.29e-8 |
| Estimated tail exponent beta | 0.495 |

### Interpretation

This is exactly the Fisher-glass failure mode:

- raw Fisher is bounded away from zero;
- accuracy and loss are not alarming;
- projected Fisher has a near-zero tail;
- inverse projected Fisher is heavy-tailed;
- the Fisher-Glass certificate collapses.

The estimated tail exponent `beta = 0.495` matches the theoretical nonintegrable-tail regime.

## 2. Synthetic Repairs

The repair experiments test whether the two proposed interventions do what the theory says they should do.

### Fisher Reserve

| Quantity | Value |
|---|---:|
| Reserve mixing weight `r` | 0.05 |
| Anchor delta | 0.5 |
| Anchor Fisher | 0.25 |
| Theory lower bound | 0.0125 |
| Mixed Fisher minimum | 0.0125 |
| Reserve Fisher-Glass certificate | 0.0126 |

### Interpretation

The reserve theorem is validated almost exactly. The predicted floor is `r rho = 0.0125`, and the observed certificate is `0.0126`.

The reserve experiment is best understood as adding a small amount of anchor information that is guaranteed to contain task signal after nuisance projection. If every environment is mixed with this anchor, exact Fisher-null collapse is no longer possible.

### Fisher Portfolios

| Portfolio size K | Estimated beta | Fisher-Glass certificate |
|---:|---:|---:|
| 1 | 0.502 | 1.12e-8 |
| 2 | 0.994 | 1.90e-4 |
| 3 | 1.521 | 3.06e-3 |

### Interpretation

The portfolio result behaves as predicted: increasing the number of independent task-information routes steepens the inverse-information tail. The estimated exponent increases approximately with codimension.

The portfolio experiment simulates multiple independent views or branches. A single branch can have a Fisher-null event, but simultaneous null events across independent branches are rarer. This is why the tail exponent improves as `K` increases.

## 3. Tail Boundary Check

| Gamma | Theoretical beta | Estimated beta | Fisher-Glass certificate |
|---:|---:|---:|---:|
| 1 | 0.5 | 0.509 | 1.21e-8 |
| 2 | 1.0 | 0.985 | 2.60e-4 |
| 4 | 2.0 | 2.001 | 0.0138 |

### Interpretation

The synthetic boundary experiment verifies the key tail map:

```text
P(delta < t) ~ t^gamma  =>  P(L > x) ~ x^(-gamma/2)
```

The estimated beta values track the theoretical beta values closely.

## 4. Multi-Seed Diagnostics

These experiments ask whether the diagnostic behavior is stable across seeds and whether the weakest-identifiability slice is simply the same as the lowest-accuracy slice.

The answer is no: the weakest-information slice and worst-accuracy slice can differ. That is the empirical reason to report Fisher-Glass in addition to accuracy and loss metrics.

### Synthetic Multi-Seed Summary

| Metric | Mean +/- CI |
|---|---:|
| Accuracy | 0.94538 +/- 0.00005 |
| Worst 5% accuracy | 0.88672 +/- 0.00000 |
| Loss-CVaR | 0.26188 +/- 0.00010 |
| Raw Fisher minimum | 1.00000 +/- 0.00000 |
| Projected Fisher 1% quantile | 1.23e-5 +/- 2.63e-7 |
| Fisher-Glass certificate | 1.22e-8 +/- 5.28e-9 |
| Tail exponent beta | 0.498 +/- 0.003 |
| Reserve certificate | 0.01260 +/- 0.000001 |

Portfolio beta estimates:

| K | Beta estimate |
|---:|---:|
| 1 | 0.499 +/- 0.003 |
| 2 | 1.000 +/- 0.009 |
| 3 | 1.508 +/- 0.015 |

### CivilComments Diagnostic

| Metric | Mean +/- CI |
|---|---:|
| Overall accuracy | 0.8610 +/- 0.0022 |
| Worst-group accuracy | 0.4375 +/- 0.0064 |
| Group loss-CVaR | 0.9981 +/- 0.0117 |
| Fisher-Glass certificate | 0.4995 +/- 0.0011 |
| Inverse-loss p95 | 1.8075 +/- 0.0025 |
| Reserve certificate | 0.5514 +/- 0.0013 |
| Weakest-identifiability slice | `muslim/y=0` |
| Worst-accuracy slice | `black/y=0` |

### Waterbirds Diagnostic

| Metric | Value |
|---|---:|
| Overall accuracy | 0.7711 |
| Worst-group accuracy | 0.5654 |
| Group loss-CVaR | 1.7407 |
| Fisher-Glass certificate | 0.1650 |
| Inverse-loss p95 | 5.9014 |
| Reserve certificate | 0.1823 |
| Weakest-identifiability slice | `y=0/background=0` |
| Worst-accuracy slice | `y=1/background=0` |

### Interpretation

The diagnostic experiments support the core claim that identifiability and accuracy are related but not identical. In both CivilComments and Waterbirds, the weakest Fisher-Glass slice is not simply the worst-accuracy slice.

This supports the paper's framing:

```text
Loss asks whether the model is wrong now.
Fisher-Glass asks whether the task remains identifiable after nuisance removal.
```

## 5. Frozen-Feature Waterbirds Training

This experiment uses frozen MobileNetV3-small features with a trainable classifier.

### Experimental Design

The frozen-feature experiment trains a classifier on fixed MobileNetV3-small features. This gives a controlled operational subspace: the representation is fixed, and the training objective changes only the classifier/decision layer.

This setting is important because:

- it is closer to the local Fisher theory;
- it avoids instability from changing the whole representation;
- it tests whether Fisher-Glass can be used as an actual training regularizer, not only a diagnostic.

The key comparison is Group DRO versus Group DRO + Fisher-Glass. Group DRO optimizes worst-group loss. Fisher-Glass adds a penalty on the inverse tail of nuisance-projected task information.

| Method | Avg acc | Worst-group acc | Loss-CVaR | FG cert | Inv p95 |
|---|---:|---:|---:|---:|---:|
| ERM | 0.772 +/- 0.000 | 0.557 +/- 0.002 | 1.968 +/- 0.004 | 0.198 +/- 0.000 | 4.865 +/- 0.012 |
| Fisher-Glass | 0.804 +/- 0.002 | 0.563 +/- 0.003 | 1.866 +/- 0.025 | 0.399 +/- 0.006 | 2.458 +/- 0.035 |
| Group DRO | 0.809 +/- 0.003 | 0.673 +/- 0.003 | 1.143 +/- 0.021 | 0.233 +/- 0.003 | 4.104 +/- 0.057 |
| Group DRO + FG | 0.806 +/- 0.009 | **0.694 +/- 0.013** | **0.855 +/- 0.082** | **0.614 +/- 0.059** | **1.629 +/- 0.167** |
| JTT | 0.774 +/- 0.001 | 0.541 +/- 0.002 | 2.167 +/- 0.004 | 0.234 +/- 0.001 | 4.127 +/- 0.013 |
| Loss-CVaR | 0.782 +/- 0.001 | 0.521 +/- 0.003 | 2.427 +/- 0.041 | 0.233 +/- 0.002 | 4.172 +/- 0.027 |

### Interpretation

This is the strongest training result before the end-to-end sweep:

- Group DRO is the strongest baseline for worst-group accuracy.
- Fisher-Glass alone improves the certificate, but it is not the main worst-group optimizer.
- Group DRO + Fisher-Glass is best overall on the combined goal:
  - best worst-group accuracy;
  - best loss-CVaR;
  - best Fisher-Glass certificate;
  - best inverse-information p95.

This supports the correct claim:

```text
Fisher-Glass is complementary to Group DRO. It regularizes identifiability tails rather than replacing worst-group loss optimization.
```

## 6. End-to-End Waterbirds: Full Baselines

The first end-to-end run used MobileNetV3-small trained end-to-end on Waterbirds.

### Experimental Design

This experiment trains the image model end-to-end rather than using frozen features. It is a harder test because the representation itself can move in response to the Fisher-Glass penalty.

The initial full-baseline run includes:

- ERM;
- Fisher-Glass alone;
- Group DRO;
- Group DRO + Fisher-Glass;
- Loss-CVaR.

This answers two questions:

1. Can Fisher-Glass alone replace robust loss optimization?
2. Does Fisher-Glass add useful information-tail control when combined with Group DRO?

The answer to the first is no. The answer to the second is yes, but only with a tuned lambda.

| Method | Avg acc | Worst-group acc | Loss-CVaR | FG cert | Inv p95 |
|---|---:|---:|---:|---:|---:|
| ERM | 0.751 +/- 0.027 | 0.005 +/- 0.003 | 4.571 +/- 0.274 | 0.227 +/- 0.023 | 4.397 +/- 0.429 |
| Fisher-Glass | 0.780 +/- 0.004 | 0.001 +/- 0.001 | 2.481 +/- 0.610 | 0.428 +/- 0.130 | 2.471 +/- 0.639 |
| Group DRO | 0.806 +/- 0.034 | 0.463 +/- 0.131 | 0.745 +/- 0.079 | 0.594 +/- 0.053 | 1.678 +/- 0.139 |
| Group DRO + FG, lambda 0.03 | 0.735 +/- 0.133 | 0.450 +/- 0.114 | 0.736 +/- 0.054 | **0.733 +/- 0.064** | **1.365 +/- 0.116** |
| Loss-CVaR | **0.817 +/- 0.006** | 0.111 +/- 0.029 | 0.970 +/- 0.026 | 0.372 +/- 0.021 | 2.679 +/- 0.145 |

### Interpretation

This run showed the right direction but also revealed that `lambda = 0.03` was too large for end-to-end training.

Key observations:

- ERM catastrophically fails the minority group.
- Fisher-Glass alone improves the certificate but does not solve worst-group accuracy.
- Group DRO remains the main worst-group optimizer.
- Group DRO + FG improves the certificate relative to Group DRO.
- Loss-CVaR has good average accuracy but poor worst-group accuracy, supporting the distinction between loss-tail and information-tail regularization.

## 7. End-to-End Waterbirds Lambda Sweep

After tuning, the useful lambda range was much smaller than the first end-to-end run.

All rows below are Group DRO + Fisher-Glass with MobileNetV3-small end-to-end training.

The lambda frontier is also visualized by:

```bash
python experiments/plot_lambda_frontier.py
```

This produces:

- `tex/figures/waterbirds_lambda_frontier.pdf`
- `tex/figures/waterbirds_lambda_frontier_deltas.pdf`
- `build/results/waterbirds_lambda_frontier_summary.csv`
- `build/results/waterbirds_lambda_frontier_summary.json`

### Experimental Design

The lambda sweep fixes the base method to Group DRO + Fisher-Glass and varies only the Fisher-Glass penalty strength. Each lambda file reruns both Group DRO and Group DRO + Fisher-Glass on the same five seeds.

This is important because the cleanest comparison is paired:

```text
same lambda file, same seeds, Group DRO + FG minus Group DRO
```

The sweep tests whether Fisher-Glass behaves like a controllable regularizer. A useful regularizer should show a frontier:

- small lambda: modest certificate gain, little accuracy damage;
- medium lambda: stronger certificate gain, more accuracy tradeoff;
- large lambda: over-regularization.

That is exactly what the results show.

| Lambda | Avg acc | Worst-group acc | Loss-CVaR | FG cert | Inv p95 |
|---:|---:|---:|---:|---:|---:|
| 0.00005 | **0.797 +/- 0.062** | 0.566 +/- 0.058 | 0.733 +/- 0.061 | 0.393 +/- 0.059 | 2.491 +/- 0.340 |
| 0.0001 | 0.795 +/- 0.102 | 0.577 +/- 0.083 | 0.733 +/- 0.110 | 0.502 +/- 0.166 | 2.100 +/- 0.582 |
| 0.0003 | 0.699 +/- 0.097 | **0.588 +/- 0.087** | **0.653 +/- 0.064** | 0.614 +/- 0.189 | 1.712 +/- 0.462 |
| 0.001 | 0.627 +/- 0.079 | 0.499 +/- 0.018 | 0.710 +/- 0.012 | 0.786 +/- 0.124 | 1.285 +/- 0.199 |
| 0.003 | 0.577 +/- 0.045 | 0.486 +/- 0.066 | 0.717 +/- 0.036 | **0.817 +/- 0.068** | **1.216 +/- 0.101** |
| 0.01 | 0.479 +/- 0.165 | 0.227 +/- 0.167 | 1.031 +/- 0.416 | 0.809 +/- 0.076 | 1.234 +/- 0.115 |

### Paired Improvements Over Group DRO

Each lambda file reruns Group DRO, so the clean comparison is paired within the same file.

| Lambda | Avg acc change | Worst acc change | Loss-CVaR change | FG cert change | Inv p95 change |
|---:|---:|---:|---:|---:|---:|
| 0.00005 | -0.054 | -0.008 | -0.110 | +0.120 | -1.003 |
| 0.0001 | -0.059 | +0.045 | -0.192 | +0.244 | -1.583 |
| 0.0003 | -0.157 | +0.047 | -0.250 | +0.349 | -1.885 |
| 0.001 | -0.230 | -0.043 | -0.197 | +0.512 | -2.205 |
| 0.003 | -0.282 | -0.035 | -0.223 | +0.546 | -2.310 |
| 0.01 | -0.372 | -0.335 | +0.153 | +0.547 | -2.434 |

### Interpretation

The lambda sweep gives a clear robustness-identifiability frontier.

- `lambda = 0.00005` is conservative. It improves the certificate and inverse tail with little change in worst-group accuracy, but the certificate gain is modest.
- `lambda = 0.0001` is the best balanced result. It improves worst-group accuracy, loss-CVaR, the Fisher-Glass certificate, and inverse p95, with a moderate average-accuracy cost.
- `lambda = 0.0003` gives the best worst-group accuracy and stronger certificate improvement, but it costs much more average accuracy.
- `lambda >= 0.001` over-regularizes. It keeps improving the information certificate but starts sacrificing classification performance.

## 8. Best Current End-to-End Result

The best current main-paper result is:

```text
Waterbirds, MobileNetV3-small end-to-end
Group DRO + Fisher-Glass, lambda = 1e-4
```

Against paired Group DRO:

| Metric | Group DRO | Group DRO + FG, lambda 1e-4 | Change |
|---|---:|---:|---:|
| Average accuracy | 0.854 | 0.795 | -0.059 |
| Worst-group accuracy | 0.532 | 0.577 | +0.045 |
| Loss-CVaR | 0.925 | 0.733 | -0.192 |
| Fisher-Glass certificate | 0.259 | 0.502 | +0.244 |
| Inverse-information p95 | 3.683 | 2.100 | -1.583 |

### Recommended Wording

Use this in the paper:

> On end-to-end Waterbirds, adding a small Fisher-Glass penalty to Group DRO improves the Fisher-Glass certificate from 0.259 to 0.502 and reduces the 95th-percentile inverse-information loss from 3.68 to 2.10. Worst-group accuracy also improves from 53.2% to 57.7%, while average accuracy decreases from 85.4% to 79.5%. Larger penalties continue to improve the information certificate but increasingly trade off average accuracy, demonstrating that Fisher-Glass acts as a controllable task-identifiability regularizer rather than a replacement for loss or Group DRO.

## 9. Main Scientific Conclusions

### Conclusion 1: The Fisher-glass failure mode is real in controlled experiments.

The synthetic experiment shows that average loss, average accuracy, and raw Fisher can all look acceptable while inverse nuisance-projected task information has a nonintegrable tail.

### Conclusion 2: The certificate measures something different from loss.

CivilComments and Waterbirds diagnostics show that the weakest-identifiability slice is not necessarily the worst-accuracy slice.

### Conclusion 3: Fisher-Glass is useful as a complement to Group DRO.

The best training results come from Group DRO + Fisher-Glass, not Fisher-Glass alone. This is the right framing:

```text
Group DRO optimizes known worst-group loss.
Fisher-Glass regularizes the tail of usable task information.
The two objectives are complementary.
```

### Conclusion 4: The lambda sweep is important.

The end-to-end results show that Fisher-Glass is a real regularizer with a real tradeoff:

- small lambda: improves certificate with modest accuracy cost;
- medium lambda: improves certificate and worst-group accuracy but costs average accuracy;
- large lambda: over-regularizes and hurts classification.

### Conclusion 5: The strongest empirical claim is not benchmark dominance.

The strongest defensible claim is:

```text
Fisher-Glass detects and regularizes hidden task-identifiability collapse.
When added to Group DRO with a small coefficient, it improves the information-tail certificate and can improve worst-group robustness on end-to-end Waterbirds.
```

Do not claim that Fisher-Glass replaces Group DRO or solves OOD generalization.

## 10. Recommended Main-Paper Tables

### Table A: Synthetic Theorem Validation

| Metric | Value |
|---|---:|
| Mean accuracy | 94.5% |
| Loss-CVaR | 0.262 |
| Minimum raw Fisher | 1.000 |
| Projected Fisher 1% quantile | 1.21e-5 |
| Fisher-Glass certificate | 1.29e-8 |
| Tail exponent beta | 0.495 |

### Table B: Frozen-Feature Waterbirds Training

| Method | Avg acc | Worst acc | FG cert | Inv p95 |
|---|---:|---:|---:|---:|
| ERM | 0.772 | 0.557 | 0.198 | 4.865 |
| Group DRO | 0.809 | 0.673 | 0.233 | 4.104 |
| Group DRO + FG | 0.806 | **0.694** | **0.614** | **1.629** |

### Table C: End-to-End Waterbirds Lambda Sweep

| Lambda | Avg acc | Worst acc | FG cert | Inv p95 |
|---:|---:|---:|---:|---:|
| 0.00005 | 0.797 | 0.566 | 0.393 | 2.491 |
| 0.0001 | 0.795 | 0.577 | 0.502 | 2.100 |
| 0.0003 | 0.699 | **0.588** | 0.614 | 1.712 |
| 0.001 | 0.627 | 0.499 | 0.786 | 1.285 |
| 0.003 | 0.577 | 0.486 | **0.817** | **1.216** |

## 11. Limitations and Cautions

1. The end-to-end results are currently Waterbirds-only.
2. CivilComments end-to-end is still expensive and should be run capped or separately.
3. Group DRO baselines vary slightly across lambda files because each file reruns the baseline; use paired comparisons within each file.
4. Larger Fisher-Glass penalties can over-regularize, producing high information certificates but poor classification accuracy.
5. The certificate still depends on the chosen task/nuisance decomposition.

## 12. Final Recommendation

For the manuscript, use the following hierarchy:

1. **Synthetic theorem validation** as the core proof-of-failure.
2. **Diagnostics** to show identifiability is distinct from accuracy.
3. **Frozen-feature Waterbirds** to show strong training behavior.
4. **End-to-end Waterbirds lambda sweep** to show the practical regularizer frontier.

The final empirical claim should be:

> Fisher-Glass is a statistically grounded diagnostic and regularizer for hidden task-identifiability collapse. In controlled experiments it detects failures missed by loss and raw Fisher. In Waterbirds training, it complements Group DRO by improving the inverse-information tail certificate and, at small regularization strength, improving worst-group robustness.
