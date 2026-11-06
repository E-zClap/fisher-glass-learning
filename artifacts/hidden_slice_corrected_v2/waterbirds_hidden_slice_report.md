# Waterbirds metadata-blind hidden-slice results

The primary row uses the fixed CLIP text nuisance, 10% within-class mass, 0.10 shrinkage, 5-fold cross-fitting, 10 gate restarts, and no Waterbirds group/background metadata. Deterministic Torch algorithms were enabled. Intervals below are over paired seeds; within-seed bootstrap intervals remain in the raw certification artifacts.

## 1. Main result

| Method | Group labels used | Selector / nuisance | Avg. acc. | Oracle worst acc. | Selected-slice loss | Projected Fisher | Inverse Fisher | Post-hoc dominant group |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ERM | No | hidden Fisher / CLIP text | 76.3% [75.6, 77.0] | 57.3% [55.9, 58.6] | 0.031 [0.014, 0.048] | 0.005 [0.004, 0.006] | 197.927 [156.339, 239.516] | landbird/land (97.0%) |
| ERM | No | high loss / CLIP text for Fisher evaluation | 76.3% [75.6, 77.0] | 57.3% [55.9, 58.6] | 5.779 [5.282, 6.275] | 0.019 [0.013, 0.025] | 55.484 [38.051, 72.916] | landbird/water (58.4%) |
| ERM | No | random equal-mass slice / CLIP text for evaluation | 76.3% [75.6, 77.0] | 57.3% [55.9, 58.6] | 1.025 [0.967, 1.084] | 0.013 [0.012, 0.014] | 76.714 [69.719, 83.708] | — |
| ERM | No | constant/uniform class gate / CLIP text for evaluation | 76.3% [75.6, 77.0] | 57.3% [55.9, 58.6] | 1.027 [0.979, 1.076] | 0.013 [0.012, 0.015] | 76.358 [69.135, 83.580] | — |
| ERM | No | hidden Fisher / random axis | 76.3% [75.6, 77.0] | 57.3% [55.9, 58.6] | 0.031 [0.016, 0.046] | 0.006 [0.004, 0.007] | 183.664 [144.697, 222.630] | landbird/land (94.4%) |
| Matched ERM continuation | No | re-discovered hidden Fisher / CLIP text | 76.3% [75.6, 77.0] | 57.3% [55.9, 58.6] | 0.031 [0.014, 0.048] | 0.005 [0.004, 0.006] | 197.927 [156.339, 239.516] | landbird/land (97.0%) |
| Loss pseudo-group repair [NO-OP selected in 3/5 seeds] | No | re-discovered hidden Fisher / CLIP text | 76.3% [75.7, 76.9] | 56.9% [55.6, 58.2] | 0.023 [0.006, 0.041] | 0.005 [0.003, 0.006] | 216.525 [147.308, 285.741] | landbird/land (97.8%) |
| Hidden-Fisher repair [NO-OP selected in all 5 seeds] | No | re-discovered hidden Fisher / CLIP text | 76.3% [75.6, 77.0] | 57.3% [55.9, 58.6] | 0.031 [0.014, 0.048] | 0.005 [0.004, 0.006] | 197.927 [156.339, 239.516] | landbird/land (97.0%) |
| Combined repair [NO-OP selected in 4/5 seeds] | No | re-discovered hidden Fisher / CLIP text | 76.4% [75.5, 77.2] | 56.7% [55.1, 58.3] | 0.031 [0.013, 0.048] | 0.005 [0.004, 0.006] | 194.203 [159.669, 228.737] | landbird/land (97.2%) |
| Oracle Group DRO | Yes—model training | re-discovered hidden Fisher / CLIP text | 80.0% [79.7, 80.3] | 64.8% [63.9, 65.7] | 0.033 [0.015, 0.051] | 0.005 [0.003, 0.007] | 206.125 [148.394, 263.856] | landbird/land (95.9%) |
| ERM | Yes—nuisance probe only | hidden Fisher / oracle metadata probe | 76.3% [75.6, 77.0] | 57.3% [55.9, 58.6] | 0.009 [0.005, 0.013] | 0.003 [0.003, 0.004] | 333.434 [279.873, 386.996] | landbird/land (99.1%) |

Lower projected Fisher is favorable for discovery selectors; higher re-discovered Fisher is favorable when evaluating repair. The neutral column heading avoids combining those opposite directions into one arrow.

## 2. Repairs on the original frozen slice

| Repair | Original-slice loss | Original-slice accuracy | Original-slice Fisher | Inv.-Fisher |
| --- | --- | --- | --- | --- |
| Matched ERM continuation | 0.031 [0.014, 0.048] | 98.8% [98.1, 99.5] | 0.005 [0.004, 0.006] | 197.927 [156.339, 239.516] |
| Loss pseudo-group repair [NO-OP selected in 3/5 seeds] | 0.027 [0.011, 0.044] | 98.9% [98.2, 99.7] | 0.005 [0.003, 0.006] | 215.344 [146.387, 284.301] |
| Hidden-Fisher repair [NO-OP selected in all 5 seeds] | 0.031 [0.014, 0.048] | 98.8% [98.1, 99.5] | 0.005 [0.004, 0.006] | 197.927 [156.339, 239.516] |
| Combined repair [NO-OP selected in 4/5 seeds] | 0.030 [0.013, 0.048] | 98.8% [98.1, 99.5] | 0.005 [0.004, 0.006] | 195.048 [159.052, 231.044] |

Every row uses the exact gate discovered against ERM; the main table instead uses a fresh adversary. Repair activity was Loss pseudo-group repair: active repair in 2/5 seeds; Hidden-Fisher repair: active repair in 0/5 seeds; Combined repair: active repair in 1/5 seeds.

## 3. Decisive paired comparisons

| Paired comparison | Metric | Mean Δ | 95% CI | Favorable direction |
| --- | --- | --- | --- | --- |
| Fisher selector versus high-loss selector | predictive_fisher | -0.0138 | [-0.0197, -0.0079] | negative (Fisher finds lower information) |
| Fisher selector versus high-loss selector | unweighted_fisher | -0.0337 | [-0.1063, 0.0389] | negative (robustness check) |
| Fisher selector versus high-loss selector | centered_unweighted_fisher | -0.0707 | [-0.1655, 0.0242] | negative (centered shift-artifact check) |
| Fisher selector versus high-loss selector | slice_loss | -5.7481 | [-6.2430, -5.2531] | negative or near zero (not merely a high-loss slice) |
| Fisher selector versus random equal-mass slice | predictive_fisher | -0.0080 | [-0.0086, -0.0074] | negative (learned adversary finds lower information than random) |
| Fisher selector versus random equal-mass slice | unweighted_fisher | -0.0059 | [-0.0449, 0.0331] | negative (unweighted optimization sanity check) |
| Fisher selector versus constant/uniform class gate | predictive_fisher | -0.0080 | [-0.0085, -0.0074] | negative (learned adversary beats a trivial feasible gate) |
| Fisher selector versus constant/uniform class gate | unweighted_fisher | -0.0066 | [-0.0460, 0.0327] | negative (unweighted optimization sanity check) |
| CLIP-nuisance Fisher selector versus random-nuisance Fisher selector | predictive_fisher | -0.0004 | [-0.0015, 0.0007] | negative (the audited nuisance exposes a lower-information slice than a random axis) |
| Loss pseudo-group repair versus ERM re-discovery | predictive_fisher | -0.0003 | [-0.0008, 0.0002] | positive (repair raises worst hidden Fisher) |
| Loss pseudo-group repair versus ERM re-discovery | inverse_fisher | 18.5973 | [-13.1992, 50.3939] | negative (repair lowers inverse Fisher) |
| Loss pseudo-group repair versus ERM re-discovery | slice_loss | -0.0076 | [-0.0238, 0.0087] | negative for loss repair; interpret jointly with Fisher |
| Loss pseudo-group repair versus ERM re-discovery | overall_accuracy | -0.0003 | [-0.0026, 0.0019] | nonnegative; any decrease must remain within the fixed accuracy budget |
| Loss pseudo-group repair versus ERM re-discovery | oracle_worst_group_accuracy | -0.0036 | [-0.0141, 0.0070] | positive |
| Hidden-Fisher repair versus ERM re-discovery | predictive_fisher | 0.0000 | [0.0000, 0.0000] | positive (repair raises worst hidden Fisher) |
| Hidden-Fisher repair versus ERM re-discovery | inverse_fisher | 0.0000 | [0.0000, 0.0000] | negative (repair lowers inverse Fisher) |
| Hidden-Fisher repair versus ERM re-discovery | slice_loss | 0.0000 | [0.0000, 0.0000] | negative for loss repair; interpret jointly with Fisher |
| Hidden-Fisher repair versus ERM re-discovery | overall_accuracy | 0.0000 | [0.0000, 0.0000] | nonnegative; any decrease must remain within the fixed accuracy budget |
| Hidden-Fisher repair versus ERM re-discovery | oracle_worst_group_accuracy | 0.0000 | [0.0000, 0.0000] | positive |
| Combined repair versus ERM re-discovery | predictive_fisher | 0.0001 | [-0.0001, 0.0003] | positive (repair raises worst hidden Fisher) |
| Combined repair versus ERM re-discovery | inverse_fisher | -3.7244 | [-14.0650, 6.6162] | negative (repair lowers inverse Fisher) |
| Combined repair versus ERM re-discovery | slice_loss | -0.0001 | [-0.0003, 0.0001] | negative for loss repair; interpret jointly with Fisher |
| Combined repair versus ERM re-discovery | overall_accuracy | 0.0008 | [-0.0014, 0.0030] | nonnegative; any decrease must remain within the fixed accuracy budget |
| Combined repair versus ERM re-discovery | oracle_worst_group_accuracy | -0.0059 | [-0.0224, 0.0105] | positive |
| Oracle Group DRO versus ERM re-discovery | predictive_fisher | -0.0001 | [-0.0010, 0.0008] | positive (repair raises worst hidden Fisher) |
| Oracle Group DRO versus ERM re-discovery | inverse_fisher | 8.1973 | [-34.3536, 50.7482] | negative (repair lowers inverse Fisher) |
| Oracle Group DRO versus ERM re-discovery | slice_loss | 0.0024 | [-0.0208, 0.0255] | negative for loss repair; interpret jointly with Fisher |
| Oracle Group DRO versus ERM re-discovery | overall_accuracy | 0.0368 | [0.0308, 0.0429] | nonnegative; any decrease must remain within the fixed accuracy budget |
| Oracle Group DRO versus ERM re-discovery | oracle_worst_group_accuracy | 0.0752 | [0.0662, 0.0842] | positive |
| Matched ERM continuation versus ERM re-discovery | predictive_fisher | 0.0000 | [0.0000, 0.0000] | near zero (matched optimization diagnostic) |
| Matched ERM continuation versus ERM re-discovery | inverse_fisher | 0.0000 | [0.0000, 0.0000] | near zero (matched optimization diagnostic) |
| Matched ERM continuation versus ERM re-discovery | slice_loss | 0.0000 | [0.0000, 0.0000] | near zero (matched optimization diagnostic) |
| Matched ERM continuation versus ERM re-discovery | overall_accuracy | 0.0000 | [0.0000, 0.0000] | near zero (matched optimization diagnostic) |
| Loss pseudo-group repair versus matched ERM continuation | predictive_fisher | -0.0003 | [-0.0008, 0.0002] | positive (repair exceeds matched continued optimization) |
| Loss pseudo-group repair versus matched ERM continuation | inverse_fisher | 18.5973 | [-13.1992, 50.3939] | negative (repair exceeds matched continued optimization) |
| Loss pseudo-group repair versus matched ERM continuation | overall_accuracy | -0.0003 | [-0.0026, 0.0019] | nonnegative; any decrease must remain within the fixed accuracy budget |
| Loss pseudo-group repair versus matched ERM continuation on original frozen slice | predictive_fisher | -0.0003 | [-0.0008, 0.0002] | positive (fixed-slice gain beyond continued optimization) |
| Loss pseudo-group repair versus matched ERM continuation on original frozen slice | slice_loss | -0.0033 | [-0.0093, 0.0027] | negative (fixed-slice loss reduction beyond continued optimization) |
| Hidden-Fisher repair versus matched ERM continuation | predictive_fisher | 0.0000 | [0.0000, 0.0000] | positive (repair exceeds matched continued optimization) |
| Hidden-Fisher repair versus matched ERM continuation | inverse_fisher | 0.0000 | [0.0000, 0.0000] | negative (repair exceeds matched continued optimization) |
| Hidden-Fisher repair versus matched ERM continuation | overall_accuracy | 0.0000 | [0.0000, 0.0000] | nonnegative; any decrease must remain within the fixed accuracy budget |
| Hidden-Fisher repair versus matched ERM continuation on original frozen slice | predictive_fisher | 0.0000 | [0.0000, 0.0000] | positive (fixed-slice gain beyond continued optimization) |
| Hidden-Fisher repair versus matched ERM continuation on original frozen slice | slice_loss | 0.0000 | [0.0000, 0.0000] | negative (fixed-slice loss reduction beyond continued optimization) |
| Combined repair versus matched ERM continuation | predictive_fisher | 0.0001 | [-0.0001, 0.0003] | positive (repair exceeds matched continued optimization) |
| Combined repair versus matched ERM continuation | inverse_fisher | -3.7244 | [-14.0650, 6.6162] | negative (repair exceeds matched continued optimization) |
| Combined repair versus matched ERM continuation | overall_accuracy | 0.0008 | [-0.0014, 0.0030] | nonnegative; any decrease must remain within the fixed accuracy budget |
| Combined repair versus matched ERM continuation on original frozen slice | predictive_fisher | 0.0001 | [-0.0001, 0.0002] | positive (fixed-slice gain beyond continued optimization) |
| Combined repair versus matched ERM continuation on original frozen slice | slice_loss | -0.0002 | [-0.0007, 0.0003] | negative (fixed-slice loss reduction beyond continued optimization) |
| Matched ERM continuation on original frozen slice versus ERM | predictive_fisher | 0.0000 | [0.0000, 0.0000] | near zero (matched optimization diagnostic) |
| Matched ERM continuation on original frozen slice versus ERM | inverse_fisher | 0.0000 | [0.0000, 0.0000] | near zero (matched optimization diagnostic) |
| Matched ERM continuation on original frozen slice versus ERM | slice_loss | 0.0000 | [0.0000, 0.0000] | near zero (matched optimization diagnostic) |
| Matched ERM continuation on original frozen slice versus ERM | slice_accuracy | 0.0000 | [0.0000, 0.0000] | near zero (matched optimization diagnostic) |
| Loss pseudo-group repair on original frozen slice versus ERM | predictive_fisher | -0.0003 | [-0.0008, 0.0002] | positive (the fixed slice gains information) |
| Loss pseudo-group repair on original frozen slice versus ERM | inverse_fisher | 17.4166 | [-12.5646, 47.3978] | negative (the fixed-slice certificate improves) |
| Loss pseudo-group repair on original frozen slice versus ERM | slice_loss | -0.0033 | [-0.0093, 0.0027] | negative (the fixed slice incurs less loss) |
| Loss pseudo-group repair on original frozen slice versus ERM | slice_accuracy | 0.0014 | [-0.0010, 0.0038] | positive (the fixed slice gains accuracy) |
| Hidden-Fisher repair on original frozen slice versus ERM | predictive_fisher | 0.0000 | [0.0000, 0.0000] | positive (the fixed slice gains information) |
| Hidden-Fisher repair on original frozen slice versus ERM | inverse_fisher | 0.0000 | [0.0000, 0.0000] | negative (the fixed-slice certificate improves) |
| Hidden-Fisher repair on original frozen slice versus ERM | slice_loss | 0.0000 | [0.0000, 0.0000] | negative (the fixed slice incurs less loss) |
| Hidden-Fisher repair on original frozen slice versus ERM | slice_accuracy | 0.0000 | [0.0000, 0.0000] | positive (the fixed slice gains accuracy) |
| Combined repair on original frozen slice versus ERM | predictive_fisher | 0.0001 | [-0.0001, 0.0002] | positive (the fixed slice gains information) |
| Combined repair on original frozen slice versus ERM | inverse_fisher | -2.8791 | [-10.8728, 5.1146] | negative (the fixed-slice certificate improves) |
| Combined repair on original frozen slice versus ERM | slice_loss | -0.0002 | [-0.0007, 0.0003] | negative (the fixed slice incurs less loss) |
| Combined repair on original frozen slice versus ERM | slice_accuracy | 0.0000 | [-0.0000, 0.0000] | positive (the fixed slice gains accuracy) |

The paired interval is wholly below zero, supporting a lower-information witness than matched high-loss discovery. The paired interval is wholly below zero, supporting a lower-information witness than random equal-mass selection. The paired interval is wholly below zero, supporting an improvement over the constant/uniform feasible gate. The paired interval includes zero, so these runs do not establish a lower-information witness under the audited nuisance than under a random nuisance axis. The paired interval includes zero, so these runs do not establish the centered, unweighted robustness check.

## 4. Predeclared mass and shrinkage sensitivity

| Sweep | Value | Selector | Predictive Fisher | Unweighted Fisher | Centered Fisher | Loss | Accuracy | Top overlap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| slice_mass | 0.05 | fisher | 0.004 [0.003, 0.005] | 0.320 [0.260, 0.380] | 0.235 [0.208, 0.263] | 0.106 [0.043, 0.169] | 97.4% [95.3, 99.5] | 0.179 [0.111, 0.246] |
| slice_mass | 0.05 | high_loss | 0.012 [0.008, 0.016] | 0.346 [0.239, 0.452] | 0.232 [0.182, 0.282] | 5.663 [5.245, 6.081] | 19.9% [13.7, 26.1] | 0.416 [0.354, 0.477] |
| slice_mass | 0.10 | fisher | 0.005 [0.003, 0.007] | 0.367 [0.316, 0.418] | 0.252 [0.221, 0.282] | 0.083 [0.043, 0.123] | 97.7% [96.8, 98.7] | 0.258 [0.225, 0.291] |
| slice_mass | 0.10 | high_loss | 0.017 [0.011, 0.022] | 0.396 [0.258, 0.535] | 0.295 [0.172, 0.417] | 4.850 [3.937, 5.763] | 23.4% [16.9, 30.0] | 0.495 [0.429, 0.562] |
| slice_mass | 0.20 | fisher | 0.005 [0.004, 0.006] | 0.346 [0.309, 0.383] | 0.238 [0.222, 0.253] | 0.053 [0.036, 0.069] | 98.3% [97.9, 98.8] | 0.395 [0.350, 0.440] |
| slice_mass | 0.20 | high_loss | 0.027 [0.023, 0.031] | 0.467 [0.344, 0.590] | 0.328 [0.260, 0.396] | 3.784 [3.638, 3.931] | 30.5% [26.6, 34.4] | 0.588 [0.494, 0.681] |
| shrinkage | 0.05 | fisher | 0.004 [0.003, 0.005] | 0.363 [0.308, 0.418] | 0.230 [0.195, 0.265] | 0.065 [0.013, 0.117] | 97.7% [95.7, 99.8] | 0.260 [0.230, 0.289] |
| shrinkage | 0.10 | fisher | 0.005 [0.003, 0.007] | 0.367 [0.316, 0.418] | 0.252 [0.221, 0.282] | 0.083 [0.043, 0.123] | 97.7% [96.8, 98.7] | 0.258 [0.225, 0.291] |
| shrinkage | 0.20 | fisher | 0.006 [0.004, 0.007] | 0.380 [0.322, 0.438] | 0.268 [0.237, 0.298] | 0.067 [0.013, 0.120] | 98.0% [96.9, 99.1] | 0.275 [0.251, 0.298] |

## 5. Stability, admissibility, and interpretation

The primary gate's selection-weighted per-class cross-fold score correlation is 0.419 [0.353, 0.485], mean top-slice overlap is 0.258 [0.225, 0.291], and selected-class agreement is 1.000 [1.000, 1.000]. Classes selected in fewer than two folds have no gate pair and are excluded from these aggregate stability summaries. Its mean effective sample size is 413.757 [382.653, 444.860]; its held-out sigmoid mass is 0.083 [0.077, 0.089], with absolute target error 0.017 [0.011, 0.023]. Fisher-versus-high-loss test top-slice Jaccard is 0.001 [0.000, 0.002]; its Student-t interval is clipped to the valid [0, 1] range for display. Post hoc, its dominant metadata group is landbird/land with mean soft purity 97.0%. The paired hidden-minus-lowest-predefined Fisher difference is -0.003 [-0.003, -0.002] and is below zero in 5/5 seeds. Across seeds, the oracle lowest-predefined-group Fisher is 0.008 [0.007, 0.009], and the oracle highest-predefined-group loss is 2.075 [1.996, 2.155]. The paired interval is wholly below zero, supporting a slice lower-Fisher than the per-seed weakest predefined group. Fisher repair changes oracle worst-group accuracy by 0.0% [0.0, 0.0]; combined repair changes it by -0.6% [-2.2, 1.1]. The paired interval includes zero, so these runs do not establish an oracle worst-group improvement from Fisher repair. The paired interval includes zero, so these runs do not establish an oracle worst-group improvement from combined repair. All three repair families satisfy the fixed overall, balanced, and per-class task-label safety envelope in every seed. Repair activity was Loss pseudo-group repair: active repair in 2/5 seeds; Hidden-Fisher repair: active repair in 0/5 seeds; Combined repair: active repair in 1/5 seeds.

## 6. Scope of controls

- **Primary:** CLIP text nuisance; no group labels in discovery, tuning, certification, or repair.
- **Random and constant controls:** equal-mass random slices and a constant/uniform feasible gate test whether learned discovery improves on non-optimized selectors.
- **Random nuisance:** a frozen random representation axis; negative control.
- **Oracle nuisance probe:** background metadata is used only after blind freezing; sensitivity analysis, not primary.
- **Oracle Group DRO:** group labels are used for training; labeled-group reference, not the hidden method.
- **Predefined groups:** metadata-only post-hoc reference rows.
- **Repair cross-fitting:** validation-fold repair diagnostics are descriptive because task-label safety and the frozen-slice objective select λ (with an exact no-op eligible); the untouched-test fresh-adversary result is confirmatory.

## 7. Reviewer-ready response

> **W3. Thank you; we agree that the original real-data experiments evaluated Fisher-Glass only on predefined metadata groups and therefore did not execute the hidden-slice adversary. We added a group-blind, cross-fitted Waterbirds experiment over five predeclared seeds.** Background and four-way group metadata are excluded from discovery, certification, hyperparameter selection, and the three pseudo-group repairs. The primary adversary observes images, task labels, frozen representations, and a fixed CLIP water-versus-land background direction; it fits a class-conditional linear gate on discovery folds and freezes that gate before held-out certification.
> 
> On the untouched test split, the Fisher selector has projected Fisher 0.005 [0.004, 0.006], versus 0.019 [0.013, 0.025] for the equal-mass high-loss selector. The paired Fisher-minus-loss difference is -0.014 [-0.020, -0.008]. The paired interval is wholly below zero, supporting a lower-information witness than matched high-loss discovery. Its slice accuracy is 98.8% [98.1, 99.5] and loss is 0.031 [0.014, 0.048]; the unweighted feature-Schur result is 0.379 [0.346, 0.412], and the centered unweighted result is 0.250 [0.237, 0.263]. The centered Fisher-minus-loss difference is -0.071 [-0.166, 0.024]. The paired interval includes zero, so these runs do not establish the centered, unweighted robustness check. Mean confidence, entropy, and absolute calibrated margin inside the slice are 0.963 [0.952, 0.973], 0.114 [0.086, 0.141], and 4.811 [4.324, 5.298], respectively. The held-out effective sample size is 413.757 [382.653, 444.860], and the transferable sigmoid gate's absolute mass error is 0.017 [0.011, 0.023]. An equal-mass random slice has Fisher 0.013 [0.012, 0.014], with paired learned-minus-random difference -0.008 [-0.009, -0.007]. The paired interval is wholly below zero, supporting a lower-information witness than random equal-mass selection. A constant/uniform feasible gate has Fisher 0.013 [0.012, 0.015], with paired difference -0.008 [-0.009, -0.007]. The paired interval is wholly below zero, supporting an improvement over the constant/uniform feasible gate. The random nuisance-axis control has Fisher 0.006 [0.004, 0.007], with audited-minus-random paired difference -0.000 [-0.001, 0.001]. The paired interval includes zero, so these runs do not establish a lower-information witness under the audited nuisance than under a random nuisance axis. At least one decisive control is unresolved; these runs execute the discovery protocol but do not yet establish the full unknown-subpopulation claim.
> 
> Only after these blind results were frozen did we reveal metadata. The primary slice is most enriched for landbird/land (mean soft purity 97.0%). Against the lowest-Fisher predefined group in each seed, the paired hidden-minus-predefined difference is -0.003 [-0.003, -0.002]. The hidden value is lower in 5/5 seeds. On average, the hidden slice is lower-Fisher than the per-seed lowest-Fisher predefined group. The Fisher/high-loss top-slice Jaccard overlap is 0.001 [0.000, 0.002], showing directly whether the two adversaries recover the same examples. The metadata-trained nuisance probe is reported separately as an oracle sensitivity analysis (Fisher 0.003 [0.003, 0.004]); it is not the primary result and does not select any primary hyperparameter.
> 
> We then froze the discovered slice, considered loss-only, Fisher-only, and combined repairs under fixed overall, balanced, and per-class task-label accuracy budgets, and evaluated both the original gate and a seed-matched adversary initialized from scratch for every active repair; an exact no-op was eligible and reuses the identical frozen ERM adversary. Repair activity was Loss pseudo-group repair: active repair in 2/5 seeds; Hidden-Fisher repair: active repair in 0/5 seeds; Combined repair: active repair in 1/5 seeds. A matched ERM continuation with the same warm start and extra optimizer steps changes the fresh certificate by 0.000 [0.000, 0.000]. Relative to that matched control, Fisher repair changes the fresh certificate by 0.000 [0.000, 0.000], while loss repair changes original-slice loss by -0.003 [-0.009, 0.003]. The paired interval includes zero, so these runs do not establish an original-slice loss improvement beyond matched continued optimization. The arm selected no-op in 3/5 seeds, so this is mixed active/no-op evidence. On the original slice, loss-only repair changes loss by -0.003 [-0.009, 0.003] and Fisher by -0.000 [-0.001, 0.000]; Fisher repair changes Fisher by 0.000 [0.000, 0.000]. Under fresh discovery, Fisher repair changes the re-discovered certificate by 0.000 [0.000, 0.000] and average accuracy by 0.0% [0.0, 0.0]; combined repair changes the certificate by 0.000 [-0.000, 0.000]. The oracle worst-group accuracy changes by 0.0% [0.0, 0.0] for Fisher repair and -0.6% [-2.2, 1.1] for combined repair. The fixed overall, balanced, and per-class task-label safety envelope was satisfied for all three repair families in every seed. The Hidden-Fisher repair arm selected the exact no-op in every seed, so it does not establish a re-discovered hidden-slice improvement beyond matched continued optimization. At least one Fisher-based arm selected the exact no-op in every seed, so these runs do not demonstrate a repair for that arm. Oracle Group DRO is shown only as a labeled-group comparator. The added experiment therefore executes the missing mechanism end to end, while the control outcomes above determine whether the present run supports—or fails to support—the stronger novelty and repair claims.
