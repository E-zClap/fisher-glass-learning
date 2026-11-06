# Waterbirds group-free Fisher-Glass lambda selection: schema-v2 matched-control sensitivity



## Protocol

Candidate repairs and lambda selection use task labels plus a frozen CLIP water-versus-land nuisance score, never Waterbirds background/group annotations. The rule retains the one-standard-error validation-NLL risk set, then maximizes the Bonferroni-adjusted cross-fit hidden-certificate lower score within it, with ties toward smaller lambda. Accuracy checks and paired fold comparisons are diagnostics only. Every lambda, including zero, receives the same continuation/checkpoint protocol; the pre-continuation ERM is a separate frozen reference. Test group labels are unsealed only post hoc.

This schema-v2 run is a post-hoc matched-control sensitivity designed after the v1 Waterbirds group/test outcomes were inspected; it is not a fresh held-out confirmation.

Seeds: `[17, 18, 19, 20, 21]`. Intervals are two-sided 95% Student-t intervals across paired training/partition seeds on the same dataset; they are not population sampling intervals.


## Policy results

| Policy | Selected lambda counts | Test avg. acc. | Test worst-group acc. | Hidden certificate |
|---|---:|---:|---:|---:|
| Frozen ERM reference | -- | 75.5 [74.3, 76.6]% | 55.7 [53.3, 58.2]% | 0.007 [0.005, 0.009] |
| Matched ERM continuation (lambda=0) | 0: 5 | 75.5 [74.3, 76.6]% | 55.7 [53.3, 58.2]% | 0.007 [0.005, 0.009] |
| Fixed lambda=0.03 (hidden-slice scale) | 0.03: 5 | 76.6 [76.1, 77.2]% | 51.5 [47.1, 55.8]% | 0.007 [0.004, 0.010] |
| Risk-only selector | 0: 1, 0.3: 4 | 77.5 [76.5, 78.6]% | 43.1 [33.1, 53.1]% | 0.012 [0.006, 0.019] |
| Group-free selector | 0: 1, 0.3: 4 | 77.5 [76.5, 78.6]% | 43.1 [33.1, 53.1]% | 0.012 [0.006, 0.019] |
| Group-supervised validation comparator | 0: 1, 0.1: 1, 0.3: 3 | 77.7 [76.6, 78.8]% | 44.2 [34.2, 54.3]% | 0.011 [0.005, 0.018] |

The group-supervised comparator uses validation group labels only after the group-free decision and blind test artifacts have been frozen. It is a diagnostic, not an available practitioner rule or an upper bound.

## Paired contrasts

- **Primary: group-free selector minus matched lambda=0 continuation:**
  - `test_accuracy`: +2.081 points (95% across-seed t interval +0.209 to +3.954).
  - `test_worst_group_accuracy`: -12.628 points (95% across-seed t interval -21.571 to -3.686).
  - `test_hidden_certificate`: +0.006 (95% across-seed t interval +0.001 to +0.010).
- **Secondary: matched lambda=0 continuation minus frozen ERM:**
  - `test_accuracy`: +0.000 points (95% across-seed t interval +0.000 to +0.000).
  - `test_worst_group_accuracy`: +0.000 points (95% across-seed t interval +0.000 to +0.000).
  - `test_hidden_certificate`: +0.000 (95% across-seed t interval +0.000 to +0.000).
- **Secondary: group-free selector minus frozen ERM:**
  - `test_accuracy`: +2.081 points (95% across-seed t interval +0.209 to +3.954).
  - `test_worst_group_accuracy`: -12.628 points (95% across-seed t interval -21.571 to -3.686).
  - `test_hidden_certificate`: +0.006 (95% across-seed t interval +0.001 to +0.010).

## Interpretation

Table 4 and this deployment question concern two different uses of lambda_FG. In Table 4, standalone Fisher-Glass obtains 56.3% worst-group accuracy and Group DRO obtains 67.3%. That table is a controlled known-group comparison: Group DRO directly optimizes the worst labeled group, while the standalone Fisher-Glass coefficient was selected by certificate among candidates within two points of the best validation worst-group accuracy. That group-supervised rule selected 0.03 in all five seeds, but it does not provide a group-free tuning rule. The certificate moves in the complementary direction (0.399 for standalone Fisher-Glass versus 0.233 for Group DRO), and the combined row reaches 69.4% worst-group accuracy with a 0.614 certificate. Thus Table 4 supports complementarity, not replacement.

To examine the practical question, we added a separate Waterbirds matched-control sensitivity: a single repair stage against a train-only slice discovered from the ERM head, followed by fresh Appendix-B cross-fitted slice rediscovery for validation. We predeclared lambda in {0, 0.01, 0.03, 0.1, 0.3}. The selector first keeps candidates whose ordinary validation NLL is within one per-example standard error of the minimum. Within that risk set, it chooses the largest Bonferroni-adjusted cross-fit hidden-certificate lower score, breaking exact ties toward the smaller coefficient. The two-point accuracy checks and paired fold comparison are reported as diagnostics and do not change the choice. The nuisance input is a frozen CLIP water-versus-land text direction: it uses no Waterbirds background or group annotation, although it does encode that semantic nuisance hypothesis. Group metadata remains sealed throughout fitting, validation, and blind test certification. Every lambda candidate, including lambda=0, now receives the same optimizer continuation and checkpoint rule; the pre-continuation ERM is retained separately. Because this matched control was designed after the v1 Waterbirds group/test outcomes had been inspected, the v2 run is a post-hoc sensitivity analysis rather than an independent held-out confirmation.

The rule selected the matched lambda=0 continuation in 1/5 seeds and a nonzero coefficient in 4/5; selected-lambda counts were lambda=0: 1, lambda=0.3: 4. It made the same choice as ordinary validation-risk minimization in 5/5 seeds. Relative to the matched lambda=0 continuation, its paired test average-accuracy change was +2.081 points (95% across-seed t interval +0.209 to +3.954); the across-seed interval is entirely above zero. Its worst-group change was -12.628 points (95% across-seed t interval -21.571 to -3.686); the across-seed interval is entirely below zero. Its hidden-certificate change was +0.006 (95% across-seed t interval +0.001 to +0.010); the across-seed interval is entirely above zero. The post-hoc group audit therefore shows worst-group harm, not evidence that the certificate rule is a safe proxy for worst-group tuning. The hidden-certificate contrast was positive. The matched lambda=0 continuation itself changed worst-group accuracy relative to the frozen ERM by +0.000 points (95% across-seed t interval +0.000 to +0.000); the across-seed interval includes zero; the total group-free-minus-frozen-ERM change was -12.628 points (95% across-seed t interval -21.571 to -3.686); the across-seed interval is entirely below zero. The worst-group effect of continuation itself was inconclusive relative to the frozen ERM. Across seeds, the group-free rule obtained test average accuracy 77.5 [76.5, 78.6]%, worst-group accuracy 43.1 [33.1, 53.1]%, and hidden certificate 0.012 [0.006, 0.019]. For context, fixed lambda=0.03 on the hidden-slice penalty scale obtained worst-group accuracy 51.5 [47.1, 55.8]%, while the group-supervised validation comparator obtained 44.2 [34.2, 54.3]%. That comparator uses validation group annotations, is not an upper bound, and is not a deployable selector.

The fixed 0.03 row is only a numerical reference on this experiment's dimensionless hidden-slice penalty scale; it is not numerically identical to the known-group Fisher-Glass objective in Table 4. All candidate heads, lambda decisions, and validation gates were hashed before any test array was deserialized or evaluated; preflight had only verified the opaque test-artifact hash and manifest attestation. Group metadata was unsealed only after blind test predictions and certificates were frozen. The practical recommendation is correspondingly conservative. A different prospective policy could predeclare a certificate requirement and choose the smallest lambda meeting it under an ordinary-risk budget, but that target-based policy was not evaluated in this audit, and the certificate should not be treated as a surrogate for unseen worst-group accuracy. Without a certificate target or group-labeled audit data, lambda=0 is the conservative choice within the matched continuation family; the frozen ERM remains the separate reference for the effect of continuation itself. These results neither replace the Table-4 group-supervised selection nor claim that standalone Fisher-Glass closes the Group DRO gap. Across-seed intervals describe training/partition-seed variability on this dataset, not population sampling uncertainty.

## Provenance

`aggregate.json` records every input path/digest, the current scientific source manifest, per-seed values, the complete lambda frontier, and all paired differences. Aggregation fails closed if any selection, blind-test, post-hoc, NPZ, or source digest is stale.
