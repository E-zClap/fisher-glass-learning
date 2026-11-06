# Table D.6: individual flip prediction

These artifacts recover and reproduce the auxiliary flip-prediction analysis of the prospective **v2 rerun, ERM training seeds 22–26**. All three variants are retained. The analysis uses already-scored donor rows; refitting its logistic predictors does not train a neural network or generate/score images.

## Reproduce

Run these commands from this directory (the directory can be copied elsewhere):

```shell
python aggregate.py
```

This standard-library command reconstructs all 15 paired seed/variant records, verifies every corresponding archived mean and nominal normal-approximation interval, and writes `paired_per_seed.csv`, `paired_summary.json`, and `table_d9.csv`. `predictive_validity.json` is the byte-identical original archive member, including both predictors' metrics before subtraction.

To refit the auxiliary predictors from the recovered original rows in an isolated Python environment:

```shell
python -m pip install -r requirements-refit.txt
python refit.py
```

An equivalent isolated command using uv is:

```shell
uv run --isolated --python 3.12 --with numpy==2.0.2 --with scikit-learn==1.6.1 --with scipy==1.13.1 python refit.py
```

`refit.py` verifies all input/source hashes, calls the recovered original implementation, and compares every run/variant metric with the original archive. Its output is `refit_verification.json`. Numerical dependencies match the versions logged by the historical run; the original Python version was 3.9.25. The verification output records the actual local Python and library versions. The SHA256 values in `provenance.json` identify the original result archive, member, recovered numeric inputs, and inspected source snapshots. No server access is needed to run either command.

The retained verification used Python 3.12.4 with those exact numerical package versions. All 15 seed/variant outputs matched, with maximum absolute difference `2.73e-12` (on an in-sample likelihood-ratio statistic). The table's rounded values are unchanged.

## Fitting protocol

One row represents one source bird and one opposite-type donor draw, restricted to predictions correct on the original image and finite stored `z`, `b`, and `beta`. Each bird contributes its two opposite-type donor rows. The binary outcome is one minus the correctness after intervention. The original pipeline keeps the two donor rows separately, including when they have different outcomes.

The baseline columns are original uncalibrated binary NLL, calibrated true-label probability, absolute original raw logit (the implemented margin), and standardized task logit `z`. The augmented design adds `beta_g * b` and `abs(z - beta_g * b)`, with validation-group coefficient `beta_g = J_TN,g / (J_NN,g + 0.001)` from the uncentered feature–Schur block. `z` uses the original training-logit moments, while stored test nuisance scores `b` use test-split moments, as in the original prospective protocol. These upstream scores are fixed before the auxiliary fits.

Variants are all groups without indicators, all groups with indicators for groups 0, 1, and 2 (group 3 is the reference), and the frozen selected waterbird/water slice without indicators. The same species split is used for the baseline and augmented predictor within each variant and seed. There are five `GroupKFold` folds, without shuffling or outcome stratification: each species, every source bird in it, and both donor rows stay in one fold. The pooled variants include 200 species and the selected-slice variant 46 species in every seed.

Every auxiliary design column, including indicators, is centered and divided by its population standard deviation (`ddof=0`) computed using only that fold's training rows. Any training-fold standard deviation below `1e-9` is replaced by one. The same transformation is applied to held-out rows. The classifier is scikit-learn 1.6.1 `LogisticRegression(max_iter=5000, random_state=0)`: its recorded-version defaults give an intercept, L2 penalty, `C=1`, L-BFGS, tolerance `1e-4`, and no class weights. No hyperparameter search, tuning, inner cross-validation, or fold-dependent parameter selection is performed. See the [versioned LogisticRegression documentation](https://scikit-learn.org/1.6/modules/generated/sklearn.linear_model.LogisticRegression.html) and [GroupKFold documentation](https://scikit-learn.org/1.6/modules/generated/sklearn.model_selection.GroupKFold.html) for the defaults used by the source invocation.

## Gain and aggregation

Let `q_A,si` and `q_B,si` be augmented and baseline out-of-fold probabilities for donor row `i` in training seed `s`, clipped to `[1e-12, 1-1e-12]`, and `y_si` its flip indicator. The gain is

```text
gain_s = (1 / N_s) * sum_i [
    y_si * log(q_A,si / q_B,si)
    + (1 - y_si) * log((1 - q_A,si) / (1 - q_B,si))
]
```

The logarithm is natural; units are **nats per donor row**. A positive value is an improvement. Out-of-fold predictions are concatenated over all rows of a seed and variant before computing mean log likelihood, average precision (`average_precision_score`, labeled AUPRC in the manuscript), and AUROC. Metrics are not unweighted averages of fold metrics. Differences are first computed between the two predictors within each seed, then averaged equally over the five training seeds. Neither predictions nor observations are pooled across training seeds. Counts and flip rates in the manuscript are also equal-weight seed means; displayed row counts are rounded.

The original uncertainty convention is preserved: nominal normal-approximation 95% half-width `1.96 * sample SD / sqrt(5)`, with sample-SD divisor four. Every uncertainty field in the newly generated paired summary names this convention explicitly. `n_auprc_improved` and `n_auroc_improved` count strictly positive within-seed differences.

## Evidence and scope

- `inputs/runs/erm_lam0_seed*/flip_records.npz`: five unmodified original numeric input files, recovered read-only from the retained v2 rerun. There are no images or model checkpoints. `species_vocab` contains CUB bird-species names.
- `source/predictive_validity.py`, `source/common.py`, and `source/evaluate_shift.py`: recovered original code; the third file documents record construction. These snapshots match the corresponding local experiment sources byte for byte.
- `predictive_validity.json`: exact retained output, including the original invocation (`folds=5`, auxiliary `seed=0`), all five training seeds, and all three variants.
- `paired_per_seed.csv`, `paired_summary.json`, `table_d9.csv`: deterministic exports of those outputs.
- `provenance.json`: checksums, recorded software versions, source relationships, and limits. The historical invocation did not record source SHA256 values; the present source hashes identify inspected snapshots, and numerical refitting checks their agreement with the archived results.

This comparison tests the incremental prediction supplied by the added nuisance-related coordinates. It does not isolate a unique contribution of the scalar information statistic. The archived likelihood-ratio statistics are in-sample calculations from regularized fits; Table D.6 uses the held-out gains, and neither the manuscript protocol nor this export treats those statistics as an additional held-out test.
