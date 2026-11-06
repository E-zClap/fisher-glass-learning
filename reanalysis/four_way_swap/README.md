# Four-way validation-summary / held-out-swap reanalysis

This post hoc analysis joins retained validation geometry to held-out background
intervention records. It performs no training, image generation, model inference,
or coefficient selection based on intervention outcomes. Python 3.10 or newer
and its standard library are sufficient.

## Replay

When this directory is published as `reanalysis/four_way_swap/` in the public
repository, run from the repository root:

```sh
python reanalysis/four_way_swap/reanalyse.py
```

The default artifact directory is resolved relative to the script, as
`Path(__file__).resolve().parents[2] / "artifacts"`. An explicit root works from
any checkout or staging layout:

```sh
python tex/camera_ready_v2/reanalysis/four_way_swap/reanalyse.py --artifact-root ../fisher-glass-learning/artifacts
```

Outputs default to the script's own directory. Use `--output-dir PATH` to write
elsewhere. Re-running with identical inputs and script bytes produces identical
outputs; there are no timestamps, absolute local paths, random seeds, or optional
scientific-library dependencies in the output.

## Inputs and provenance

The only inputs are the following three JSON files in each of
`artifacts/prospective_v2/` (primary seeds 22--26) and
`artifacts/prospective_v1/` (development seeds 17--21):

- `frozen_selection.json`: all 20 candidate records, validation groups, per-seed
  FG choices, accuracy-budget fallback flags, checkpoint identifiers, and frozen
  protocol hash.
- `shift_summary.json`: the ten intervention records, joined by model tag.
- `paired_generation.json`: the two recompositions per source image and arm,
  plus source/generated-image totals and group counts.

The retained inputs used for the checked-in outputs match the Git blobs at
[source commit 7f129365db05c69917383eeca9e1af2eeae4873f](https://github.com/E-zClap/fisher-glass-learning/tree/7f129365db05c69917383eeca9e1af2eeae4873f).
`results.json` records that reference commit, the SHA-256 and byte count of each
actual input file, and the analysis script's SHA-256. The reference commit is a
baseline citation; a replay's recorded per-file hashes identify the inputs it
actually consumed. `models.csv` retains each selected model's stored checkpoint
SHA-256; checkpoint binaries are not needed or re-hashed by this analysis.

The source definitions were checked against
[`select_slice.py`](https://github.com/E-zClap/fisher-glass-learning/blob/7f129365db05c69917383eeca9e1af2eeae4873f/experiments/prospective_nuisance_shift/select_slice.py),
[`common.py`](https://github.com/E-zClap/fisher-glass-learning/blob/7f129365db05c69917383eeca9e1af2eeae4873f/experiments/prospective_nuisance_shift/common.py),
[`evaluate_shift.py`](https://github.com/E-zClap/fisher-glass-learning/blob/7f129365db05c69917383eeca9e1af2eeae4873f/experiments/prospective_nuisance_shift/evaluate_shift.py),
and [`build_paired_images.py`](https://github.com/E-zClap/fisher-glass-learning/blob/7f129365db05c69917383eeca9e1af2eeae4873f/experiments/prospective_nuisance_shift/build_paired_images.py).

## Definitions and analysis units

For each model, use all four equally weighted predefined validation groups.
Let `F_g = projected_fisher`, `J_g = raw_fisher`, and `L_g = 1/(F_g + epsilon)`.
The retained geometry consists of uncentered, unweighted feature moments and
their scalar nuisance-ridge Schur complement. The nuisance ridge is `0.001`,
reciprocal stabilization is `epsilon = 1e-8`, and CVaR level is `alpha = 0.95`.
These are distinct from calibrated predictive-Fisher constructions.

| Symbol | Summary | Computation |
| --- | --- | --- |
| A | Mean raw task second moment | `mean_g J_g` |
| P | Mean projected information | `mean_g F_g` |
| H | Reciprocal mean inverse information | `1 / mean_g L_g` |
| C | Inverse-CVaR certificate | `1 / upper_CVaR_0.95(L_g)` |

`H` is a stabilized harmonic mean in information units, not mean inverse
information. The latter is exported separately as `mean_inverse` in
`models.csv`. The script computes CVaR using the discrete upper-tail integral;
with four mass-0.25 atoms and a 0.05 tail, it equals the largest inverse term.
Thus `C = min_g F_g + epsilon`. This also agrees with the retained implementation's
mean above the empirical 0.95 quantile in this particular four-group setting.
It should not be assumed that those two CVaR implementations agree generally.

Each cohort contains five ERM models and the five **stored per-seed selected FG
models**. The original selection maximizes the certificate among FG candidates
within 0.01 of ERM validation accuracy; when none qualifies, it maximizes over
all three FG candidates. The analysis keeps those recorded fallbacks: primary
seeds 22 and 24, and development seed 20. It verifies the original choice but
does not make a new choice. The original headline group rule used surviving
fraction in the primary cohort and absolute projected information in development;
the four-summary comparison consistently uses all four groups.

Panel A reports means and nominal 95% normal-approximation half-widths,
`1.96 * sample_sd / sqrt(5)`. FG-minus-ERM changes are calculated within seed
before aggregation. These are across-seed summaries, not the stored per-source
bootstrap intervals. The paired intervals for P, H, and C contain zero in both
cohorts.

Panel B uses only the five ERM models per cohort. It selects the minimum raw
group, the minimum projected group, or the largest inverse contribution among
eligible groups. Eligibility means group validation accuracy at least overall
validation accuracy and group uncalibrated NLL at most the four-group median.
The minimum projected, largest inverse, and CVaR-active group rules over all
four groups are checked to agree and combined in one row. H does not itself
define a group selector. Drops remain signed, in percentage points:
`100 * (accuracy_same - accuracy_opposite)`; a negative drop means improved
accuracy under opposite-type recomposition. The largest inverse share is
`max_g L_g / sum_g L_g`, averaged over the five ERM models, not pooled across
groups or models. Its means are 38.298% (primary) and 40.620% (development).

Panel C gives descriptive Spearman correlations across ten models arranged in
five ERM/FG seed pairs per cohort. No independence-based significance test is
performed. Outcomes are `D_max = 100 * max_g(accuracy_same - accuracy_opposite)`
and `W_opp = min_g accuracy_opposite`, both over all four groups. All ranks use
average ranks for ties. These comparisons do not establish superior prediction
of swap damage by inverse-CVaR.

## Exact outcome ties

The generation protocol produces two draws for each source image in each arm.
The evaluator averages correctness over draws and then over source images.
Consequently each reported group accuracy corresponds to an integer count over
`2 * source_n` trials. The script recovers that integer by rounding only after
checking its distance from the stored count scale is at most `1e-8` (plus a
`1e-12` relative tolerance). It constructs `Fraction(count, 2 * source_n)` and
takes differences, extrema, and outcome ranks using exact rational arithmetic.
This is not decimal rounding of outcome ranks.

For primary ERM seeds 22 and 25, the largest drops are respectively
`(1185 - 489)/1284` and `(1195 - 499)/1284`, both `58/107` before conversion to
percentage points. The stored floating-point differences differ at the final
bit. Respecting their exact tie gives primary D-max correlations A/P/H/C of
`-0.097264887`, `-0.085106776`, `-0.145897330`, and `+0.145897330`.
Development ERM seeds 20 and 21 also tie on `W_opp = 203/428`; the average-rank
calculation preserves this tie. Full tie metadata is in `results.json`.

## Outputs and checks

- `results.json`: definitions, all three table panels, per-seed group choices,
  cohort and fallback metadata, inverse shares, exact outcome ties, provenance.
- `panel_a.csv`, `panel_b.csv`, `panel_c.csv`: machine-readable table panels
  without display rounding.
- `models.csv`: the 20 analysed models, four summaries, mean inverse information,
  exact outcome fractions, coefficients, fallback flags, checkpoint identifiers.
- `groups.csv`: all 80 joined validation/intervention group records, inverse
  contributions and shares, eligibility, source counts, recovered integer arm
  counts, and exact signed-drop fractions.

Before writing outputs, the script checks all 40 candidate models and 160
candidate validation groups, reconstructs both frozen hashes, replays original
group eligibility and FG selection, checks the ten-model selection/intervention
join in each cohort, and verifies certificates and inverse terms. It checks
validation group counts and recoverable validation correctness totals, all
intervention source and generation counts, integer arm-count recovery, signed
drops, duplicate bootstrap point estimates, and the three equivalent all-group
selectors. Each analysed model has 5,794 source images; the source counts by
group are 2,255/2,255/642/642, and each model has 23,176 generated-image trials
across both arms. A group-selection tie or missing/changed required record
causes failure instead of a silent tie break or exclusion.

These checks establish internal consistency of the retained summaries; they do
not rerun checkpoint predictions or inspect the original per-image manifest.
The generation metadata and evaluator code justify the count denominator, and
integer recovery is checked for every consumed accuracy. Nonfinite values in
unused retained confidence/bin fields are tolerated on input; every consumed
numeric field is checked, and outputs are strict finite JSON.
