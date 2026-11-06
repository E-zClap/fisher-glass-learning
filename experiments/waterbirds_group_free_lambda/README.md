# Waterbirds group-free Fisher-Glass lambda selection: matched-control sensitivity

Table 4 uses the known-group Fisher-Glass objective and a group-supervised
validation rule for its coefficient; this experiment instead answers the
deployment question with a single repair stage against an Appendix-B-discovered
hidden slice, followed by fresh cross-fitted rediscovery: if
group labels are unavailable, how is its Fisher coefficient selected? It does
not retrospectively replace Table 4's group-supervised choice, claim that
standalone Fisher-Glass replaces Group DRO, or erase the reported worst-group gap.

The schema-v2 candidate grid is `0, 0.01, 0.03, 0.1, 0.3`. Every grid point,
including `lambda=0`, receives the same optimizer continuation, randomness, and
checkpoint rule. Thus `lambda=0` is the matched zero-penalty continuation, not
an exact no-op. The pre-continuation ERM head is frozen and reported separately;
it never enters lambda selection. `lambda=0.03` remains a fixed
hidden-slice-scale numerical reference. This experiment's
dimensionless hidden-slice penalty is not numerically identical to the
known-group Fisher-Glass objective used for Table 4. All blind stages consume
only `row_id`, frozen features `h`, task label `y`, and `b_vl`: a frozen CLIP
water-versus-land text direction. This nuisance score uses no example-level
background/group annotation, but it is semantically specified rather than
nuisance-concept-free.

This v2 experiment was designed after the v1 Waterbirds group/test outcomes
had been inspected, specifically to diagnose the unmatched-continuation
baseline. It is therefore a **post-hoc matched-control sensitivity analysis**,
not a fresh held-out confirmation. A prospective rerun on new untouched data
would be needed for independent confirmation.

## Selection rule

For each of five seeds, the script trains the predeclared candidate repairs and
scores them on the ordinary validation task risk and a held-out, cross-fitted
hidden-slice Fisher certificate.  It then:

1. finds the lowest mean validation NLL and admits candidates within its
   one-standard-error risk ceiling;
2. chooses the highest Bonferroni-adjusted cross-fit certificate lower score
   among those candidates, breaking an exact tie toward the smaller lambda.

The two-percentage-point overall/balanced/per-class accuracy screen and the
paired inverse-Fisher comparison to the matched `lambda=0` continuation are reported diagnostics, not
extra gates on the primary choice. Because cross-fit gate-training folds
overlap, the foldwise Student-t quantity is used conservatively as a ranking
score rather than claimed as a formally calibrated independent-fold interval.

The selected lambda, fitted heads, validation scores, and final audit gates are
hashed and frozen before any test array is deserialized or evaluated. Preflight
only verifies the test artifact's opaque hash and manifest attestation. Test
group metadata is still sealed. `certify.py` evaluates every frozen candidate on blind test data;
only afterward may `posthoc.py` verify the chain, unseal group labels, and add
worst-group accuracy plus a group-supervised validation comparator.

The distinction in supervision is important:

| Row or policy | Group labels in fitting | Group labels in lambda selection | Role |
|---|---:|---:|---|
| Table 4 Group DRO | yes | yes | known-group robust baseline |
| Table 4 standalone Fisher-Glass | yes | group-supervised validation; selected `lambda=0.03` in all seeds | controlled known-group training comparison |
| New frozen ERM reference | no | no | pre-continuation deployment reference; outside the lambda grid |
| New matched `lambda=0` continuation | no | no | primary zero-penalty control with matched optimization |
| New fixed `lambda=0.03` policy | no | no | predeclared hidden-slice-scale reference |
| New risk-only selector | no | no | ordinary-label validation control |
| New group-free selector | no | no | task labels + fixed CLIP nuisance score; rule under evaluation |
| New group-supervised comparator | no | **yes, validation only** | post-hoc diagnostic reference; never a deployable rule |

Thus the new run evaluates a group-label-free operating rule. Its supervised
comparator and test group metrics are descriptive checks revealed after
selection, not inputs to the rule. The across-seed intervals describe
training/partition-seed variability on this dataset, not population sampling
uncertainty, and the experiment is not presented as closing the Group DRO gap.
The primary matched contrast is the group-free selector minus the matched
`lambda=0` continuation. The matched-continuation-minus-frozen-ERM and
group-free-minus-frozen-ERM contrasts are secondary controls that expose the
effect of continuation itself.

## Run on Linux with the project environment

The experiment reuses the attested prepared bundle from the corrected
hidden-slice experiment. It has no download path. The launcher first looks for
`build/results/waterbirds_hidden_slice_v2/prepared`; if that manifest is absent,
it accepts the transferred-bundle path
`build/results/waterbirds_hidden_slice/prepared`. Set `DATA_DIR` explicitly to
override either location. From the repository root:

```bash
. .venv/bin/activate

PYTHON="$PWD/.venv/bin/python" \
DEVICE=cuda \
bash experiments/waterbirds_group_free_lambda/run_all.sh
```

The defaults are the complete five-seed sensitivity protocol: seeds 17–21, five cross-fit folds,
ten gate restarts, and the five-point lambda grid.  On an H100, budget several
hours; exact time depends on the MIG allocation and software stack.  The gate
optimizers operate on small frozen-feature matrices, so low or intermittent
`nvidia-smi` utilization is normal even while the process is making progress.

The blind-test and metadata-reveal stages require all five default selections
to be frozen before either reveal begins. Partial seed cohorts are rejected
unless `--smoke` is explicit; that mode records the incomplete cohort in every
downstream artifact and cannot produce a reportable aggregate. Each reveal is
also bound to the hashes of every artifact in the frozen cohort, so interrupted
runs cannot silently replace a later seed after an earlier result is visible.

To check wiring first:

```bash
PYTHON="$PWD/.venv/bin/python" DEVICE=cuda \
bash experiments/waterbirds_group_free_lambda/run_all.sh --smoke
```

Smoke mode writes to `build/results/waterbirds_group_free_lambda_v2_smoke` and
uses one seed, two folds, two lambda values, and tiny optimizers. Its
aggregate JSON and generated prose/TeX outputs are explicitly marked
**NON-REPORTABLE**, and the non-default protocol fails the reportability audit.
Keep all files in its separate default output directory out of the paper.

Useful full-run overrides include `DATA_DIR`, `OUT_DIR`, `LAMBDAS`, `FOLDS`,
`HEAD_STEPS`, `GATE_STEPS`, `GATE_RESTARTS`, `BOOTSTRAP_REPLICATES`, `DEVICE`,
and `RESUME=0`. The end-to-end reveal cohort is fixed to seeds 17–21. In smoke
mode, `SMOKE_OUT_DIR` controls the destination and the launcher pins its own
seed and small runtime settings. A scientifically changed configuration or
changed source should use a fresh `OUT_DIR`; the hash checks intentionally
reject mixed old/new artifacts. The aggregate records every departure from the
frozen five-seed defaults and marks such outputs **NON-REPORTABLE**; path,
device, and resume choices do not change that status.

## Outputs

The default output directory is `build/results/waterbirds_group_free_lambda_v2`.
The final stage writes:

- `aggregate.json`: validated seed summaries, paired differences, selection
  counts, provenance, and the full candidate frontier;
- `main_table.csv`: frozen ERM, matched `lambda=0`, fixed, risk-only,
  group-free, and group-supervised comparator policies;
- `frontier.csv`: every predeclared lambda, whether or not it was selected;
- `waterbirds_group_free_lambda_table.tex`: compact paper table;
- `report.md`: readable protocol, results, and interpretation;
- `rebuttal_lambda_fg.txt`: restrained reviewer-facing draft generated from the
  actual outcomes.

Per-seed artifacts remain under `runs/seed_<seed>/`.  To transfer a complete
result bundle without copying the already shared preparation data:

```bash
tar -czf waterbirds_group_free_lambda_v2_results.tar.gz \
  -C build/results waterbirds_group_free_lambda_v2

tar -tzf waterbirds_group_free_lambda_v2_results.tar.gz | head -80
```

Do not copy only the prose files: the per-seed JSON/NPZ chain is needed to audit
the metadata firewall and reproduce every reported row.

## CPU-only checks

These tests do not load Waterbirds or require a GPU:

```bash
PYTHON="$PWD/.venv/bin/python" -m unittest discover \
  -s experiments/waterbirds_group_free_lambda/tests -v
```

`preflight.py` is also read-only.  It parses the stage sources, imports required
packages, checks CUDA visibility, validates all prepared blind hashes and exact
schemas, verifies the opaque metadata-seal digest without deserializing it, and
checks the static blind-stage firewall.
