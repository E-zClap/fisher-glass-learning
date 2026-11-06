# Frozen-coordinate matrix reanalysis

Run `python scripts/reproduce_retained_tables.py` from the repository root. The command reads the exact compact NPZ coordinates, verifies all published input hashes, recomputes all four matrix definitions and the seed-specific selections, and checks agreement with `results.json`. It writes regenerated numerical tables under `build/reproduced_tables/matrix_constructions/`.

The cohort is the November prospective v2 rerun, ERM seeds 22–26. Eligibility is fixed to each seed's original validation eligible set. For the original globally standardized coordinates `v=(z,b)`, the constructions are `mean(v v^T)`, `mean((v-mean(v))(v-mean(v))^T)`, and `mean(p*(1-p)*v v^T)`. The last is the predictive Fisher of `Bernoulli(sigmoid(raw_logit/T + theta*z + eta*b))` at zero perturbation. The archived fourth comparator divides its weights by their within-slice sum. All four use a nuisance ridge of 0.001; therefore weight normalization changes the effective ridge.

The uncentered and predictive constructions select waterbird/water in all five seeds; within-slice centering selects landbird/land in all five. Their associated intervention outcomes are the existing frozen-checkpoint outcomes. These are post hoc changes in which population is selected, not new interventions or refitted models.

`source_provenance.json` preserves original source-file, archive and checkpoint hashes. The compact NPZs contain exact original validation coordinates, logits, row IDs, group/class labels, temperature, and task-score normalization constants. Full feature caches and checkpoints are not distributed. `results.json` and the two original CSV files retain full precision output; reported seed intervals are nominal normal-approximation intervals.
