# Matched validation selectors

Run `python scripts/reproduce_retained_tables.py` from the repository root. It recomputes each seed's eligibility from `artifacts/prospective_v2/runs/erm_lam0_seed*/selection.json`, applies all six selectors both to the same eligible set and separately to all four groups, and recovers each selected group's existing outcome from `shift_results.json`. Regenerated tables go under `build/reproduced_tables/matched_selectors/`.

Eligibility requires validation group accuracy at least the overall validation accuracy and uncalibrated group NLL no larger than the median across all four groups. The eligible groups are landbird/land and waterbird/water in every seed, with no fallback. Minimum surviving fraction, minimum validation accuracy and maximum validation NLL select waterbird/water in all five seeds. This establishes agreement, not additional selector discrimination.

The retained CSVs and `results.json` include all group statistics, 60 seed-level selector decisions, original per-run bootstrap intervals and checkpoint hashes. Across-seed intervals retain the nominal normal approximation `1.96 * sample_SD / sqrt(5)`. Original archive/member hashes are retained in `results.json` and the release manifest; the numeric archive members are distributed as directories under `artifacts/`.
