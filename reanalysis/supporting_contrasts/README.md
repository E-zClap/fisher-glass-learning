# Supporting paired contrast and certificate means

Recompute the numerical additions in Appendices B.4 and D.4 from retained public records, without training:

```shell
python reanalysis/supporting_contrasts/recompute.py
```

The script requires SciPy (listed in `requirements-replay.txt`). Use `--artifact-root PATH` for another artifact location or `--output-dir PATH` to keep the distributed outputs untouched.

For each hidden-slice seed 17–21, the script subtracts the minimum of the four predefined-group projected predictive information values from the discovered slice value, then calculates a paired 95% Student-t interval across five seeds. It checks matching Fisher settings and same-class pooling, and reconstructs every Schur value from its stored scalar blocks.

For CivilComments seeds 13–17, it reconstructs ridge and unregularized certificates from the 18 uniformly weighted environment matrices. At alpha 0.95 the upper tail fits within the worst group's mass; the reciprocal CVaR therefore equals the minimum information plus epsilon. These means belong to the fresh diagnostic-audit cohort, distinct from the older diagnostic summary.

`results.json` contains the per-seed calculations, aggregate values and SHA256 of all 20 input files. These are arithmetic reanalyses of retained records, not additional trained models or independently collected observations.
