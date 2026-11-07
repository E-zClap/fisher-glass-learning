# Worked classification mechanism: derivation checks

The active statement is Section 3.2 of the camera-ready manuscript; the full proof is Appendix A.4 in `tex/camera_ready_v2/appendix/transferred_theory.tex`. This is a theoretical addition, not a new empirical experiment.

For known `0 < delta <= 1/2`, the family has uniform X on [-1,1] and regression probability `1/2 + x/4 + theta + mu * 1{|x| > delta}`, with both parameters bounded by delta/8. The proof establishes exact information and separation identities, a KL bound, and uniform matching minimax rates, including the case with no observed points in the central region.

Run deterministic sanity checks from this directory:

```shell
python verify.py
```

The script requires Python and NumPy. It checks information integrals, decision separation, KL inequalities, the exact excess-risk identity, and the binomial reciprocal identity by quadrature and finite sums. `verification.json` records the results. These checks support the analytic derivation; they do not replace its uniform proof. No simulation, model training, intervention inference, or new empirical measurement is performed.

The example's delta is the half-width and probability mass of the boundary region, distinct from the earlier score-distance parameter delta_e. Its nuisance preserves the Bayes rule at fixed theta. It does not model the physical image-background swap.
