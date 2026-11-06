# W2 Rebuttal Report: Exact versus Ridge Efficient Information

- Date: 29 July 2026
- Audit ridge: \(\lambda_{\mathrm{rid}}=10^{-3}\)
- Numerical floor: \(\epsilon=10^{-8}\)
- Tail level: \(\alpha=0.95\)

## Paste-ready response to the reviewer

**W2. Exact versus ridge efficient information.** Thank you for raising this point. You are right that a ridge value alone cannot support a positive claim about the exact efficient information. The relevant distinction is between the exact object analyzed in Theorems 1–2 and the differentiable ridge surrogate used for numerical stability in Equations 10, 33, and 36.

We used

\[
\lambda_{\mathrm{rid}}=10^{-3},\qquad \epsilon=10^{-8},
\]

with task and nuisance coordinates normalized using the training split. The analytic synthetic experiment computes the exact Schur complement without ridge.

Your observation about the direction of the approximation is correct. Writing \(A=J_{TT}\), \(B=J_{TN}\), and \(C=J_{NN}\succ0\),

\[
\begin{aligned}
F^{\mathrm{rid}}-F^{\mathrm{eff}}
&=
B\left[C^{-1}-(C+\lambda_{\mathrm{rid}}I)^{-1}\right]B^\top\\
&=
\lambda_{\mathrm{rid}}
BC^{-1}(C+\lambda_{\mathrm{rid}}I)^{-1}B^\top
\succeq0.
\end{aligned}
\]

If \(C\succeq\kappa I\) and \(\lVert B\rVert_{\mathrm{op}}\le M\), then

\[
\left\lVert F^{\mathrm{rid}}-F^{\mathrm{eff}}\right\rVert_{\mathrm{op}}
\le
\frac{\lambda_{\mathrm{rid}}M^2}
{\kappa(\kappa+\lambda_{\mathrm{rid}})}.
\]

Without a positive conditioning margin \(\kappa\), there is no useful uniform guarantee that the two quantities are close. We therefore addressed your central concern by computing both quantities directly rather than assuming that the ridge gap is small.

In a new environment-level audit, we retained the Fisher blocks, evaluated both Schur complements, and constructed a conservative lower certificate accounting for the ridge-to-exact gap and sampling uncertainty. Across the 75 held-out test audits using \(\lambda_{\mathrm{rid}}=10^{-3}\), the minimum nuisance eigenvalue was \(0.211349\), with an approximate simultaneous 95% lower bound of \(0.178152\). The largest observed environment-level ridge-to-exact gap was \(0.002520\), with an approximate 95% upper bound of \(0.003002\), and the largest aggregate certificate inflation was \(0.001403\). The smallest ridge, plug-in exact, and conservative lower certificates were

\[
0.164630,\qquad 0.163919,\qquad 0.150410,
\]

respectively. Every nuisance-conditioning lower bound remained positive, and none of the audited exact or conservative certificates collapsed.

For the diagnostics corresponding directly to the submitted values, the Waterbirds certificate changes from \(0.164630\) ridge to \(0.163919\) exact, with conservative lower certificate \(0.150410\). On CivilComments, the five-seed ridge and exact means are \(0.499276\) and \(0.499095\), while the minimum conservative certificate is \(0.461847\). The conservative audit uses \(100{,}000\) Bonferroni-corrected Gaussian/Wald draws and is simultaneous within each fitted-model/environment-set run.

For the training experiments, the original environment-level Fisher blocks were not retained. The exact comparison is therefore based on fresh same-protocol reruns rather than the byte-identical historical fits. We consequently attach the positive exact-identifiability conclusion to the models on which the exact and conservative audit was performed, rather than using a ridge-only historical value for that conclusion.

Finally, your concern about optimism applies directly to the positive identifiability interpretation of Theorem 1. For Theorem 2, the fragility lower bound scales as \((n\lambda_e)^{-q/2}\); substituting the larger ridge information makes this lower bound weaker, not artificially stronger.

Thus, we rule out ridge-hidden collapse in the audited runs through direct exact-Schur evaluation, a positive nuisance-conditioning audit, and a conservative lower certificate—not by treating the ridge surrogate as exact.

## Direct answers to W2

### What ridge strength was used?

All main real-data diagnostics and training audits used

\[
\boxed{\lambda_{\mathrm{rid}}=10^{-3}}
\]

with \(\epsilon=10^{-8}\) and training-normalized task and nuisance coordinates. The frozen-training sensitivity audit additionally evaluated \(10^{-4}\) and \(10^{-2}\).

### Is the ridge approximation optimistic?

Yes. For \(C\succ0\),

\[
F^{\mathrm{rid}}\succeq F^{\mathrm{eff}}.
\]

The unsafe interpretation is therefore to treat a positive ridge certificate as sufficient evidence for exact identifiability. The audit does not make this inference: it evaluates the exact empirical Schur complement and a conservative lower value separately.

### Can the gap be bounded uniformly?

Only under a conditioning assumption. If \(C\succeq\kappa I\), the operator-norm bound is

\[
\left\lVert F^{\mathrm{rid}}-F^{\mathrm{eff}}\right\rVert_{\mathrm{op}}
\le
\frac{\lambda_{\mathrm{rid}}\lVert B\rVert_{\mathrm{op}}^2}
{\kappa(\kappa+\lambda_{\mathrm{rid}})}.
\]

The need for \(\kappa>0\) is essential. In the scalar construction

\[
A=1,\qquad C=c,\qquad B=\sqrt c,
\]

the exact information is zero, whereas

\[
F^{\mathrm{rid}}=\frac{\lambda_{\mathrm{rid}}}{c+\lambda_{\mathrm{rid}}},
\]

which approaches one as \(c/\lambda_{\mathrm{rid}}\to0\). This is why nuisance conditioning and the exact Schur complement are both included in the audit.

### How is hidden exact collapse ruled out?

It is ruled out for an audited fitted model only when all three checks succeed:

1. the nuisance block has a positive lower conditioning bound;
2. the exact plug-in Schur complement is computed directly; and
3. the conservative lower certificate remains above the numerical floor.

For environment \(e\), the audit constructs

\[
\underline s_e=
\left[
\lambda_{\min}(\widehat F^{\mathrm{rid}}_e)
-\delta^{\mathrm{rid}}_e
-\eta_e
\right]_+,
\]

where \(\delta^{\mathrm{rid}}_e\) is an upper audit of the ridge-to-exact gap and \(\eta_e\) accounts for estimation error in the ridge information. It then reports

\[
\underline I_\alpha=
\left[
\operatorname{CVaR}_\alpha
\left(
\frac{1}{\underline s_e+\epsilon}
\right)
\right]^{-1}.
\]

The observed ordering was

\[
\underline I_\alpha
\le
I_\alpha^{\mathrm{exact}}
\le
I_\alpha^{\mathrm{ridge}}
\]

for every audited record.

## Numerical audit results

All values in this table use the default Schur ridge \(10^{-3}\). Parentheses contain the approximate simultaneous 95% within-run lower or upper audit.

| Scope | Test rows | Minimum \(J_{NN}\) (LCB) | Maximum environment gap (UCB) | Maximum certificate inflation | Minimum exact certificate | Minimum conservative certificate |
|---|---:|---:|---:|---:|---:|---:|
| CivilComments diagnostics | 5 | 1.45594 (1.41696) | 0.00115951 (0.00132552) | 0.00018243 | 0.497034 | 0.461847 |
| Waterbirds diagnostics | 5 | 0.211349 (0.178152) | 0.00125011 (0.00166114) | 0.000711254 | 0.163919 | 0.150410 |
| Fresh frozen-feature training audit | 30 | 0.227201 (0.191597) | 0.00143740 (0.00198801) | 0.000617340 | 0.197157 | 0.181409 |
| Fresh end-to-end training audit | 35 | 0.349184 (0.349184) | 0.00252044 (0.00300185) | 0.00140313 | 0.218458 | 0.200382 |
| **All default-ridge test audits** | **75** | **0.211349 (0.178152)** | **0.00252044 (0.00300185)** | **0.00140313** | **0.163919** | **0.150410** |

### Diagnostic values corresponding directly to the submitted results

| Diagnostic | Ridge certificate | Plug-in exact certificate | Conservative lower certificate |
|---|---:|---:|---:|
| CivilComments, five-seed mean | 0.499276 | 0.499095 | 0.463848 mean |
| CivilComments, minimum across seeds | 0.497216 | 0.497034 | 0.461847 |
| Waterbirds | 0.164630 | 0.163919 | 0.150410 |

### Fresh training audit at manuscript-aligned hyperparameters

These are results for newly fitted models under the same protocol; they are not post-hoc values from the original training checkpoints.

| Setting | Ridge certificate | Plug-in exact certificate | Conservative lower certificate |
|---|---:|---:|---:|
| Frozen Fisher-Glass, fixed \(\lambda_{\mathrm{FG}}=0.03\), five-seed mean | 0.400243 | 0.399875 | 0.366300 |
| Frozen Group-DRO+FG, candidates restricted to \(\lambda_{\mathrm{FG}}\in\{0.1,0.3\}\), five-seed mean | 0.625390 | 0.625222 | 0.566367 |
| End-to-end Group-DRO+FG, \(\lambda_{\mathrm{FG}}=10^{-4}\), five-seed mean | 0.544693 | 0.544152 | 0.489027 |

### Ridge sensitivity

The fixed protocol was also evaluated at smaller and larger Schur ridge values. The maximum environment-level gap grows in the expected direction:

| Schur ridge | Maximum observed gap | Approximate 95% UCB |
|---:|---:|---:|
| \(10^{-4}\) | 0.000144309 | 0.000199624 |
| \(10^{-3}\) | 0.00143740 | 0.00198801 |
| \(10^{-2}\) | 0.0138286 | 0.0191165 |

## Interpretation of Theorems 1 and 2

Theorem 1 gives a positive high-coverage identifiability interpretation. Because the ridge information is larger, that interpretation should be attached to the exact or conservative certificate, not to ridge alone.

Theorem 2 gives a lower bound on classification fragility of the form

\[
\mathfrak R_e(n)\gtrsim
\min\{1,(n\lambda_e)^{-q/2}\}.
\]

Using the larger ridge value for \(\lambda_e\) decreases the right-hand side. It therefore weakens the fragility lower bound rather than making the theoretical lower bound artificially stronger.

The ridge objective remains useful as a stable and differentiable training surrogate. The exact and conservative quantities serve a different role: post-training evaluation of the positive identifiability claim.

## Scope and reporting language

The diagnostic reruns reproduce the submitted CivilComments and Waterbirds diagnostic certificates and directly supply their exact and conservative counterparts.

The training portion of the audit uses newly fitted models. The original training checkpoints, score arrays, and environment-level Fisher blocks were not retained. In addition, the fresh hyperparameter selection and end-to-end aggregates are not identical to the submitted training aggregates. The training results should therefore be described as a fresh same-protocol audit, not as a reconstruction of the exact historical fits.

Recommended language:

> The new audit rules out ridge-hidden collapse in the diagnostic experiments and in fresh same-protocol training reruns. We attach the positive exact-identifiability conclusion to the models on which the exact and conservative audit was performed.

Language to avoid:

> The audit retroactively proves that every originally submitted ridge-only training certificate was exact-collapse-free.

## Audit integrity and provenance

- Archive: `build/fisherglass_w2_audit_bundle.tar.gz`
- Archive SHA-256: `713C3FE127C736ED2B952BD04FF10C9DA6B347851C4471FAD77E54FE86D82424`
- Source commit recorded by the run: `aacc84765c2f650e31d7b7042e605a43b07a1470`
- Raw validation: 510 audit objects covering 2,180 environment records
- Generated report: 260 rows, comprising 125 validation and 135 test rows
- Default-ridge report: 140 validation/test rows, including 75 test rows
- Audit simulation: 100,000 Gaussian/Wald draws, confidence 0.95
- Status: every inspected audit object was `audited`; no object was `uncertifiable`
- Algebra: every stored exact, ridge, and ridge-gap identity recomputed successfully
- Certificate ordering: conservative \(\le\) exact \(\le\) ridge for every record
- Report consistency: JSON, CSV, and Markdown outputs agree with the raw records

The confidence calculation is conditional on each fitted model and fixed evaluated environment set and is simultaneous within each run. It is not a joint confidence statement across all seeds and methods, nor a replacement for the population-level assumptions of Theorems 1–2.

## Supporting files

- Full generated audit: [`../build/audit_inspection_20260729/w2_ridge_exact_audit/report/ridge_exact_audit.md`](../build/audit_inspection_20260729/w2_ridge_exact_audit/report/ridge_exact_audit.md)
- Machine-readable report: [`../build/audit_inspection_20260729/w2_ridge_exact_audit/report/ridge_exact_audit.json`](../build/audit_inspection_20260729/w2_ridge_exact_audit/report/ridge_exact_audit.json)
- Tabular report: [`../build/audit_inspection_20260729/w2_ridge_exact_audit/report/ridge_exact_audit.csv`](../build/audit_inspection_20260729/w2_ridge_exact_audit/report/ridge_exact_audit.csv)
- Run configuration: [`../build/audit_inspection_20260729/w2_ridge_exact_audit/run_config.txt`](../build/audit_inspection_20260729/w2_ridge_exact_audit/run_config.txt)
- System information: [`../build/audit_inspection_20260729/w2_ridge_exact_audit/system_info.txt`](../build/audit_inspection_20260729/w2_ridge_exact_audit/system_info.txt)
