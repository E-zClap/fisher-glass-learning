# An easy, low-identifiability slice fails under a held-out nuisance intervention

Prospective Waterbirds experiment: the slice is chosen from validation data alone, everything is frozen, and only then is a label-preserving background intervention applied to held-out images.

## What was frozen before any shifted image was scored

- **Selection rule**: Acc_val(g) >= Acc_val(overall) and NLL_val(g) <= median_h NLL_val(h), then argmin_g projected_over_raw on validation.
- **Selected slice** `waterbird/water` (5/5 seeds, unanimous; votes: {'waterbird/water': 5}).
- **Nuisance coordinate**: cross-fitted logistic background probe on the trained pooled representation, validation only.
- **Constants**: alpha=0.95, ridge=0.001, eps=1e-08.
- **Selection statistic**: `projected_over_raw`. The absolute projected Fisher is confounded by the within-slice logit spread `J_TT`, which varies several-fold across groups; the scale-free ratio is not.
- **A loss-based rule would instead have selected** `waterbird/land`; a worst-accuracy rule would have selected `waterbird/land`; the calibrated variant selects `waterbird/water`; the absolute-projected-Fisher variant selects `landbird/land`.
- **Selected slice sits 2.11x below the median slice's selection statistic.**
- **Eligible (benign) slices**: ['landbird/land', 'waterbird/water'] — note how many candidates the accuracy/loss filter left, since the statistic only ranks within that set.
- **Frozen hash**: `4c36c7529df0affc5a9f648150f3e36d462574933c39fcfd88c7731c2460615e`.

## 1. Validation-only slice table (ERM, mean ± 95% CI over seeds)

|Validation slice|n|Accuracy|NLL|Raw Fisher J_TT|Projected Fisher|Projected/raw|Calibrated projected|
|---|---|---|---|---|---|---|---|
|landbird/land|467|0.9897 ± 0.0050|0.0383 ± 0.0159|0.7369 ± 0.0458|0.4357 ± 0.0453|0.590 ± 0.027|0.3436 ± 0.0267|
|landbird/water|466|0.7030 ± 0.0189|1.6699 ± 0.1561|0.7942 ± 0.0320|0.2889 ± 0.0137|0.365 ± 0.026|0.2366 ± 0.0205|
|waterbird/land|133|0.3895 ± 0.0205|7.7175 ± 0.8003|0.9726 ± 0.0325|0.7637 ± 0.0314|0.786 ± 0.036|0.6761 ± 0.0380|
|waterbird/water|133|0.9203 ± 0.0128|0.6377 ± 0.0382|2.0633 ± 0.0544|0.5107 ± 0.0207|0.248 ± 0.015|0.5357 ± 0.0215|

## 2. Prospective held-out outcome on the frozen slice

`Δ_nuisance = Acc(same-type swap) − Acc(opposite-type swap)`; the same-type arm pays exactly the same recomposition cost, so it is the artifact floor.

|Model (slice is the frozen one throughout)|Source acc.|Source NLL|Projected Fisher|Proj./raw|Val. certificate|Same-type acc.|Opposite-type acc.|Δ_nuisance|
|---|---|---|---|---|---|---|---|---|
|ERM (n=5)|0.9199 ± 0.0062|0.7690 ± 0.0470|0.5107 ± 0.0207|0.248 ± 0.015|0.2889 ± 0.0137|0.9321 ± 0.0072|0.4062 ± 0.0198|0.5259 ± 0.0139|
|ERM + FG (validation-chosen λ) (n=5)|0.9131 ± 0.0175|0.8750 ± 0.2795|0.7375 ± 0.1254|0.458 ± 0.019|0.3298 ± 0.0434|0.9202 ± 0.0117|0.3910 ± 0.0505|0.5293 ± 0.0405|
|ERM + FG (λ=0.003) — breakdown (n=1)|0.9377|0.5571|0.7930|0.433|0.3437|0.9385|0.4813|0.4572|
|ERM + FG (λ=0.01) — breakdown (n=3)|0.9133 ± 0.0152|0.9079 ± 0.4037|0.7640 ± 0.1935|0.466 ± 0.025|0.3492 ± 0.0459|0.9200 ± 0.0065|0.3738 ± 0.0417|0.5462 ± 0.0360|
|ERM + FG (λ=0.03) — breakdown (n=1)|0.8879|1.0939|0.6024|0.457|0.2579|0.9026|0.3520|0.5506|

### 2b. Failure among predictions the source metrics call safe

All three columns condition on the prediction being **correct on the original image**, so neither source accuracy nor source loss can explain them. They differ only in how much confidence is additionally required, and each is shown with its denominator `n`:

- `Flip | correct` — every originally-correct prediction. Always well populated.
- `Flip | confident half` — the most-confident 50% of correct predictions *within the slice*. Keeps the 'confidently correct' framing with a guaranteed denominator.
- `Flip | p≥0.90` — fixed calibrated-probability threshold. **Check `n` before quoting this**: a single global temperature can leave a slice with almost nothing above the threshold, in which case the number is noise.

|Model|Flip \| correct|control|n|Flip \| confident half|control|n|Flip \| p≥0.9|control|n|
|---|---|---|---|---|---|---|---|---|---|
|ERM (n=5)|0.5642 ± 0.0217|0.0330 ± 0.0085|591 ± 4|0.4176 ± 0.0244|0.0070 ± 0.0031|299 ± 4|n/a|n/a|0 ± 0|
|ERM + FG (validation-chosen λ) (n=5)|0.5772 ± 0.0488|0.0419 ± 0.0080|586 ± 11|0.4779 ± 0.0630|0.0201 ± 0.0050|294 ± 6|0.4469 ± 0.0489|0.0162 ± 0.0041|219 ± 43|
|ERM + FG (λ=0.003) — breakdown (n=1)|0.4892|0.0291|602|0.3588|0.0133|301|0.3545|0.0093|268|
|ERM + FG (λ=0.01) — breakdown (n=3)|0.5961 ± 0.0406|0.0424 ± 0.0044|586 ± 10|0.5079 ± 0.0432|0.0203 ± 0.0055|295 ± 6|0.4728 ± 0.0326|0.0190 ± 0.0027|206 ± 66|
|ERM + FG (λ=0.03) — breakdown (n=1)|0.6088|0.0535|570|0.5070|0.0262|286|0.4615|0.0144|208|

## 3. Every held-out group (ERM), including the ones the rule did not select

|Held-out group|n|Original|Same-type|Opposite-type|Δ_nuisance|Flip \| correct|Flip \| confident half|n|Flip \| p≥0.9|n|
|---|---|---|---|---|---|---|---|---|---|---|
|landbird/land|2255|0.9887 ± 0.0023|0.9934 ± 0.0006|0.7264 ± 0.0181|0.2670 ± 0.0177|0.2679 ± 0.0180|0.1475 ± 0.0210|1122 ± 4|0.2009 ± 0.0183|1752 ± 39|
|landbird/water|2255|0.7125 ± 0.0177|0.7228 ± 0.0203|0.9918 ± 0.0014|-0.2690 ± 0.0199|0.0017 ± 0.0004|0.0008 ± 0.0006|805 ± 19|0.0005 ± 0.0007|362 ± 48|
|waterbird/land|642|0.4234 ± 0.0280|0.4058 ± 0.0245|0.9229 ± 0.0092|-0.5171 ± 0.0202|0.0115 ± 0.0031|0.0028 ± 0.0033|136 ± 9|n/a|0 ± 0|
|waterbird/water|642|0.9199 ± 0.0062|0.9321 ± 0.0072|0.4062 ± 0.0198|0.5259 ± 0.0139|0.5642 ± 0.0217|0.4176 ± 0.0244|299 ± 4|n/a|0 ± 0|

## 4. Refined probe-score bins inside `waterbird/water`

Bin edges come from validation; the flip rates are held-out. Accuracy and loss barely vary across these bins, so a monotone relationship between a validation identifiability statistic and held-out failure is much stronger evidence than the four-group comparison, where the accuracy/loss filter leaves only two candidates.

_no refined bins (validation slice too small for the requested bin count)_

## 5. Unpaired natural-image cross-check (no image generation)

Same frozen slice, untouched official test split, compared against the same-label / opposite-background slice. Causally weaker (different birds) but free of any compositing.

|Model|Selected slice|Accuracy|Same-label / other background|Accuracy|Δ accuracy|Confident acc. (selected)|Confident acc. (partner)|
|---|---|---|---|---|---|---|---|
|ERM|waterbird/water|0.9199 ± 0.0062|waterbird/land|0.4234 ± 0.0280|0.4966 ± 0.0240|0.0000 ± 0.0000|0.0000 ± 0.0000|
|fisher_glass (λ=0.003)|waterbird/water|0.9377|waterbird/land|0.4938|0.4439|0.9675|0.8633|
|fisher_glass (λ=0.01)|waterbird/water|0.9133 ± 0.0152|waterbird/land|0.3863 ± 0.0340|0.5270 ± 0.0297|0.9565 ± 0.0114|0.7355 ± 0.0328|
|fisher_glass (λ=0.03)|waterbird/water|0.8879|waterbird/land|0.3692|0.5187|0.9498|0.6691|

## 6. Incremental predictive validity at the individual-image level

Outcome: does an originally-**correct** held-out prediction flip under the opposite-type swap? Baseline predictors are nll, confidence_true, margin, z; the Fisher arm adds nuisance_component, abs_transverse_residual, both built from the same 2x2 block as the certificate. Cross-fitted with folds grouped by CUB species. All three variants are computed unconditionally and all are reported:

- `pooled_with_group` — all groups; group indicators in the baseline (most stringent: the baseline already knows the slice, which is what the certificate outputs)
- `pooled_no_group` — all groups; no group indicator (the practitioner's situation, where the subpopulation structure is unknown)
- `selected_slice` — restricted to the frozen selected slice, matching the bin analysis

|Variant|n rows|flip rate|loglik base|loglik +Fisher|**loglik gain**|AUPRC base|AUPRC +Fisher|**ΔAUPRC (paired)**|AUROC base|AUROC +Fisher|**ΔAUROC (paired)**|
|---|---|---|---|---|---|---|---|---|---|---|---|
|pooled_with_group (n=5 runs)|9398 ± 88|0.199 ± 0.010|-0.3343 ± 0.0147|-0.3334 ± 0.0145|0.0009 ± 0.0008|0.5986 ± 0.0163|0.5919 ± 0.0156|-0.0067 ± 0.0019 (0/5)|0.8707 ± 0.0089|0.8733 ± 0.0083|0.0027 ± 0.0015 (5/5)|
|pooled_no_group (n=5 runs)|9398 ± 88|0.199 ± 0.010|-0.4794 ± 0.0165|-0.4630 ± 0.0167|0.0164 ± 0.0033|0.2966 ± 0.0136|0.3321 ± 0.0128|0.0354 ± 0.0072 (5/5)|0.6361 ± 0.0105|0.7325 ± 0.0150|0.0964 ± 0.0068 (5/5)|
|selected_slice (n=5 runs)|1181 ± 8|0.564 ± 0.022|-0.6179 ± 0.0085|-0.6051 ± 0.0097|0.0128 ± 0.0019|0.7349 ± 0.0212|0.7606 ± 0.0194|0.0257 ± 0.0046 (5/5)|0.7081 ± 0.0112|0.7230 ± 0.0120|0.0150 ± 0.0029 (5/5)|

> Quote the paired columns. Held-out log-likelihood gain and ΔAUPRC/ΔAUROC are within-seed differences with the count of seeds that improved; the separate per-arm CIs are across-seed spreads and overlap even when every seed improves. Likelihood-ratio p-values are in `predictive_validity.json` but are not worth quoting: at several thousand rows per run the test detects effects far too small to matter.

> Scope: `beta_g` is constant within a group, so the Fisher arm contributes the per-image nuisance coordinate weighted by that slice's loading. A slice-level scalar cannot be tested this way at all, being collinear with the group indicator by construction.

## Intervention quality control

- Generated **23176** images from **5794** held-out birds (2 independent background draws per bird per arm).
- Visibly inpainted background area: mean 4.13%, p95 8.45%, max 20.37%.
- Matched across arms: same-type 4.13% vs opposite-type 4.12%.
- Bird foreground, matte and label are byte-identical across arms; only background pixels differ.

## Fisher-Glass penalty chosen on validation

> **Not accuracy-matched.** No lambda stayed within 0.010 of ERM validation accuracy for seed(s) ['22', '24'], so the intervention arm below must not be described as accuracy-matched. Try smaller lambdas.

```
{'per_seed': {'22': {'chosen_lambda': 0.01, 'chosen_tag': 'fisher_glass_lam0.01_seed22', 'val_accuracy': 0.7823185988323603, 'val_certificate': 0.3023552243177918, 'erm_val_accuracy': 0.8190158465387823, 'accuracy_floor': 0.8090158465387823, 'n_eligible': 0, 'fell_back': True}, '23': {'chosen_lambda': 0.03, 'chosen_tag': 'fisher_glass_lam0.03_seed23', 'val_accuracy': 0.8123436196830692, 'val_certificate': 0.25786285991134, 'erm_val_accuracy': 0.8056713928273561, 'accuracy_floor': 0.7956713928273561, 'n_eligible': 2, 'fell_back': False}, '24': {'chosen_lambda': 0.01, 'chosen_tag': 'fisher_glass_lam0.01_seed24', 'val_accuracy': 0.780650542118432, 'val_certificate': 0.37217770096244424, 'erm_val_accuracy': 0.8031693077564637, 'accuracy_floor': 0.7931693077564637, 'n_eligible': 0, 'fell_back': True}, '25': {'chosen_lambda': 0.003, 'chosen_tag': 'fisher_glass_lam0.003_seed25', 'val_accuracy': 0.798999165971643, 'val_certificate': 0.3436997773533371, 'erm_val_accuracy': 0.7906588824020017, 'accuracy_floor': 0.7806588824020017, 'n_eligible': 3, 'fell_back': False}, '26': {'chosen_lambda': 0.01, 'chosen_tag': 'fisher_glass_lam0.01_seed26', 'val_accuracy': 0.8023352793994996, 'val_certificate': 0.37305157638222414, 'erm_val_accuracy': 0.8015012510425354, 'accuracy_floor': 0.7915012510425354, 'n_eligible': 2, 'fell_back': False}}, 'consensus_lambda': 0.01, 'accuracy_tolerance': 0.01, 'accuracy_matched': False, 'fell_back_seeds': ['22', '24']}
```

## Rebuttal paragraph

> We agree that the previous version did not empirically close the diagnostic-to-failure loop. We have added a prospective held-out nuisance-intervention experiment on a fully fine-tuned mobilenet_v3_small (5 seeds). Using validation data only, we select the lowest-Fisher-Glass slice among slices with above-average accuracy and below-median loss, and then freeze the checkpoint, the temperature, the cross-fitted nuisance probe, the slice definition and every numerical constant (frozen hash `4c36c7529df0affc`). The selected slice is `waterbird/water`, with 92.0% held-out source accuracy, source NLL 0.769, and validation projected-Fisher score 0.511 (0.248 of its raw Fisher). On held-out paired images, replacing the background with another background of the same type changes accuracy to 93.2%, whereas changing only the background type while preserving the bird foreground and label reduces it to 40.6%, a drop of 52.6% points relative to the artifact-matched control. A fixed calibrated-confidence threshold is not usable on this slice (only 0 predictions clear it), so we report the most-confident 50% of the correct predictions instead (n=299): 41.8% become incorrect after the nuisance flip, against 0.7% for the same-type control. Across all originally-correct predictions the flip rate is 56.4%. A source-loss-based rule does not select this slice. A Fisher-Glass-regularized model (ERM + FG (validation-chosen λ), 5 seeds, validation accuracy 0.7953 vs 0.8040 for ERM, NOT within the matching tolerance for every seed) raises the validation certificate from 0.289 to 0.330 and increases the nuisance-shift drop from 52.6% to 52.9% (flip among originally-correct predictions 56.4% -> 57.7%). NOTE: this does not support the intended 'raising the certificate reduces the failure it predicted' claim -- do not write it up as an intervention result without re-checking. The intervention is pixel-level: it inverts the official Waterbirds composition using the CUB mattes, so the bird foreground and label are byte-identical across arms and only background pixels change; the mean visibly-inpainted background area is 4.1% and is matched across arms (4.1% same-type vs 4.1% opposite-type). This provides a trained-model, real-image example in which source accuracy and loss are benign but the certificate anticipates a specific held-out failure.

