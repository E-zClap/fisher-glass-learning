# An easy, low-identifiability slice fails under a held-out nuisance intervention

Prospective Waterbirds experiment: the slice is chosen from validation data alone, everything is frozen, and only then is a label-preserving background intervention applied to held-out images.

## What was frozen before any shifted image was scored

- **Selection rule**: Acc_val(g) >= Acc_val(overall) and NLL_val(g) <= median_h NLL_val(h), then argmin_g projected_fisher on validation.
- **Selected slice** `landbird/land` (5/5 seeds, unanimous; votes: {'landbird/land': 5}).
- **Nuisance coordinate**: cross-fitted logistic background probe on the trained pooled representation, validation only.
- **Constants**: alpha=0.95, ridge=0.001, eps=1e-08.
- **Selection statistic**: `projected_fisher`. The absolute projected Fisher is confounded by the within-slice logit spread `J_TT`, which varies several-fold across groups; the scale-free ratio is not.
- **A loss-based rule would instead have selected** `waterbird/land`; a worst-accuracy rule would have selected `waterbird/land`; the calibrated variant selects `landbird/land`; the absolute-projected-Fisher variant selects `landbird/land`.
- **Selected slice sits 1.30x below the median slice's selection statistic.**
- **Eligible (benign) slices**: ['landbird/land', 'waterbird/water'] — note how many candidates the accuracy/loss filter left, since the statistic only ranks within that set.
- **Frozen hash**: `6989773ddf82d93c9657b19e1ebf5b3b9e09012ce47c6a7ca8ac81f537062d26`.

## 1. Validation-only slice table (ERM, mean ± 95% CI over seeds)

|Validation slice|n|Accuracy|NLL|Raw Fisher J_TT|Projected Fisher|Projected/raw|Calibrated projected|
|---|---|---|---|---|---|---|---|
|landbird/land|467|0.9880 ± 0.0010|0.0404 ± 0.0164|0.7396 ± 0.0419|0.3993 ± 0.0362|0.539 ± 0.018|0.3145 ± 0.0115|
|landbird/water|466|0.6605 ± 0.0147|2.0737 ± 0.1402|0.7825 ± 0.0325|0.2550 ± 0.0180|0.326 ± 0.022|0.2195 ± 0.0137|
|waterbird/land|133|0.4571 ± 0.0331|6.8029 ± 0.2158|0.9764 ± 0.0266|0.7836 ± 0.0403|0.802 ± 0.023|0.6931 ± 0.0253|
|waterbird/water|133|0.9353 ± 0.0137|0.5474 ± 0.1557|2.1106 ± 0.0523|0.5178 ± 0.0489|0.245 ± 0.020|0.5381 ± 0.0545|

## 2. Prospective held-out outcome on the frozen slice

`Δ_nuisance = Acc(same-type swap) − Acc(opposite-type swap)`; the same-type arm pays exactly the same recomposition cost, so it is the artifact floor.

|Model (slice is the frozen one throughout)|Source acc.|Source NLL|Projected Fisher|Proj./raw|Val. certificate|Same-type acc.|Opposite-type acc.|Δ_nuisance|
|---|---|---|---|---|---|---|---|---|
|ERM (n=5)|0.9870 ± 0.0014|0.0485 ± 0.0049|0.3993 ± 0.0362|0.539 ± 0.018|0.2550 ± 0.0180|0.9918 ± 0.0025|0.7000 ± 0.0298|0.2918 ± 0.0276|
|ERM + FG (validation-chosen λ) (n=5)|0.9869 ± 0.0034|0.0912 ± 0.0494|0.4329 ± 0.0747|0.946 ± 0.035|0.3027 ± 0.0824|0.9911 ± 0.0019|0.7333 ± 0.0652|0.2578 ± 0.0647|
|ERM + FG (λ=0.003) — breakdown (n=1)|0.9902|0.0471|0.4021|0.907|0.2490|0.9925|0.7129|0.2796|
|ERM + FG (λ=0.01) — breakdown (n=1)|0.9871|0.0721|0.2967|0.906|0.2112|0.9922|0.7217|0.2705|
|ERM + FG (λ=0.03) — breakdown (n=3)|0.9857 ± 0.0053|0.1123 ± 0.0762|0.4886 ± 0.0114|0.973 ± 0.024|0.3510 ± 0.1046|0.9902 ± 0.0028|0.7440 ± 0.1167|0.2463 ± 0.1153|

### 2b. Failure among predictions the source metrics call safe

All three columns condition on the prediction being **correct on the original image**, so neither source accuracy nor source loss can explain them. They differ only in how much confidence is additionally required, and each is shown with its denominator `n`:

- `Flip | correct` — every originally-correct prediction. Always well populated.
- `Flip | confident half` — the most-confident 50% of correct predictions *within the slice*. Keeps the 'confidently correct' framing with a guaranteed denominator.
- `Flip | p≥0.90` — fixed calibrated-probability threshold. **Check `n` before quoting this**: a single global temperature can leave a slice with almost nothing above the threshold, in which case the number is noise.

|Model|Flip \| correct|control|n|Flip \| confident half|control|n|Flip \| p≥0.9|control|n|
|---|---|---|---|---|---|---|---|---|---|
|ERM (n=5)|0.2938 ± 0.0302|0.0076 ± 0.0027|2226 ± 3|0.1771 ± 0.0154|0.0015 ± 0.0004|1119 ± 5|0.2220 ± 0.0209|0.0027 ± 0.0014|1695 ± 21|
|ERM + FG (validation-chosen λ) (n=5)|0.2608 ± 0.0641|0.0081 ± 0.0019|2225 ± 8|0.2061 ± 0.0554|0.0045 ± 0.0015|1116 ± 4|0.1679 ± 0.0444|0.0026 ± 0.0009|723 ± 142|
|ERM + FG (λ=0.003) — breakdown (n=1)|0.2835|0.0065|2233|0.2022|0.0027|1120|0.1899|0.0020|998|
|ERM + FG (λ=0.01) — breakdown (n=1)|0.2718|0.0070|2226|0.2177|0.0040|1116|0.1794|0.0042|722|
|ERM + FG (λ=0.03) — breakdown (n=3)|0.2495 ± 0.1143|0.0090 ± 0.0029|2223 ± 12|0.2034 ± 0.1005|0.0052 ± 0.0019|1115 ± 5|0.1568 ± 0.0772|0.0024 ± 0.0008|631 ± 54|

## 3. Every held-out group (ERM), including the ones the rule did not select

|Held-out group|n|Original|Same-type|Opposite-type|Δ_nuisance|Flip \| correct|Flip \| confident half|n|Flip \| p≥0.9|n|
|---|---|---|---|---|---|---|---|---|---|---|
|landbird/land|2255|0.9870 ± 0.0014|0.9918 ± 0.0025|0.7000 ± 0.0298|0.2918 ± 0.0276|0.2938 ± 0.0302|0.1771 ± 0.0154|1119 ± 5|0.2220 ± 0.0209|1695 ± 21|
|landbird/water|2255|0.6693 ± 0.0186|0.6940 ± 0.0247|0.9907 ± 0.0033|-0.2967 ± 0.0219|0.0020 ± 0.0009|0.0012 ± 0.0008|756 ± 21|0.0007 ± 0.0014|285 ± 19|
|waterbird/land|642|0.4595 ± 0.0258|0.4276 ± 0.0545|0.9302 ± 0.0140|-0.5026 ± 0.0427|0.0134 ± 0.0074|0.0028 ± 0.0041|148 ± 8|n/a|0 ± 0|
|waterbird/water|642|0.9290 ± 0.0044|0.9372 ± 0.0079|0.4336 ± 0.0478|0.5036 ± 0.0406|0.5381 ± 0.0502|0.4032 ± 0.0551|299 ± 1|0.1236 ± 0.1052|8 ± 7|

## 4. Refined probe-score bins inside `landbird/land`

Bin edges come from validation; the flip rates are held-out. Accuracy and loss barely vary across these bins, so a monotone relationship between a validation identifiability statistic and held-out failure is much stronger evidence than the four-group comparison, where the accuracy/loss filter leaves only two candidates.

|Probe-score bin (validation edges)|n (test)|Val. acc.|Val. NLL|Val. projected Fisher|Val. proj./raw|Val. selection statistic|Held-out original acc.|Δ_nuisance|Flip \| correct|Flip \| confident half|
|---|---|---|---|---|---|---|---|---|---|---|
|landbird/land | probe bin 1|581|1.0000 ± 0.0000|0.0002 ± 0.0002|0.4169 ± 0.0469|0.301 ± 0.025|0.417 ± 0.047|0.9982 ± 0.0011|0.3097 ± 0.0347|0.3144 ± 0.0377|0.1908 ± 0.0307|
|landbird/land | probe bin 2|546|0.9966 ± 0.0041|0.0190 ± 0.0289|0.4138 ± 0.0856|0.608 ± 0.056|0.414 ± 0.086|0.9983 ± 0.0021|0.2969 ± 0.0329|0.3049 ± 0.0348|0.1695 ± 0.0240|
|landbird/land | probe bin 3|536|0.9949 ± 0.0067|0.0183 ± 0.0203|0.3878 ± 0.0283|0.775 ± 0.053|0.388 ± 0.028|0.9915 ± 0.0030|0.2824 ± 0.0206|0.2867 ± 0.0234|0.1365 ± 0.0128|
|landbird/land | probe bin 4|592|0.9607 ± 0.0085|0.1240 ± 0.0396|0.3701 ± 0.0295|0.943 ± 0.024|0.370 ± 0.030|0.9606 ± 0.0051|0.2757 ± 0.0323|0.2660 ± 0.0339|0.1334 ± 0.0165|

## 5. Unpaired natural-image cross-check (no image generation)

Same frozen slice, untouched official test split, compared against the same-label / opposite-background slice. Causally weaker (different birds) but free of any compositing.

|Model|Selected slice|Accuracy|Same-label / other background|Accuracy|Δ accuracy|Confident acc. (selected)|Confident acc. (partner)|
|---|---|---|---|---|---|---|---|
|ERM|landbird/land|0.9870 ± 0.0014|landbird/water|0.6693 ± 0.0186|0.3177 ± 0.0179|1.0000 ± 0.0000|1.0000 ± 0.0000|
|fisher_glass (λ=0.003)|landbird/land|0.9902|landbird/water|0.6585|0.3317|0.9990|0.9167|
|fisher_glass (λ=0.01)|landbird/land|0.9871|landbird/water|0.6794|0.3078|0.9972|0.8599|
|fisher_glass (λ=0.03)|landbird/land|0.9857 ± 0.0053|landbird/water|0.7298 ± 0.1390|0.2559 ± 0.1347|0.9961 ± 0.0046|0.8519 ± 0.1036|

## 6. Incremental predictive validity at the individual-image level

Outcome: does an originally-**correct** held-out prediction flip under the opposite-type swap? Baseline predictors are nll, confidence_true, margin, z; the Fisher arm adds nuisance_component, abs_transverse_residual, both built from the same 2x2 block as the certificate. Cross-fitted with folds grouped by CUB species. All three variants are computed unconditionally and all are reported:

- `pooled_with_group` — all groups; group indicators in the baseline (most stringent: the baseline already knows the slice, which is what the certificate outputs)
- `pooled_no_group` — all groups; no group indicator (the practitioner's situation, where the subpopulation structure is unknown)
- `selected_slice` — restricted to the frozen selected slice, matching the bin analysis

|Variant|n rows|flip rate|loglik base|loglik +Fisher|**loglik gain**|AUPRC base|AUPRC +Fisher|**ΔAUPRC (paired)**|AUROC base|AUROC +Fisher|**ΔAUROC (paired)**|
|---|---|---|---|---|---|---|---|---|---|---|---|
|pooled_with_group (n=5 runs)|9252 ± 105|0.212 ± 0.013|-0.3551 ± 0.0091|-0.3539 ± 0.0084|0.0011 ± 0.0012|0.5862 ± 0.0240|0.5813 ± 0.0241|-0.0049 ± 0.0018 (0/5)|0.8584 ± 0.0057|0.8618 ± 0.0048|0.0034 ± 0.0023 (5/5)|
|pooled_no_group (n=5 runs)|9252 ± 105|0.212 ± 0.013|-0.4998 ± 0.0186|-0.4817 ± 0.0152|0.0181 ± 0.0044|0.2968 ± 0.0249|0.3320 ± 0.0223|0.0352 ± 0.0128 (5/5)|0.6234 ± 0.0201|0.7135 ± 0.0190|0.0902 ± 0.0069 (5/5)|
|selected_slice (n=5 runs)|4451 ± 6|0.294 ± 0.030|-0.5496 ± 0.0161|-0.5275 ± 0.0117|0.0222 ± 0.0048|0.4945 ± 0.0465|0.5429 ± 0.0510|0.0484 ± 0.0060 (5/5)|0.6982 ± 0.0172|0.7344 ± 0.0167|0.0362 ± 0.0076 (5/5)|

> Quote the paired columns. Held-out log-likelihood gain and ΔAUPRC/ΔAUROC are within-seed differences with the count of seeds that improved; the separate per-arm CIs are across-seed spreads and overlap even when every seed improves. Likelihood-ratio p-values are in `predictive_validity.json` but are not worth quoting: at several thousand rows per run the test detects effects far too small to matter.

> Scope: `beta_g` is constant within a group, so the Fisher arm contributes the per-image nuisance coordinate weighted by that slice's loading. A slice-level scalar cannot be tested this way at all, being collinear with the group indicator by construction.

## Intervention quality control

- Generated **23176** images from **5794** held-out birds (2 independent background draws per bird per arm).
- Visibly inpainted background area: mean 4.13%, p95 8.45%, max 20.37%.
- Matched across arms: same-type 4.13% vs opposite-type 4.12%.
- Bird foreground, matte and label are byte-identical across arms; only background pixels differ.

## Fisher-Glass penalty chosen on validation

> **Not accuracy-matched.** No lambda stayed within 0.010 of ERM validation accuracy for seed(s) ['20'], so the intervention arm below must not be described as accuracy-matched. Try smaller lambdas.

```
{'per_seed': {'17': {'chosen_lambda': 0.01, 'chosen_tag': 'fisher_glass_lam0.01_seed17', 'val_accuracy': 0.7898248540450375, 'val_certificate': 0.21123512221282398, 'erm_val_accuracy': 0.798999165971643, 'accuracy_floor': 0.788999165971643, 'n_eligible': 1, 'fell_back': False}, '18': {'chosen_lambda': 0.03, 'chosen_tag': 'fisher_glass_lam0.03_seed18', 'val_accuracy': 0.8457047539616347, 'val_certificate': 0.40466671741725296, 'erm_val_accuracy': 0.7948290241868223, 'accuracy_floor': 0.7848290241868223, 'n_eligible': 2, 'fell_back': False}, '19': {'chosen_lambda': 0.03, 'chosen_tag': 'fisher_glass_lam0.03_seed19', 'val_accuracy': 0.7998331943286072, 'val_certificate': 0.2442638174231984, 'erm_val_accuracy': 0.7839866555462885, 'accuracy_floor': 0.7739866555462885, 'n_eligible': 3, 'fell_back': False}, '20': {'chosen_lambda': 0.03, 'chosen_tag': 'fisher_glass_lam0.03_seed20', 'val_accuracy': 0.7764804003336113, 'val_certificate': 0.404113098981292, 'erm_val_accuracy': 0.8006672226855713, 'accuracy_floor': 0.7906672226855713, 'n_eligible': 0, 'fell_back': True}, '21': {'chosen_lambda': 0.003, 'chosen_tag': 'fisher_glass_lam0.003_seed21', 'val_accuracy': 0.804837364470392, 'val_certificate': 0.24901440489488494, 'erm_val_accuracy': 0.8015012510425354, 'accuracy_floor': 0.7915012510425354, 'n_eligible': 2, 'fell_back': False}}, 'consensus_lambda': 0.03, 'accuracy_tolerance': 0.01, 'accuracy_matched': False, 'fell_back_seeds': ['20']}
```

## Rebuttal paragraph

> We agree that the previous version did not empirically close the diagnostic-to-failure loop. We have added a prospective held-out nuisance-intervention experiment on a fully fine-tuned mobilenet_v3_small (5 seeds). Using validation data only, we select the lowest-Fisher-Glass slice among slices with above-average accuracy and below-median loss, and then freeze the checkpoint, the temperature, the cross-fitted nuisance probe, the slice definition and every numerical constant (frozen hash `6989773ddf82d93c`). The selected slice is `landbird/land`, with 98.7% held-out source accuracy, source NLL 0.049, and validation projected-Fisher score 0.399 (0.539 of its raw Fisher). On held-out paired images, replacing the background with another background of the same type changes accuracy to 99.2%, whereas changing only the background type while preserving the bird foreground and label reduces it to 70.0%, a drop of 29.2% points relative to the artifact-matched control. Among source-correct predictions with calibrated confidence above 0.90 (n=1695), 22.2% become incorrect after the nuisance flip, against 0.3% for the same-type control. A source-loss-based rule does not select this slice. A Fisher-Glass-regularized model (ERM + FG (validation-chosen λ), 5 seeds, validation accuracy 0.8033 vs 0.7960 for ERM, NOT within the matching tolerance for every seed) raises the validation certificate from 0.255 to 0.303 and reduces the nuisance-shift drop from 29.2% to 25.8% (flip among originally-correct predictions 29.4% -> 26.1%). The intervention is pixel-level: it inverts the official Waterbirds composition using the CUB mattes, so the bird foreground and label are byte-identical across arms and only background pixels change; the mean visibly-inpainted background area is 4.1% and is matched across arms (4.1% same-type vs 4.1% opposite-type). This provides a trained-model, real-image example in which source accuracy and loss are benign but the certificate anticipates a specific held-out failure.

