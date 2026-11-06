# Asset provenance and rights status

This release distributes experiment source code, manuscript sources, derived numerical audit inputs and outputs, and retained reports. It does not redistribute benchmark images/text, pretrained model weights, fitted model weights, or feature caches. Compact numeric inputs contain derived coordinates and, where needed, numeric labels and bird-species names.

No repository-wide software license has been assigned. The absence of a license should not be interpreted as a grant of reuse rights. Dataset and model links below describe external assets; they do not assign their rights to this repository. The historical runs did not retain a complete software bill of materials or all source-version/license records.

| External asset | Experiment use | Upstream source and terms |
|---|---|---|
| WILDS | Dataset interfaces and benchmark definitions | [WILDS datasets](https://wilds.stanford.edu/datasets/) and [software repository](https://github.com/p-lambda/wilds) |
| CivilComments-WILDS | Text diagnostics | [Official WILDS dataset information](https://wilds.stanford.edu/datasets/#civilcomments) |
| Waterbirds | Image classification, nuisance interventions, discovery and training | [Original group DRO/Waterbirds repository](https://github.com/kohpangwei/group_DRO) |
| CUB-200-2011 and its segmentations | Bird images, labels, species grouping, foreground masks | [Caltech CUB dataset](https://www.vision.caltech.edu/datasets/cub_200_2011/); the official project credits the segmentations to Ryan Farrell |
| Places | Waterbirds background images | [Places project](http://places2.csail.mit.edu/) and original Waterbirds documentation |
| ImageNet-pretrained MobileNetV3-small | Feature extraction and initialization | [Torchvision MobileNetV3-small documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html) and associated upstream model/data terms |
| CLIP ViT-B/32 | Fixed water-versus-land text nuisance direction | [OpenAI CLIP source and license](https://github.com/openai/CLIP) and [model card](https://huggingface.co/openai/clip-vit-base-patch32) |
| DistilBERT | Optional end-to-end CivilComments harness | [Model card](https://huggingface.co/distilbert/distilbert-base-uncased) |

Download external assets from their providers after checking the applicable original terms. Dependency packages retain their own licenses. Prior methods and the Fisher-glass framework are cited in the manuscript bibliography. Historical reports are retained as evidence and can contain earlier interpretations; the release README and final manuscript state the current scope.
