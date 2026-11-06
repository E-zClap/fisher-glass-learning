# Reproducing the primary prospective and hidden-slice experiments

Run from the repository root. These commands produce **new complete runs** from raw public images and pretrained model assets. The retained-table commands in the root README independently reproduce the published numbers without training. Use fresh output directories: the curated `artifacts/` records omit checkpoints/features and must not be used as resumable training directories.

## Environment and data

Use Python 3.9 or newer and install a CUDA-enabled PyTorch/torchvision pair appropriate to the machine before installing the remaining dependencies:

```shell
python -m pip install -r requirements.txt
python scripts/prepare_waterbirds.py --root data/wilds
```

The second command uses only Python's standard library. It downloads the original `waterbird_complete95_forest2water2.tar.gz` from [Stanford's download host](https://downloads.cs.stanford.edu/nlp/data/dro/waterbird_complete95_forest2water2.tar.gz), corresponding to the Waterbirds archive linked by the [dataset authors](https://github.com/kohpangwei/group_DRO#waterbirds). The historical WILDS/CodaLab endpoints returned HTTP 404 during this release check. The script safely stages the archive, verifies all 11,788 metadata-referenced image files, 200 CUB species, official train/validation/test counts 4,795/1,199/5,794 and every split/group count, then creates the WILDS compatibility marker. It will not overwrite an existing dataset. Existing datasets are validated before a missing marker is added. No dataset images are distributed with this repository.

For an already downloaded archive, or to validate existing data without writing anything:

```shell
python scripts/prepare_waterbirds.py --root data/wilds --archive /path/to/waterbird_complete95_forest2water2.tar.gz
python scripts/prepare_waterbirds.py --root data/wilds --verify-only
```

Both pipelines use `data/wilds/waterbirds_v1.0/metadata.csv` and the adjacent species/image paths. Prospective compositing additionally needs the original CUB soft segmentation mattes; no Places download is needed:

```shell
python experiments/prospective_nuisance_shift/fetch_cub_masks.py --out-dir data/cub_segmentations --metadata data/wilds/waterbirds_v1.0/metadata.csv
```

This obtains `segmentations.tgz` from [Caltech's original CUB record](https://data.caltech.edu/records/w9d68-gec53), extracts it, and checks matte coverage for every Waterbirds image. Species IDs used in grouped folds come from the CUB directory component of `img_filename`, not an additional private file. On Linux, `bash experiments/prospective_nuisance_shift/download_data.sh` performs both data commands.

The historical prospective run recorded Python 3.9.25, PyTorch 2.8.0+cu128, torchvision 0.23.0+cu128, NumPy 2.0.2, scikit-learn 1.6.1, SciPy 1.13.1, OpenCV 4.11.0, and Pillow 11.3.0. It used bf16 on an H100. The current broad dependency file is an installation route, not a claim of bitwise identity on every future library/hardware combination. `reanalysis/flip_prediction/requirements-refit.txt` pins the exact numerical versions for the independently verified auxiliary refit.

## Prospective primary v2 run: seeds 22–26, 50 epochs

The unqualified launcher defaults to development settings (seeds 17–21, 20 epochs). The paper's primary cohort requires explicit settings:

```bash
python experiments/prospective_nuisance_shift/preflight.py --skip-compose
bash experiments/prospective_nuisance_shift/run_all.sh --skip-download \
  --seeds "22 23 24 25 26" --epochs 50 --batch-size 32 --lr 0.0001 \
  --methods "erm fisher_glass" --fg-lambdas "0.003 0.01 0.03" \
  --selection-statistic projected_over_raw --groups "0 1 2 3" --draws 2 \
  --max-per-group 0 --compose-workers 12 --device cuda \
  --out-dir build/prospective_shift_v2_fresh \
  --report build/prospective_shift_v2_fresh/report.md \
  --figure-stem waterbirds_prospective_shift_v2_fresh
```

The launcher fits five ERM checkpoints and 15 Fisher-Glass candidates. Selection uses only validation information and writes `frozen_selection.json` before intervention scoring. It generates both arms for all held-out birds, verifies checkpoint hashes, evaluates the selected validation-chosen models, then computes natural-image controls, individual flip prediction, figures, and the report. The original pilot used absolute `projected_fisher`; that is a separate cohort.

The underlying stages can also be run individually, including from PowerShell (each line is one complete command):

```shell
python experiments/prospective_nuisance_shift/mask_stats.py --cache build/prospective_shift_v2_fresh/matte_stats.npz
python experiments/prospective_nuisance_shift/train_end_to_end.py --out-dir build/prospective_shift_v2_fresh --arch mobilenet_v3_small --methods erm fisher_glass --fg-lambdas 0.003 0.01 0.03 --seeds 22 23 24 25 26 --epochs 50 --batch-size 32 --lr 0.0001 --device cuda
python experiments/prospective_nuisance_shift/select_slice.py --out-dir build/prospective_shift_v2_fresh --selection-statistic projected_over_raw --probe-folds 5 --n-bins 4 --min-bin 100
python experiments/prospective_nuisance_shift/build_paired_images.py --out-dir build/prospective_shift_v2_fresh --groups 0 1 2 3 --draws 2 --max-per-group 0 --workers 12
python experiments/prospective_nuisance_shift/evaluate_shift.py --out-dir build/prospective_shift_v2_fresh --confidence-threshold 0.9 --n-boot 2000 --device cuda --num-workers 8 --batch-size 128
python experiments/prospective_nuisance_shift/natural_shift_fallback.py --out-dir build/prospective_shift_v2_fresh --confidence-threshold 0.9 --n-boot 2000
python experiments/prospective_nuisance_shift/predictive_validity.py --out-dir build/prospective_shift_v2_fresh --folds 5 --seed 0
python experiments/prospective_nuisance_shift/make_report.py --out-dir build/prospective_shift_v2_fresh --report build/prospective_shift_v2_fresh/report.md
python experiments/prospective_nuisance_shift/make_figure.py --out-dir build/prospective_shift_v2_fresh --stem waterbirds_prospective_shift_v2_fresh
```

Retain each `runs/*/checkpoint.pt`, `scores.npz`, `diagnostics.npz`, `selection.json`, `shift_results.json`, and `flip_records.npz`, together with the frozen selection, paired manifest/generation record, and final summaries. Fresh checkpoints need not have the historical hashes; each evaluation must match its own frozen selection. The paper's checkpoint identifiers and original outputs remain under `artifacts/prospective_v2/`.

## Corrected hidden-slice discovery and repair: seeds 17–21

Download/validate Waterbirds with the portable command above **before** invoking the hidden-slice launcher. Its preflight checks raw data before preparation, so setting its legacy `DOWNLOAD=1` alone does not bootstrap an empty machine. CUB mattes are not required for hidden discovery.

Preparation extracts 576-dimensional ImageNet MobileNetV3-small features and the fixed CLIP `openai/clip-vit-base-patch32` prompt-direction score from raw images. The complete extraction implementations and fixed prompt ensemble are in `prepare_data.py`; precomputed/private feature files are not required. `ALLOW_MODEL_DOWNLOAD=1` allows public pretrained model retrieval. An immutable CLIP revision can be supplied through `CLIP_REVISION`; the retained numerical cohort does not include its preparation manifest/resolved revision, so no exact historical revision is invented here. New preparation records the resolved revision, cache hashes, and package versions.

```bash
ROOT_DIR=data/wilds OUT_DIR=build/results/waterbirds_hidden_slice_v2_fresh \
  DEVICE=cuda ALLOW_MODEL_DOWNLOAD=1 DOWNLOAD=0 \
  SEEDS="17 18 19 20 21" FOLDS=5 GATE_RESTARTS=10 \
  HEAD_STEPS=1200 HEAD_LR=0.01 GATE_STEPS=1000 GATE_LR=0.03 \
  MAIN_MASS=0.10 MASSES="0.05 0.10 0.20" \
  MAIN_SHRINKAGE=0.10 SHRINKAGES="0.05 0.10 0.20" \
  BOOTSTRAP_REPLICATES=2000 RANDOM_SLICES=200 DETERMINISTIC=1 \
  bash experiments/waterbirds_hidden_slice/run_all.sh
```

The equivalent PowerShell launcher supports the same environment variables:

```powershell
$env:ROOT_DIR = "data/wilds"
$env:OUT_DIR = "build/results/waterbirds_hidden_slice_v2_fresh"
$env:DEVICE = "cuda"
$env:ALLOW_MODEL_DOWNLOAD = "1"
$env:DOWNLOAD = "0"
$env:SEEDS = "17 18 19 20 21"
& experiments/waterbirds_hidden_slice/run_all.ps1
```

Other defaults match the retained corrected-v2 protocol: calibration fraction 0.15, head/gate weight decay 0.0001, nuisance ridge 0.001, epsilon 1e-8, ESS fraction 0.8, weight-cap multiplier 2, repair loss grid 0.1/0.3/1, Fisher grid 0.01/0.03/0.1, and 0.02 overall/balanced/per-class accuracy budgets. `artifacts/hidden_slice_corrected_v2/blind_protocol.json` and per-seed repair/certification protocols record the historical settings.

The launcher executes preparation, discovery, repair, certification, post-hoc metadata reveal, aggregation, report, and figure stages. Preparation creates exactly `row_id,h,y,b_vl` in each blind split plus separately sealed metadata. Test certification follows frozen gates, and post-hoc metadata access follows all requested seeds' blind certification. The source also supports group-free fold-gate persistence; its default is off in the primary pipeline. Accordingly, fresh source digests differ from historical retained records. Use fresh outputs rather than bypassing integrity checks to mix cohorts.

Keep the generated `prepared/` inputs/manifests, `blind_protocol.json`, per-seed discovery/repair/certification/posthoc records and score arrays, and aggregate/main-table/paired-comparison/sensitivity outputs. To regenerate reports from a complete fresh run:

```shell
python experiments/waterbirds_hidden_slice/aggregate.py --out-dir build/results/waterbirds_hidden_slice_v2_fresh --seeds 17 18 19 20 21
python experiments/waterbirds_hidden_slice/make_report.py --out-dir build/results/waterbirds_hidden_slice_v2_fresh
```

## Checks performed for this release

- All 20 primary pipeline Python command-line entrypoints successfully parsed `--help`.
- All 39 hidden-slice unit tests passed, covering blind boundaries, geometry, gate constraints/gradients, repairs, and reporting. Run `python -m unittest discover -s experiments/waterbirds_hidden_slice/tests -v`.
- The portable downloader's five fixture tests passed, including incomplete data, wrong counts, path escape/symlink rejection, and refusal to overwrite an existing dataset. Run `python -m unittest discover -s scripts/tests -p test_prepare_waterbirds.py -v`.
- Stanford returned HTTP 206 for a range request, reported the expected 489,698,023-byte archive, and its downloaded prefix was opened as gzip/tar with the expected dataset/species/JPEG paths. Caltech masks and public MobileNet/CLIP endpoints responded successfully. This verifies acquisition access and the extractor on fixtures; a complete new raw-image/GPU run was not performed for this release audit.
- Table D.7 was fully refit from retained original numeric donor rows; all 15 seed/variant metrics agreed with the archive to maximum absolute difference 2.73e-12.

The code needed for a new primary run is included; external dataset/model downloads and GPU computation remain the reader's execution requirements. These checks do not claim a newly completed five-seed training or hidden-discovery run.
