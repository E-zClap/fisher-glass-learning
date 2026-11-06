#!/usr/bin/env bash
# End-to-end driver for the prospective Waterbirds nuisance-shift experiment.
#
# Run from the repository root:
#
#     bash experiments/prospective_nuisance_shift/run_all.sh --smoke      # ~10 min, validates the pipeline
#     bash experiments/prospective_nuisance_shift/run_all.sh              # the full experiment
#
# Pipeline
#   0. download Waterbirds + CUB mattes                     (download_data.sh)
#   1. preflight: versions, imports, data, compositor       (preflight.py)
#   2. cache matte statistics                               (mask_stats.py)
#   3. fine-tune end to end, several seeds and lambdas      (train_end_to_end.py)
#   4. validation-only slice + lambda selection, then FREEZE (select_slice.py)
#   5. build paired background interventions on held-out images (build_paired_images.py)
#   6. score the frozen models on the frozen images         (evaluate_shift.py)
#   7. unpaired natural-image cross-check                   (natural_shift_fallback.py)
#   8. figures                                              (make_figure.py)
#   9. markdown report + filled rebuttal paragraph          (make_report.py)
#
# Steps 3-9 are resumable: rerunning skips completed training runs.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"

# ---- defaults (full experiment) -------------------------------------------- #
OUT_DIR="build/prospective_shift"
ARCH="mobilenet_v3_small"
SEEDS="17 18 19 20 21"
METHODS="erm fisher_glass"
FG_LAMBDAS="0.003 0.01 0.03"
EPOCHS=20
BATCH_SIZE=32
EVAL_BATCH_SIZE=128
LR=1e-4
DEVICE="auto"
NUM_WORKERS=8
# Each compose worker caches full-resolution donor backgrounds, so cap the
# default rather than taking every core on a shared node. Raise with
# --compose-workers if you have the machine to yourself.
COMPOSE_WORKERS="$(( $(nproc 2>/dev/null || echo 4) - 2 ))"
[[ "${COMPOSE_WORKERS}" -gt 16 ]] && COMPOSE_WORKERS=16
[[ "${COMPOSE_WORKERS}" -lt 1 ]] && COMPOSE_WORKERS=1
DRAWS=2
MAX_PER_GROUP=0
# NB: do not call this GROUPS -- that is a bash builtin array holding the
# current user's Unix group ids, and assignments to it are silently ignored.
IMAGE_GROUPS="0 1 2 3"
N_BOOT=2000
N_BINS=4
MIN_BIN=100
CONFIDENCE=0.9
REPORT="reports/prospective_nuisance_shift_report.md"
FIGURE_STEM="waterbirds_prospective_shift"
SKIP_DOWNLOAD=0
SMOKE=0
REUSE_IMAGES=""
SELECTION_STATISTIC="projected_over_raw"
MANIFEST_ARG=()

usage() {
  sed -n '2,22p' "$0"
  cat <<'EOF'

Options:
  --smoke                 fast validation run (1 seed, 1 epoch, 40 birds/group, separate out-dir)
  --skip-download         assume the data is already in place
  --arch NAME             mobilenet_v3_small (default) | resnet50
  --seeds "17 18 19"      training seeds
  --methods "erm fisher_glass"
  --fg-lambdas "0.01 0.03 0.1"
  --epochs N              training epochs (default 20)
  --batch-size N          training batch size (default 32)
  --lr F                  learning rate (default 1e-4)
  --device auto|cuda|cpu  default auto
  --num-workers N         dataloader workers (default 8)
  --compose-workers N     processes used to generate intervention images
  --draws N               background draws per bird per arm (default 2)
  --max-per-group N       cap held-out birds per group (0 = all)
  --groups "0 1 2 3"      which held-out groups to generate images for
  --out-dir PATH          artifact directory (default build/prospective_shift)
  --report PATH           markdown report path
  --figure-stem NAME      figure filename stem
  --reuse-images PATH     reuse an earlier run's paired_manifest.csv instead of regenerating
                          (the images depend only on the dataset, never on a model)
  --selection-statistic S projected_over_raw (default) | projected_fisher
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) SMOKE=1; shift ;;
    --skip-download) SKIP_DOWNLOAD=1; shift ;;
    --arch) ARCH="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --methods) METHODS="$2"; shift 2 ;;
    --fg-lambdas) FG_LAMBDAS="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --compose-workers) COMPOSE_WORKERS="$2"; shift 2 ;;
    --draws) DRAWS="$2"; shift 2 ;;
    --max-per-group) MAX_PER_GROUP="$2"; shift 2 ;;
    --groups) IMAGE_GROUPS="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --report) REPORT="$2"; shift 2 ;;
    --reuse-images) REUSE_IMAGES="$2"; shift 2 ;;
    --selection-statistic) SELECTION_STATISTIC="$2"; shift 2 ;;
    --figure-stem) FIGURE_STEM="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ "${SMOKE}" -eq 1 ]]; then
  OUT_DIR="build/prospective_shift_smoke"
  SEEDS="17"
  FG_LAMBDAS="0.03"
  EPOCHS=1
  DRAWS=1
  MAX_PER_GROUP=40
  N_BOOT=200
  REPORT="reports/prospective_nuisance_shift_smoke_report.md"
  FIGURE_STEM="waterbirds_prospective_shift_smoke"
  echo "### SMOKE MODE: results are meaningless, this only checks the pipeline runs ###"
fi

echo "=============================================================="
echo " prospective Waterbirds nuisance-shift experiment"
echo " out-dir       : ${OUT_DIR}"
echo " arch          : ${ARCH}"
echo " seeds         : ${SEEDS}"
echo " methods       : ${METHODS}   lambdas: ${FG_LAMBDAS}"
echo " epochs        : ${EPOCHS}    batch: ${BATCH_SIZE}   lr: ${LR}"
echo " device        : ${DEVICE}    dataloader workers: ${NUM_WORKERS}"
echo " compose procs : ${COMPOSE_WORKERS}   draws/arm: ${DRAWS}"
echo "=============================================================="

# ---- 0. data --------------------------------------------------------------- #
if [[ "${SKIP_DOWNLOAD}" -eq 0 ]]; then
  echo "### [0/9] downloading data"
  bash "${HERE}/download_data.sh"
else
  echo "### [0/9] skipping download"
fi

# ---- 1. preflight ---------------------------------------------------------- #
echo
echo "### [1/9] preflight: versions, imports, data layout, compositor"
${PYTHON} -u "${HERE}/preflight.py" --out-dir "${OUT_DIR}"

# ---- 2. matte statistics --------------------------------------------------- #
echo
echo "### [2/9] caching CUB matte statistics"
${PYTHON} -u "${HERE}/mask_stats.py" --cache "${OUT_DIR}/matte_stats.npz"

# ---- 3. training ----------------------------------------------------------- #
echo
echo "### [3/9] end-to-end training"
${PYTHON} -u "${HERE}/train_end_to_end.py" \
  --out-dir "${OUT_DIR}" \
  --arch "${ARCH}" \
  --methods ${METHODS} \
  --fg-lambdas ${FG_LAMBDAS} \
  --seeds ${SEEDS} \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --lr "${LR}" \
  --device "${DEVICE}" \
  --num-workers "${NUM_WORKERS}"

# ---- 4. selection, then freeze --------------------------------------------- #
echo
echo "### [4/9] validation-only selection (freezes checkpoint, temperature, probe, slice)"
${PYTHON} -u "${HERE}/select_slice.py" \
  --out-dir "${OUT_DIR}" \
  --selection-statistic "${SELECTION_STATISTIC}" \
  --n-bins "${N_BINS}" \
  --min-bin "${MIN_BIN}"

# ---- 5. paired intervention images ----------------------------------------- #
# The images depend only on the dataset, never on a model, so a previous run's
# manifest can be reused verbatim across seeds and architectures.
echo
if [[ -n "${REUSE_IMAGES}" ]]; then
  echo "### [5/9] reusing existing intervention images: ${REUSE_IMAGES}"
  if [[ ! -f "${REUSE_IMAGES}" ]]; then
    echo "ERROR: manifest not found: ${REUSE_IMAGES}" >&2
    exit 1
  fi
  MANIFEST_ARG=(--manifest "${REUSE_IMAGES}")
  mkdir -p "${OUT_DIR}"
  cp "$(dirname "${REUSE_IMAGES}")/paired_generation.json" "${OUT_DIR}/paired_generation.json"
else
  echo "### [5/9] building paired background interventions"
  ${PYTHON} -u "${HERE}/build_paired_images.py" \
    --out-dir "${OUT_DIR}" \
    --groups ${IMAGE_GROUPS} \
    --draws "${DRAWS}" \
    --max-per-group "${MAX_PER_GROUP}" \
    --workers "${COMPOSE_WORKERS}"
  MANIFEST_ARG=()
fi

# ---- 6. prospective evaluation --------------------------------------------- #
echo
echo "### [6/9] scoring the frozen models on the frozen images"
${PYTHON} -u "${HERE}/evaluate_shift.py" \
  --out-dir "${OUT_DIR}" \
  ${MANIFEST_ARG[@]+"${MANIFEST_ARG[@]}"} \
  --confidence-threshold "${CONFIDENCE}" \
  --n-boot "${N_BOOT}" \
  --device "${DEVICE}" \
  --num-workers "${NUM_WORKERS}" \
  --batch-size "${EVAL_BATCH_SIZE}"

# ---- 7. natural-image cross-check + individual-level predictive validity ---- #
echo
echo "### [7/9] unpaired natural-image cross-check"
${PYTHON} -u "${HERE}/natural_shift_fallback.py" \
  --out-dir "${OUT_DIR}" \
  --confidence-threshold "${CONFIDENCE}" \
  --n-boot "${N_BOOT}"

echo
echo "### [7b/9] does the Fisher geometry predict individual flips beyond loss and confidence?"
${PYTHON} -u "${HERE}/predictive_validity.py" --out-dir "${OUT_DIR}"

# ---- 8. figures ------------------------------------------------------------ #
echo
echo "### [8/9] figures"
${PYTHON} -u "${HERE}/make_figure.py" --out-dir "${OUT_DIR}" --stem "${FIGURE_STEM}"

# ---- 9. report ------------------------------------------------------------- #
echo
echo "### [9/9] report"
${PYTHON} -u "${HERE}/make_report.py" --out-dir "${OUT_DIR}" --report "${REPORT}"

echo
echo "=============================================================="
echo " done."
echo
echo " Send these back:"
echo "   ${REPORT}"
echo "   ${OUT_DIR}/frozen_selection.json"
echo "   ${OUT_DIR}/shift_summary.json"
echo "   ${OUT_DIR}/natural_shift_summary.json"
echo "   ${OUT_DIR}/paired_generation.json"
echo "   tex/figures/${FIGURE_STEM}.pdf"
echo "   ${OUT_DIR}/previews/*.jpg      (visual check of the interventions)"
echo
echo " Small tarball with everything except the generated images:"
echo "   tar czf prospective_shift_results.tar.gz \\"
echo "     ${REPORT} tex/figures/${FIGURE_STEM}*.pdf ${OUT_DIR}/previews \\"
echo "     ${OUT_DIR}/*.json ${OUT_DIR}/*.csv ${OUT_DIR}/runs/*/run.json \\"
echo "     ${OUT_DIR}/runs/*/selection.json ${OUT_DIR}/runs/*/shift_results.json \\"
echo "     ${OUT_DIR}/runs/*/natural_shift.json"
echo "=============================================================="
