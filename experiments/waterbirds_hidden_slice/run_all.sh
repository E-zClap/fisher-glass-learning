#!/usr/bin/env bash
# End-to-end launcher for metadata-blind Waterbirds hidden-slice discovery.
# Shared arguments are centralized below and may be overridden by environment
# variables, e.g. DEVICE=cuda SEEDS="17 18" bash run_all.sh.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"

if ! PYTHON_VERSION="$("${PYTHON}" -c 'import platform, sys; print(platform.python_version()); raise SystemExit(0 if sys.version_info >= (3, 8) else 2)')"; then
  echo "ERROR: ${PYTHON} is Python ${PYTHON_VERSION:-unknown}; Python 3.8 or newer is required." >&2
  echo "Set PYTHON to a compatible interpreter, for example PYTHON=python3.10." >&2
  exit 2
fi

# Paths and runtime.
ROOT_DIR="${ROOT_DIR:-${DATA_ROOT:-data/wilds}}"
OUT_DIR="${OUT_DIR:-build/results/waterbirds_hidden_slice_v2}"
DATA_DIR="${DATA_DIR:-${PREPARED_DIR:-${OUT_DIR}/prepared}}"
IMAGE_ROOT="${IMAGE_ROOT:-${ROOT_DIR}/waterbirds_v1.0}"
FEATURE_CACHE="${FEATURE_CACHE:-build/features/waterbirds_mobilenetv3_small_attested.npz}"
NUISANCE_CACHE="${NUISANCE_CACHE:-build/features/waterbirds_clip_vit_base_patch32_nuisance.npz}"
DEVICE="${DEVICE:-auto}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PREPARE_SEED="${PREPARE_SEED:-17}"
DOWNLOAD="${DOWNLOAD:-0}"
ALLOW_MODEL_DOWNLOAD="${ALLOW_MODEL_DOWNLOAD:-0}"
CLIP_REVISION="${CLIP_REVISION:-}"
AMP="${AMP:-1}"
PIN_MEMORY="${PIN_MEMORY:-1}"
DETERMINISTIC="${DETERMINISTIC:-1}"

# Frozen scientific protocol.
SEEDS="${SEEDS:-17 18 19 20 21}"
FOLDS="${FOLDS:-5}"
GATE_RESTARTS="${GATE_RESTARTS:-${RESTARTS:-10}}"
MAIN_MASS="${MAIN_MASS:-0.10}"
MASSES="${MASSES:-0.05 0.10 0.20}"
MAIN_SHRINKAGE="${MAIN_SHRINKAGE:-0.10}"
SHRINKAGES="${SHRINKAGES:-0.05 0.10 0.20}"
CLIP_MODEL="${CLIP_MODEL:-openai/clip-vit-base-patch32}"
RIDGE="${RIDGE:-0.001}"
EPSILON="${EPSILON:-${EPS:-1e-8}}"
ESS_FRACTION="${ESS_FRACTION:-${MIN_ESS_FRACTION:-0.8}}"
CAP_MULTIPLIER="${CAP_MULTIPLIER:-${MAX_WEIGHT_MULTIPLIER:-2.0}}"

# Shared head, gate, repair, and inference settings.
CALIBRATION_FRACTION="${CALIBRATION_FRACTION:-0.15}"
HEAD_STEPS="${HEAD_STEPS:-1200}"
HEAD_LR="${HEAD_LR:-0.01}"
HEAD_WEIGHT_DECAY="${HEAD_WEIGHT_DECAY:-0.0001}"
GATE_STEPS="${GATE_STEPS:-1000}"
GATE_LR="${GATE_LR:-0.03}"
LOSS_LAMBDAS="${LOSS_LAMBDAS:-0.1 0.3 1.0}"
FISHER_LAMBDAS="${FISHER_LAMBDAS:-0.01 0.03 0.1}"
COMBINED_LOSS_LAMBDAS="${COMBINED_LOSS_LAMBDAS:-${LOSS_LAMBDAS}}"
COMBINED_FISHER_LAMBDAS="${COMBINED_FISHER_LAMBDAS:-${FISHER_LAMBDAS}}"
ACCURACY_BUDGET="${ACCURACY_BUDGET:-0.02}"
BALANCED_ACCURACY_BUDGET="${BALANCED_ACCURACY_BUDGET:-0.02}"
PER_CLASS_ACCURACY_BUDGET="${PER_CLASS_ACCURACY_BUDGET:-0.02}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-12}"
EARLY_STOPPING_EVAL_EVERY="${EARLY_STOPPING_EVAL_EVERY:-25}"
EARLY_STOPPING_MIN_DELTA="${EARLY_STOPPING_MIN_DELTA:-1e-5}"
BOOTSTRAP_REPLICATES="${BOOTSTRAP_REPLICATES:-${N_BOOT:-2000}}"
BOOTSTRAP_CONFIDENCE="${BOOTSTRAP_CONFIDENCE:-0.95}"
RANDOM_SLICES="${RANDOM_SLICES:-200}"
AGGREGATE_CONFIDENCE="${AGGREGATE_CONFIDENCE:-0.95}"
FIGURE_SEED="${FIGURE_SEED:-17}"
TOP_K="${TOP_K:-6}"
RESUME="${RESUME:-1}"

# Preparation batch size. It does not change the predeclared headline row.
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"

SMOKE=0
usage() {
  cat <<'EOF'
Usage: bash experiments/waterbirds_hidden_slice/run_all.sh [--smoke]

Environment variables centralize configuration; see README.md for the full
list. --smoke validates plumbing only and produces no reportable results.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) SMOKE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "${SMOKE}" -eq 1 ]]; then
  OUT_DIR="${SMOKE_OUT_DIR:-build/results/waterbirds_hidden_slice_v2_smoke}"
  DATA_DIR="${SMOKE_DATA_DIR:-${OUT_DIR}/prepared}"
  SEEDS="17"
  FOLDS="2"
  GATE_RESTARTS="1"
  MASSES="0.10"
  SHRINKAGES="0.10"
  HEAD_STEPS="${SMOKE_HEAD_STEPS:-20}"
  GATE_STEPS="${SMOKE_GATE_STEPS:-20}"
  BOOTSTRAP_REPLICATES="${SMOKE_BOOTSTRAP_REPLICATES:-100}"
  RANDOM_SLICES="${SMOKE_RANDOM_SLICES:-5}"
  echo "WARNING: SMOKE MODE checks plumbing only; its scientific outputs are meaningless."
fi

read -r -a SEED_ARGS <<< "${SEEDS}"
read -r -a MASS_ARGS <<< "${MASSES}"
read -r -a SHRINKAGE_ARGS <<< "${SHRINKAGES}"
read -r -a LOSS_LAMBDA_ARGS <<< "${LOSS_LAMBDAS}"
read -r -a FISHER_LAMBDA_ARGS <<< "${FISHER_LAMBDAS}"
read -r -a COMBINED_LOSS_ARGS <<< "${COMBINED_LOSS_LAMBDAS}"
read -r -a COMBINED_FISHER_ARGS <<< "${COMBINED_FISHER_LAMBDAS}"

if [[ "${RESUME}" == "1" ]]; then
  RESUME_ARGS=(--resume)
else
  RESUME_ARGS=(--no-resume)
fi

run_stage() {
  local script="$1"
  shift
  echo
  echo "### ${script}"
  "${PYTHON}" -u "${HERE}/${script}" "$@"
}

echo "=============================================================="
echo " Waterbirds metadata-blind hidden-slice pipeline (corrected v2)"
echo " python        : ${PYTHON} (${PYTHON_VERSION})"
echo " out-dir       : ${OUT_DIR}"
echo " device        : ${DEVICE}"
echo " seeds/folds   : ${SEEDS} / ${FOLDS}"
echo " mass          : ${MAIN_MASS} (sweep: ${MASSES})"
echo " shrinkage     : ${MAIN_SHRINKAGE} (sweep: ${SHRINKAGES})"
echo " nuisances     : CLIP text (primary), random axis, oracle probe"
echo " gate restarts : ${GATE_RESTARTS}"
echo " deterministic : ${DETERMINISTIC}"
echo "=============================================================="

PREPARE_EXTRA_ARGS=()
if [[ -n "${CLIP_REVISION}" ]]; then PREPARE_EXTRA_ARGS+=(--clip-revision "${CLIP_REVISION}"); fi
if [[ "${DOWNLOAD}" == "1" ]]; then PREPARE_EXTRA_ARGS+=(--download); fi
if [[ "${ALLOW_MODEL_DOWNLOAD}" == "1" ]]; then PREPARE_EXTRA_ARGS+=(--allow-model-download); fi
if [[ "${AMP}" == "1" ]]; then PREPARE_EXTRA_ARGS+=(--amp); else PREPARE_EXTRA_ARGS+=(--no-amp); fi
if [[ "${PIN_MEMORY}" == "1" ]]; then PREPARE_EXTRA_ARGS+=(--pin-memory); else PREPARE_EXTRA_ARGS+=(--no-pin-memory); fi
if [[ "${DETERMINISTIC}" == "1" ]]; then DETERMINISM_ARGS=(--deterministic); else DETERMINISM_ARGS=(--no-deterministic); fi
PREPARE_EXTRA_ARGS+=("${DETERMINISM_ARGS[@]}")

PREFLIGHT_EXTRA_ARGS=()
if [[ -n "${CLIP_REVISION}" ]]; then PREFLIGHT_EXTRA_ARGS+=(--clip-revision "${CLIP_REVISION}"); fi
if [[ "${ALLOW_MODEL_DOWNLOAD}" == "1" ]]; then PREFLIGHT_EXTRA_ARGS+=(--allow-model-download); fi
run_stage preflight.py \
  --root-dir "${ROOT_DIR}" \
  --out-dir "${DATA_DIR}" \
  --feature-cache "${FEATURE_CACHE}" \
  --nuisance-cache "${NUISANCE_CACHE}" \
  --clip-model "${CLIP_MODEL}" \
  --device "${DEVICE}" \
  "${PREFLIGHT_EXTRA_ARGS[@]}"

run_stage prepare_data.py \
  --root-dir "${ROOT_DIR}" \
  --out-dir "${DATA_DIR}" \
  --feature-cache "${FEATURE_CACHE}" \
  --nuisance-cache "${NUISANCE_CACHE}" \
  --clip-model "${CLIP_MODEL}" \
  --seed "${PREPARE_SEED}" \
  --batch-size "${EVAL_BATCH_SIZE}" \
  --device "${DEVICE}" \
  --num-workers "${NUM_WORKERS}" \
  "${PREPARE_EXTRA_ARGS[@]}"

run_stage discover.py \
  --data-dir "${DATA_DIR}" \
  --out-dir "${OUT_DIR}" \
  --seeds "${SEED_ARGS[@]}" \
  --device "${DEVICE}" \
  --calibration-fraction "${CALIBRATION_FRACTION}" \
  --head-steps "${HEAD_STEPS}" \
  --head-lr "${HEAD_LR}" \
  --head-weight-decay "${HEAD_WEIGHT_DECAY}" \
  --folds "${FOLDS}" \
  --gate-steps "${GATE_STEPS}" \
  --gate-lr "${GATE_LR}" \
  --gate-restarts "${GATE_RESTARTS}" \
  --slice-mass "${MAIN_MASS}" \
  --mass-sensitivity "${MASS_ARGS[@]}" \
  --shrinkage "${MAIN_SHRINKAGE}" \
  --shrinkage-sensitivity "${SHRINKAGE_ARGS[@]}" \
  --ess-fraction "${ESS_FRACTION}" \
  --cap-multiplier "${CAP_MULTIPLIER}" \
  --ridge "${RIDGE}" \
  --epsilon "${EPSILON}" \
  "${DETERMINISM_ARGS[@]}" \
  "${RESUME_ARGS[@]}"

run_stage repair.py \
  --data-dir "${DATA_DIR}" \
  --out-dir "${OUT_DIR}" \
  --seeds "${SEED_ARGS[@]}" \
  --device "${DEVICE}" \
  --loss-lambdas "${LOSS_LAMBDA_ARGS[@]}" \
  --fisher-lambdas "${FISHER_LAMBDA_ARGS[@]}" \
  --combined-loss-lambdas "${COMBINED_LOSS_ARGS[@]}" \
  --combined-fisher-lambdas "${COMBINED_FISHER_ARGS[@]}" \
  --accuracy-budget "${ACCURACY_BUDGET}" \
  --balanced-accuracy-budget "${BALANCED_ACCURACY_BUDGET}" \
  --per-class-accuracy-budget "${PER_CLASS_ACCURACY_BUDGET}" \
  --head-steps "${HEAD_STEPS}" \
  --head-lr "${HEAD_LR}" \
  --head-weight-decay "${HEAD_WEIGHT_DECAY}" \
  --early-stopping-patience "${EARLY_STOPPING_PATIENCE}" \
  --early-stopping-eval-every "${EARLY_STOPPING_EVAL_EVERY}" \
  --early-stopping-min-delta "${EARLY_STOPPING_MIN_DELTA}" \
  "${DETERMINISM_ARGS[@]}" \
  "${RESUME_ARGS[@]}"

run_stage certify.py \
  --data-dir "${DATA_DIR}" \
  --out-dir "${OUT_DIR}" \
  --seeds "${SEED_ARGS[@]}" \
  --bootstrap-replicates "${BOOTSTRAP_REPLICATES}" \
  --bootstrap-confidence "${BOOTSTRAP_CONFIDENCE}" \
  --random-slices "${RANDOM_SLICES}" \
  "${RESUME_ARGS[@]}"

# This is the first stage allowed to reveal metadata; oracle controls are produced here.
run_stage posthoc.py \
  --data-dir "${DATA_DIR}" \
  --out-dir "${OUT_DIR}" \
  --seeds "${SEED_ARGS[@]}" \
  --device "${DEVICE}" \
  --head-steps "${HEAD_STEPS}" \
  --head-lr "${HEAD_LR}" \
  --head-weight-decay "${HEAD_WEIGHT_DECAY}" \
  "${DETERMINISM_ARGS[@]}" \
  "${RESUME_ARGS[@]}"

run_stage aggregate.py \
  --out-dir "${OUT_DIR}" \
  --seeds "${SEED_ARGS[@]}" \
  --confidence "${AGGREGATE_CONFIDENCE}"

run_stage make_report.py \
  --out-dir "${OUT_DIR}" \
  --report "${OUT_DIR}/waterbirds_hidden_slice_report.md" \
  --latex "${OUT_DIR}/waterbirds_hidden_slice_table.tex" \
  --rebuttal "${OUT_DIR}/rebuttal_w3.txt"

if [[ "${SMOKE}" -eq 0 ]]; then
  run_stage make_figure.py \
    --data-dir "${DATA_DIR}" \
    --out-dir "${OUT_DIR}" \
    --image-root "${IMAGE_ROOT}" \
    --seed "${FIGURE_SEED}" \
    --top-k "${TOP_K}"
else
  echo
  echo "### make_figure.py (skipped: incomplete smoke runs cannot emit paper figures)"
fi

echo
if [[ "${SMOKE}" -eq 0 ]]; then
  echo "Done. Review ${OUT_DIR}/aggregate.json, ${OUT_DIR}/main_table.csv,"
  echo "${OUT_DIR}/waterbirds_hidden_slice_report.md, ${OUT_DIR}/rebuttal_w3.txt, and the figure files."
else
  echo "Smoke check done. Outputs under ${OUT_DIR} are diagnostic-only; no paper figure was created."
fi
