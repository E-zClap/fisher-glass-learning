#!/usr/bin/env bash
# End-to-end metadata-blind Fisher-Glass lambda-selection audit.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"
cd "${REPO_ROOT}"
PYTHON="${PYTHON:-python}"

if ! PYTHON_VERSION="$("${PYTHON}" -c 'import platform, sys; print(platform.python_version()); raise SystemExit(0 if sys.version_info >= (3, 9) else 2)')"; then
  echo "ERROR: ${PYTHON} is Python ${PYTHON_VERSION:-unknown}; Python 3.9 or newer is required." >&2
  echo "Activate the project environment or set PYTHON=/absolute/path/to/.venv/bin/python." >&2
  exit 2
fi

# This experiment consumes the already prepared, exact-schema blind bundle.
# It never downloads a model or dataset. Corrected runs normally use the v2
# path, while older transferred audit bundles used the unsuffixed path.
if [[ -n "${DATA_DIR:-}" ]]; then
  DATA_DIR="${DATA_DIR}"
elif [[ -f "build/results/waterbirds_hidden_slice_v2/prepared/preparation_manifest.json" ]]; then
  DATA_DIR="build/results/waterbirds_hidden_slice_v2/prepared"
elif [[ -f "build/results/waterbirds_hidden_slice/prepared/preparation_manifest.json" ]]; then
  DATA_DIR="build/results/waterbirds_hidden_slice/prepared"
else
  DATA_DIR="build/results/waterbirds_hidden_slice_v2/prepared"
fi
OUT_DIR="${OUT_DIR:-build/results/waterbirds_group_free_lambda_v2}"
DEVICE="${DEVICE:-cuda}"
SEEDS="${SEEDS:-17 18 19 20 21}"
LAMBDAS="${LAMBDAS:-0 0.01 0.03 0.1 0.3}"
# Frozen reference required by the reporting schema; intentionally not an
# environment override.
FIXED_LAMBDA="0.03"
FOLDS="${FOLDS:-5}"
# Frozen to match the simultaneous selection and reporting convention.
CONFIDENCE="0.95"

FIT_FRACTION="${FIT_FRACTION:-0.70}"
CALIBRATION_FRACTION="${CALIBRATION_FRACTION:-0.15}"
GATE_DISCOVERY_FRACTION="${GATE_DISCOVERY_FRACTION:-0.15}"
ACCURACY_DIAGNOSTIC_TOLERANCE="${ACCURACY_DIAGNOSTIC_TOLERANCE:-0.02}"
HEAD_STEPS="${HEAD_STEPS:-1200}"
HEAD_LR="${HEAD_LR:-0.01}"
HEAD_WEIGHT_DECAY="${HEAD_WEIGHT_DECAY:-0.0001}"
GRADIENT_CLIP="${GRADIENT_CLIP:-10.0}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-12}"
EARLY_STOPPING_EVAL_EVERY="${EARLY_STOPPING_EVAL_EVERY:-25}"
EARLY_STOPPING_MIN_DELTA="${EARLY_STOPPING_MIN_DELTA:-1e-5}"
GATE_STEPS="${GATE_STEPS:-1000}"
GATE_LR="${GATE_LR:-0.03}"
GATE_RESTARTS="${GATE_RESTARTS:-10}"
GATE_WEIGHT_DECAY="${GATE_WEIGHT_DECAY:-0.0001}"
GATE_MASS_PENALTY="${GATE_MASS_PENALTY:-10.0}"
SLICE_MASS="${SLICE_MASS:-0.10}"
SHRINKAGE="${SHRINKAGE:-0.10}"
ESS_FRACTION="${ESS_FRACTION:-0.8}"
CAP_MULTIPLIER="${CAP_MULTIPLIER:-2.0}"
RIDGE="${RIDGE:-0.001}"
EPSILON="${EPSILON:-1e-8}"
BOOTSTRAP_REPLICATES="${BOOTSTRAP_REPLICATES:-2000}"
BOOTSTRAP_CONFIDENCE="${BOOTSTRAP_CONFIDENCE:-0.95}"
DETERMINISTIC="${DETERMINISTIC:-1}"
RESUME="${RESUME:-1}"

SMOKE=0
usage() {
  cat <<'EOF'
Usage: bash experiments/waterbirds_group_free_lambda/run_all.sh [--smoke]

The default is the complete five-seed sensitivity protocol. Environment variables above
may override paths/runtime settings. --smoke checks plumbing with one seed,
two folds, and tiny optimizers; its numbers must not enter a paper or rebuttal.
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
  OUT_DIR="${SMOKE_OUT_DIR:-build/results/waterbirds_group_free_lambda_v2_smoke}"
  SEEDS="17"
  LAMBDAS="0 0.03"
  FOLDS="2"
  HEAD_STEPS="${SMOKE_HEAD_STEPS:-20}"
  GATE_STEPS="${SMOKE_GATE_STEPS:-20}"
  GATE_RESTARTS="${SMOKE_GATE_RESTARTS:-1}"
  BOOTSTRAP_REPLICATES="${SMOKE_BOOTSTRAP_REPLICATES:-100}"
  echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
  echo "!! SMOKE MODE: DIAGNOSTIC PLUMBING ONLY; NOT REPORTABLE.    !!"
  echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
fi

read -r -a SEED_ARGS <<< "${SEEDS}"
read -r -a LAMBDA_ARGS <<< "${LAMBDAS}"

if [[ "${DETERMINISTIC}" == "1" ]]; then
  DETERMINISM_ARGS=(--deterministic)
else
  DETERMINISM_ARGS=(--no-deterministic)
fi
if [[ "${RESUME}" == "1" ]]; then
  RESUME_ARGS=(--resume)
else
  RESUME_ARGS=(--no-resume)
fi
MODE_ARGS=()
if [[ "${SMOKE}" -eq 1 ]]; then MODE_ARGS+=(--smoke); fi

run_stage() {
  local script="$1"
  shift
  echo
  echo "### ${script}"
  "${PYTHON}" -u "${HERE}/${script}" "$@"
}

echo "=============================================================="
echo " Waterbirds group-free Fisher-Glass lambda-selection audit (schema v2)"
echo " python        : ${PYTHON} (${PYTHON_VERSION})"
echo " prepared data : ${DATA_DIR}"
echo " out-dir       : ${OUT_DIR}"
echo " device        : ${DEVICE}"
echo " seeds/folds   : ${SEEDS} / ${FOLDS}"
echo " lambda grid   : ${LAMBDAS} (matched lambda=0; fixed reference ${FIXED_LAMBDA})"
echo " ERM control   : frozen pre-continuation reference reported separately"
echo " selection     : one-SE task risk, then adjusted hidden-certificate lower score"
echo " analysis      : post-hoc matched-control sensitivity; not fresh held-out confirmation"
echo " downloads     : disabled (this launcher has no download path)"
echo "=============================================================="

run_stage preflight.py --data-dir "${DATA_DIR}" --device "${DEVICE}"

run_stage select_lambda.py \
  --data-dir "${DATA_DIR}" \
  --out-dir "${OUT_DIR}" \
  --seeds "${SEED_ARGS[@]}" \
  --lambdas "${LAMBDA_ARGS[@]}" \
  --fixed-lambda "${FIXED_LAMBDA}" \
  --folds "${FOLDS}" \
  --confidence "${CONFIDENCE}" \
  --device "${DEVICE}" \
  --fit-fraction "${FIT_FRACTION}" \
  --calibration-fraction "${CALIBRATION_FRACTION}" \
  --gate-discovery-fraction "${GATE_DISCOVERY_FRACTION}" \
  --accuracy-diagnostic-tolerance "${ACCURACY_DIAGNOSTIC_TOLERANCE}" \
  --head-steps "${HEAD_STEPS}" \
  --head-lr "${HEAD_LR}" \
  --head-weight-decay "${HEAD_WEIGHT_DECAY}" \
  --gradient-clip "${GRADIENT_CLIP}" \
  --early-stopping-patience "${EARLY_STOPPING_PATIENCE}" \
  --early-stopping-eval-every "${EARLY_STOPPING_EVAL_EVERY}" \
  --early-stopping-min-delta "${EARLY_STOPPING_MIN_DELTA}" \
  --gate-steps "${GATE_STEPS}" \
  --gate-lr "${GATE_LR}" \
  --gate-restarts "${GATE_RESTARTS}" \
  --gate-weight-decay "${GATE_WEIGHT_DECAY}" \
  --gate-mass-penalty "${GATE_MASS_PENALTY}" \
  --slice-mass "${SLICE_MASS}" \
  --shrinkage "${SHRINKAGE}" \
  --ess-fraction "${ESS_FRACTION}" \
  --cap-multiplier "${CAP_MULTIPLIER}" \
  --ridge "${RIDGE}" \
  --epsilon "${EPSILON}" \
  "${DETERMINISM_ARGS[@]}" \
  "${RESUME_ARGS[@]}"

run_stage certify.py \
  --data-dir "${DATA_DIR}" \
  --out-dir "${OUT_DIR}" \
  --seeds "${SEED_ARGS[@]}" \
  --bootstrap-replicates "${BOOTSTRAP_REPLICATES}" \
  --bootstrap-confidence "${BOOTSTRAP_CONFIDENCE}" \
  "${MODE_ARGS[@]}" \
  "${RESUME_ARGS[@]}"

# This is the first and only stage allowed to deserialize group metadata.
run_stage posthoc.py \
  --data-dir "${DATA_DIR}" \
  --out-dir "${OUT_DIR}" \
  --seeds "${SEED_ARGS[@]}" \
  "${MODE_ARGS[@]}" \
  "${RESUME_ARGS[@]}"

run_stage aggregate.py \
  --out-dir "${OUT_DIR}" \
  --seeds "${SEED_ARGS[@]}" \
  --confidence "${CONFIDENCE}" \
  "${MODE_ARGS[@]}"

echo
if [[ "${SMOKE}" -eq 1 ]]; then
  echo "SMOKE CHECK COMPLETE. Every output in ${OUT_DIR} is NON-REPORTABLE."
else
  echo "Complete. Review ${OUT_DIR}/report.md and ${OUT_DIR}/rebuttal_lambda_fg.txt."
  echo "Machine-readable tables: ${OUT_DIR}/main_table.csv and ${OUT_DIR}/frontier.csv."
fi
