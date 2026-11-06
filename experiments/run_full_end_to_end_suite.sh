#!/usr/bin/env bash
set -euo pipefail

# Rootless Linux/HPC launcher for the end-to-end Fisher-Glass benchmark suite.
# Run with:
#   bash experiments/run_full_end_to_end_suite.sh

ROOT_DIR="${ROOT_DIR:-data/wilds}"
PYTHON="${PYTHON:-python}"
OUTPUT="${OUTPUT:-build/results/end_to_end_deep_suite_results.json}"
LOG_PATH="${LOG_PATH:-build/results/end_to_end_deep_suite.log}"
BENCHMARKS="${BENCHMARKS:-waterbirds civilcomments celeba multinli}"
METHODS="${METHODS:-erm group_dro loss_cvar fisher_glass group_dro_fisher_glass}"
SEEDS="${SEEDS:-17 18 19 20 21}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-32}"
TEXT_BATCH_SIZE="${TEXT_BATCH_SIZE:-16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
TEXT_EVAL_BATCH_SIZE="${TEXT_EVAL_BATCH_SIZE:-32}"
DEVICE="${DEVICE:-cuda}"
RESUME="${RESUME:-1}"
FG_LAMBDA="${FG_LAMBDA:-0.03}"
RIDGE="${RIDGE:-1e-3}"
EPS="${EPS:-1e-8}"
ALPHA="${ALPHA:-0.95}"
NUM_WORKERS="${NUM_WORKERS:-8}"
TEXT_NUM_WORKERS="${TEXT_NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
PIN_MEMORY="${PIN_MEMORY:-1}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-1}"
AMP="${AMP:-1}"
TF32="${TF32:-1}"
COMPILE_MODEL="${COMPILE_MODEL:-0}"
CHANNELS_LAST="${CHANNELS_LAST:-1}"
DOWNLOAD="${DOWNLOAD:-1}"
ALLOW_MODEL_DOWNLOAD="${ALLOW_MODEL_DOWNLOAD:-1}"
AUDIT_BOOTSTRAP_REPLICATES="${AUDIT_BOOTSTRAP_REPLICATES:-100000}"
AUDIT_CONFIDENCE="${AUDIT_CONFIDENCE:-0.95}"
AUDIT_SEED="${AUDIT_SEED:-20260729}"

# Keep downloaded model/data caches inside the repository by default so no
# system-wide locations or root permissions are needed.
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export WILDS_CACHE_DIR="${WILDS_CACHE_DIR:-$PWD/.cache/wilds}"

mkdir -p "$(dirname "$OUTPUT")" "$(dirname "$LOG_PATH")" "$HF_HOME" "$TRANSFORMERS_CACHE" "$WILDS_CACHE_DIR" "$ROOT_DIR"

read -r -a BENCHMARK_ARGS <<< "$BENCHMARKS"
read -r -a METHOD_ARGS <<< "$METHODS"
read -r -a SEED_ARGS <<< "$SEEDS"

EXTRA_ARGS=()
if [[ -n "${MAX_TRAIN:-}" ]]; then EXTRA_ARGS+=(--max-train "$MAX_TRAIN"); fi
if [[ -n "${MAX_VAL:-}" ]]; then EXTRA_ARGS+=(--max-val "$MAX_VAL"); fi
if [[ -n "${MAX_TEST:-}" ]]; then EXTRA_ARGS+=(--max-test "$MAX_TEST"); fi
if [[ -n "${WATERBIRDS_MAX_TRAIN:-}" ]]; then EXTRA_ARGS+=(--waterbirds-max-train "$WATERBIRDS_MAX_TRAIN"); fi
if [[ -n "${WATERBIRDS_MAX_VAL:-}" ]]; then EXTRA_ARGS+=(--waterbirds-max-val "$WATERBIRDS_MAX_VAL"); fi
if [[ -n "${WATERBIRDS_MAX_TEST:-}" ]]; then EXTRA_ARGS+=(--waterbirds-max-test "$WATERBIRDS_MAX_TEST"); fi
if [[ -n "${CIVIL_MAX_TRAIN:-}" ]]; then EXTRA_ARGS+=(--civilcomments-max-train "$CIVIL_MAX_TRAIN"); fi
if [[ -n "${CIVIL_MAX_VAL:-}" ]]; then EXTRA_ARGS+=(--civilcomments-max-val "$CIVIL_MAX_VAL"); fi
if [[ -n "${CIVIL_MAX_TEST:-}" ]]; then EXTRA_ARGS+=(--civilcomments-max-test "$CIVIL_MAX_TEST"); fi
if [[ -n "${FREEZE_BACKBONE:-}" ]]; then EXTRA_ARGS+=(--freeze-backbone); fi
if [[ "$RESUME" != "0" ]]; then EXTRA_ARGS+=(--resume); fi
if [[ "$PIN_MEMORY" != "0" ]]; then EXTRA_ARGS+=(--pin-memory); fi
if [[ "$PERSISTENT_WORKERS" != "0" ]]; then EXTRA_ARGS+=(--persistent-workers); fi
if [[ "$AMP" != "0" ]]; then EXTRA_ARGS+=(--amp); fi
if [[ "$TF32" != "0" ]]; then EXTRA_ARGS+=(--tf32); fi
if [[ "$CHANNELS_LAST" != "0" ]]; then EXTRA_ARGS+=(--channels-last); fi
if [[ "$COMPILE_MODEL" != "0" ]]; then EXTRA_ARGS+=(--compile-model); fi
if [[ "$DOWNLOAD" != "0" ]]; then EXTRA_ARGS+=(--download); fi
if [[ "$ALLOW_MODEL_DOWNLOAD" != "0" ]]; then EXTRA_ARGS+=(--allow-model-download); fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] launching Fisher-Glass end-to-end suite"
echo "  root:       $ROOT_DIR"
echo "  output:     $OUTPUT"
echo "  log:        $LOG_PATH"
echo "  benchmarks: $BENCHMARKS"
echo "  methods:    $METHODS"
echo "  seeds:      $SEEDS"
echo "  epochs:     $EPOCHS"
echo "  device:     $DEVICE"
echo "  python:     $PYTHON"
echo "  resume:     $RESUME"
echo "  fg_lambda:  $FG_LAMBDA"
echo "  certificate: ridge=$RIDGE eps=$EPS alpha=$ALPHA"
echo "  workers:    image=$NUM_WORKERS text=$TEXT_NUM_WORKERS prefetch=$PREFETCH_FACTOR pin=$PIN_MEMORY persistent=$PERSISTENT_WORKERS"
echo "  precision:  amp_bf16=$AMP tf32=$TF32 channels_last=$CHANNELS_LAST compile=$COMPILE_MODEL"
echo "  audit:      draws=$AUDIT_BOOTSTRAP_REPLICATES confidence=$AUDIT_CONFIDENCE seed=$AUDIT_SEED"
echo "  downloads:  data=$DOWNLOAD models=$ALLOW_MODEL_DOWNLOAD"
if [[ -n "${CIVIL_MAX_TRAIN:-}${CIVIL_MAX_VAL:-}${CIVIL_MAX_TEST:-}" ]]; then
  echo "  civil caps: train=${CIVIL_MAX_TRAIN:-none} val=${CIVIL_MAX_VAL:-none} test=${CIVIL_MAX_TEST:-none}"
fi
echo

"$PYTHON" -u experiments/end_to_end_deep_suite.py \
  --root-dir "$ROOT_DIR" \
  --benchmarks "${BENCHMARK_ARGS[@]}" \
  --methods "${METHOD_ARGS[@]}" \
  --seeds "${SEED_ARGS[@]}" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --text-batch-size "$TEXT_BATCH_SIZE" \
  --eval-batch-size "$EVAL_BATCH_SIZE" \
  --text-eval-batch-size "$TEXT_EVAL_BATCH_SIZE" \
  --device "$DEVICE" \
  --num-workers "$NUM_WORKERS" \
  --text-num-workers "$TEXT_NUM_WORKERS" \
  --prefetch-factor "$PREFETCH_FACTOR" \
  --fg-lambda "$FG_LAMBDA" \
  --ridge "$RIDGE" \
  --eps "$EPS" \
  --alpha "$ALPHA" \
  --audit-bootstrap-replicates "$AUDIT_BOOTSTRAP_REPLICATES" \
  --audit-confidence "$AUDIT_CONFIDENCE" \
  --audit-seed "$AUDIT_SEED" \
  --output "$OUTPUT" \
  "${EXTRA_ARGS[@]}" 2>&1 | tee "$LOG_PATH"
