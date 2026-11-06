#!/usr/bin/env bash
set -euo pipefail

# One-command Linux/NVIDIA rerun for the W2 ridge-versus-exact audit.
# The default profile runs both real-data diagnostics and the five-seed frozen
# Waterbirds training sweep.  Set RUN_END_TO_END=1 for the six paper-matched
# end-to-end Waterbirds lambda runs.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python}"
ROOT_DIR="${ROOT_DIR:-data/wilds}"
OUT_DIR="${OUT_DIR:-build/results/w2_ridge_exact_audit}"
BUNDLE_PATH="${BUNDLE_PATH:-build/results/fisherglass_w2_audit_bundle.tar.gz}"
FEATURE_CACHE="${FEATURE_CACHE:-build/features/waterbirds_mobilenetv3_small.npz}"
DEVICE="${DEVICE:-cuda}"

RIDGE="${RIDGE:-1e-3}"
EPS="${EPS:-1e-8}"
ALPHA="${ALPHA:-0.95}"
AUDIT_BOOTSTRAP_REPLICATES="${AUDIT_BOOTSTRAP_REPLICATES:-100000}"
AUDIT_CONFIDENCE="${AUDIT_CONFIDENCE:-0.95}"
AUDIT_SEED="${AUDIT_SEED:-20260729}"

WATERBIRDS_SEEDS="${WATERBIRDS_SEEDS:-17 18 19 20 21}"
CIVILCOMMENTS_SEEDS="${CIVILCOMMENTS_SEEDS:-13 14 15 16 17}"
FEATURE_BATCH_SIZE="${FEATURE_BATCH_SIZE:-128}"
FROZEN_EPOCHS="${FROZEN_EPOCHS:-1200}"
FROZEN_FG_LAMBDAS="${FROZEN_FG_LAMBDAS:-0.01 0.03 0.1 0.3}"
FROZEN_GDRO_FG_LAMBDAS="${FROZEN_GDRO_FG_LAMBDAS:-0.01 0.03 0.1 0.3}"
TRAINING_RIDGE_VALUES="${TRAINING_RIDGE_VALUES:-1e-4 1e-3 1e-2}"

RUN_WATERBIRDS_DIAGNOSTICS="${RUN_WATERBIRDS_DIAGNOSTICS:-1}"
RUN_CIVILCOMMENTS_DIAGNOSTICS="${RUN_CIVILCOMMENTS_DIAGNOSTICS:-1}"
RUN_FROZEN_TRAINING="${RUN_FROZEN_TRAINING:-1}"
RUN_TRAINING_RIDGE_SENSITIVITY="${RUN_TRAINING_RIDGE_SENSITIVITY:-1}"
RUN_END_TO_END="${RUN_END_TO_END:-0}"
DOWNLOAD="${DOWNLOAD:-0}"
REQUIRE_CUDA="${REQUIRE_CUDA:-1}"

END_TO_END_LAMBDAS="${END_TO_END_LAMBDAS:-5e-5 1e-4 3e-4 1e-3 3e-3 1e-2}"
END_TO_END_EPOCHS="${END_TO_END_EPOCHS:-10}"
END_TO_END_BATCH_SIZE="${END_TO_END_BATCH_SIZE:-128}"
END_TO_END_EVAL_BATCH_SIZE="${END_TO_END_EVAL_BATCH_SIZE:-512}"
NUM_WORKERS="${NUM_WORKERS:-8}"

export TORCH_HOME="${TORCH_HOME:-$REPO_ROOT/.cache/torch}"
export HF_HOME="${HF_HOME:-$REPO_ROOT/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export WILDS_CACHE_DIR="${WILDS_CACHE_DIR:-$REPO_ROOT/.cache/wilds}"

DIAGNOSTIC_DIR="$OUT_DIR/diagnostics"
END_TO_END_DIR="$OUT_DIR/end_to_end"
RIDGE_SENSITIVITY_DIR="$OUT_DIR/training_ridge_sensitivity"
LOG_DIR="$OUT_DIR/logs"
REPORT_DIR="$OUT_DIR/report"
mkdir -p \
  "$DIAGNOSTIC_DIR" "$END_TO_END_DIR" "$RIDGE_SENSITIVITY_DIR" "$LOG_DIR" "$REPORT_DIR" \
  "$(dirname "$FEATURE_CACHE")" "$(dirname "$BUNDLE_PATH")" \
  "$TORCH_HOME" "$HF_HOME" "$TRANSFORMERS_CACHE" "$WILDS_CACHE_DIR" "$ROOT_DIR"

read -r -a WATERBIRDS_SEED_ARGS <<< "$WATERBIRDS_SEEDS"
read -r -a CIVILCOMMENTS_SEED_ARGS <<< "$CIVILCOMMENTS_SEEDS"
read -r -a FROZEN_FG_LAMBDA_ARGS <<< "$FROZEN_FG_LAMBDAS"
read -r -a FROZEN_GDRO_FG_LAMBDA_ARGS <<< "$FROZEN_GDRO_FG_LAMBDAS"
read -r -a TRAINING_RIDGE_ARGS <<< "$TRAINING_RIDGE_VALUES"
read -r -a END_TO_END_LAMBDA_ARGS <<< "$END_TO_END_LAMBDAS"

DATA_ARGS=()
if [[ "$DOWNLOAD" != "0" ]]; then DATA_ARGS+=(--download); fi

run_logged() {
  local log_path="$1"
  shift
  {
    printf 'command:'
    printf ' %q' "$@"
    printf '\n'
  } | tee "$log_path"
  "$@" 2>&1 | tee -a "$log_path"
}

need_waterbirds=0
if [[ "$RUN_WATERBIRDS_DIAGNOSTICS" != "0" || "$RUN_FROZEN_TRAINING" != "0" || "$RUN_END_TO_END" != "0" ]]; then
  need_waterbirds=1
fi

if [[ "$DOWNLOAD" == "0" ]]; then
  if [[ "$need_waterbirds" == "1" && ! -d "$ROOT_DIR/waterbirds_v1.0" ]]; then
    echo "Missing $ROOT_DIR/waterbirds_v1.0. Stage the WILDS dataset there or rerun with DOWNLOAD=1." >&2
    exit 2
  fi
  if [[ "$RUN_CIVILCOMMENTS_DIAGNOSTICS" != "0" && ! -d "$ROOT_DIR/civilcomments_v1.0" ]]; then
    echo "Missing $ROOT_DIR/civilcomments_v1.0. Stage the WILDS dataset there or rerun with DOWNLOAD=1." >&2
    exit 2
  fi
fi

run_logged "$LOG_DIR/dependency_check.log" "$PYTHON" -c \
  'import numpy, sklearn, torch, torchvision, wilds; print("dependencies: ok"); print("torch", torch.__version__); print("cuda", torch.cuda.is_available(), torch.version.cuda); print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")'

if [[ "$REQUIRE_CUDA" != "0" && "$need_waterbirds" == "1" && "$DEVICE" == cuda* ]]; then
  if ! "$PYTHON" -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)'; then
    echo "CUDA PyTorch is required for this profile, but torch.cuda.is_available() is false." >&2
    echo "Install the CUDA build of PyTorch for this NVIDIA machine, or set REQUIRE_CUDA=0." >&2
    exit 2
  fi
fi

run_logged "$LOG_DIR/audit_self_test.log" "$PYTHON" experiments/ridge_exact_audit.py --self-test

{
  echo "created_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "repo_root=$REPO_ROOT"
  echo "root_dir=$ROOT_DIR"
  echo "python=$PYTHON"
  echo "device=$DEVICE"
  echo "ridge=$RIDGE"
  echo "eps=$EPS"
  echo "alpha=$ALPHA"
  echo "audit_draws=$AUDIT_BOOTSTRAP_REPLICATES"
  echo "audit_confidence=$AUDIT_CONFIDENCE"
  echo "audit_seed=$AUDIT_SEED"
  echo "waterbirds_seeds=$WATERBIRDS_SEEDS"
  echo "civilcomments_seeds=$CIVILCOMMENTS_SEEDS"
  echo "run_end_to_end=$RUN_END_TO_END"
  echo "run_training_ridge_sensitivity=$RUN_TRAINING_RIDGE_SENSITIVITY"
  echo "training_ridge_values=$TRAINING_RIDGE_VALUES"
  echo "end_to_end_lambdas=$END_TO_END_LAMBDAS"
  git rev-parse HEAD 2>/dev/null | sed 's/^/git_commit=/' || true
} > "$OUT_DIR/run_config.txt"

{
  uname -a
  "$PYTHON" --version
  "$PYTHON" -m pip freeze || true
  nvidia-smi || true
} > "$OUT_DIR/system_info.txt" 2>&1

if [[ "$RUN_WATERBIRDS_DIAGNOSTICS" != "0" ]]; then
  for seed in "${WATERBIRDS_SEED_ARGS[@]}"; do
    output="$DIAGNOSTIC_DIR/waterbirds_seed_${seed}.json"
    run_logged "$LOG_DIR/waterbirds_diagnostic_seed_${seed}.log" \
      "$PYTHON" -u experiments/waterbirds_fisher_glass.py \
      --root-dir "$ROOT_DIR" \
      --cache "$FEATURE_CACHE" \
      --output "$output" \
      --seed "$seed" \
      --batch-size "$FEATURE_BATCH_SIZE" \
      --ridge "$RIDGE" --eps "$EPS" --alpha "$ALPHA" \
      --audit-bootstrap-replicates "$AUDIT_BOOTSTRAP_REPLICATES" \
      --audit-confidence "$AUDIT_CONFIDENCE" --audit-seed "$AUDIT_SEED" \
      "${DATA_ARGS[@]}"
  done
fi

if [[ "$RUN_CIVILCOMMENTS_DIAGNOSTICS" != "0" ]]; then
  for seed in "${CIVILCOMMENTS_SEED_ARGS[@]}"; do
    output="$DIAGNOSTIC_DIR/civilcomments_seed_${seed}.json"
    run_logged "$LOG_DIR/civilcomments_diagnostic_seed_${seed}.log" \
      "$PYTHON" -u experiments/civilcomments_fisher_glass.py \
      --root-dir "$ROOT_DIR" \
      --output "$output" \
      --seed "$seed" --max-features 50000 \
      --ridge "$RIDGE" --eps "$EPS" --alpha "$ALPHA" \
      --audit-bootstrap-replicates "$AUDIT_BOOTSTRAP_REPLICATES" \
      --audit-confidence "$AUDIT_CONFIDENCE" --audit-seed "$AUDIT_SEED" \
      "${DATA_ARGS[@]}"
  done
fi

if [[ "$RUN_FROZEN_TRAINING" != "0" ]]; then
  run_logged "$LOG_DIR/waterbirds_frozen_training.log" \
    "$PYTHON" -u experiments/waterbirds_fisher_glass_training.py \
    --root-dir "$ROOT_DIR" \
    --cache "$FEATURE_CACHE" \
    --output "$OUT_DIR/waterbirds_frozen_training.json" \
    --device "$DEVICE" --batch-size "$FEATURE_BATCH_SIZE" \
    --seeds "${WATERBIRDS_SEED_ARGS[@]}" \
    --epochs "$FROZEN_EPOCHS" \
    --fg-lambdas "${FROZEN_FG_LAMBDA_ARGS[@]}" \
    --gdro-fg-lambdas "${FROZEN_GDRO_FG_LAMBDA_ARGS[@]}" \
    --selection-worst-tolerance 0.02 \
    --ridge "$RIDGE" --eps "$EPS" --alpha "$ALPHA" \
    --audit-bootstrap-replicates "$AUDIT_BOOTSTRAP_REPLICATES" \
    --audit-confidence "$AUDIT_CONFIDENCE" --audit-seed "$AUDIT_SEED" \
    "${DATA_ARGS[@]}"
fi

if [[ "$RUN_FROZEN_TRAINING" != "0" && "$RUN_TRAINING_RIDGE_SENSITIVITY" != "0" ]]; then
  for sensitivity_ridge in "${TRAINING_RIDGE_ARGS[@]}"; do
    if "$PYTHON" -c 'import sys; sys.exit(0 if float(sys.argv[1]) == float(sys.argv[2]) else 1)' \
      "$sensitivity_ridge" "$RIDGE"; then
      continue
    fi
    ridge_tag="${sensitivity_ridge//-/m}"
    ridge_tag="${ridge_tag//./p}"
    run_logged "$LOG_DIR/waterbirds_frozen_training_ridge_${ridge_tag}.log" \
      "$PYTHON" -u experiments/waterbirds_fisher_glass_training.py \
      --root-dir "$ROOT_DIR" \
      --cache "$FEATURE_CACHE" \
      --output "$RIDGE_SENSITIVITY_DIR/waterbirds_frozen_training_ridge_${ridge_tag}.json" \
      --device "$DEVICE" --batch-size "$FEATURE_BATCH_SIZE" \
      --seeds "${WATERBIRDS_SEED_ARGS[@]}" \
      --epochs "$FROZEN_EPOCHS" \
      --fg-lambdas "${FROZEN_FG_LAMBDA_ARGS[@]}" \
      --gdro-fg-lambdas "${FROZEN_GDRO_FG_LAMBDA_ARGS[@]}" \
      --selection-worst-tolerance 0.02 \
      --ridge "$sensitivity_ridge" --eps "$EPS" --alpha "$ALPHA" \
      --audit-bootstrap-replicates "$AUDIT_BOOTSTRAP_REPLICATES" \
      --audit-confidence "$AUDIT_CONFIDENCE" --audit-seed "$AUDIT_SEED" \
      "${DATA_ARGS[@]}"
  done
fi

if [[ "$RUN_END_TO_END" != "0" ]]; then
  first_lambda=1
  for lambda_fg in "${END_TO_END_LAMBDA_ARGS[@]}"; do
    lambda_tag="${lambda_fg//-/m}"
    lambda_tag="${lambda_tag//./p}"
    output="$END_TO_END_DIR/waterbirds_lambda_${lambda_tag}.json"
    log_path="$LOG_DIR/waterbirds_end_to_end_lambda_${lambda_tag}.log"
    methods="group_dro_fisher_glass"
    if [[ "$first_lambda" == "1" ]]; then
      methods="group_dro group_dro_fisher_glass"
      first_lambda=0
    fi
    run_logged "$log_path" env \
      PYTHON="$PYTHON" ROOT_DIR="$ROOT_DIR" OUTPUT="$output" LOG_PATH="$log_path.inner" \
      BENCHMARKS=waterbirds METHODS="$methods" SEEDS="$WATERBIRDS_SEEDS" \
      EPOCHS="$END_TO_END_EPOCHS" BATCH_SIZE="$END_TO_END_BATCH_SIZE" \
      EVAL_BATCH_SIZE="$END_TO_END_EVAL_BATCH_SIZE" DEVICE="$DEVICE" \
      NUM_WORKERS="$NUM_WORKERS" TEXT_NUM_WORKERS=0 \
      RIDGE="$RIDGE" EPS="$EPS" ALPHA="$ALPHA" FG_LAMBDA="$lambda_fg" \
      AUDIT_BOOTSTRAP_REPLICATES="$AUDIT_BOOTSTRAP_REPLICATES" \
      AUDIT_CONFIDENCE="$AUDIT_CONFIDENCE" AUDIT_SEED="$AUDIT_SEED" \
      DOWNLOAD="$DOWNLOAD" ALLOW_MODEL_DOWNLOAD=0 RESUME=1 \
      bash experiments/run_full_end_to_end_suite.sh
  done
fi

shopt -s nullglob
REPORT_INPUTS=("$DIAGNOSTIC_DIR"/*.json)
if [[ -f "$OUT_DIR/waterbirds_frozen_training.json" ]]; then
  REPORT_INPUTS+=("$OUT_DIR/waterbirds_frozen_training.json")
fi
REPORT_INPUTS+=("$END_TO_END_DIR"/*.json)
REPORT_INPUTS+=("$RIDGE_SENSITIVITY_DIR"/*.json)
shopt -u nullglob

if [[ "${#REPORT_INPUTS[@]}" == "0" ]]; then
  echo "No audit JSON files were produced." >&2
  exit 3
fi

run_logged "$LOG_DIR/make_report.log" \
  "$PYTHON" experiments/make_ridge_exact_audit_report.py \
  "${REPORT_INPUTS[@]}" --output-dir "$REPORT_DIR"

tar -czf "$BUNDLE_PATH" -C "$(dirname "$OUT_DIR")" "$(basename "$OUT_DIR")"
sha256sum "$BUNDLE_PATH" > "$BUNDLE_PATH.sha256"

echo
echo "W2 audit complete."
echo "Report: $REPORT_DIR/ridge_exact_audit.md"
echo "Return bundle: $BUNDLE_PATH"
echo "Checksum: $BUNDLE_PATH.sha256"
