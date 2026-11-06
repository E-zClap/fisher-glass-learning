#!/usr/bin/env bash
# Prepare original Waterbirds and the CUB mattes used by the paired intervention.
# Run from the repository root; no Places download is needed.
# Uses the portable validated Stanford downloader, avoiding retired CodaLab URLs.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
DATA_ROOT="data/wilds"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --root) DATA_ROOT="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: bash experiments/prospective_nuisance_shift/download_data.sh [--root data/wilds]"
      exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac
done

"${PYTHON}" "${HERE}/../../scripts/prepare_waterbirds.py" --root "${DATA_ROOT}"
"${PYTHON}" "${HERE}/fetch_cub_masks.py" \
  --out-dir data/cub_segmentations \
  --metadata "${DATA_ROOT}/waterbirds_v1.0/metadata.csv"
