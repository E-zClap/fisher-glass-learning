#!/usr/bin/env bash
# Download everything the prospective nuisance-shift experiment needs.
#
#   1. Waterbirds v1.0  (466 MB compressed, ~1.1 GB on disk) -- composited
#      images + metadata.csv
#   2. CUB-200-2011 segmentations (39 MB) -- the exact alpha mattes that were
#      used to build Waterbirds, which is what makes a pixel-exact,
#      label-preserving background swap possible.
#
# Places365 is NOT needed: donor backgrounds are recovered from training-split
# Waterbirds images by inpainting their own (small) bird hole.
#
# The CodaLab bundle's internal layout is not guaranteed, so the archive is
# unpacked into a staging directory and the real dataset root is located by
# finding metadata.csv. An earlier flat extraction directly into --root is
# detected and repaired without re-downloading.
#
# Usage:  bash experiments/prospective_nuisance_shift/download_data.sh [--root data/wilds]

set -euo pipefail

ROOT="data/wilds"
PYTHON="${PYTHON:-python}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root) ROOT="$2"; shift 2 ;;
    -h|--help) sed -n '2,19p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac
done

WATERBIRDS_DIR="${ROOT}/waterbirds_v1.0"
WATERBIRDS_URL="https://worksheets.codalab.org/rest/bundles/0xb922b6c2d39c48bab4516780e06d5649/contents/blob/"
TARBALL="${ROOT}/waterbirds_v1.0.tar.gz"
STAGE="${ROOT}/_waterbirds_extract"
EXPECTED_ROWS=11788

echo "==> repository root: $(pwd)"
mkdir -p "${ROOT}"
echo "==> current contents of ${ROOT}:"
# `... | head` would SIGPIPE the producer; with `set -o pipefail` that aborts the
# script. Keep the `|| true`.
ls -la "${ROOT}" 2>/dev/null | head -12 || true

# --------------------------------------------------------------------------- #
# Repair: a previous run may have unpacked the bundle flat into ${ROOT}
# --------------------------------------------------------------------------- #
repair_flat_extraction() {
  [[ -f "${ROOT}/metadata.csv" ]] || return 1
  echo "==> found a flat extraction in ${ROOT}; moving it into ${WATERBIRDS_DIR}"
  mkdir -p "${WATERBIRDS_DIR}"
  mv "${ROOT}/metadata.csv" "${WATERBIRDS_DIR}/"
  if [[ -f "${ROOT}/RELEASE_v1.0.txt" ]]; then
    mv "${ROOT}/RELEASE_v1.0.txt" "${WATERBIRDS_DIR}/"
  fi
  # CUB class directories look like 001.Black_footed_Albatross
  local had_nullglob=0
  shopt -q nullglob && had_nullglob=1
  shopt -s nullglob
  local moved=0
  for d in "${ROOT}"/[0-9][0-9][0-9].*; do
    [[ -d "$d" ]] || continue
    mv "$d" "${WATERBIRDS_DIR}/"
    moved=$((moved + 1))
  done
  [[ "${had_nullglob}" -eq 1 ]] || shopt -u nullglob
  echo "    moved ${moved} class directories"
  return 0
}

# A subdirectory holds metadata.csv (e.g. waterbird_complete95_forest2water2).
# Only accept a directory that also has CUB class folders, so an unrelated
# metadata.csv (civilcomments, for instance) is never mistaken for Waterbirds.
looks_like_waterbirds() {
  local dir="$1"
  [[ -f "${dir}/metadata.csv" ]] || return 1
  compgen -G "${dir}/[0-9][0-9][0-9].*" > /dev/null
}

repair_nested_extraction() {
  local found src
  while IFS= read -r found; do
    [[ -n "${found}" ]] || continue
    src="$(dirname "${found}")"
    looks_like_waterbirds "${src}" || continue
    [[ "${src}" == "${WATERBIRDS_DIR}" ]] && return 0
    echo "==> found dataset at ${src}; moving it to ${WATERBIRDS_DIR}"
    rm -rf "${WATERBIRDS_DIR}"
    mv "${src}" "${WATERBIRDS_DIR}"
    return 0
  done < <(find "${ROOT}" -mindepth 2 -maxdepth 4 -name metadata.csv 2>/dev/null || true)
  return 1
}

# --------------------------------------------------------------------------- #
# 1. Waterbirds
# --------------------------------------------------------------------------- #
if [[ -f "${WATERBIRDS_DIR}/metadata.csv" ]]; then
  echo "==> Waterbirds already present at ${WATERBIRDS_DIR}"
elif repair_flat_extraction; then
  echo "==> repaired without re-downloading"
elif repair_nested_extraction; then
  echo "==> repaired without re-downloading"
else
  echo "==> downloading Waterbirds v1.0 (466 MB) into ${ROOT}"
  if ${PYTHON} -c "import wilds" 2>/dev/null; then
    echo "    using the WILDS downloader"
    ${PYTHON} - "$ROOT" <<'PY'
import sys
from wilds import get_dataset
get_dataset(dataset="waterbirds", root_dir=sys.argv[1], download=True)
PY
  else
    echo "    wilds not installed; fetching the archive directly"
    if [[ ! -s "${TARBALL}" ]]; then
      curl -L --fail --retry 3 --continue-at - -o "${TARBALL}" "${WATERBIRDS_URL}"
    else
      echo "    reusing existing ${TARBALL}"
    fi

    echo "    unpacking into a staging directory"
    rm -rf "${STAGE}"
    mkdir -p "${STAGE}"
    tar -xzf "${TARBALL}" -C "${STAGE}"

    META_PATH="$(find "${STAGE}" -maxdepth 3 -name metadata.csv -print -quit)"
    if [[ -z "${META_PATH}" ]]; then
      echo "ERROR: no metadata.csv anywhere in the archive. Top level was:" >&2
      ls -la "${STAGE}" >&2
      echo "The tarball has been kept at ${TARBALL} for inspection." >&2
      exit 1
    fi
    SRC_DIR="$(dirname "${META_PATH}")"
    echo "    dataset root inside archive: ${SRC_DIR#${STAGE}/}"
    rm -rf "${WATERBIRDS_DIR}"
    mkdir -p "$(dirname "${WATERBIRDS_DIR}")"
    mv "${SRC_DIR}" "${WATERBIRDS_DIR}"
    rm -rf "${STAGE}"
  fi
fi

if [[ ! -f "${WATERBIRDS_DIR}/metadata.csv" ]]; then
  echo "ERROR: ${WATERBIRDS_DIR}/metadata.csv not found after download" >&2
  echo "Contents of ${ROOT}:" >&2
  ls -la "${ROOT}" >&2
  exit 1
fi

ROWS=$(($(wc -l < "${WATERBIRDS_DIR}/metadata.csv") - 1))
CLASS_DIRS=$(find "${WATERBIRDS_DIR}" -mindepth 1 -maxdepth 1 -type d | wc -l)
echo "==> Waterbirds: ${ROWS} metadata rows (expected ${EXPECTED_ROWS}), ${CLASS_DIRS} class directories (expected 200)"
if [[ "${ROWS}" -ne "${EXPECTED_ROWS}" ]]; then
  echo "WARNING: unexpected row count; split sizes may differ from the paper" >&2
fi

# Spot-check that the image referenced by the first metadata row really exists.
FIRST_IMG="$(awk -F, 'NR==2 {print $2}' "${WATERBIRDS_DIR}/metadata.csv")"
if [[ ! -f "${WATERBIRDS_DIR}/${FIRST_IMG}" ]]; then
  echo "ERROR: metadata references ${FIRST_IMG} but that file is missing under ${WATERBIRDS_DIR}" >&2
  exit 1
fi
echo "==> image spot-check OK: ${FIRST_IMG}"
rm -f "${TARBALL}"

# --------------------------------------------------------------------------- #
# 2. CUB segmentation mattes
# --------------------------------------------------------------------------- #
echo "==> downloading CUB-200-2011 segmentations (39 MB) and verifying coverage"
${PYTHON} -u experiments/prospective_nuisance_shift/fetch_cub_masks.py \
  --out-dir data/cub_segmentations \
  --metadata "${WATERBIRDS_DIR}/metadata.csv"

echo
echo "==> all data ready:"
echo "    images : ${WATERBIRDS_DIR}"
echo "    mattes : data/cub_segmentations/segmentations"
