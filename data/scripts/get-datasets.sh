#!/usr/bin/env bash
# Download the challenge FineWeb token shards (and matching tokenizer) into
# data/datasets/ and data/tokenizers/.
#
# Defaults match the repo baseline:
#   DATA_PATH=./data/datasets/fineweb10B_sp1024
#   80 train shards (~8B tokens) + full val split
#
# Usage:
#   ./data/scripts/get-datasets.sh
#   ./data/scripts/get-datasets.sh --variant sp1024 --train-shards 80
#   ./data/scripts/get-datasets.sh --with-docs
#   VARIANT=sp8192 TRAIN_SHARDS=40 ./data/scripts/get-datasets.sh
#
# Environment (optional, forwarded to the Python downloader):
#   MATCHED_FINEWEB_REPO_ID              default: willdepueoai/parameter-golf
#   MATCHED_FINEWEB_REMOTE_ROOT_PREFIX   default: datasets
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

VARIANT="${VARIANT:-sp1024}"
TRAIN_SHARDS="${TRAIN_SHARDS:-80}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "error: python3 is required" >&2
  exit 1
fi

if ! python3 -c "import huggingface_hub" >/dev/null 2>&1; then
  echo "error: python package 'huggingface_hub' is required (pip install huggingface_hub)" >&2
  exit 1
fi

echo "Downloading FineWeb challenge data"
echo "  repo root:   $REPO_ROOT"
echo "  variant:     $VARIANT"
echo "  train shards: $TRAIN_SHARDS"
echo "  output:      data/datasets/fineweb10B_${VARIANT}/"
echo

python3 data/cached_challenge_fineweb.py \
  --variant "$VARIANT" \
  --train-shards "$TRAIN_SHARDS" \
  "$@"

dataset_dir="data/datasets/fineweb10B_${VARIANT}"
if [[ "$VARIANT" == "byte260" ]]; then
  dataset_dir="data/datasets/fineweb10B_byte260"
fi

train_count="$(find "$dataset_dir" -maxdepth 1 -name 'fineweb_train_*.bin' 2>/dev/null | wc -l | tr -d ' ')"
val_count="$(find "$dataset_dir" -maxdepth 1 -name 'fineweb_val_*.bin' 2>/dev/null | wc -l | tr -d ' ')"

echo
echo "Done."
echo "  train shards: $train_count"
echo "  val shards:   $val_count"
echo "  dataset path: $dataset_dir"
echo
echo "Train with:"
echo "  DATA_PATH=./${dataset_dir} python3 train_gpt.py"
echo
echo "Optional math-mix corpus (requires FineWeb above plus math sources under postraining/data/):"
echo "  python3 build_math_mix_dataset.py --output data/datasets/mathmix_v4_sp1024"
