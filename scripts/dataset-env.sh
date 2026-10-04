# Source this before dataset download/preparation commands. Model weights remain
# in the existing model cache; every dataset cache and temporary file uses the NAS.
export MICROGPT_DATA_ROOT=/mnt/gigachad/microgpt/dataset
if ! mountpoint -q /mnt/gigachad; then
  echo 'NAS is not mounted at /mnt/gigachad; refusing local fallback' >&2
  return 1
fi
mkdir -p "$MICROGPT_DATA_ROOT/cache/datasets" "$MICROGPT_DATA_ROOT/cache/hub" "$MICROGPT_DATA_ROOT/cache/xet" "$MICROGPT_DATA_ROOT/tmp"
export HF_DATASETS_CACHE="$MICROGPT_DATA_ROOT/cache/datasets"
export MICROGPT_DATA_HUB_CACHE="$MICROGPT_DATA_ROOT/cache/hub"
export HF_XET_CACHE="$MICROGPT_DATA_ROOT/cache/xet"
export TMPDIR="$MICROGPT_DATA_ROOT/tmp"
export TIKTOKEN_CACHE_DIR="$MICROGPT_DATA_ROOT/cache/tiktoken"
