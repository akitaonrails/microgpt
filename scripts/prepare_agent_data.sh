#!/usr/bin/env bash
# Only the sources needed for the measured foundation + narrow agent recipe.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/dataset-env.sh
agent_python=.venv/bin/python
agent_raw="$MICROGPT_DATA_ROOT/raw"
agent_out="$MICROGPT_DATA_ROOT/project-data"
mkdir -p "$agent_out"
if [[ ! -e data && ! -L data ]]; then ln -s "$agent_out" data; fi

ensure_file() {
  local marker="$1"
  shift
  if [[ ! -f "$marker" ]]; then "$@"; fi
}

for spec in codeparrot:8:32 cosmopedia:1.3:1 opencoder:0.6:2 smoltalk:1:4 multilingual:0.4:2 magicoder:0.3:1; do
  IFS=: read -r source gb files <<< "$spec"
  ensure_file "$agent_raw/$source/manifest.json" "$agent_python" -m microgpt.download_corpus \
    --source "$source" --max-gb "$gb" --max-files "$files"
done
for spec in codeparrot:codeparrot:600000000 opencoder:opencoder:160000000 cosmopedia:cosmopedia-v2:120000000 smoltalk:smoltalk:80000000 multilingual:multilingual:40000000; do
  IFS=: read -r source directory tokens <<< "$spec"
  ensure_file "$agent_out/large-v1/$directory/meta.json" "$agent_python" -m microgpt.prepare_large \
    --source "$source" --out "$agent_out/large-v1/$directory" --tokens "$tokens"
done
ensure_file "$agent_out/large-v1/meta.json" "$agent_python" -m microgpt.mixture \
  --data "$agent_out/large-v1" --source codeparrot:0.60 --source opencoder:0.16 \
  --source cosmopedia-v2:0.12 --source smoltalk:0.08 --source multilingual:0.04
ensure_file "$agent_out/smoltalk-large.source.json" "$agent_python" -m microgpt.curate \
  --input "$agent_raw/smoltalk" --kind smoltalk --out "$agent_out/smoltalk-large.jsonl" \
  --limit 12000 --scan-limit 500000
ensure_file "$agent_out/magicoder-large.source.json" "$agent_python" -m microgpt.download_sft \
  --source magicoder --local "$agent_raw/magicoder" --out "$agent_out/magicoder-large.jsonl" \
  --limit 20000 --scan-limit 80000

ensure_file "$agent_out/opencode-tools-v2.json" "$agent_python" scripts/capture_opencode_tools.py \
  --out "$agent_out/opencode-tools-v2.json"
ensure_file "$agent_out/grounding-wire-v1.source.json" "$agent_python" -m microgpt.grounding \
  --tools "$agent_out/opencode-tools-v2.json" --out "$agent_out/grounding-wire-v1.jsonl" --wire --effort none
ensure_file "$agent_out/horizon-related-v1.source.json" "$agent_python" -m microgpt.horizon_data \
  --tools "$agent_out/opencode-tools-v2.json" --out "$agent_out/horizon-related-v1.jsonl" \
  --vocabulary-data "$agent_out/magicoder-large.jsonl" --count 2000 --wire --effort none \
  --recover-errors --line-anchors --related-names
# Keep this training corpus identical to the measured run. Export separately
# enables canonical runtime IDs; new experiments may canonicalize training too.
ensure_file "$agent_out/agent-related-v1/manifest.json" "$agent_python" -m microgpt.trajectories \
  --input "$agent_out/horizon-related-v1.jsonl" "$agent_out/grounding-wire-v1.jsonl" \
  "$agent_out/smoltalk-large.jsonl" --out "$agent_out/agent-related-v1" \
  --tokenizer "$agent_out/large-v1" --max-length 32768 --limit 14000 --pack-episodes
