#!/usr/bin/env bash
# Reproduce the larger curriculum. Completed artifacts are reused; incomplete
# directories fail visibly rather than silently becoming training inputs.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/dataset-env.sh
PY=.venv/bin/python
RAW="$MICROGPT_DATA_ROOT/raw"
OUT="$MICROGPT_DATA_ROOT/project-data"
mkdir -p "$OUT"
if [[ ! -e data && ! -L data ]]; then ln -s "$OUT" data; fi

ensure() {
  local marker="$1"
  shift
  if [[ ! -f "$marker" ]]; then "$@"; fi
}

for spec in codeparrot:8:32 cosmopedia:1.3:1 opencoder:0.6:2 smoltalk:1:4 multilingual:0.4:2 conversation:0.1:1 commitpack:1:3 magicoder:0.3:1 swesmith:1.5:20 reasoning-code:0.45:2 reasoning-math:0.45:2; do
  IFS=: read -r source gb files <<< "$spec"
  "$PY" -m microgpt.download_corpus --source "$source" --max-gb "$gb" --max-files "$files"
done

for spec in codeparrot:codeparrot:600000000 opencoder:opencoder:160000000 cosmopedia:cosmopedia-v2:120000000 smoltalk:smoltalk:80000000 multilingual:multilingual:40000000; do
  IFS=: read -r source directory tokens <<< "$spec"
  ensure "$OUT/large-v1/$directory/meta.json" "$PY" -m microgpt.prepare_large --source "$source" --out "$OUT/large-v1/$directory" --tokens "$tokens"
done
ensure "$OUT/large-v1/meta.json" "$PY" -m microgpt.mixture --data "$OUT/large-v1" --source codeparrot:0.60 --source opencoder:0.16 --source cosmopedia-v2:0.12 --source smoltalk:0.08 --source multilingual:0.04

for spec in opencoder:opencoder-large:20000:400000 smoltalk:smoltalk-large:12000:500000 multilingual:multilingual-large:4000:300000 conversation:conversation-complete:200:300000 commitpack:commits-large:5000:300000; do
  IFS=: read -r source name count scan <<< "$spec"
  ensure "$OUT/$name.source.json" "$PY" -m microgpt.curate --input "$RAW/$source" --kind "$source" --out "$OUT/$name.jsonl" --limit "$count" --scan-limit "$scan"
done
for spec in magicoder:magicoder-large:20000:80000 swesmith:swesmith-native:6000:25000; do
  IFS=: read -r source name count scan <<< "$spec"
  ensure "$OUT/$name.source.json" "$PY" -m microgpt.download_sft --source "$source" --local "$RAW/$source" --out "$OUT/$name.jsonl" --limit "$count" --scan-limit "$scan"
done
ensure "$OUT/commit-tools-large.jsonl" "$PY" -m microgpt.tool_examples --input "$OUT/commits-large.jsonl" --out "$OUT/commit-tools-large.jsonl"
ensure "$OUT/practice-v2.source.json" "$PY" -m microgpt.practice --out "$OUT/practice-v2.jsonl" --count 3000
ensure "$OUT/practice-handoff.source.json" "$PY" -m microgpt.practice --out "$OUT/practice-handoff.jsonl" --count 1000 --start-seed 100000 --handoffs
for source in reasoning-code reasoning-math reasoning-trace; do
  ensure "$OUT/$source.source.json" "$PY" -m microgpt.reasoning_data --source "$source" --out "$OUT/$source.jsonl" --limit 6000
 done
ensure "$OUT/reasoning-tools.source.json" "$PY" -m microgpt.reasoning_data --source reasoning-tools --input "$OUT/practice-v2.jsonl" "$OUT/practice-handoff.jsonl" --out "$OUT/reasoning-tools.jsonl" --limit 4000 --max-length 32768
ensure "$OUT/agent-sft-v4/manifest.json" "$PY" -m microgpt.trajectories --input \
  "$OUT/magicoder-large.jsonl" "$OUT/opencoder-large.jsonl" "$OUT/smoltalk-large.jsonl" \
  "$OUT/multilingual-large.jsonl" "$OUT/conversation-complete.jsonl" "$OUT/commit-tools-large.jsonl" \
  "$OUT/swesmith-native.jsonl" "$OUT/practice-v2.jsonl" "$OUT/practice-handoff.jsonl" \
  "$OUT/reasoning-code.jsonl" "$OUT/reasoning-math.jsonl" "$OUT/reasoning-trace.jsonl" "$OUT/reasoning-tools.jsonl" \
  --out "$OUT/agent-sft-v4" --tokenizer "$OUT/large-v1" --max-length 32768 --limit 120000 --pack-episodes
