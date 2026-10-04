#!/usr/bin/env bash
# Reproduce the programming-focused correction. Store all datasets on the NAS.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/dataset-env.sh
coding_python=.venv/bin/python
coding_raw="$MICROGPT_DATA_ROOT/raw"
# Existing foundation/tokenizer and Magicoder/short conversation preparation:
bash scripts/prepare_agent_data.sh
ensure() {
  local marker="$1"
  shift
  if [[ ! -f "$marker" ]]; then "$@"; fi
}
ensure "$coding_raw/opencodeinstruct/manifest.json" "$coding_python" -m microgpt.download_corpus \
  --source opencodeinstruct --max-gb 8 --max-files 50
ensure "$coding_raw/code-feedback/manifest.json" "$coding_python" -m microgpt.download_corpus \
  --source code-feedback --max-gb 0.5 --max-files 1
ensure "$coding_raw/opencoder-educational/manifest.json" "$coding_python" -m microgpt.download_corpus \
  --source opencoder-educational --max-gb 0.1 --max-files 1
ensure data/opencodeinstruct-v1.source.json "$coding_python" -m microgpt.curate \
  --input "$coding_raw/opencodeinstruct" --kind opencodeinstruct \
  --out data/opencodeinstruct-v1.jsonl --limit 100000 --scan-limit 5000000
ensure data/code-feedback-v1.source.json "$coding_python" -m microgpt.curate \
  --input "$coding_raw/code-feedback" --kind code-feedback \
  --out data/code-feedback-v1.jsonl --limit 68000 --scan-limit 100000
ensure data/opencoder-educational-v1.source.json "$coding_python" -m microgpt.curate \
  --input "$coding_raw/opencoder-educational" --kind opencoder-educational \
  --out data/opencoder-educational-v1.jsonl --limit 120000 --scan-limit 200000
ensure data/educational-tools-v1.source.json "$coding_python" scripts/prepare_programming_tools.py \
  --input "$coding_raw/opencoder-educational/educational_instruct/train-00000-of-00001.parquet" \
  --tools data/opencode-tools-v2.json --out data/educational-tools-v1.jsonl --count 4000
if [[ ! -f data/programming-chat-replay-v1.jsonl ]]; then
  "$coding_python" - <<'PY'
import json
with open('data/programming-chat-replay-v1.jsonl', 'x') as target:
    for path in ['data/smoltalk-large.jsonl', 'data/multilingual-large.jsonl']:
        count = 0
        for line in open(path):
            record = json.loads(line)
            if sum(len(m.get('content') or '') for m in record['messages']) > 4000:
                continue
            record['source'] = 'conversation-replay'
            target.write(json.dumps(record) + '\n')
            count += 1
            if count == 250:
                break
PY
fi
ensure data/programming-v1.source.json "$coding_python" -m microgpt.programming_data \
  --tools data/opencode-tools-v2.json --out data/programming-v1.jsonl \
  --public data/opencodeinstruct-v1.jsonl data/opencoder-educational-v1.jsonl \
  data/code-feedback-v1.jsonl data/magicoder-large.jsonl data/programming-chat-replay-v1.jsonl \
  --count 8000 --per-source 100000 --max-chars 12000
ensure data/programming-sft-v1/manifest.json "$coding_python" -m microgpt.trajectories \
  --input data/programming-v1.jsonl data/educational-tools-v1.jsonl data/horizon-related-v1.jsonl \
  --out data/programming-sft-v1 --tokenizer data/large-v1 --max-length 32768 \
  --limit 400000 --pack-episodes --canonical-tool-ids
