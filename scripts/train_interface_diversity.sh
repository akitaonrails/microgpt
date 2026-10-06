#!/usr/bin/env bash
# Same scratch parent and SFT settings; broaden checked source-code diversity.
set -euo pipefail
cd "$(dirname "$0")/.."
.venv/bin/python - <<'PY'
import hashlib
import json
from pathlib import Path
import subprocess

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

run = Path('runs/interface-diverse-v2')
data = Path('data/programming-interface-diverse-sft-v3')
parent = 'runs/capacity-pilot-v1/candidate-adapter/last.pt'
checked = json.loads(Path('runs/interface-diverse-filtered-checks-v2.json').read_text())
manifest = json.loads((data / 'manifest.json').read_text())
assert checked['selected'] > 0 and checked['selected'] == checked['passed']
assert checked['group_by_code'], 'Link duplicate code and instruction groups before splitting'
assert checked['input_sha256'] == manifest['sources'][0]['sha256']
assert checked['input_sha256'] == sha('data/programming-interface-diverse-checked-v2.jsonl')
assert not Path('runs/interface-diverse-continuation-v2').exists()
if run.exists():
    raise ValueError('Use a fresh training output directory')
experiment = {
    'hypothesis': 'Broader checked source-code diversity improves direct programming at fixed update budget.',
    'parent': parent, 'parent_sha256': sha(parent),
    'data_manifest_sha256': sha(data / 'manifest.json'),
    'reference_checks_sha256': sha('runs/interface-diverse-filtered-checks-v2.json'),
    'development_sha256': sha('evaluation/programming-v1/development.jsonl'),
    'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    'milestones': [5000, 10000, 20000], 'continuation_hours': 2,
    'parent_and_optimizer': 'Same parent as interface-v1, fresh AdamW, LR 3e-5, accumulation 2, seed 42.',
    'limitations': 'Math SDPA from the start; original comparison switched from Flash at 2K. Equal updates do not mean equal token exposures. Source order and AST deduplication also differ.',
    'progression': 'At least 20/100 direct, 70/100 matching parameter names, and passes in at least 5 families before extending this branch. No automatic promotion.',
    'acceptance_set_used': False,
}
with Path('runs/interface-diverse-experiment-v2.json').open('x') as output:
    output.write(json.dumps(experiment, indent=2) + '\n')
PY
.venv/bin/python -m microgpt.finetune --checkpoint runs/capacity-pilot-v1/candidate-adapter/last.pt \
  --data data/programming-interface-diverse-sft-v3 --out runs/interface-diverse-v2 --steps 1000 \
  --accumulation 2 --lr 0.00003 --eval-every 500 --eval-examples 80 \
  --max-length 8192 --device cuda --attention math
.venv/bin/python scripts/continue_programming.py --run runs/interface-diverse-v2 \
  --data data/programming-interface-diverse-sft-v3 --out runs/interface-diverse-continuation-v2 \
  --milestones 5000 10000 20000 --hours 2 --port 8024 \
  --tasks-file evaluation/programming-v1/development.jsonl --direct-only
