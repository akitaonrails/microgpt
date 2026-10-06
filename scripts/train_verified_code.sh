#!/usr/bin/env bash
# A bounded public-data pilot, with execution checks before longer training.
set -euo pipefail
cd "$(dirname "$0")/.."
.venv/bin/python - <<'PY'
import hashlib
import json
from pathlib import Path
import subprocess

def sha(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()

source = json.loads(Path('data/verified-opencodeinstruct-v1.source.json').read_text())
assert source['retained'] >= 50000
assert source['split_grouping'].startswith('Transitive union')
parent = 'runs/interface-diverse-v2/checkpoint-10000.pt'
assert not Path('runs/verified-code-learning-v1').exists()
experiment = {
    'parent': parent, 'parent_sha256': sha(parent), 'baseline_direct': 17,
    'data_manifest_sha256': sha('data/verified-opencodeinstruct-sft-v1/manifest.json'),
    'hypothesis': 'Broader independently checked public programming problems improve executable code beyond the narrow OpenCoder interface curriculum.',
    'microbatch': 4, 'accumulation': 8, 'learning_rate': 1e-5, 'seed': 20261006,
    'initial_updates': 2000, 'evaluation_milestones': [500, 1000, 2000],
    'data_source_report': source, 'validation_seed': 20261007,
    'progression': 'At least 22/100 direct passes, 75/100 parameter-name matches, 90/100 valid Python outputs, and passes in at least five families.',
    'continuation_if_passed': 'Evaluate at 5K, 10K and 20K total updates, with a three-hour continuation ceiling.',
    'limitations': 'Same parent, learning rate, effective batch and training seed as the failed batched pilot. Data changes; variable sequence lengths mean equal updates do not imply equal tokens or compute. Microbatch partition changes for memory headroom; CPU gradient equivalence is tested. This is not a matched-token causal comparison.',
    'sampling': 'Sample the new checked source with replacement; no eightfold identifier expansion. Validation uses a separate fixed random seed and does not alter training RNG.',
    'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    'acceptance_used': False, 'automatic_promotion': False,
}
with Path('runs/verified-code-learning-experiment-v1.json').open('x') as output:
    output.write(json.dumps(experiment, indent=2) + '\n')
PY
.venv/bin/python -m microgpt.finetune \
  --checkpoint runs/interface-diverse-v2/checkpoint-10000.pt \
  --data data/verified-opencodeinstruct-sft-v1 --out runs/verified-code-learning-v1 \
  --steps 250 --lr 0.00001 --seed 20261006 --microbatch 4 --accumulation 8 \
  --eval-every 250 --eval-examples 256 --validation-seed 20261007 --max-length 8192 --device cuda \
  --attention math --max-seconds 1200
.venv/bin/python scripts/continue_programming.py --run runs/verified-code-learning-v1 \
  --data data/verified-opencodeinstruct-sft-v1 --out runs/verified-code-learning-pilot-v1 \
  --milestones 500 1000 2000 --hours 2 --port 8028 \
  --tasks-file evaluation/programming-v1/development.jsonl --direct-only
.venv/bin/python - <<'PY'
import json
from pathlib import Path
state = json.loads(Path('runs/verified-code-learning-pilot-v1/status.json').read_text())
assert state['status'] == 'milestones_exhausted'
stage = state['stages'][-1]
assert stage['metrics']['step'] == 2000
analysis = json.loads(Path('runs/verified-code-learning-pilot-v1/eval-2000/direct/analysis.json').read_text())
counts = analysis['counts']
families = sum(row['passed'] > 0 for row in analysis['families'].values())
passed = counts['passed'] >= 22 and counts['parameter_names_match'] >= 75 and counts['syntax_valid'] >= 90 and families >= 5
decision = {'extend': passed, 'counts': counts, 'families_with_passes': families,
            'reason': 'Predeclared progression gate passed' if passed else 'Stop this data pilot: progression gate unmet'}
Path('runs/verified-code-learning-pilot-decision.json').write_text(json.dumps(decision, indent=2) + '\n')
print(json.dumps(decision), flush=True)
PY
if .venv/bin/python -c "import json,sys; sys.exit(not json.load(open('runs/verified-code-learning-pilot-decision.json'))['extend'])"; then
  .venv/bin/python scripts/continue_programming.py --run runs/verified-code-learning-v1 \
    --data data/verified-opencodeinstruct-sft-v1 --out runs/verified-code-learning-continuation-v1 \
    --milestones 5000 10000 20000 --hours 3 --port 8028 \
    --tasks-file evaluation/programming-v1/development.jsonl --direct-only
fi
