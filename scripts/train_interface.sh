#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
.venv/bin/python -m microgpt.finetune --checkpoint runs/capacity-pilot-v1/candidate-adapter/last.pt \
  --data data/programming-interface-sft-v1 --out runs/interface-v1 --steps 1000 \
  --accumulation 2 --lr 0.00003 --eval-every 500 --eval-examples 80 \
  --max-length 8192 --device cuda --attention flash
.venv/bin/python scripts/continue_programming.py --run runs/interface-v1 \
  --data data/programming-interface-sft-v1 --out runs/interface-continuation-v1 \
  --milestones 5000 10000 20000 --hours 2 --port 8024 \
  --tasks-file evaluation/programming-v1/development.jsonl --direct-only
