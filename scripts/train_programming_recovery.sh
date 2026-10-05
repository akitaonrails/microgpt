#!/usr/bin/env bash
# Explicit new SFT stage: own scratch-trained 80K checkpoint, fresh optimizer.
set -euo pipefail
cd "$(dirname "$0")/.."
.venv/bin/python -m microgpt.trajectories \
  --input data/programming-v2.jsonl data/educational-tools-v1.jsonl data/horizon-related-v1.jsonl data/programming-recovery-v1.jsonl \
  --out data/programming-sft-v3 --tokenizer data/large-v1 --max-length 32768 \
  --limit 420000 --pack-episodes --canonical-tool-ids
.venv/bin/python -m microgpt.finetune --data data/programming-sft-v3 \
  --checkpoint runs/programming-v2/checkpoint-80000.pt --out runs/programming-v3 --steps 1000 \
  --accumulation 2 --lr 0.00003 --eval-every 500 --eval-examples 80 \
  --device cuda --attention flash --max-length 32768 --balanced-lengths \
  --source-weight opencodeinstruct:0.15 --source-weight opencoder-educational:0.10 \
  --source-weight code-feedback:0.10 --source-weight magicoder:0.05 \
  --source-weight educational-tools:0.15 --source-weight programming-tools:0.10 \
  --source-weight programming-direct:0.015 --source-weight horizon-session:0.015 \
  --source-weight horizon-handoff:0.01 --source-weight conversation-replay:0.01 \
  --source-weight programming-recovery:0.20 --source-weight programming-switch:0.10
.venv/bin/python scripts/continue_programming.py --run runs/programming-v3 \
  --data data/programming-sft-v3 --out runs/programming-recovery-continuation \
  --milestones 5000 10000 20000 40000 --hours 4
