#!/usr/bin/env bash
# Broad programming SFT on our random-origin foundation; never pretrained weights.
set -euo pipefail
cd "$(dirname "$0")/.."
agent_checkpoint="${1:-runs/coding-v1/foundation/last.pt}"
agent_output="${2:-runs/programming-v1}"
agent_steps="${3:-6000}"
.venv/bin/python -m microgpt.finetune --data data/programming-sft-v1 \
  --checkpoint "$agent_checkpoint" --out "$agent_output" --steps "$agent_steps" \
  --accumulation 2 --lr 0.0001 --eval-every 500 --eval-examples 40 \
  --device cuda --attention flash --max-length 32768 --balanced-lengths \
  --source-weight opencodeinstruct:0.35 --source-weight opencoder-educational:0.20 \
  --source-weight code-feedback:0.15 --source-weight magicoder:0.10 \
  --source-weight educational-tools:0.10 --source-weight programming-tools:0.05 \
  --source-weight programming-direct:0.015 --source-weight horizon-session:0.015 \
  --source-weight horizon-handoff:0.01 --source-weight conversation-replay:0.01
