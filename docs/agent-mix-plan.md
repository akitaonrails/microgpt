# Longer mixed agent training — October 7, 2026

The selected scratch-trained 238,568,448-parameter checkpoint solves 24/100
standalone development problems, but solves 0/4 real OpenCode repair smoke tasks
with the general assistant prompt. All four produced zero tool calls. The
code-only continuation has not learned the interaction policy needed by OpenCode.
The baseline is preserved in `agent-best24-baseline.json`; this is a four-task
smoke test, not an estimate of broad programming competence.

## Frozen experiment

Start a fresh optimizer from `runs/verified-code-learning-v1/checkpoint-500.pt`.
Use `data/programming-agent-mix-v1`: 92,214 training code examples, 14,061 executed
repair trajectories, 11,319 SmolTalk and 3,796 multilingual examples. Validation
has disjoint source groups; public data has not been semantically decontaminated
against every possible benchmark. No acceptance tasks enter training.

Sample sources with probabilities 0.50 code, 0.40 repairs, 0.08 SmolTalk and
0.02 multilingual. These are episode probabilities, not token proportions. Each
repair demonstrates reading the actual file, observing a failed check, making a
scoped edit, passing the check and completing. These are reference-policy
trajectories, not successful model rollouts.

Use LR 1e-5, effective batch 16 (microbatch one, accumulation 16), seed 20261008,
validation seed 20261007, 256 validation episodes, assistant-only loss and an 8K
whole-episode limit. The model's allocated context remains 32K; this experiment
makes no claim about reliable 32K use. Keep 75% soft GPU pacing and nice 10.

Budget up to six hours including checkpoint evaluations, with updates at 250,
1,000, 3,000, 6,000 and 12,000. The last evaluation may finish beyond the budget;
the trainer saves when its remaining time expires. At 12K updates, 192K episode
draws would give about 76,800 repair draws (5.46 per stored repair on average).
Sampling is with replacement, so this is not 5.46 complete epochs. A direct-code
plateau alone does not terminate this agent-policy curriculum before coverage.
Do not automatically extend a deteriorating recipe or promote any checkpoint.

A disposable weighted-data profile measured ~1.84 seconds/update with math
attention and ~0.84 with Flash at microbatch one (three measured updates only).
Math peaked at 6.03 GB allocated. Choose math for this long desktop run because
previous training hit driver watchdog timeouts and the math fallback has a longer
stable record; Flash was not proven to cause those timeouts. Inference continues
using Flash. The profile does not prove sustained throughput or worst-case VRAM.
Checkpoints go under the authorized NAS dataset directory, not the nearly-full
local data disk. Cold NAS loads get a five-minute evaluation startup allowance.

## Evidence and decisions

At every milestone, run the same frozen 100 direct-code development tasks and
four real OpenCode repairs with the unchanged general prompt. Record final-code
correctness, tool use, successful final check after the last edit, and normal
completion. Preserve every checkpoint; choose based on behavior, not loss alone.
The controller's direct-only switch suppresses the much larger fresh-agent suite;
its explicit repair-evaluation switch still runs actual OpenCode.

First-stage progress means tool use on at least three repairs and at least two
fully successful repairs while retaining at least 19/100 direct successes. These
are diagnostic progression thresholds, not project readiness. Inspect regressions
and malformed calls before extending. If no checkpoint learns tool use, inspect
training/serving protocol alignment and held-out repair action likelihood before
spending more time on the same curriculum. Never train on these four solutions.

A qualifying checkpoint proceeds to fresh programming tasks and then the prepared
source-disjoint cumulative-session curriculum with short-code replay. Evaluate
4-, 8- and 32-request sessions, earlier-code preservation, recovery and handoffs.
The existing broad readiness gates remain unchanged. Long context allocation,
valid tool syntax and explanatory text do not establish long-horizon competence
or deep reasoning.

Verification ownership: the experiment controller records commands, hashes and
checkpoint outcomes; the restricted Docker grader establishes code correctness;
OpenCode event traces establish actual tool use and completion. Existing training
regressions cover weighted sampling and resume identity. Run their focused suite
and Python compilation before launch. Existing 66-test grading evidence remains
valid unless grading code changes. The live service on port 8000 stays unchanged.

Run with `nice -n 10 .venv/bin/python scripts/train_agent_mix.py`. Status is in
`runs/agent-mix-v1/experiment.json` and `runs/agent-mix-v1/pilot/status.json`.
