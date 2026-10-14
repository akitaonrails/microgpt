# Executed repair refinement plan — October 7, 2026

This implements the next comparison from `repair-investigation.md`. The objective
is improved executed repair and recovery while retaining the preserved 24/100
coding baseline. The weights remain our 238,568,448-parameter scratch lineage;
no external model, cloud rental or service promotion is involved.

## Data and frozen evaluation

The new structural builder selected 6,000 original training sources and 192
validation sources, with one source per original split group. It retained 5,513
training and 177 validation sources after executing broken and repaired programs
under hash seeds 1, 2 and 3. It requires a completed reference check (exit 73), a
failed broken check (exit 1), and preservation of an unrelated helper. Defects
cover omitted guard bodies, omitted loop bodies, wrong returns and empty function
implementations. The first attempt encountered an unchanged `return None`
mutation; the fixed generator rejects unchanged text, and a regression covers it.
The failed attempt is preserved separately; it does not enter training.

Each retained source supplies a plain structural-repair demonstration and one
reference-policy recovery variant: rejected no-op edit, stale edit text, or a
single transient checker failure. Incorrect edits are context with masked target
labels. Rejection and retry behavior is executed inside the sandbox; these
constructed lessons are explicitly not model rollouts.

`data/repair-diagnostic-v1/evaluation.jsonl` freezes 96 distinct validation groups:
32 old single-mutation repairs, 32 structural repairs and 32 structural repairs
with a real one-shot checker fault. It was frozen before new training. Initial
broken/reference controls come from the multiseed-checked source artifacts.
The original four broad repairs remain a separate regression set. Runtime fault
injection uses the existing isolated OpenCode check tool, after the initial
fixture sanity check. Each case requires correct final code, a successful check
after the last edit and a normal finish.

A separate 64-case **training-group** sample runs through actual OpenCode using
the previous 11,718-update model. A loopback proxy captures its actual requests
and the solution.py contents immediately before each model action. Snapshot reads
are restricted to this evaluator's temporary work directories. No unrelated
session, user data or credentials are collected. A reference correction is built
from each failing captured state and must pass the same three-seed controls.
Previous model actions are masked, so failures are not taught as correct outputs.
The appended verified correction is a teacher continuation, not a model success.

The comparison keeps the old overall source probabilities: 50% checked coding,
40% repairs, 8% SmolTalk and 2% multilingual, sampled by episode. Within repairs,
roughly one third are old simple repairs, one third structural repairs and one
third constructed recovery episodes; the bounded model-failure corrections are
appended. All original split groups are preserved. The runner checks train/val
group separation and rejects a prepared dataset that drops every actual model
correction at the 8K whole-episode limit.

## A runtime-format finding

Captured OpenCode 1.18.34 CLI requests wrap user text in quotes and escape embedded
quotes, but preserve actual newlines and existing backslashes. Older generated repair prompts used JSON-escaped
strings. The refined corpus corrects generated repair prompts to match captured
requests; actual captured histories are preserved. This means the data comparison
includes both coverage/recovery improvements and a prompt-serialization correction;
it cannot isolate their individual contributions. This mismatch does not explain
the four older broad failures by itself: those prompts are simple single-line
text. The earlier synthetic packing/serving equality checks were valid for their
supplied messages but did not establish equality with actual multiline OpenCode
requests. This distinction belongs in the educational article.

## Run and decision sequence

`scripts/run_refined_agent.py` prepares the captured corrections, assembles and
packs the new curriculum, and evaluates the preserved old-data 1K and 3K
checkpoints on the new frozen 96-case diagnostic. These control weights are reused;
there is no need to repeat the old training run.

The new run starts from `runs/verified-code-learning-v1/checkpoint-500.pt`, using
a fresh optimizer, LR 1e-5, seed 20261008, microbatch 1, accumulation 16, math
attention, 8K whole episodes, 256 validation examples and validation seed
20261007. This matches the previous mixed run's optimization settings. Counts,
target tokens and wall time are recorded; different episode lengths mean equal
updates do not imply equal tokens or compute. GPU duty-cycle pacing stays at 75%
and processes run with nice 10. Training checkpoints remain inside the authorized
NAS dataset path.

After a 250-update save, evaluate at 1K and 3K: 100 direct coding tasks, the four
broad repairs and all 96 stratified repairs. The training/evaluation phase has a
six-hour budget, separate from data preparation and old-checkpoint evaluation;
an evaluation already started can finish beyond the deadline. Extension to 6K
and 12K uses the remainder of that same budget, only if the **current 3K state**:

- retains at least 24/100 direct coding successes;
- improves at least 12/96 over the corresponding old-data control;
- reaches at least 16/32 in structural repair and 16/32 in recovery;
- has no explicit unsupported PASS-claim flags anywhere in the diagnostic.

The last check is deliberately stricter than checking recovery cases alone.
The flag is a text heuristic for explicit claims that tests/checks passed when
verification disagrees; it is not a complete truthfulness judge and flagged
language requires review. No checkpoint is automatically promoted, even if the
progression gate passes. A regressed 3K checkpoint is not extended merely because
an earlier checkpoint passed. If the gate fails, preserve the result for the next
diagnosis rather than silently extending the same recipe.

Fresh construction and then 4/8/32-request sessions remain subsequent gates.
Long-context training and reasoning capability are not established by this stage.
Source tests may be incomplete; source-group separation is not semantic benchmark
decontamination. The model's size and foundation quality remain possible limits.

## Verification and artifacts

Authored regressions cover meaningful mutations, rejected edits, masking failed
actions, snapshot scope, captured prompt decoding and explicit unsupported success
claims. Public Python executes only inside the restricted Docker boundary. The
96-case evaluation exercises actual OpenCode, including the one-shot fault.
Data manifests contain output hashes, and the experiment records the parent,
diagnostic, plan and prepared-data hashes plus exact subprocess commands.

Prepared data: `data/structural-repairs-v2`, `data/repair-diagnostic-v1`,
`data/repair-training-rollouts-v1`, then `data/rollout-corrections-v1` and
`data/refined-agent-sft-v1`. Runtime state: `runs/refined-agent-v1/status.json`;
checkpoint evaluations: its `pilot/status.json`. Checkpoints:
`data/experiments/refined-agent-v1`.
