# Programming learnability and optimization control — 2026-10-07

The 250M-token foundation continuation finished with coding scores 4, 8, and
11/100 at 500, 1000, and 2000 SFT updates. Its matched original-foundation control
scored 7, 5, and 12. This does not justify extending that pretraining recipe.
Our best earlier recipe remains 24/100; no candidate has been promoted.

First test whether the original scratch 239M foundation can learn 16 short public
programming examples. `scripts/prepare_learnability.py` selects unique groups
from the multiseed-checked public corpus, preserves its hash split, and freezes
16 training examples, 16 unseen-source validation examples, and 16 renamed
versions of the training examples. Only the original training examples receive
updates. All 48 reference solutions passed sandbox tests and all empty controls
failed before training. The ordinary evaluator uses the exact stored prompt.
Seen examples test memorization; renamed examples test limited interface transfer;
unseen-source examples test transfer within this curriculum. None is an independent
acceptance benchmark; earlier foundation exposure is possible.

The diagnostic starts from `runs/capacity-pilot-v1/foundation/last.pt`, LR 1e-4,
math attention, microbatch 4, accumulation 8, seed 20261006, validation seed
20261007, 75% GPU pacing. Evaluate executable correctness at updates 100, 300,
and 600. Maximum training/controller budget one hour. A score of at least 15/16
on seen examples establishes narrow learnability, not general programming skill.
If this is unmet, inspect per-example outputs, loss, targets and serving parity
before launching broader adaptation. Higher unseen/renamed scores are diagnostic,
not a condition for passing this basic memorization check. No diagnostic checkpoint
will become the parent of a broader candidate.

Once learnability is established, compare LR 1e-4 against the completed clean
LR 1e-5 control from the same original foundation. Keep the original frozen
98,694-example data, seed, batch, math backend, validation subset and milestones
500/1000/2000 unchanged. Filtering the corpus in only one arm would confound this
comparison, so the newly rechecked corpus is reserved for the subsequent recipe.
A gain of at least five development tasks over 12/100 warrants extending the
higher-rate arm to 4000 and 6000 updates (192,000 draws, about two dataset lengths,
with replacement, not two exhaustive epochs). Stop extension if correctness
declines at both checkpoints. No automatic promotion or acceptance-set access.
This is one fixed-seed optimization diagnostic, not statistical proof of optimality.

Before agent training, select a demonstrably stronger direct-code checkpoint and
measure fresh OpenCode completion, then train checked repairs/tool workflows and
progressively evaluate cumulative 4/8/32-request sessions. Keep earlier-function
regression checks, requirements handoffs and recovery observations explicit.
A 32K allocation is not evidence of useful recall or deep reasoning.

New checkpoints live under `data/experiments/` on the authorized NAS, preserving
local disk headroom. Public and generated Python runs only in restricted Docker.
Verification owner: the main agent. Evidence: source/split manifests, sandbox
reference and negative controls, actual generated-code results, existing training
and pacing regression tests, and hashes/settings recorded before each run.

Initial execution evidence: all 32 original diagnostic examples have exactly
matching training/evaluation prompt token prefixes, with assistant-only labels.
The existing 12 training regression tests pass. At 100 updates the diagnostic
scores 16/16 seen, 0/16 renamed, 1/16 unseen-source. This establishes narrow
memorization while directly showing poor transfer. The diagnostic is not a
candidate model. NAS cold loading of its optimizer checkpoint added roughly two
minutes to controller startup; subsequent evaluation loading was faster. Storage
capacity and checkpoint I/O latency are separate constraints.

The diagnostic completed all three checkpoints with identical scores: 16/16 seen,
0/16 renamed, 1/16 unseen-source. At 600 updates it had sampled 2,336,082 assistant
tokens from just 1,942 stored training target tokens; more repetition did not
improve transfer. Full evidence is in `learnability-outcome.json`. The higher-rate
control started after this run ended, using the original foundation, not these
memorized weights. Its live record is `runs/optimization-lr1e4-v1/experiment.json`.

The higher-rate arm scored 9/100 at 500 updates versus 7/100 in the matched
control. It consumed exactly the same 2,454,569 assistant tokens. At 1000 updates,
training saved successfully, but cold NAS loading exceeded the evaluation server's
old 60-second startup allowance. `continue_programming.py --evaluate-current`
now evaluates the already-saved first milestone without another optimizer update;
reusing an existing checkpoint requires it to be the same hardlinked file as
`last.pt`. The startup allowance is now 300 seconds. Recovery uses a fresh output
directory, preserving the failed attempt, then continues to 2000 with the saved
optimizer/RNG and identical training settings. This infrastructure failure is not
a model accuracy result.

The repair-only evaluator now requires a correct final file, the most recent
check to pass after the last edit, and a clean final stop. Five regression tests
cover unchecked success, recovery, errors, exhausted runs and stale checks.
Offline regrading of eight archived OpenCode runs preserved their 7/8 score.
The direct-code comparison is unaffected. Apply the same completion rule to the
fresh/cumulative agent evaluators before the next agent trial.

An additional CPU-only check of the original foundation scored 0/16 on the exact
seen-training prompts, versus 16/16 after diagnostic adaptation. This is a
qualitative baseline because CPU SDPA and GPU Flash use different numerical
execution. The first CPU launch rejected `--attention math`: the server CLI accepts
`sdpa` or `flash`, while the trainer also exposes `math`. Retrying with CPU SDPA
completed all 16 requests; no training checkpoint changed.

Next, evaluate the strongest retained direct-code checkpoint on actual OpenCode
repairs before further agent training. `eval_opencode.py --general-prompt` selects
the general assistant instruction used in the public repair curriculum and records
that choice. The older specialized repair-lesson prompt remains the default for
historical reproducibility. Keep the general prompt fixed for the new before/after
comparison; these four repairs are a smoke test, not the broader readiness gate.

The higher-rate control finished at 9, 10, and 7/100 (low-rate control: 7, 5, 12).
Its requested-parameter matches fell from 74 to 47 to 38/100. Lower validation loss
therefore did not establish better instruction following. Do not extend this
higher-rate raw-foundation recipe. Full evidence: `optimization-control-outcome.json`.
The user explicitly authorized longer beneficial training on October 7. Longer
runs remain appropriate for a stronger recipe; the short diagnostic budgets are
not a project-wide training limit. The strongest retained checkpoint is still
`runs/verified-code-learning-v1/checkpoint-500.pt` at 24/100.

An authored sandbox control exposed another grader issue: `SystemExit(0)` during
solution import bypassed all tests but was accepted as a pass. The Docker wrapper
now returns status 73 only after the grader finishes; both offline grading and the
actual OpenCode check plugin use that completion status. Correct, incorrect, and
early-exit fixtures were tested through real OpenCode using scripted local replies
(no model or cloud inference). This rejects accidental early exits, not deliberate
grader manipulation. All 700 saved outputs from the six matched checkpoints and
the selected 24/100 checkpoint were regraded; no score changed. Fresh and cumulative
agent evaluators now share the strict last-check-after-last-edit completion rule.
