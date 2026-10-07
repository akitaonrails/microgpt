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
