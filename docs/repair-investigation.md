# Why the longer agent run failed, and what to change

October 7, 2026. This supersedes the next-step recommendation in
`agent-mix-plan.md`; that file remains the frozen plan for the completed run.

## Finding

The model learned the repair task represented in the curriculum much better than
its 0/4 broader smoke score suggested. It has not learned general program repair.
The immediate priority is broader defects and recovery from actual failures,
with separate measurements for locating the edit and producing its replacement.
There is no evidence yet that increasing model size or reweighting loss alone
will solve the problem.

The 239M model remains randomly initialized in its ancestry. The best retained
standalone coding checkpoint still scores 24/100. No candidate has met the
OpenCode readiness gates, and no service was promoted. No new training was
launched during this investigation; GPU use was limited to diagnostic inference.

## Completed run

The six-hour budget ended at 11,718 updates and 35,583,002 assistant target tokens.
The controller's historical status says `milestones_exhausted` although its last
requested milestone was 12,000: use the saved actual step and `budget_reached`
field, not that status label, when reporting exposure.

| Update | Direct coding /100 | Broad OpenCode repairs /4 |
| --- | ---: | ---: |
| Parent | 24 | 0, no tools |
| 250 | 18 | 0, tools on all four |
| 1,000 | 18 | 0 |
| 3,000 | 20 | 0 |
| 6,000 | 13 | 0 |
| 11,718 | 14 | 0 |

At the final checkpoint, three repairs attempted identical old/new strings. The
fourth changed a constant incorrectly. Three final replies explicitly claimed
checks passed despite failure. Strict grading rejected all four. Lower validation
loss accompanied worse standalone coding, so loss is not a model selection rule.
Full frozen outcome: [agent-mix-outcome.json](agent-mix-outcome.json).

## What the investigation establishes

The 14,770 public repair demonstrations all follow the same sequence:
`read -> check FAIL -> edit -> check PASS -> final success claim`.
Defects consist of 8,389 integer mutations, 3,564 comparison mutations and 2,817
arithmetic mutations. Each starts from a working reference with one AST node
mutated. There are no rejected edits or still-failing repairs after the edit in
these demonstrations, and no examples requiring multi-step recovery.

Only 15,025 of 2,089,674 assistant target tokens (0.719%) differ, by BPE alignment,
from the corresponding no-op edit. This is a comparison against copying the
old string, not a semantic classification of every token. The trainer averages
assistant-token losses within each episode; tool-result observations are masked.
Consequently, excellent aggregate repair loss can conceal the few decisions that
matter. This does not by itself prove loss weighting caused the failures.

A fixed sample of 64 source-held-out repairs was scored with correct histories.
The diagnostic supplies the correct edit location and all correct tokens before
the decision; it asks which token the model prefers where the correct edit and a
no-op first diverge. It does not measure autonomous patch localization.

| Diagnostic | Before mixed training | After 11,718 updates |
| --- | ---: | ---: |
| Correct edit preferred over no-op | 24/64 | 63/64 |
| Correct first divergent token is top prediction | 23/64 | 63/64 |
| Top prediction accuracy on changed tokens | 35.94% | 98.44% |
| All assistant-token negative log likelihood | 4.4077 | 0.00850 |
| Changed-token negative log likelihood | 1.6478 | 0.03922 |

This refutes the strong claim that the model simply failed to learn these narrow
edits under correct context. It leaves edit localization, generated histories,
more complex defects and general coding competence unresolved.

Eight of those held-out sources were then run through actual OpenCode. Each
broken fixture failed and each reference repair passed in restricted Docker
before model evaluation. The final checkpoint completed **7/8**, including a
passing check after the last edit and a normal finish. The remaining failure
selected the wrong line and proposed a no-op. These eight are a diagnostic sample,
not a new benchmark or independent replication of the 64-example sample.

The first diagnostic wrapper embedded all tests in one `exec(...)` source line,
which inflated failure tracebacks. It scored 5/8. Keeping the same eight cases
and using ordinary test lines scored 7/8. Both results are preserved. This small
paired observation shows sensitivity to tool-result presentation; it does not
establish a general improvement rate. Future evaluation must freeze grader output
format, and training should cover reasonable variations rather than depend on
one traceback layout.

Evidence, hashes, sampled IDs, controls and tool traces are in
[repair-investigation-evidence.json](repair-investigation-evidence.json).
The original 160-turn packing/serving template check also remains valid. These
results reduce suspicion of a universal tokenizer or loss-mask failure; they do
not rule out every possible protocol mismatch.

## Revised execution plan

1. **Freeze a broader diagnostic before preparing the next training mix.** Use
   original source-group validation only, with 32 single-mutation repairs,
   32 structural repairs and 32 recovery episodes. Keep all variants of a source
   in its original split. Confirm reference success and broken-code failure in
   restricted Docker with the existing multiseed controls. Freeze grader output,
   prompts, tools and IDs. Keep the four older repairs as a regression smoke
   test, never as a source of training answers. Acceptance tasks remain untouched.
   Report each stratum separately; a gain on easy mutations cannot hide failure
   on structural repair or recovery.
   Record whether the proposed old string exists uniquely, whether the edit
   changes the file, whether the changed region overlaps the injected defect,
   whether tests pass, and whether a failed check is followed by recovery or a
   false success claim. Region agreement is diagnostic only: a different valid
   patch can still pass execution tests. Add a free-generation probe immediately
   before the edit turn so localization is tested without supplying the old string.

2. **Broaden defects using the public programming sources already checked.** Add
   removed guards, omitted statements, wrong returns and empty implementations,
   where applicable. Derive them from source ASTs, not the four smoke solutions.
   Keep only mutations rejected by the source tests whose reference repair passes.
   Vary whether repair requires a local replacement or a function rewrite, while
   preserving an unrelated helper. Preserve source-group splits and record defect
   families. These are proposed additions, not yet implemented data.

3. **Collect failure-and-correction trajectories on training sources.** Run our
   existing model against training fixtures in the isolated harness. Preserve
   actual rejected/no-op edits and failed checks as context, but mask incorrect
   assistant actions from the supervised target (`supervise: false` is already
   supported). Derive a corrective edit from the current file to the checked
   reference, execute it, and retain it only when tests and helper preservation
   pass. Include re-reading after stale-string failures and retrying a transient
   check failure. A final success claim is supervised only after an observed PASS.
   Do not invent recovery evidence or apply reference corrections to evaluation
   attempts. Start with a bounded collection and audit it before scaling.

4. **Run one controlled data comparison.** Start from the preserved 24/100 parent,
   with the previous LR 1e-5, effective batch 16, source probabilities, seeds,
   context limit and attention backend. Change the repair data, not all training
   settings together. Keep the checked code and conversational replay. Reuse the
   completed mixed run as the old-data control, and evaluate its 1K/3K snapshots
   on the new frozen diagnostic. Evaluate the new curriculum at 1K and 3K updates.
   Record examples, target tokens and elapsed time: equal updates with different
   episode lengths are not matched compute or tokens. No immediate PPO, critic
   model or changed-token loss weighting is justified by the current evidence.

5. **Decide before extending to another long run.** Practical progression requires
   at least 24/100 direct coding, an improvement of at least 12/96 repair successes
   over the corresponding old-data control, and at least 16/32 in each new
   structural/recovery stratum. Require no false PASS claims in the recovery
   diagnostic. These are engineering gates, not statistical guarantees. If met,
   continue to 6K/12K within a six-hour budget with the same checkpoint checks.
   If not, inspect localization, replacement correctness and recovery separately.
   Adjust replay only in a subsequent controlled comparison if repair improves
   but coding retention fails. This avoids hiding forgetting behind a repair gain.

6. **Then test fresh construction and cumulative sessions.** Basic repair is not
   sufficient for writing a function from a user request. Evaluate the existing
   fresh-task suite before the prepared cumulative curriculum. Advance through
   4-, 8- and 32-request sessions with code replay, earlier-function preservation,
   transient errors and both continuous sessions and handoffs. Use Flash Attention
   for long-context work after a resource/stability check, retaining 75% desktop
   pacing. The original broad readiness gates stay in force. A 32K allocation,
   valid tool JSON or reasoning tags do not establish usable context or reasoning.

No new dataset download or larger model is the first step: we have evidence that
our existing public repair distribution is too narrow for the requested tasks.
Foundation quality and the model's roughly 85M non-embedding parameters may still
limit generalization. The proposed comparison tests a specific data remedy; it
does not promise that this model will meet all project goals.

## Research that informs the plan

Sequential imitation can encounter observations caused by its own mistakes that
were absent from expert demonstrations. DAgger studies this distribution problem
and motivates collecting corrections on visited states. Our proposed reference-
verified correction dataset is an application of that idea, not a claim to
implement DAgger's algorithm or inherit its guarantees.
[Ross, Gordon and Bagnell, 2011](https://proceedings.mlr.press/v15/ross11a.html).

CodeRL uses unit-test feedback with an actor/critic training method and regeneration
at inference. It supports investigating execution feedback, but its pretrained
CodeT5 results do not establish that RL will rescue our scratch student. Verified
supervised corrections are the smaller experiment to try first.
[Le et al., 2022](https://arxiv.org/abs/2207.01780).

## Reproduction and verification

`scripts/investigate_repairs.py` audits the corpus and scores the fixed held-out
sample without optimizer updates. `scripts/prepare_repair_probe.py` reconstructs
held-out fixtures and validates them exclusively in restricted Docker.
`scripts/eval_opencode.py --tasks-file ... --general-prompt --effort none` runs
those frozen fixtures in real OpenCode. The counterfactual helper has an authored
regression test; the five existing strict-completion tests pass. Both eight-case
wrapper versions had eight positive and eight negative Docker controls. Diagnostic
servers were stopped after evaluation; the user's port-8000 service was untouched.
