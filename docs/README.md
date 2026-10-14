# Documentation

Five folders, by purpose. If you only read one file, read the
[experiment closeout](report/experiment-closeout.md).

| Folder | What it holds |
|---|---|
| [`report/`](report/) | The final results: closeout report, training-time ledger, and the [`closeout/`](report/closeout/) evidence behind them |
| [`guides/`](guides/) | How to run each part of the pipeline |
| [`history/`](history/) | Experiment notebooks and plans, in the order they were written |
| [`operations/`](operations/) | Logs of pauses, recoveries and resumes during the long run |
| [`evidence/`](evidence/) | Machine-readable results, hashes and controls referenced by the other documents |

Numbers in older documents describe the model they were written about. The
project went through three model lineages (a 91M model, a 239M model, and the
final 337M model), and a result for one is not a result for another. When in
doubt, the closeout report is the authority.

## Reading paths

- **New to the project:** the top-level [README](../README.md), then
  [Follow one prediction](guides/walkthrough.md), then the
  [closeout](report/experiment-closeout.md).
- **Want to train something:** [training sequence](guides/finetuning.md),
  [datasets](guides/datasets.md), [hardware](guides/hardware.md).
- **Want the lessons, not the code:** the closeout's sections on failures and
  next steps, then [repair investigation](history/repair-investigation.md) and
  [capacity pilot](history/capacity-pilot.md).

## Report

| Document | Contents |
|---|---|
| [experiment-closeout.md](report/experiment-closeout.md) | Final model card, data, training time, checkpoints, benchmarks, failures, next steps, and how to reproduce the scores |
| [training-time-ledger.md](report/training-time-ledger.md) | Wall-clock accounting for the 337M foundation run and its limits |
| [closeout/](report/closeout/) | JSON evidence: selected checkpoint, development and holdout results, code examples, OpenCode, reasoning, dialogue and context probes, checkpoint inventory |

## Guides

| Document | Contents |
|---|---|
| [walkthrough.md](guides/walkthrough.md) | Follow one prediction from documents to a batch, through the model, and back |
| [finetuning.md](guides/finetuning.md) | The minimal scratch-to-assistant training sequence |
| [datasets.md](guides/datasets.md) | Data sources, licenses and the storage layout |
| [hardware.md](guides/hardware.md) | Local GPU settings and an optional Runpod route |
| [opencode.md](guides/opencode.md) | Serving a scratch-trained model to OpenCode, independent checks, longer runs, remote machines |
| [reasoning.md](guides/reasoning.md) | The reasoning curriculum: data, training, serving and what counts as improvement |
| [agentic-minimum.md](guides/agentic-minimum.md) | Runbook and notebook for the 91M narrow arithmetic agent |
| [agentic-training.md](guides/agentic-training.md) | Programming and sustained agent runs on the earlier models |

## History

The experiments as they happened, October 2026. Each records its plan or result
at the time; later documents supersede earlier conclusions.

| Date | Document | Contents |
|---|---|---|
| Oct 4 | [research.md](history/research.md) | Research notes: microgpt, nanoGPT, nanochat and other references |
| Oct 4 | [training-plan.md](history/training-plan.md) | Larger scratch training and OpenCode capability gates |
| Oct 4 | [verification.md](history/verification.md) | Scratch-only verification: the evidence plan and the earlier failed and narrow runs |
| Oct 4 | [next-run.md](history/next-run.md) | Correcting the first full run |
| Oct 4 | [programming-improvement.md](history/programming-improvement.md) | First programming improvement experiment |
| Oct 5 | [revised-programming-plan.md](history/revised-programming-plan.md) | Programming model reset: evidence and gated plan |
| Oct 5 | [capacity-pilot.md](history/capacity-pilot.md) | Controlled capacity pilot on the 239M model |
| Oct 6 | [learnability-plan.md](history/learnability-plan.md) | Programming learnability and optimization control (frozen plan) |
| Oct 7 | [agent-mix-plan.md](history/agent-mix-plan.md) | Longer mixed agent training (frozen plan) |
| Oct 7 | [repair-investigation.md](history/repair-investigation.md) | Why the longer agent run failed, and what to change |
| Oct 7 | [refined-agent-execution.md](history/refined-agent-execution.md) | Executed repair refinement plan (frozen plan) |
| Oct 7 | [foundation-pivot.md](history/foundation-pivot.md) | The 337M foundation-first pivot (frozen plan) |
| Oct 7 | [foundation-pivot-start.md](history/foundation-pivot-start.md) | Execution notes for the pivot |
| Oct 8 | [pivot-evaluation-recovery.md](history/pivot-evaluation-recovery.md) | 2B evaluation interruption and recovery |
| Oct 9 | [pivot-next-experiments.md](history/pivot-next-experiments.md) | Larger next steps considered after the pivot |

Files marked "frozen plan" are hashed by the training scripts that executed
them (`plan_sha256`), so they are kept byte-for-byte as written.

## Operations

Pauses and recoveries during the six-day foundation run, kept because they
explain gaps in the timing and why every resume checks a checkpoint's hash and
step.

| Document | Event |
|---|---|
| [maintenance-pause-2026-10-11.md](operations/maintenance-pause-2026-10-11.md) | Maintenance pause |
| [maintenance-resume-2026-10-11.md](operations/maintenance-resume-2026-10-11.md) | Recovery and the recurring GPU schedule |
| [nas-pause-2026-10-11.md](operations/nas-pause-2026-10-11.md) | Second pause for NAS maintenance |
| [nas-remount-resume-2026-10-11.md](operations/nas-remount-resume-2026-10-11.md) | Resume after the NAS was remounted |
| [pc-pause-2026-10-12.md](operations/pc-pause-2026-10-12.md) | PC maintenance pause |
| [pc-resume-2026-10-12.md](operations/pc-resume-2026-10-12.md) | Resume after PC maintenance |
| [training-resume-2026-10-13.md](operations/training-resume-2026-10-13.md) | Recovery of training and the evaluation queue |

## Evidence

[`evidence/`](evidence/) holds the JSON the documents above cite: corpus
manifests, per-experiment outcomes, grader and checker controls
(`*controls*.json`), profiles and resume records. Two of them are read by
training scripts (`expanded-foundation-corpus.json`,
`clean-foundation-adapter-outcome.json`). The evidence for the final model is in
[`report/closeout/`](report/closeout/) instead.
