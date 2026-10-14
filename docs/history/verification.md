# Scratch-only verification

Current evidence: [capacity pilot and diagnostics](capacity-pilot.md), with 35 passing
unit tests. General programming and long-session acceptance gates remain unmet.
The entries below preserve the earlier teaching checkpoint and experiments.

## Historical served checkpoint

The selected deployment is `runs/agent-ready-v1/model.pt`: 90,969,600 parameters,
random-origin foundation followed by one focused SFT stage, selected at step 2250.
It passed four 16-round cumulative OpenCode sessions and a 32-round session
(130 calls, 27,528-token peak prompt), plus 8/8 narrow arithmetic repairs.
Broader repairs scored 0/4; reasoning scored 0/12 both directly and with high effort;
arbitrary exact retrieval scored 0/9 across 8K/16K/32K. Harness-maintained filesystem
state supports the successful sessions. These results do not establish general
coding, autonomous summarization, conversation, or deep reasoning capability.

All 25 CPU tests pass. Exported weights equal the selected training checkpoint;
the bundled tokenizer loads without the tiktoken registry/network. Exact-prefix
cache checks and profiled Flash dispatch through a 32K allocation passed.
See the [reproducible runbook](../guides/agentic-minimum.md) and [evidence](../evidence/agentic-evidence.json).
The dated progress entries below preserve the experimental history.

The sole implementation target is our GPT initialized randomly. The earlier
pretrained-model experiment has been removed, including its project checkpoints,
prepared tokenized conversations and results. It supplies no evidence for this
model. Public source datasets and the original scratch-demo artifacts are retained.

Owner of these checks: the implementing agent. Verification distinguishes code
correctness, training execution, protocol compatibility and actual model ability.

## Historical progress: larger curriculum

The original run was `runs/coding-v1`; consult `status.json` and each stage's
`metrics.jsonl` for live progress. It continues our random-origin pilot, with
the same 90,969,600 parameters and frozen o200k_base tokenizer. No external model
weights or rented GPU are involved. Its final capability results failed the targets; see the selected follow-up above.

Prepared data: 947,532,764 training tokens; 62,366 training / 3,475 held-out SFT
episodes, with 33,785,800 / 2,240,035 assistant target tokens. The complete-episode
preparation retains 762 training examples with 32+ calls and 249 with more than
16K tokens. Sources, exclusions and limits are in [the runbook](training-plan.md).

The first 12,000 updates consumed 98,304,000 additional token exposures and lowered
the new mixture's fixed validation loss from 6.5590 to 3.0778. Throughput was
roughly 55K tokens/s, peak PyTorch allocation 7.77 GB. This loss is on a different
corpus from the original pilot and must not be compared directly with its loss.
Neither this improvement nor its throughput establishes model usefulness.

Checks completed for the new pipeline:

- 14 focused CPU checks pass, including packed/unpacked assistant targets,
  indexed-episode integrity/selection, source-mixture RNG replay, and the
  constructed error-recovery/handoff records. Compilation and dependency checks pass.
- Real 5090 execution of short SFT and the final native long-horizon preparation
  (`agent-sft-v3`), with source/length balancing across all three bins. The finite
  budget run with 10% short replay saved and stopped after one update. Its resume
  restored optimizer/RNG and reproduced validation loss 5.566626799923048 before
  continuing to step 2. These are execution checks, not model capability results.
- A real native SWE-smith episode with 32,698 input tokens, 5,303 assistant targets
  and 25 tool calls completed backward propagation with forced Flash Attention
  and finite gradients. Peak PyTorch allocation was 1.96 GB, excluding optimizer
  state in this isolated check (`runs/native-32k-check.json`).
- The foundation runner saved on SIGTERM at step 4,725 and resumed with the same
  validation loss (3.581614). It preserves the original token schedule and subtracts
  elapsed time from the remaining stage budget.
- The 16-round horizon grader rejects the empty fixture and accepts its trusted
  reference implementation inside Docker. This is explicitly a harness self-test,
  not a model result (`runs/horizon-harness-check/results.json`).
- Installed OpenCode accepts the 64-step agent, compaction reserve and restricted
  read/edit/check configuration. End-to-end model results are queued after training.

The new long-horizon gate checks actual tool calls, cumulative correctness, a
transient tool failure and an explicit state-file handoff. A separate retrieval
evaluation tests 8K/16K/32K positions. Passing either would still be narrower than
general long-horizon coding; automatic model-written summaries are not established
by the harness-managed state file.

## Evidence plan

| Claim | Minimum evidence |
| --- | --- |
| Model weights originate here | Inspect `GPT` initialization, training lineage and checkpoint load paths; no external model dependency |
| Correct autoregression | Causal-prefix and shifted-target checks |
| Correct assistant objective | Mask test plus selected-position/full-logit loss and gradient equivalence on our GPT |
| One training/serving format | Compare the SFT prefix directly with the serving serializer |
| Usable checkpoints | Pretrain → full-weight SFT → resume → serve, with matching tokenizer identities |
| Flash/cache correctness | Cold/warm and changed-prefix equivalence; profiler at 8K and 32K |
| Actual agent ability | Real OpenCode attempts, independently graded; failures reported without qualification |

Ten focused CPU tests passed after removing external model dependencies, in about
0.45 seconds after imports. They cover causal masking, cached attention alignment,
activation-checkpoint gradients, tiny-sequence learning/resume, data splits,
UTF-8/o200k round trips, assistant masking, and selected-logit loss/gradients.
Additional integration evidence and any limitations are recorded below.

## Data and initial scratch pilot

The NAS holds the source dataset selection (1,656,295,759 compressed/raw bytes),
curated JSONL and prepared arrays under `/mnt/gigachad/microgpt/dataset`.
The mixed scratch pretraining corpus contains 6,073 train and 327 validation
documents, 7,848,251 and 460,255 o200k_base tokens. It includes raw code, commit
pairs, coding instructions, everyday chat and multilingual instructions.
Pretraining and SFT use the same stable 5% group partition.

`configs/student.toml` yields **90,969,600 parameters** with o200k_base, all updated
in both training stages. Context ceiling 32,768; first-stage windows 2,048; eight
layers, width 384, six heads. No learned weights were downloaded.

The initial 100-update GPU pilot reduced validation loss from 12.2787 to 7.9851,
processing 819,200 token exposures. Reported throughput was 50,598 tokens/s with
7.395 GB peak PyTorch allocation. It was continued to a larger pilot budget; that
schedule extension is not identical to planning the larger cosine schedule at
step zero. These are execution/learning checks, not assistant-quality evidence.

Prepared scratch SFT uses the exact native protocol and fixed tokenizer: 2,701
training / 98 validation assistant turns; 475,861 / 22,996 target tokens. It retains
1,341 / 60 complete episodes and rejects 79 overlength episodes at 8K. Training
contains 378 coding turns, 1,143 constructed commit-tool turns, 786 conversation
turns, 381 multilingual turns and 13 SWE-smith turns. Validation lacks retained
everyday/SWE-smith groups at this tiny sample size, so its loss cannot establish
those abilities. Broaden held-out groups before a serious quality claim.

## Original offline lesson

The independently trained byte-tokenizer lesson has 886,016 parameters. Its
600-step run reduced validation loss from 5.5915 to 0.2470 and completed a familiar
addition pattern. The corpus has near-identical templates, so this establishes
pattern learning rather than unseen algorithm solving.

## Full native training path

The continued pretraining pilot reached step 1,000 (8,192,000 token exposures),
with validation loss **4.96085**. The continued segment reported about 49,225
tokens/s and 7.395 GB peak allocation. It used the original unchunked head in
that already-running process; the chunked loss was verified independently after
implementation. Checkpoint: `runs/scratch-pretrain/last.pt`.

Full-weight SFT started from those weights, using no imported backbone. Its fixed
16-example validation loss fell from **6.87336 to 6.22151** over 500 updates,
with 2.146 GB peak allocation and 55.8 seconds elapsed (including concurrent local
verification activity). Resuming restored the identical initial validation loss;
step 501 reached **6.21884**. This is a resume smoke, not proof of bitwise GPU
trajectory equivalence. Checkpoint: `runs/scratch-sft/last.pt`; its lineage records
random initialization and pretraining step 1,000.

A new stage initialized from the pretraining weights completed one actual
**32,768-token training-window** update with four accumulated microbatches,
forced Flash Attention and checkpointed vocabulary-loss chunks. Peak model-process
allocation was **2.330 GB**; validation changed from 5.6586 to 5.6494. This proves
that the long-window update runs locally, not that one update teaches long-context
reasoning. Artifact: `runs/scratch-32k-check/last.pt`. This checkpoint is a separate
capacity check and is not the parent of the SFT pilot.

The vocabulary-head change is checked against the full masked loss and gradients,
including multiple chunks with wholly masked targets. The serving prefix is also
compared directly to the prepared SFT prefix. All ten focused tests pass without
Transformers, PEFT or Accelerate installed; package requirements pass `uv pip check`.

The native SFT checkpoint passed cold/warm and changed-prefix cache-equivalence
checks. At 8,192 prompt tokens plus 16 output tokens, peak allocation was 0.675 GB;
at 32,752 plus 16 it was **1.015 GB**. The profiler observed Flash Attention kernels
in both cases. Profiled times were 2.17 and 4.94 seconds; they include profiling
overhead and are not a throughput benchmark. Filler inputs establish capacity,
not useful long-context retrieval. Raw output is stored beside the SFT checkpoint
in `inference-check.jsonl`.

## Actual OpenCode result

OpenCode 1.18.34 connected to the native checkpoint and completed all four
regression sessions through the local HTTP/SSE endpoint. **0/4 repairs passed**.
The model made no tool calls and generated text until the 2,048-token output cap.
That establishes endpoint compatibility but refutes a claim that this pilot is
already an operational coding agent. Raw events, unchanged files and independent
grades are in `runs/scratch-eval`. Background title requests occasionally received
429 while the single-request server was busy; the main sessions completed.

The project now contains the necessary native training/serving stages, but the
minimal data/compute needed for useful agent behavior has not been established.
The next training experiments need broader data and more learning, followed by
these behavioral gates. No external model is used to conceal that gap.

The native dialogue probes (`runs/scratch-dialogue.json`) returned the same English
greeting to Portuguese, Spanish and English requests and failed contextual name
recall. This small pilot has learned some response format, not useful multilingual
conversation. The server was stopped after checks; no cloud GPU was rented.

## Reasoning extension

The v4 curriculum has 78,192 training and 4,343 validation complete episodes.
Sixteen CPU tests pass, including original program oracles, masked thinking targets,
shared problem groups, and thinking/tool separation. `runs/reasoning-smoke.log`
and `runs/reasoning-resume-smoke.log` record a one-step CUDA Flash Attention run and
resume to step 2 with the four-source selector and 20% replay; the resumed validation
loss exactly matches the saved step-1 value (5.722826563727058). These are plumbing
checks, not evidence of useful reasoning.

`runs/reasoning-wire-check.json` records a scripted response through installed
OpenCode, separate from model capability tests. The curriculum runner was gracefully
restarted at foundation step 65,421 (535,928,832 tokens in this stage), preserving
its checkpoint, optimizer and remaining 12-hour budget. It now uses v4 and includes
the reasoning stage and final equal-budget reasoning evaluation. Full-run reasoning
and agent results are pending.

## Corrective tool training

See [next-run.md](next-run.md) for reproduction and limitations. On the same eight
held-out arithmetic repair instances, the original instruction checkpoint passed
0/8 and the frozen `runs/grounding-v1/candidate.pt` passed 8/8 through real OpenCode
with Docker grading. These are familiar template families with new seeds, not a
general coding benchmark. The four broader repairs still score 0/4. That candidate's
program-tracing scores are 2/12 direct and 0/12 thinking at both output budgets.
The subsequent horizon candidate retains 7/8 arithmetic repairs but fails the
cumulative task; it is not promoted as a successful long-horizon model.
