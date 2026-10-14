# MicroGPT Lab: experiment closeout

Closed at the user's request on October 14, 2026. No further training is queued.
This report describes the final artifacts and measured results; earlier plans
remain intact as historical records. Numbers refer to the **337M pivot** unless
explicitly labeled otherwise. GB means decimal bytes.

## Outcome

We built a readable, working pipeline that trains a language model from random
weights, prepares public data, fine-tunes on checked programming answers, serves
an OpenAI-compatible API, and connects to actual OpenCode. The final model can
generate useful small Python functions for some requests. It is **not a useful
general OpenCode agent**, and the original goal of reliable simple programming
tasks over long sessions was not achieved.

| Final selected model, one greedy attempt per case | Result |
|---|---:|
| Development programming suite, used for checkpoint selection | 68/100 |
| Reserved programming holdout, evaluated after selection | 58/100 |
| Holdout outputs that parse as Python | 100/100 |
| Holdout outputs with requested function and parameter names | 97/100 |
| Actual OpenCode repository-repair smoke suite | **0/4; zero tool calls** |
| Exact program-tracing answers, 256 output tokens, effort `none` | 0/12 |
| Same tracing problems and budget, effort `high` | 0/12 |
| Synthetic retrieval at 8K / 16K / 32K | 0/3 / 0/3 / 0/3 |

The last four rows were measured during closeout, without further training or
checkpoint selection. Long prompts completed without HTTP errors, but answers
were repetitive or irrelevant. In OpenCode, the model produced chat/code text
instead of reading, editing, or checking the file. The server/client connection
worked; the requested agent behavior did not.

These are small local diagnostics, **not HumanEval, MBPP, SWE-bench, or a general
58% programming success rate**. Each 100-case coding split has only 20 families
with five correlated naming/wording variants. The development set was repeatedly
used to make decisions. Holdout families differ, but public-training-data overlap
was not exhaustively excluded. A single seed, greedy generation, limited tests,
and small suites limit conclusions. No statistically controlled ablation isolates
which part of the final recipe caused the improvement.

## Goal and project history

The initial educational target was a cleaner, modular counterpart to Karpathy's
microgpt, progressing from a visible training loop to a small practical coding
model. The scope grew to conversation, multilingual text, reasoning, tool calls,
prompt caching, and long OpenCode sessions. The user explicitly required training
from scratch; the retained model lineages use our own random-origin weights.
No Qwen or Genesys weights are in the final model. Public teacher-generated
answers are nevertheless supervision: scratch initialization does not mean that
all training text was authored without other models.

We studied microgpt, nanoGPT, nanochat, LLMs from Scratch, small-model recipes, and
the local `../llm-coding-benchmark/lua-forensics` assets. The Genesys material
provided questions about curriculum and reproducibility, not an independently
verified architecture or reusable training recipe. Token accounting at an API
does not establish internal tokenizer identity or weight ancestry. No Genesys
corpus was used. See [the original research notes](../history/research.md).

| Stage | What happened and what it taught us |
|---|---|
| Byte-tokenizer lesson, approximately 0.886M parameters | A CPU-sized implementation makes tokenization, autograd, next-token prediction and sampling inspectable; narrow patterns are not an assistant. |
| 90,969,600-parameter o200k student | Foundation and multiple assistant/reasoning/handoff curricula produced a narrow arithmetic agent. One retained model passed 8/8 arithmetic repairs and a 32-round, 130-tool-call session. It still failed 0/4 broader repairs and 0/12 tracing questions in both effort modes. |
| User-discovered task switch | Asking that arithmetic agent for a greeting function triggered repeated irrelevant arithmetic edits, nonexistent paths, and finally invalid tool JSON. The lesson's long-session score did not transfer to arbitrary simple requests. |
| 238,568,448-parameter capacity candidate | Twelve layers, width 768, o200k vocabulary: about 154M parameters were embedding/output and only 85M were transformer/norm. More parameters and lower loss alone did not solve coding. |
| Interface/optimization experiments | Broader interface data reached 17/100 at 10K SFT updates, then 14/100 at 20K. A larger-batch recipe reached 15/100. Prompt-layout continuations both scored 7/100 despite some interface improvements. |
| Independently checked public-code SFT | The older 239M lineage reached 24/100 at 500 and 1,000 updates; 2,000 regressed to 21. A clean foundation-to-SFT control peaked at 12; an extra 250M foundation-token tranche peaked at 11. A separate higher-LR control peaked at 10. These are recipe comparisons, not universal limits. |
| Six-hour mixed agent run | Ended at 11,718 updates / 35,583,002 assistant tokens. Tool use appeared, but all four broad repairs still failed and direct coding fell to 14/100. Some outputs claimed tests passed when they had not. |
| Repair investigation | On supplied correct histories the model strongly preferred the correct narrow edit; actual OpenCode passed 7/8 similarly distributed held-out repairs. It failed on broader structural/recovery tasks. This separated narrow learning from autonomous generalization. |
| Refined repair preparation | Prepared structural defects and recovery examples, including 64 corrections of actual training-source model failures. The old 1K control scored 14/32 simple, 0/32 structural, 0/32 recovery cases. The new run was superseded at step 829 before a conclusive behavioral comparison. Interruption is not a negative result. |
| Foundation-first pivot | Randomly initialized a 337M model with a smaller 32K tokenizer and much more transformer capacity, then trained through 2B, 5B and 10B tokens. Fixed coding probes rose from 39 to 52 to 65/100. Longer SFT selected 68/100, followed by 58/100 holdout. |

Detailed earlier results remain in [capacity-pilot.md](../history/capacity-pilot.md),
[agentic-minimum.md](../guides/agentic-minimum.md),
[repair-investigation.md](../history/repair-investigation.md), and
[refined-agent-execution.md](../history/refined-agent-execution.md). These are different
lineages: the 337M model did **not** inherit the 91M model's arithmetic tool skill
or the 239M model's mixed-agent training.

## Final model specification and identity

| Property | Selected artifact |
|---|---|
| Parameters | **337,167,360** |
| Tied embedding/output parameters | 33,554,432 |
| Transformer/norm parameters | 303,612,928 |
| Decoder layers / width | 24 / 1,024 |
| Attention | 16 full multi-head attention heads, head dimension 64; no GQA |
| MLP | SwiGLU, hidden width 2,752 |
| Normalization / position | RMSNorm / RoPE, base 10,000 |
| Vocabulary | 32,768-entry local byte-level BPE, EOS ID 0 |
| Initialization | Random, seed 20261007; residual output/down projection scaling by `1/sqrt(2*layers)` |
| Context allocation ceiling | 32,768 tokens including prompt and generated output |
| Actual foundation training window | **2,048 tokens** |
| Code-SFT maximum episode length | 8,192; this is a limit, not evidence that most examples are that long |
| Attention implementation | PyTorch SDPA with Flash backend on CUDA, not a custom FlashAttention build |
| Weight storage | FP32; BF16 autocast for CUDA computation |
| Foundation exposure | 10,000,007,168 tokens, 305,176 updates |
| Selected SFT exposure | 4,000 updates, 10,341,997 supervised assistant tokens |
| Full selected checkpoint size | 4,046,192,859 bytes, including optimizer/training state |
| Weight payload alone | 1,348,669,440 bytes theoretically at FP32, before serialization metadata |

Selected checkpoint:
`data/experiments/foundation-pivot-v1/adapter-10b/checkpoint-4000.pt`

SHA-256, rechecked at closeout:
`0a8768e96cb766cb41587c704a263498a0f9438f0eb5c1249825d59d7795f081`.

Foundation parent:
`data/experiments/foundation-pivot-v1/foundation-resumed-5b/checkpoint-10b.pt`,
SHA-256 `60afbcbab54aa111587821c4c0d071a134e2e0ad583ff14d670a336b529e854f`.
Tokenizer fingerprint:
`09c11b442be4ba8ce7a70839ded42805953a99ccc6133bca50c044ac7865682a`.

The selected model is **not** `adapter-10b/last.pt`: that is the later 8,000-update
continuation, which scored worse. “Adapter” is a directory name here; SFT updates
the full model, not a LoRA module. The old `runs/agent-ready-v1/model.pt` is a
different 91M arithmetic lesson, not this selected model.

Earlier recipes supported `o200k_base` and `o200k_harmony`. The final pivot moved
to local BPE because the old large vocabulary consumed most parameters. UTF-8
coverage remains, but multilingual ability is not implied. Harmony vocabulary
support never meant implementation of the Harmony wire format. OpenCode uses our
HTTP/chat/tool protocol and does not require OpenAI tokenizer IDs.

Exact settings and ancestry: [selected-checkpoint.json](closeout/selected-checkpoint.json).

## Data: downloaded, prepared, and consumed are different numbers

The NAS root is `/mnt/gigachad/microgpt/dataset`; repository `data` resolves to its
`project-data` subtree. All large datasets and final-pivot checkpoints live there.
The 14 raw-download manifests account for **20,404,443,820 bytes (20.40 GB)**.
This is the sum of selected manifests, not total NAS usage, unpacked text, cache
size, or the amount consumed by the selected model. The five foundation sources'
raw selections account for 10.69 GB of that total.

### Final foundation corpus

| Source | Sampling weight | Stored training tokens | Training documents | Validation tokens |
|---|---:|---:|---:|---:|
| CodeParrot clean Python | 60% | 3,057,457,819 | 1,767,534 | 169,577,626 |
| OpenCoder stage-1 instruction/code | 16% | 155,161,279 | 250,663 | 8,285,139 |
| Cosmopedia v2 | 12% | 128,895,319 | 165,764 | 6,748,940 |
| SmolTalk | 8% | 83,112,546 | 89,693 | 4,269,487 |
| Multilingual SmolTalk2 | 4% | 51,024,432 | 75,856 | 2,713,491 |
| **Total** | **100%** | **3,475,651,395** | **2,349,510** | **191,594,683** |

Validation contains 125,892 documents. Decoded training text is 13,086,503,731
UTF-8 bytes; validation is 716,548,997 bytes. The uint32 token arrays occupy
14,668,984,312 bytes across both splits. “Documents” and “stored tokens” do not
prove globally unique information after near-duplicate removal.

The prior expanded corpus contained 3,057,260,620 training tokens under o200k.
Retokenization preserved documents, order and split membership; the increase to
3.476B is primarily a **tokenization change**, not newly downloaded knowledge.
The BPE was fitted on 92,047,594 training-only UTF-8 bytes, sampled across sources.
Document boundaries and text round trips were checked, including literal EOS
spellings inside source text.

The 10B training budget is about 2.88 times stored training-token count, but it is
not 2.88 uniform epochs. Sources are sampled at the weights above and windows
repeat. In expectation, CodeParrot receives about 6B tokens (~1.96 stored-corpus
equivalents), OpenCoder 1.6B (~10.31), Cosmopedia 1.2B (~9.31), SmolTalk 0.8B
(~9.63), and multilingual 0.4B (~7.84). These are mixture-based expectations;
per-source realized exposure was not logged as an exact total.

### Final supervised coding corpus

The source is public `nvidia/OpenCodeInstruct`, distinct from OpenCoder. We
downloaded its pinned 50-shard selection, checked source answers in restricted
Docker, required an empty-program negative control to fail, then rechecked under
hash seeds 1, 2 and 3. Of 98,694 candidates in the multiseed pass, 97,074 survived.

Retokenization for this model retained **92,214 training episodes** across 90,213
source groups and **4,860 validation episodes** across 4,786 groups. Stored
assistant targets are **14,912,011 train / 790,849 validation tokens**. The selected
4K checkpoint consumed 10,341,997 assistant targets across 64,000 sampled episode
draws. Draws can repeat; that is not 64,000 unique problems or one full epoch.
The later 8K checkpoint consumed 20,711,969 assistant tokens but was not selected.

User/system/tool text is supplied as context but masked out of the SFT loss.
Crucially, this final SFT mix contains **coding answers only**: no broad
conversation replay, tool trajectories, recovery curriculum, or long-session
curriculum. Its tendency to answer unrelated requests with code is consistent
with that narrow training, though we did not isolate the causal contribution.

### Other material prepared or used in earlier lineages

Earlier downloads included CommitPackFT, Magicoder, SWE-smith, CodeFeedback,
Mixture-of-Thoughts and conversational/multilingual subsets. The 91M reasoning
curriculum included 699 complete code traces, 6,000 math traces, 6,000 procedural
direct/thinking records and 3,995 planning/tool variants. Its combined historical
SFT corpus held 78,192 training episodes / 59,773,809 assistant tokens.

We also prepared 14,770 checked scoped-repair demonstrations; 187 cumulative
sessions containing 5,984 user turns, packed into 2,040 complete episodes; and a
refined repair mix containing 17,134 repair/recovery episodes including 64 actual
failure corrections. These counts are **not additive unique training data**:
many derive from the same public programs. They were not all used, and none of
these agent curricula trained the final 337M checkpoint. See
[reasoning.md](../guides/reasoning.md), [public-horizon-corpus.json](../evidence/public-horizon-corpus.json),
and [refined-agent-data.json](../evidence/refined-agent-data.json).

Manifests preserve revisions, source identities, selected licenses and hashes.
Repository/problem grouping and exact deduplication reduce leakage; they do not
eliminate copied repositories, near-duplicates, or public benchmark overlap.
Passing supplied tests is not proof of a correct specification. Dataset/source
license conditions remain attached to their original material; this report does
not relicense the corpus. Raw inventory: [raw-downloads.json](closeout/raw-downloads.json).

## Training procedure, time, hardware, and checkpoints

Foundation: AdamW, betas `(0.9, 0.95)`, weight decay 0.1, peak LR 3e-4, 1,000-update
warmup, cosine decay to 10% over the fixed 305,176-update schedule, gradient
clipping at 1.0. Batch 2 × accumulation 8 × 2,048 positions = **32,768 tokens per
update**. Validation uses 16 batches every 500 updates. Gradient checkpointing
was disabled for this foundation run. We retained one continuous optimizer/RNG
schedule across milestones and valid resumes.

At 2B, 5B and 10B we froze a foundation snapshot and trained separate full-weight
coding branches, each with LR 1e-5, microbatch 1 × accumulation 16, AdamW weight
decay 0.01, seed 20261007, Flash attention, 8K episode limit and 256 validation
examples. Those branches do not modify the continuing foundation optimizer.
SFT uses a constant LR in this recipe. After the fixed 500/1,000-update comparison,
the strongest foundation's coding branch continued through 2K/4K/8K updates.
Selection used development correctness; only the selected checkpoint saw the
reserved holdout.

All reported training used **one local RTX 5090**. No H100 was rented. The tested
stack was PyTorch 2.7.1+cu128, CUDA runtime 12.8 and driver 610.57.04. Foundation
peak allocated VRAM was about **13.88 GB**; final coding SFT about **6.78 GB**.
These are PyTorch allocation measurements, not total system/GPU usage.

| Time measurement | Final value |
|---|---:|
| First foundation launch | October 7, 2026, 16:01:29 São Paulo |
| Foundation completion | October 14, 2026, 02:00:27 São Paulo |
| Calendar elapsed to 10B | **153.98 hours / 6.42 days** |
| Sum of foundation process intervals, including discarded attempt | **136.68 hours** |
| Outside foundation process intervals | 17.31 hours: maintenance, recovery, checks, probes and other gaps |
| First 2B process interval | 28.80 hours |
| Next 3B process interval | 47.30 hours |
| Remaining attempts/resumes through 10B | 60.57 hours, including recovery cost |
| Logged SFT training-segment time across all nine measured pivot stages | **3.40 hours** |
| Logged 10B-to-selected-4K SFT training segments | **1.36 hours** |
| Longer-SFT controller, including its three development evaluations | 2.22 hours |

The selected foundation ancestry excludes the discarded 1.24-hour stale-checkpoint
attempt and six-second failed restart, giving approximately **135.44 hours** of
foundation process wall time; plus selected SFT segments gives **136.80 hours**.
This is a useful lineage estimate, not a complete rerun quotation. Trainer segment
timers omit startup/initial validation and the final checkpoint write; controller
intervals have different boundaries. Evaluation, preparation, retries, hashing,
and failed alternatives must be budgeted separately.

A defensible all-project total including every earlier 91M/239M experiment is
**not available as one complete ledger**. Do not label 136.68 hours as the whole
project. The older mixed-agent run alone used a six-hour budget, and the expanded
239M foundation tranche logged 9,877 seconds. Historical outcome files retain
additional partial timings, but incomplete/overlapping phase accounting prevents
an honest exact grand total. GPU busy-hours, energy use and monetary cost were
not metered.

Short profiles varied from 29,988 tokens/sec on the complete mixture to 43,882 on
a cached single-source prefix. The actual run was much slower at times: after NAS
maintenance a restart averaged ~5.4K, later recovered through ~13K and ~22K; the
final restart averaged 23,061 tokens/sec. The metric is **cumulative since each
restart**, not instantaneous. Earlier chat ETAs based on it were rough estimates.

Desktop contention caused browser hangs despite free VRAM. Cooperative pacing
synchronizes GPU work and sleeps between batches. The later authorized schedule
was full duty from 03:00–09:00 São Paulo and 75% otherwise. This is neither a
hardware partition nor a guaranteed UI latency limit. NAS mmap reads and 4GB
checkpoint writes also contributed to elapsed time; adding GPU utilization cannot
fix an I/O stall. See [final timing ledger](training-time-ledger.md).

### How many checkpoints?

The final-pivot directory retains **20 `.pt` paths, representing 17 distinct
filesystem inodes**. Logical file lengths total **80.92 GB**, or **68.79 GB** after
hardlink deduplication. Neither is a measurement of compressed/deduplicated NAS
physical allocation, and separate inodes can still contain duplicate weights.

There are **12 named experimental milestone snapshots**: three foundation
milestones (2B/5B/10B) and nine coding-SFT milestones (500/1,000 for 2B and 5B;
500/1,000/2,000/4,000/8,000 for 10B). Other paths are `best.pt` / `last.pt`
working aliases or recovery outputs. Foundation evaluates and overwrites
`last.pt` every 500 steps; it does not retain a new permanent file for every save.
The three phase logs contain 620 metric records including maintenance/recovery
records, not 620 independent retained models. Full path inventory is in
[experiment-evidence.json](closeout/experiment-evidence.json).

Across the broader retained `runs/` and `data/experiments/` archive, including
older models, there are **168 `.pt` paths / 134 distinct inodes**, totaling
347.49 GB of logical file lengths or 274.94 GB after hardlink deduplication.
This scoped inventory excludes files elsewhere and does not represent 168
independent training experiments. No checkpoints were deleted during closeout.
See [checkpoint-archive.json](closeout/checkpoint-archive.json).

## Measured learning curve

| Foundation exposure | Raw completion /100 | Coding SFT 500 /100 | Coding SFT 1,000 /100 | Foundation validation loss |
|---|---:|---:|---:|---:|
| 2,000,027,648 tokens | 0 | 24 | 39 | 1.3261 |
| 5,000,003,584 tokens | 0 | 49 | 52 | 1.2108 |
| 10,000,007,168 tokens | 8 | 56 | 65 | 1.0854 |

| Extra 10B-foundation coding comparison | Development /100 | Assistant tokens consumed |
|---|---:|---:|
| 2,000 total SFT updates | 61 | 5,167,227 |
| **4,000 total SFT updates, selected** | **68** | **10,341,997** |
| 8,000 total SFT updates | 62 | 20,711,969 |

Raw completion and instruction prompting are different conditions. The gain from
8 raw passes to 65 after short SFT demonstrates the importance of conditioning on
the requested task in this recipe; it is not an isolated estimate of knowledge
added by SFT. Lower SFT loss continued through 8K while executable correctness
fell. The selected 68 score must not be confused with the separate 58 holdout score.

The direct coding evaluator uses a general system prompt, appends “Return only
the Python code in a fenced block,” allows 768 output tokens, and uses temperature
zero with effort `none`. It extracts the fenced code and runs fixed assertions
in Docker. A pass requires correct code and a normal completed response; there
is no retry or model-written self-correction in this condition.

### What the selected model can do

Holdout family scores, out of five variants each:

| Family | Passes | Family | Passes |
|---|---:|---|---:|
| Median | 5 | Factorial | 5 |
| GCD | 5 | Binary conversion | 5 |
| Invert mapping | 5 | Line count | 5 |
| JSON keys | 5 | Leap year | 5 |
| Seconds conversion | 5 | Running total | 4 |
| Pairwise | 3 | Anagram | 3 |
| Transpose | 2 | Intersection | 1 |
| Deduplicate | 0 | Group by length | 0 |
| Pad | 0 | Split once | 0 |
| Basename | 0 | Strip comments | 0 |

Nine families passed all five variants, five were mixed, and six failed every
variant. The model often emits syntactically plausible Python and docstrings.
The hard part is satisfying the requested semantics and edge cases. Of the 42
holdout failures, three lack the requested function/parameter interface; **39
have valid syntax and matching names but still fail behavior/completion checks**.

Representative saved failures illustrate the limits: deduplication used a set
despite required unhashable-list support; grouping by length counted words instead;
padding attempted arithmetic on each element; split-once returned `(text, '')`
regardless of the separator; basename removed the final character; a transpose
copied rows unchanged. These are saved outputs, not hypothetical examples. Read
[coding-examples.json](closeout/coding-examples.json) for prompts, code, pass/fail
and grader traces, including successful median/factorial cases.

### Agent, reasoning, conversation, and context diagnostics

Closeout used OpenCode **1.19.0**, a private Flash server on port 8037, greedy
generation, effort `none`, constrained tool syntax, read/edit/check tools, a
maximum of eight agent steps and a 180-second per-task timeout. The fixed broken
fixtures were independently rejected by Docker graders before model attempts.
All four tasks (mean, chunks, unique, word counts) failed. No tools were called,
no files were repaired, and no passing check was observed. The mean response
even renamed the function to `fix_mean`. The first three tasks returned promptly;
the fourth took about 42 seconds and exhausted its 4,096-token output budget.
These are historically reused smoke tasks,
not a new independent repository benchmark. Full records:
[opencode-repairs.json](closeout/opencode-repairs.json).

The tracing suite asks for exact integers from short programs, using 12 generated
inputs over four templates, equal 256-token budgets and both effort settings.
It scored 0/12 for each setting. Many answers were code that purported to compute
something rather than the requested integer. This combines instruction-following
and calculation failure; it is not a pure measure of latent arithmetic ability.
The suite's “held-out template” flag refers to the older procedural curriculum;
the final model did not train on that curriculum. No deep reasoning was established.
See [reasoning.json](closeout/reasoning.json).

Four conversation probes are qualitative observations, not a validated fluency
score. Portuguese sky-color output was repetitive and incorrect; a Spanish rainy-
day activity request produced irrelevant Python; an English breakfast request
repeated “oats” in a Python docstring until the token limit. In a short Portuguese
exchange, it correctly recalled the cat's name Pipoca but replied in a sentence
despite “only the name.” Thus there is some short-context text recall, not robust
multilingual assistant behavior. See [dialogue.json](closeout/dialogue.json).

The context probe inserts a six-digit setting at 5%, 50% and 95% positions in
repeated distractor text, near 8K, 16K and 32K prompt lengths. All nine requests
were accepted and all nine failed exact retrieval. No short-window baseline or
context-length training comparison was run, so this does not identify the precise
cause or an effective maximum context length. It **does** reject claiming useful
32K retrieval from a configuration ceiling. See [context.json](closeout/context.json).

No final-model long agent-session test was run after its short agent smoke test
failed. The old arithmetic model's 32-round success does not establish final-model
long-horizon competence. Three of the four smoke requests reported 1,446 cached
prefix tokens each, confirming reuse of their shared prompt. These were separate
short tasks, not a long-session caching benchmark; no paired latency comparison
establishes a productivity gain.

## Engineering implemented, and the difference from learned behavior

The repository includes data downloads and provenance, grouped splits, BPE and
o200k tokenizers, pretraining, full-weight assistant-only SFT, checkpoint/resume,
portable inference export, KV/prefix reuse, chunked prefill, Flash SDPA, chat
completions, HTTP/SSE formatting, reasoning-content separation, schema-validated
tool calls, optional grammar constraints, OpenCode configuration and isolated
executable graders. The simple model and loops remain inspectable Python/PyTorch.

These components are useful educational results. A valid HTTP response or valid
tool JSON is not a correct action. A context allocation is not learned long
memory. Thinking tags are not reasoning. Source success labels are not independent
verification. Passing test fixtures is not a guarantee on arbitrary user prompts.

The server uses a single serialized backend/cache trust domain, not a production
multi-user service. The safe evaluation harness restricts file edits and uses a
read-only, no-network Docker grader with dropped capabilities, CPU/memory/PID
limits and a non-root user. It is a limited educational harness, not a full
unrestricted developer desktop workload. Generated code was never run on the host
for these closeout tests.

## Failures and operational lessons worth preserving

1. **We initially measured too close to the training distribution.** Arithmetic
   sessions appeared agent-ready until the user switched task family. Later
   diagnostics explicitly separated syntax, interface, semantics, tool use and
   complete task success. The earlier narrow score was real but overbroad as a
   proxy for the user's goal.
2. **Loss and teacher forcing were weak readiness proxies.** Correct-history
   narrow repair accuracy rose strongly while free repair and coding stayed weak.
   A six-hour mix taught tool syntax without broad repair; longer final SFT lowered
   loss while reducing development correctness. Select by frozen executed tasks.
3. **Some data and graders were misleading.** Independent source re-execution
   rejected nominally successful examples. Random/hash-order-sensitive tests,
   shadowed built-ins, `SystemExit(0)`, and copied no-op edits needed controls.
   Python `-I` ignores `PYTHONHASHSEED`; the multiseed checker uses an explicit
   environment with `-s -P -B`. A completion sentinel distinguishes finishing
   tests from merely exiting zero. Full control evidence remains in `docs/evidence/*controls*.json`.
4. **Runtime and training formats must match.** Handoff announcements, tool-call
   IDs, edit anchors, sibling identifiers and OpenCode's request quoting affected
   old agents. Format changes sometimes improved names without improving code.
   Fixing one session or auto-repairing model actions would hide model failures.
5. **Tokenization crossed several correctness boundaries.** Literal EOS text,
   UTF-8 document reconstruction and XGrammar byte-level vocabulary support all
   required fixes. The first 2B coding evaluation failed at grammar construction;
   that infrastructure failure was repaired and the evaluation rerun, rather than
   scored as a model failure.
6. **NAS maintenance exposed stale-checkpoint risk.** An early 10B-stage resume
   loaded step 81,500 instead of frozen 152,588. NAS mount/source changes coincided
   with regressed mutable checkpoint timestamps; root cause was not proven. We
   discarded that continuation, verified the frozen 5B hash, and resumed into a
   new output directory. Later resumes required hash, exact step, optimizer/RNG,
   dataset checks and a write/fsync probe. NFS mapping initially prevented writes
   until the user corrected it. Never resume solely because a file is named `last.pt`.
7. **Desktop GPU and storage throughput mattered.** Free VRAM did not prevent
   application hangs; pacing helped share the GPU. NAS reads blocked the trainer
   with an idle GPU. Read-heavy audits and checkpoint writes took meaningful time.
   Store active working data locally if space allows, while preserving NAS archives.
8. **Queue recovery and stopping require explicit state.** An upstream interruption
   made the waiting evaluation queue `failed`, while its resume path expected
   `interrupted`. We added a narrow recovery case with no prior commands and an
   unchanged holdout, with three regression tests. SIGTERM checkpoints at update
   boundaries; SIGKILL cannot guarantee a safe save. The October 13 interruption's
   initiating cause remains unknown.
9. **We stopped at an intermediate pipeline boundary.** The unattended pivot
   runner implemented foundation, coding SFT and holdout, not subsequent agent
   training. Its clean completion did not fulfill the larger OpenCode goal. We
   should have communicated that boundary more prominently, and tested the final
   checkpoint through the harness before calling it useful for agent work.

The work was not entirely blind trial and error: we added matched controls,
spending gates, independent execution, diagnostics and frozen holdout selection.
But the experiment grew well beyond a minimal tutorial, and several early recipe
changes were confounded or overly narrow. The final pivot changes vocabulary,
capacity distribution, initialization and exposure together. It is evidence of a
better practical recipe, not proof of any single architectural explanation.

## What could improve it, with bounded estimates

These are **unexecuted engineering estimates**, not a commitment to resume and
not predicted pass rates. No evidence supports promising that another fixed
number of hours would produce a reliable agent. Keep the selected checkpoint and
old holdout frozen; any future selection needs fresh acceptance tasks.

| Candidate next experiment | Scope and rough budget on this RTX | What would justify continuing |
|---|---|---|
| Train actual tool use on the stronger model | Retokenize/adapt 10K–30K diverse verified short repository trajectories, with 30–50% code replay plus some conversation. Budget 1–3 engineering days and an initial 2–6 training hours, not an unbounded run. Longer episodes can cost much more than code-only SFT. | Correctly read/edit/check on fresh tasks; retain coding performance; no false PASS claims. |
| Correct the model's own mistakes | Collect hundreds to a few thousand training-only failures, reconstruct current file state, verify corrections, mask wrong actions, train a controlled comparison. Budget 1–3 days for collection/audit and 2–8 training hours initially. | Improvement on unseen localization, stale-edit and recovery tasks, rather than repeating canned read/edit/check sequences. |
| Restore conversational/task-switch behavior | Compare a small balanced conversation/tool/code mix against the frozen coding checkpoint; 1–4 initial training hours plus evaluation. | Follows non-code requests, switches task families, preserves coding score and obeys output constraints. |
| Longer context and sessions | Only after short tasks pass: pilot 4K then 8K, then consider 16K/32K, using retrieval, earlier constraints, edits and real compaction/handoff trajectories. Reserve 1–2 engineering days and 6–24 initial training hours, with memory/throughput measured at each window. | Accurate retrieval versus short-context controls and preserved earlier work in 4/8/16/32-task sessions. Allocation alone is never the gate. |
| More foundation exposure and better data | First improve unique educational code breadth and holdout hygiene. At 20K–30K tokens/sec, each additional 10B costs **93–139 hours** before extra preparation/evaluation. Another 20B is roughly 7.7–11.6 days. | Matched SFT probes improve on fresh tasks enough to justify days of compute. Do not merely repeat the same 3.476B-token mixture indefinitely. |
| Remove avoidable I/O overhead | Stage roughly 14.67 GB of foundation arrays locally if free space permits; keep immutable NAS copies and verify hashes. Profile I/O, checkpoint intervals, microbatch and compilation separately. Budget hours to a day. | Same correctness/lineage with a measured sustained throughput gain. This improves cost, not learned ability by itself. |

The observed final code-SFT segments ranged roughly 1.0–1.6 seconds per update;
4K updates took 1.36 logged training hours across the selected branch. Tool
trajectories can be much longer, so that timing is a lower-complexity reference,
not a reliable estimate for agent SFT. A tiny profile should precede any larger
budget. GQA, quantization or a faster server may reduce inference cost, but none
substitutes for missing tool-use supervision or semantics.

For scale context, the public SmolLM2-360M card reports **4 trillion** pretraining
tokens—400 times this model's 10B. This does not establish a necessary minimum
or predict our outcome; it explains why similar parameter counts are not similar
training efforts. At 20K–30K tokens/sec, 4T would take roughly 4.2–6.3 years of
uninterrupted token processing on this throughput range.
[SmolLM2-360M model card](https://huggingface.co/HuggingFaceTB/SmolLM2-360M).

For future agent data, SWE-smith's public workflow is relevant because it couples
repository environments, generated tasks and trajectories. Its results on much
larger models are not evidence that a 337M model will match them. Use its methods
as a reference and adapt bounded, independently executable tasks to our harness.
[SWE-smith training guide](https://swesmith.com/guides/train_swe_agent/).

Stack-Edu and related code collections were investigated earlier, but identifier/
metadata downloads do not automatically deliver source text. Access, provenance,
retrieval and storage work remain; see [datasets.md](../guides/datasets.md). A larger download
is not the first remedy for a final checkpoint that has never received agent SFT.
For an H100, no speedup or dollar estimate was measured. Run the same profile on
the actual rental, then compute `tokens / sustained_tokens_per_second / 3600 ×
hourly_quote`, adding storage, transfer, startup and evaluation. No cloud expense
was incurred by this experiment.

## Reproduce the small benchmark and inspect the model

Use the existing environment and original checkpoint/tokenizer directory. The
private evaluation server is closed after each wrapper run. These commands
evaluate; they do not train or promote a model. Choose fresh output paths.

```sh
# Standalone code holdout: repeats published evidence; no longer an unseen test.
.venv/bin/python scripts/evaluate_repair_checkpoint.py \
  --checkpoint data/experiments/foundation-pivot-v1/adapter-10b/checkpoint-4000.pt \
  --tasks-file evaluation/programming-v1/acceptance.jsonl \
  --programming --out runs/repeat-closeout-code --port 8037

# Private server for the small behavioral probes; stop with Ctrl-C afterwards.
.venv/bin/python -m microgpt.serve \
  --checkpoint data/experiments/foundation-pivot-v1/adapter-10b/checkpoint-4000.pt \
  --device cuda --attention flash --constrain-tools --port 8037
```

In another terminal while that server is running:

```sh
.venv/bin/python scripts/eval_opencode.py --url http://127.0.0.1:8037/v1 \
  --out runs/repeat-closeout-repairs --label '337M 10B + SFT4000' \
  --suite basic --general-prompt --effort none
.venv/bin/python scripts/eval_reasoning.py --url http://127.0.0.1:8037/v1 \
  --out runs/repeat-closeout-reasoning.json --label '337M 10B + SFT4000' \
  --tasks 12 --budgets 256
.venv/bin/python scripts/eval_dialogue.py --url http://127.0.0.1:8037/v1 \
  --out runs/repeat-closeout-dialogue.json --label '337M 10B + SFT4000'
.venv/bin/python scripts/eval_context.py --url http://127.0.0.1:8037/v1 \
  --checkpoint data/experiments/foundation-pivot-v1/adapter-10b/checkpoint-4000.pt \
  --out runs/repeat-closeout-context.json
```

The Docker image is pinned in the evaluators; OpenCode's version is recorded in
the repair result. The existing tests/controls cover grader completion, negative
fixtures, source checking, tokenizer identity, checkpoint continuation, protocol
and pacing. Closeout reused those evaluators and introduced no training-code
change. The benchmark makes generated code execute only in restricted Docker.
No extra grader/repair model supplies hidden answers at inference.

For archiving or later teaching, keep the selected full checkpoint, its tokenizer
files, the 10B foundation parent, manifests, source licenses, and this repository.
`scripts/export_agent.py` can create an inference-only bundle without optimizer
state; no extra export was needed or created for closeout. Never retain only a
`.pt` file while discarding its matching tokenizer. The model is not a standard
Transformers/GGUF artifact; use the project's native server unless a conversion
is separately implemented and verified. Existing Runpod instructions describe an
optional execution route, not a trained cloud artifact.

## Evidence map and publication guidance

The committed [closeout directory](closeout/) preserves compact machine-readable
model identity, final statuses, corpus totals, timing intervals, checkpoint
inventory, development/holdout results, example outputs and the new diagnostic
responses. Local ignored `runs/` retains full logs and per-task generations;
large model/data artifacts remain on the NAS. The committed report is not a
backup of those weights or corpora.

| Question | Evidence |
|---|---|
| Exact model/config/hash/ancestry | [selected-checkpoint.json](closeout/selected-checkpoint.json) |
| Foundation and SFT stages, data counts, checkpoint paths, timings | [experiment-evidence.json](closeout/experiment-evidence.json) |
| Holdout outcomes and interface diagnostics | [holdout-results.json](closeout/holdout-results.json), [holdout-analysis.json](closeout/holdout-analysis.json) |
| Selected development outcomes | [development-results.json](closeout/development-results.json) |
| Concrete correct and incorrect code | [coding-examples.json](closeout/coding-examples.json) |
| Actual final-model OpenCode behavior | [opencode-repairs.json](closeout/opencode-repairs.json) |
| Reasoning, conversation, long prompts | [reasoning.json](closeout/reasoning.json), [dialogue.json](closeout/dialogue.json), [context.json](closeout/context.json) |
| Frozen plan and implementation diary | [foundation-pivot.md](../history/foundation-pivot.md), [foundation-pivot-start.md](../history/foundation-pivot-start.md), `runs/pivot-v1/journal.md` |
| Earlier failed/successful narrow experiments | [verification.md](../history/verification.md), [capacity-pilot.md](../history/capacity-pilot.md), [repair-investigation.md](../history/repair-investigation.md) |

An educational article can honestly demonstrate the complete scratch-model
lifecycle and the substantial gap between generating a plausible function and
acting reliably in a repository. The strongest result is the measured coding
improvement, alongside concrete counterexamples to confusing syntax with
semantics, protocol support with tool skill, and context capacity with memory.
We did not establish the minimum model size, minimum training budget, or minimal
recipe that achieves the original agent goal. Closing the experiment preserves
that finding rather than turning an unfinished goal into a success claim.
