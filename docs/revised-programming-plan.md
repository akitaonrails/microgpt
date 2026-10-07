# Programming model reset: evidence and gated plan

**Current work (October 7):** the [learnability and optimization plan](learnability-plan.md)
follows the completed 250M-token continuation, which did not improve the matched
coding control (11/100 versus 12/100). A 16-example diagnostic can reproduce all
16 training solutions but fails all 16 renamed variants at its first checkpoint.
The completed higher-rate comparison peaked at 10/100 versus the control's 12/100;
it is not being extended. The best 24/100 checkpoint makes no tool calls on four
actual OpenCode repairs. The [longer mixed agent curriculum](agent-mix-plan.md)
now targets tool policy while retaining coding and conversational practice.
The [controlled experiment notebook](capacity-pilot.md)
supersedes the status snapshot below. The scratch candidate is now 238,568,448
parameters; its best broader direct-code score is 24/100. Programming correctness
remains the bottleneck. Checked public-code adaptation, an ancestry control,
expanded foundation data, and executed scoped-repair lessons address distinct
questions; no new candidate has passed the OpenCode readiness gates. The original
acceptance criteria below remain in force.

Audit date: 2026-10-05. This plan supersedes the next-run recommendations in
`next-run.md` and the automatic continuation recommendations in
`programming-improvement.md`. Those files remain experiment history.

## Judgment

We made useful infrastructure progress, but recent capability work became
insufficiently controlled trial and error. We changed data, sampling, optimization,
and sometimes decoding together, repeatedly tested eight development prompts,
and continued training without isolating why real tasks failed. The recovery run
has now finished too: 2/8 direct and 0/8 cumulative agent tasks at 5K, 10K, 20K and
40K. Another automatic SFT extension is not justified by these results.

This is not evidence that scratch training cannot work. It is evidence that our
current recipe has not worked and our measurements cannot yet select the remedy.
No training process remains active at audit time. No new training or rental is
launched by this document. Next work is diagnostics, not another unattended run.

## Preserve the goal; make success falsifiable

Build and explain a small, randomly initialized model that uses actual OpenCode
for simple Python programming across a sustained session. Keep o200k_base,
Flash Attention, caching, and a 32K supported context. No Qwen or other pretrained
weights. Existing synthetic public answers are teacher-generated data; scratch
initialization does not mean all knowledge comes from raw human-written text.

The initial task distribution is ordinary Python standard-library functions,
small file changes, tests, bug repairs, and follow-up changes in tiny repositories.
It includes varied names, wording, paths and combinations, rather than a closed
list of exercise templates. English requests first, a small Portuguese regression
set retained. Broad multilingual coding and arbitrary repository maintenance are
later extensions, not silent prerequisites for the first milestone.

A reasonable response to an unsupported request can be a specific clarification or
an honest limitation. Correctly solving **any** user prompt is not a feasible
acceptance criterion. Handling arbitrary text without malformed tools, stale task
replay, or fabricated success is a separate interface requirement.

Deep reasoning remains an objective, measured by improved correctness on multi-step
coding and repair under a matched compute budget. Producing a think tag or a long
explanation does not meet it. Long horizon remains required, but will be tested
only after short tasks work, and with real context compaction and retained files.

## What we actually have

| Item | Evidence | What it does not establish |
| --- | --- | --- |
| Scratch foundation | 1,000,005,632 token exposures; final validation loss 2.5781 | Instruction following or runnable code quality |
| Pretraining corpus | 947,532,764 stored training tokens across five sources | Unique semantic content; no repeated windows |
| Model | 90,969,600 parameters; 8 layers, width 384 | Capacity comparable to a 91M model with a much smaller vocabulary |
| Parameter allocation | 76,807,296 tied embedding/head; 14,162,304 transformer/norm | Proven capacity bottleneck; this is a strong hypothesis |
| Programming v2 | Best 4/8 direct at 80K; final 2/8 direct, 0/8 agent at 160K | Stable gains or readiness |
| Programming v3 | 37,940,173 additional target exposures; 2/8 direct, 0/8 agent at every milestone | Benefit from the new recovery curriculum |
| Older served checkpoint | Narrow arithmetic sessions up to 32 rounds passed | Unrelated programming tasks, autonomous task-state maintenance |
| Context | 32K allocation/Flash/cache checks; old arbitrary retrieval 0/9 | Useful recall throughout 32K on current candidates |
| Protocol | Parsing, schema constraints, tool execution, loop bounds | Correct function behavior or good tool decisions |

Local evidence: `runs/coding-v1/foundation/metrics.jsonl`, `data/large-v1/meta.json`,
`runs/programming-continuation/status.json`,
`runs/programming-recovery-continuation/status.json`, and their evaluation artifacts.
The v2 final had 110,623,221 assistant target exposures. V3 branched from v2 **80K**,
not 160K; these are branches, not one additive training history.

The foundation mixture was 60% CodeParrot, 16% OpenCoder, 12% Cosmopedia, 8% SmolTalk,
4% multilingual by configured sampling weights. The later large coding downloads
mostly expanded **SFT**, not foundation pretraining. Downloaded bytes, stored tokens,
actual exposures, and independent verified solutions must be reported separately.

## Failure analysis and confidence

**Established: basic programming is weak before tools are involved.** Direct
answers fail ordinary functions. An agent wrapper cannot reliably repair a missing
underlying capability. Direct and tool scores need paired task-level analysis.

**Established: successful tool syntax is not successful execution.** The 80K agent
read and edited the correct file, but emitted a literal `{name}` placeholder and
could not recover after a failing test. At 160K repeated tool cycles and eventual
context overflow were also recorded. Do not train on these evaluation solutions.

**Established: aggregate loss obscures the relevant errors.** The SFT validator
uses a small fixed source/length round-robin sample (40, then 80 episodes), with
assistant-token-weighted reporting. Long, easy traces and copied edit text can
have tiny loss while free generation fails. The optimizer averages episode losses,
whereas reported aggregate validation weights tokens: even the weighting differs.
Per-source loss helps but still cannot replace execution.

**Established: the recovery intervention poorly matched the observed failures.**
It teaches restoring a NotImplementedError body from a reference and switching
between independent files. The failures involve semantic bugs, failed-test
interpretation, and keeping cumulative code correct in the same file. Its PASS
observations were simulated; only original references were sandbox-tested.
A larger count of these lessons was not a demonstrated increase in task diversity.

**Established: evaluation conflates individual and session competence.** One early
failure causes every cumulative grade to fail. The current-task grader helps but
later generation still sees the failed history. Add independent fresh-session
runs and controlled successful-history prefixes. Keep cumulative grading too.

**Likely, not proven: foundation/capacity limitations.** A shallow 14M transformer
trained for 1B exposures is a plausible bottleneck. We have not measured a proper
raw-completion baseline or isolated instruction-format failure, so declaring that
more parameters alone fixes it would repeat the same mistake.

**Not yet ruled out: implementation or distribution mismatch.** Existing unit and
cache checks are useful, but verify causal masking, shifted labels, packed-vs-single
loss, training-vs-serving prefixes, and teacher-forced-vs-free output on the same
examples. Do not diagnose every low score as insufficient scale.

## Research that changes the plan

- [Phi-1 paper](https://arxiv.org/abs/2306.11644) describes a 1.3B coding model using
  a roughly 7B-token selected/synthetic corpus. Its [model card](https://huggingface.co/microsoft/phi-1)
  reports 54B training-token exposures and warns of poor alternate-format behavior.
  This supports distinguishing code completion from chat/agent competence; it does
  not prescribe a minimum size for our simpler tasks.
- [SmolLM2 report](https://arxiv.org/abs/2502.02737) documents data-centric staged
  training and substantially larger budgets. Its smaller 135M/360M variants were
  trained on 2T/4T tokens. These are reference points, not a requirement that this
  educational project spend trillions of tokens. The report also explicitly removes
  complex function-calling examples from post-training for the 135M/360M variants.
  Their existence is therefore not evidence that a 239M model will be a reliable
  coding agent; this remains a constrained research target.
- [OpenCoder](https://arxiv.org/abs/2411.04905) provides a reproducible data pipeline,
  with cleaning/deduplication, code-related prose, and synthetic annealing/SFT as
  distinct ingredients. Simply adding more instruction rows misses that structure.
- [nanochat](https://github.com/karpathy/nanochat) is the closer educational reference
  for an end-to-end assistant; its scaling experiments separately record transformer,
  embedding and head parameters and evaluate capability. Hardware-specific speedrun
  numbers cannot be copied to our 5090.
- [SWE-smith](https://arxiv.org/abs/2504.21798) generates executable repository tasks
  by introducing test-breaking changes, and reports 50K instances across 128 repos.
  Its trained agent is 32B, so its results do not validate our model size. Borrow the
  verified task-generation method, initially on tiny repositories.
- [RULER](https://arxiv.org/abs/2404.06654) distinguishes advertised from effective
  context with tasks beyond simple retrieval. Test context use, not just allocation.

The local Lua/Genesys forensics are black-box behavioral/token-count observations,
not access to weights, architecture, data or compute. Their strong tokenizer
ancestry claims exceed what API accounting alone establishes. Our existing
`research.md` already records this caveat. o200k compatibility gives neither
Genesys capability nor proof of shared internals; Harmony formatting adds no
learned reasoning skill. Keep it for the user's tokenizer preference, with its
capacity cost visible.

## Dataset decisions

| Source | Role and next action |
| --- | --- |
| Existing CodeParrot/source corpus | Audit language, license selection, duplication and held-out repo separation; expand selected source tokens only after the diagnostic baseline |
| [Stack-Edu](https://huggingface.co/datasets/HuggingFaceTB/stack-edu) | Preferred candidate for educational Python pretraining; files are identifiers/metadata, with separate Software Heritage content retrieval. Prove access and throughput on a bounded sample first |
| [OpenCoder annealing corpus](https://huggingface.co/datasets/OpenCoder-LLM/opc-annealing-corpus) | Candidate for explanations and small algorithmic implementations before chat SFT; inspect subset quality and source terms, deduplicate against evaluation; its card labels ODC-BY |
| [OpenCoder code-related web](https://huggingface.co/datasets/OpenCoder-LLM/opc-fineweb-code-corpus) | Supporting prose, not a substitute for clean source code; sampled viewer rows include noisy forum/error pages, so filter rather than trusting the dataset title |
| [OpenCodeInstruct](https://huggingface.co/datasets/nvidia/OpenCodeInstruct) | Keep as instruction data. Its five million rows are already downloaded; independently test a stratified sample. Upstream perfect test scores are not proof of spec correctness |
| [OpenCoder educational SFT](https://huggingface.co/datasets/OpenCoder-LLM/opc-sft-stage2) | Keep checked small examples; require callable signatures, task difficulty bins, and additional tests where possible |
| [Code-Feedback](https://huggingface.co/datasets/m-a-p/Code-Feedback) | Useful feedback conversations, but their text roles are not native OpenCode action/observation traces. Conversion must be executed and verified |
| SWE-smith/local generated repairs | Collect real read/edit/test trajectories with actual observations; split by repository/task family before generating variants |
| [Mixture-of-Thoughts](https://huggingface.co/datasets/open-r1/Mixture-of-Thoughts) | Later optional reasoning supervision; long teacher traces are poorly matched to the current weak student. Filter for short independently verifiable code tasks first |

No more bulk downloading as a proxy for progress. Preserve revision, source license,
file hashes, selected counts and verified counts. All datasets stay under
`/mnt/gigachad/microgpt/dataset`; no other NAS location is touched.

## Ordered experiments and decision gates

All thresholds below are proposed engineering acceptance criteria, not established
scientific constants. The primary agent owns implementation and evidence for every
phase. Each run must name one hypothesis, control, budget and rejection condition.

### 1. Establish a trustworthy diagnosis before another training run

Build 100 development tasks and a separate 100-task locked acceptance set across
strings, collections, numeric boundaries, parsing, files and simple multi-function
changes. Split by task family/repository before making mutations or paraphrases;
check normalized code/text duplication. Keep the existing eight prompts as visible
regressions, not the only selection benchmark. Public-benchmark contamination cannot
be excluded just by changing variable names; disclose residual uncertainty.

Run foundation, v2-80K, v2-160K and v3-40K under the same harness:

1. Raw function/docstring completion (foundation-appropriate format).
2. Chat code answer, no tools, on the same tasks.
3. Fresh OpenCode session per task.
4. Diagnostic oracle controls: supply the correct code and ask for the edit; supply
   a successful prefix and ask for the next action. Label these assisted diagnostics,
   never model success. Also run a deterministic scripted client to validate graders.
5. Independent short repair episodes with real failing tests, then mixed sessions.

Report parsing, requested-interface accuracy, execution correctness, exact-edit
success, tool selection, stopping, retries, latency and cumulative preservation
separately. Score the first failure and classify it. Stratified teacher-forced losses
should distinguish code from tool wrappers/copied context. Seed prompt perturbations;
repeat the same deterministic request only to check nondeterminism, not inflate N.

Minimum checks: causal future-token invariance; packed/single loss equivalence on
a tiny fixture; full-vs-cached logits; representative normalized train/serve prefix
identity; sandbox graders pass references and reject known incorrect mutations.
Reuse existing tests where they establish the exact claim. Run only the four
checkpoint comparisons and expand if a specific unresolved failure demands it.

Decision: if oracle editing fails, fix representation/protocol first. If raw completion
works but chat fails, focus instruction mapping. If raw completion also fails,
prioritize foundation/capacity. Mixed outcomes can require both; quantify before
choosing. This prevents another recovery-only intervention on a weak code generator.

### 2. Establish code competence with bounded foundation experiments

Keep the existing 91M student as a control. A reasonable next capacity candidate is
**238,568,448 parameters: 12 layers, width 768, 12 heads**, preserving o200k and the
same basic model code. It has **84,953,856 transformer/norm parameters**, about six
times the current core. This is a hypothesis, not a promised fix or a decision to
start an expensive run immediately. Random initialization is required for changed
architecture; no padding/transplanting old weights and calling that a clean control.

First profile both shapes for a short representative 2K training workload, record
end-to-end tokens/sec, VRAM, output-head cost and data stalls. Then compare at matched
compute using the same cleaned corpus. A bounded pilot of roughly 250M–1B exposures
per candidate can reject broken settings, but cannot establish a well-trained model.
Compare learning curves and execution after the same small instruction adapter.
Use a second seed only for a promising result before committing the main budget.
Report both matched-compute and matched-token points where available; comparing
a continued 91M model with a fresh larger model alone confounds initialization and
training exposure. Preserve a fresh 91M baseline for the architecture comparison.

Select a candidate only after a predeclared meaningful gain (initially at least
10 percentage points on development code execution without an interface regression),
then train in evaluated tranches toward a **5–10B-token initial budget**, not an
unbounded run. This budget is a starting experiment, not a claimed sufficiency law.
If two tranches improve held-out loss but not execution, revisit data/format/capacity
instead of adding tool templates. Track genuinely distinct content separately.

Use 2K then 4K sequences initially. Keep long-context inference support, but do not
pay 32K training costs while simple functions fail. Predeclare the LR schedule and
effective token batch. Correction after full trainer inspection: pretraining already
uses warmup plus cosine decay to 10% of peak; SFT uses a constant rate. Benchmark
AdamW first; a new optimizer is a separate experiment, not another simultaneous change.

The old 91M foundation reported about 55.7K tokens/sec: 1B tokens is about five hours,
10B about fifty hours of that training workload, excluding overhead. **These are not
estimates for the larger candidate.** Measure it on the RTX5090 before calculating
RunPod hours or costs. Rent an H100 only when the measured plan demonstrates why its
memory/throughput is needed; no cloud resources are provisioned by this audit.

### 3. Teach tools with executed trajectories after direct competence

Progression gate: at least 80/100 direct development tasks, including held-out families,
and no collapse on paraphrases. Then train a small native OpenCode curriculum:
read actual file → make scoped edit → execute tests → inspect failure → revise →
report accurately. Include wrong comparisons, off-by-one errors, missing imports,
wrong return types, literal formatting mistakes, and preservation of other functions.
Generate failures from training tasks, never from locked evaluation answers.

Every PASS/FAIL observation must come from the sandbox run. Keep bad model actions
as context with loss masked when teaching a correction. A reference patch can give
supervision without loading a teacher model; public or future teacher-generated
traces must disclose their provenance. Do not fabricate tests passing after a patch.
Initially budget 1–2K diverse verified episodes and a short SFT run, with substantial
coding replay; expand only if observed next-action/repair accuracy improves.

Progression gate: at least 70/100 fresh-session agent tasks, at least 99% valid
structural calls, no false success on deliberately failing checks, and direct code
score within five points of the starting checkpoint. One hundred successes cannot
prove a zero failure rate; report counts and uncertainty. Do not select solely on
validation loss or the direct score as the current continuation runner does.

### 4. Long sessions and reasoning, with separate measurements

Extend training context 4K → 8K → 16K → 32K using varied code, instructions and
retrieval positions. At each length, test beginning/middle/end dependencies and
changes that invalidate old requirements. Measure exact copying and semantic use
separately. Advertise the highest *verified useful* length alongside the supported
allocation limit. Flash and prefix cache are performance mechanisms, not memory skills.

Add OpenCode compaction before the prompt plus reserved output reaches the limit.
Keep requirements and test status in inspectable files. Require the model to update
and use those files; distinguish harness-written memory from model-written memory.
Test cold-cache equivalence, new sessions, compacted history, transient tool failures,
user task switches, interrupted/resumed work and contradictory follow-up requirements.

Initial long-session gate: ten 32-request sessions on varied small programming tasks;
at least eight sessions complete at least 90% of requests within the fixed tool budget,
with no unintended regressions to already-correct requirements or manual intervention.
Also report the strict all-requests-pass session count, not just average turn success.
The locked acceptance set is used only after development gates pass.

Reasoning experiment: compare direct output against short test-grounded planning and
revision under equal token/tool budgets on held-out multi-step tasks. Reward execution,
not verbose text. Add verified reasoning SFT only when this comparison can measure a
benefit; postpone RL until successful trajectories are common enough to supply signal.
Neither general deep reasoning nor long-horizon reliability is guaranteed at this size.

## Experiment discipline and educational deliverable

Freeze a run manifest before each experiment: hypothesis, parent hash, data hashes,
source/token weights, seed, schedule, eval revision, compute ceiling, progression and
stop rules. Record examples and first failures, not just loss. Retain controls and
never silently promote a checkpoint. A null result is useful when it rules out a
hypothesis; changing four variables and retesting eight prompts does not do that.

Keep the readable minimal model and a short runnable happy path. Put large-data
curation and experiment history in separate scripts/docs. The article should explain
four milestones independently: working transformer; code-capable base; instruction/
tool-capable assistant; resilient agent system. At present we have substantial
infrastructure for the latter two, without the prerequisite coding capability.
