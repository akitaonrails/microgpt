# Controlled capacity pilot

This implements the first diagnostic decision in [the revised plan](revised-programming-plan.md).
It does not claim that a larger model will solve the OpenCode goal.

## Diagnostic evidence

The original eight development prompts remain unchanged; the new evaluations add
fresh sessions and explicitly assisted reference-code controls.

| Checkpoint / condition | Passed |
| --- | ---: |
| Original 1B-token foundation, raw signature/docstring completion | 1/8 |
| Programming v2 80K, fresh independent OpenCode session per task | 3/8 |
| Same checkpoint, correct reference code supplied to OpenCode | 5/8 |
| Same checkpoint, broader direct-code development set | 2/100 |
| Original foundation, broader raw-completion development set | 2/100 |

The 100-task set has **20 families with five name/wording variants each**. These
are correlated variants, not 100 independent task families. Another 20 families
provide 100 acceptance variants; their references were validated, but no candidate
has been run on that acceptance set. Neither split is claimed uncontaminated
relative to public coding datasets. Both are frozen in `evaluation/programming-v1`.

All 200 reference implementations pass their tests in restricted Docker; all 200
empty implementations fail. The old eight references pass too. Training generators
must not import evaluation references. A fixed grader is weaker than proof of full
spec correctness; these small cases are diagnostic, not a universal coding benchmark.

Post-SFT checkpoints scored 0/8 in the raw completion format. Many generated closing
Markdown fences despite being prompted with plain source code; do not interpret
this as proof that SFT erased all code knowledge. It demonstrates format sensitivity.
Raw and chat results are separate measurements, not interchangeable scores.
On the 100-variant set, the SFT model had 58 invalid-Python outputs; the foundation
had 18. Both passed only two, with other failures including wrong results and
incorrect interfaces. This supports both a code-semantics bottleneck and a format/
instruction bottleneck; merely changing JSON constraints cannot solve them.

These results support weak code generation and unreliable instruction/copying as
bottlenecks, rather than a cumulative-session-only bug. Assisted 5/8 is not unassisted
agent competence. In particular, valid JSON can still contain corrupted Python quotes.
The grammar accepts all eight correct reference edits token by token with o200k;
its allowed language does not exclude those answers.

A separate grammar bug allowed `</tool_call>` as free text with tools enabled.
It is now excluded outside calls while remaining legal inside JSON code strings.
The 3/8 and 5/8 runs predate this fix; the broader direct evaluation follows it.
Direct no-tool grammar is unchanged. Existing deployment was not replaced.

Artifacts live under `runs/diagnostic-v1`. Committed summary:
`docs/diagnostic-evidence.json`. The 35 unit tests include causal behavior,
full/cached attention, target shifting, packed/individual-turn loss equivalence,
training/serving prefixes, and exact resume equivalence across schedule-preserving
stops. One correction to the earlier audit: pretraining already used warmup plus
cosine decay; SFT used a constant learning rate.

## Preregistered comparison

Hypothesis: at fixed corpus and token exposures, increasing transformer capacity
improves executable code and instruction transfer sufficiently to justify further
investment. No dataset expansion, new optimizer, tokenizer change or reasoning
curriculum is combined with this experiment.

| | Control | Candidate |
| --- | ---: | ---: |
| Total parameters | 90,969,600 | 238,568,448 |
| Transformer/norm parameters | 14,162,304 | 84,953,856 |
| Layers / width / heads | 8 / 384 / 6 | 12 / 768 / 12 |
| Initialization | Existing documented random-origin foundation | Fresh random weights |
| Training exposures | 1,000,005,632 | At most 1,000,005,632 |
| Corpus | `data/large-v1` | Identical |
| Training window | 2,048 | 2,048 |
| Supported inference context | 32,768 | 32,768, not yet a verified useful context |

Use `configs/capacity-239m.toml`: same batch/accumulation, LR, warmup, cosine horizon,
seed and validation sampling as the original foundation. Independent batch RNG
keeps the sampled windows comparable across architecture shapes. The historical
control is reused; this is a matched-token comparison, **not matched compute**.
Short profiling also measures the cost difference. Optimizer settings are fixed
for this experiment, not claimed optimal for each size.

On this RTX5090, a 20-step real-data profile measured 42.3K tokens/sec and 7.8GB peak
allocated memory for the control, versus 32.9K and 11.2GB for the candidate. These
exclude warmup and are not sustained-run guarantees. Candidate 1B-token arithmetic
projects about 8.5 hours of training before overhead; the runner allows 12 hours of
training-command wall time, including data loading/checkpoint time. Evaluations may
extend elapsed time beyond that training budget. No H100 is needed for this pilot.

At 30,518, 61,036 and 122,071 updates, preserve an immutable checkpoint and evaluate
raw code on the same development set. `--stop-at-step` pauses at a milestone without
shortening the full learning-rate schedule. A regression test verifies bitwise
identical CPU weights after staged versus uninterrupted training.

Reject a nonfinite validation loss or a loss more than 0.2 above the nearest-step
historical control. This early filter rejects clearly poor configurations; passing
it does not establish coding improvement. Otherwise complete the bounded 1B pilot.

Next, give **both** foundations an identical 5,000-update code-only instruction
adapter from the same four public sources (OpenCodeInstruct, OpenCoder educational,
Code-Feedback, Magicoder), with identical sampling seed and optimizer settings.
Compare direct execution on 100 development variants. At least a 10-point gain is
required to call the capacity intervention promising; 80/100 is the separate gate
for proceeding to tool training. Report per-family results when interpreting the
comparison. A short adapter can undertrain both models, so a null result limits
this recipe rather than proving a capacity law.

If these gates fail, the runner records that result instead of silently extending
training or claiming completion. The next decision is based on raw/code-format and
adapter results: stronger foundation data versus instruction mapping versus
capacity/optimization. Long-session training remains downstream of code and tool
gates from the revised plan. No automatic deployment occurs.

## Reproduction and status

```sh
.venv/bin/python scripts/run_capacity_pilot.py \
  --out runs/capacity-pilot-v1 --hours 12
```

Use a fresh output directory. The runner records the control weight hash, config,
data manifest and benchmark hashes, implementation commit, stage results and PID
in `runs/capacity-pilot-v1/status.json`. It evaluates the original foundation on
all 100 development variants before starting candidate training. SIGTERM to the
controller forwards a graceful stop to its child. Evaluation servers are temporary,
use port 8023, verify the checkpoint identity, and are shut down on completion.
Training logs are in `pretraining.log`; intermediate results are in `raw-STEP`.

This is a capability experiment, not a completed agent. The existing port-8000
service continues to serve the previously documented narrow checkpoint.


## CUDA interruption and recovery

The first controller failed after the durable step-114000 checkpoint (933,888,000
exposures): PyTorch reported a CUDA launch timeout, not a nonfinite loss or a
capability rejection. The 250M and 500M evaluations had each scored 0/100 raw;
validation loss improved relative to the matched-step control. Neither result
establishes agent competence. The final instruction comparison had not run.

The controller now supports `--resume` for a failed/interrupted pretraining phase,
verifies the control/config/data/evaluation identities, and skips completed stages.
It accounts failed-command wall time and permits one checkpoint-based retry per
stage for this specific CUDA timeout. It does not retry other errors blindly.
The historical failure budget was reconstructed from recorded throughput plus a
300-second allowance, conservatively recording about 8.36 hours already used.
The exact failure-time snapshot remains in `failure-before-recovery.json`.

Recovery command (already launched; do not duplicate while running):

```sh
.venv/bin/python scripts/run_capacity_pilot.py \
  --out runs/capacity-pilot-v1 --hours 12 --resume
```

Recovery launch output: `runs/capacity-pilot-resume.log`. The CUDA timeout's underlying
system cause has not been established; a successful retry would not diagnose it.


## Completed outcome and next instruction experiment

The recovered run completed 1,000,005,632 exposures. Candidate validation loss
improved from the control's 2.5781 to 2.3020, but raw execution was only 1/100
(control 2/100). Identical 5,000-update instruction adapters scored 0/100 control
and 1/100 candidate. The predefined capacity gain threshold was not met. The
runner correctly recorded `no_meaningful_capacity_gain`; this experiment does
not justify promoting the model or claiming that capacity alone fixes the task.

The next intervention targets observed interface errors: wrong function names,
signatures and output format. `scripts/prepare_interface_curriculum.py` derives
28,592 code-only instruction records from 3,574 of the previously checked public
references, varying the requested function name with AST renaming. Nested scopes,
classes, decorators and local rebinding of the function name are excluded. Related
variants retain their original group. No evaluation references are imported.
Original references were checked; one renamed variant per source was subsequently
executed against upstream tests, as recorded below. Reflective code remains a
limitation of this text transformation.

Prepared data has 27,163 training and 1,384 validation episodes. This is a small
instruction intervention, not a new foundation corpus. It changes the response
format and name diversity together; any improvement cannot be attributed to
renaming alone without a separate ablation. General conversation/tool retention
is not established by a direct-code gain.

`scripts/train_interface.sh` starts from the candidate's short instruction adapter
with a new optimizer, LR 3e-5 and the fixed new curriculum. The bounded controller
evaluates the unchanged 100-task development set at 5K, 10K and 20K updates, within
a two-hour continuation budget (plus the first 1K bootstrap updates). Direct-only
mode cannot mark agent gates passed. Acceptance remains untouched by model runs.
Status: `runs/interface-continuation-v1/status.json`; bootstrap log:
`runs/interface-launch.log`. Failure to improve is retained as a negative result,
not grounds for extending the same training indefinitely.

### Interface data validation and second CUDA interruption

All 3,574 first-renamed variants passed their upstream tests in restricted Docker;
`scripts/check_interface_references.py` renames test function references too, without
adding compatibility aliases. This tests one transformed variant per source, not
every random identifier or independent specification completeness. Results are in
`runs/interface-reference-checks.json`.

The interface run saved step 2,000 and then hit another CUDA launch timeout.
Kernel logs recorded NVIDIA Xid 8 and a seven-second RC watchdog timeout for the
trainer PID. The underlying driver/kernel cause is not established. Repeatedly
restarting the same path is insufficient diagnosis.

Training is being resumed with explicit math SDPA, retaining optimizer and RNG.
`--allow-attention-change` permits only this explicit resume-setting difference and
records it in `origin.attention_changes`; other resume identity checks remain strict.
This changes numerical execution, not the dataset or optimizer schedule. It is a
stability experiment, not proof that Flash caused the timeout. Inference continues
to use Flash. PyTorch documents explicit backend selection and possible numerical
differences in its [SDPA reference](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html).
No GPU watchdog, driver or desktop settings were changed.

### First interface checkpoint: better structure, weak semantics

The math-backend continuation is recorded in
`runs/interface-math-continuation-v1/status.json`. At 5,000 updates it has consumed
792,422 assistant target tokens. The same 100 development variants give:

| Measurement | Parent short adapter | Interface 5K |
| --- | ---: | ---: |
| Pass source-independent execution tests | 1/100 | 8/100 |
| Parse as Python | 34/100 | 92/100 |
| Define requested function | 10/100 | 50/100 |
| Match requested parameter names | 8/100 | 50/100 |

At 10K updates, execution regressed to 1/100, syntax to 85/100, and both requested
function and parameter-name matches to 37/100. Assistant target exposures reached
1,584,702. Validation loss also worsened (0.1947 at 5K to 0.2275), although the
sampled training loss was just 0.0072. The 5K checkpoint remains preferable on
development execution; additional training is not monotonic improvement. The
predeclared final 20K comparison subsequently finished at **0/100**, with 85/100
valid Python outputs and 40/100 parameter-name matches after 3,164,356 assistant
target exposures. This branch is finished. Its best checkpoint remains 5K;
extending the same small curriculum did not improve programming. The frozen
summary is [interface-outcome.json](interface-outcome.json).

Passing variants cover four of twenty families: negatives 1/5, product 2/5,
longest 1/5, and is_sorted 4/5. None of the twenty families passes all five
variants. These are correlated name/wording variants, not 100 independent
programming problems. A rotation request returns elementwise multiplication;
another answer recurses without a base case. Syntax improvements therefore do
not establish programming competence or readiness for tool training.

Artifacts: `runs/interface-math-continuation-v1/eval-5000/direct/results.json`
and `analysis.json`; analysis command is `scripts/analyze_programming.py`.
No further watchdog errors were observed through the completed 20K training run.
That does not establish a root cause or a permanent fix. The broader checked-data
branch uses `runs/interface-diverse-continuation-v2`, with its
preregistered manifest in `runs/interface-diverse-experiment-v2.json` and bootstrap
output in `runs/interface-diverse-launch-v2.log`.

### Preparing a broader checked selection

An AST-only audit of the complete local OpenCoder educational shard found
118,278 rows, 118,117 meeting the existing short-example filters, and 107,465
distinct parsed ASTs among those rows. Names remain significant in that count;
it is not semantic deduplication. The interface experiment uses just 3,574 source
records with renamed variants (2,732 ASTs after normalizing the function name).

The next data preparation checks the first 24,000 distinct eligible ASTs from
the same pinned shard in restricted Docker, using `prepare_programming_tools.py
--deduplicate-code`. This reuses existing public data rather than downloading
more bytes. Source order is a sampling limitation. Passing upstream tests means
only that those supplied tests passed, not that a reference fully satisfies its
natural-language specification. These intermediate tool traces are simulated;
their code-only extraction is intended for a subsequent direct-code experiment.
No development or acceptance solutions are added to training.

The expanded source check initially reported 23,957/24,000 successful references.
Structural filtering retained 20,974 source programs, producing 167,792 variants.
The separate renamed-reference check then exposed a test-harness gotcha: some
upstream assertions begin with indentation. Appending those strings directly after
a reference can put them inside the function, including after a return, and report
a false pass. An AST audit found nine original test texts that did not parse on
their own and 37 selected references without usable assertions after normalization.
None of the rejected test sets belongs to the original 3,574-program curriculum.

`normalize_reference_tests` now dedents each test string while preserving relative
indentation, parses the resulting module, and requires assertions. Authored
regression tests verify that an indented failing assertion really fails and that
multiline loops keep their structure. This still does not prove comprehensive
test coverage or that every assertion executes; source tests remain fallible.

The expanded renamed-reference run executes normalized tests. Its failures and
assertion-free sources are removed by `filter_interface_curriculum.py`, preserving
the already checked random names rather than regenerating them. The filtered
report links back to the original sandbox report and checks both source hashes.
Only `programming-interface-diverse-sft-v3` is eligible for training; the earlier
unfiltered tokenization is retained as preparation history.

`scripts/train_interface_diversity.sh` records parent/data/evaluation hashes before
starting the next branch. It uses the same scratch parent, eight name variants,
LR 3e-5, accumulation two, seed 42 and 5K/10K/20K evaluations, with a two-hour
continuation ceiling. Math SDPA is used from the start for stability; the earlier
branch switched at 2K. Source ordering and AST deduplication also differ. This is
a broader-selection experiment at matched update counts, not an isolated proof
about any one filtering choice or equal token exposure. Extension requires at
least 20/100 direct passes, 70/100 parameter-name matches, and passes in five
families. Those are research progression criteria, not OpenCode readiness gates.

The corrected renamed check passed 20,969/20,974 references. Filtering also removed
two assertion-free cases, leaving **20,967 source programs / 167,736 variants**.
The five execution failures included assertions that had been hidden after a
return, missing `TreeNode`/`PriorityHeap` dependencies, a set-to-list ordering
assumption, and random password properties not guaranteed by the implementation.
A successful single execution cannot exclude flaky tests; this remains a data
quality limitation rather than a claim of complete reference correctness.

Historical raw selection was prepared with `prepare_programming_tools.py` as in
commit `220ed20`. The corrected current preparer can select different subsequent
rows because it rejects malformed/assertion-free tests earlier. The exact run
is identified by hashes rather than by assuming row counts reproduce across
preparer versions. Relevant artifacts:

- `data/educational-tools-diverse-v1.source.json`: initial source selection.
- `runs/interface-diverse-reference-checks.json`: all renamed execution outcomes.
- `runs/interface-diverse-test-audit.json`: assertion/indentation audit.
- `runs/interface-diverse-filtered-checks-v2.json`: retained IDs, code grouping and parent report hash.
- `data/programming-interface-diverse-sft-v3/manifest.json`: actual training split.

The filtered curriculum preserves the checked identifiers:

```sh
source scripts/dataset-env.sh
.venv/bin/python scripts/filter_interface_curriculum.py \
  --input data/programming-interface-diverse-v1.jsonl \
  --checks runs/interface-diverse-reference-checks.json \
  --source "$MICROGPT_DATA_ROOT/raw/opencoder-educational/educational_instruct/train-00000-of-00001.parquet" \
  --out data/programming-interface-diverse-checked-v2.jsonl \
  --out-checks runs/interface-diverse-filtered-checks-v2.json --group-by-code
.venv/bin/python -m microgpt.trajectories \
  --input data/programming-interface-diverse-checked-v2.jsonl \
  --out data/programming-interface-diverse-sft-v3 --tokenizer data/large-v1 \
  --max-length 8192 --limit 200000 --pack-episodes --canonical-tool-ids
bash scripts/train_interface_diversity.sh
```

These commands refuse existing outputs. Do not rerun them over the recorded run.

### Split audit before capability selection

The first broader run was stopped at step 2,640, before any development execution
evaluation, after an audit found 231 code ASTs shared across its train/validation
splits when normalizing function names. Original instruction groups alone did
not prevent duplicated implementations under different descriptions.

The filtered curriculum now links both original instruction groups and matching
normalized-code groups before splitting. This keeps transitive duplicate groups
together. The independent audit confirmed zero matching normalized ASTs across
splits, preserved all original instruction groups, and verified that every field
other than split-group metadata was unchanged. The 20,967 references contain
17,056 such ASTs: 16,242 train and 814 validation. This does not prove semantic
deduplication or exclude exposure during the earlier foundation/adapter stages.

The corrected branch restarts from the same original parent, not from the
interrupted run. Evidence: `runs/interface-diverse-v2-diversity.json`; interrupted
run: `runs/interface-diverse-continuation-v1/status.json`. This split correction
was made before observing a capability score for the broader-data branch.

### Training-distribution generation diagnostic

`prepare_curriculum_probe.py` freezes 20 early examples sampled by the actual
seed-42 unweighted training sequence and 20 examples from the corrected validation
split. `data/interface-diverse-probe-v1.jsonl` is a diagnostic of learning and
format transfer, explicitly not independent capability evidence. An example seen
once is not an overfitting test, and validation examples may have earlier foundation
exposure. The source-provided tests have the limitations discussed above.

All 40 reference solutions passed and all empty implementations failed in Docker
(`runs/interface-diverse-probe-controls-v2.json`). Some public references print
values at module scope; the control checker now suppresses that output while
executing code so it cannot be parsed as checker JSON. The initial noisy report
is retained separately. The direct evaluator's optional `--exact-prompt` preserves
these original curriculum requests; default development evaluation is unchanged.
No diagnostic answers are added to training.

The corrected broader branch scored 10/100 at 5K (92 valid Python outputs,
47 matching parameter lists) and 17/100 at 10K (96 valid outputs, 65 matching
parameter lists). At 5K, passing variants cover only product, flatten and longest.
The original-prompt diagnostic scored 5/20 early training examples and 6/20
validation examples, with 38/40 matching parameter lists. These tasks differ from
the development set, so that comparison alone does not isolate prompt effects.

A paired diagnostic on the same 5K checkpoint appended `Use the exact function
signature SIGNATURE.` to each unchanged development request. Signatures were
extracted from the user prompt, not reference answers. It produced 86/100 matching
parameter lists but only 16/100 correct programs (eight new passes, two lost passes,
eight passes shared with the original requests). Repeating an existing cue helps
interface copying, while program semantics remain weak. These diagnostic scores
do not replace the unmodified development score. Frozen results:
[interface-diagnostics.json](interface-diagnostics.json).

`vary_interface_prompts.py` prepares eight generic request layouts from the checked
public curriculum. It reads no benchmark material. All 167,736 assistant answers,
source IDs and split groups remain unchanged; 146,769 user requests change and
20,967 retain the original layout. The byte/content comparison is recorded in
`runs/programming-format-variety-audit.json`. Prepared data is
`data/programming-format-variety-sft-v1`; this is a subsequent intervention, not
a mid-run change to the broader-data comparison.

### Microbatching short lessons

The original SFT loop processes one episode per forward/backward pass. Optional
`--microbatch` now right-pads independent sequences and gathers each example's
supervised positions before projecting to the large vocabulary. Padding lies
after real tokens, so causal attention cannot let it influence those tokens.
The loss averages each episode's own token mean, preserving the existing training
objective rather than silently weighting longer answers more heavily.

An authored regression compares losses and all parameter gradients against
separate passes, including unequal lengths, gaps in supervised labels, multiple
vocabulary chunks and activation checkpointing. All 40 project tests pass.
GPU floating-point execution can still differ; this is objective equivalence,
not a promise of bitwise-identical training trajectories. Default microbatch one
preserves the existing path, and old checkpoints resume with that default.

`profile_finetune.py` measures disposable training copies at fixed example order
and effective batch (`microbatch × accumulation`). It saves no model weights.
The GPU profiles run after the capability experiment finishes. Their token rate
counts assistant targets, not all input tokens; it cannot be compared directly
with the foundation's all-token throughput. Larger effective batches are a
separate optimization choice, not an automatic model-quality improvement.

The broader comparison completed with 10/100, 17/100 and 14/100 at 5K, 10K and
20K. The final checkpoint had 63/100 matching parameter lists despite validation
loss falling to 0.1176. Its 10K checkpoint remains preferable on development
execution. [interface-diversity-outcome.json](interface-diversity-outcome.json)
preserves the complete result; the capability goal remains unmet.

Measured math-SDPA throughput on the 5090 (preloaded examples; short profiling runs):

| Effective batch | Microbatch | Accumulation | Assistant targets/s | Peak allocated GB |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 1 | 2 | 1,981 | 5.08 |
| 2 | 2 | 1 | 3,400 | 4.81 |
| 32 | 1 | 32 | 2,139 | 5.08 |
| 32 | 4 | 8 | 6,944 | 5.81 |
| 32 | 8 | 4 | 7,913 | 7.16 |
| 32 | 16 | 2 | 7,333 | 9.84 |
| 32 | 32 | 1 | 7,189 | 14.26 |

Thus larger microbatches were not always faster. The matched format comparison
uses microbatch two / accumulation one, preserving the effective batch of two.
The real control bootstrap consumed the same 167,311 target tokens in 58.6 seconds,
versus 99.5 seconds in the earlier unbatched bootstrap; model starting weights
differ, so that timing is supporting operational evidence rather than the matched
profile itself. Full measurements are in [sft-throughput.json](sft-throughput.json).

### Matched request-format continuation

`scripts/compare_request_formats.py --out runs/format-comparison-v1` starts both
arms from the broader run's frozen 10K checkpoint (17/100). Each gets 5,000 updates,
a fresh AdamW optimizer, LR 3e-5, seed 42, effective batch two and math SDPA. Source
programs, split groups and ordered target answers are unchanged; only user-request
formatting differs. The two arms run sequentially on the GPU. This separates a
format benefit from simply adding more training.

The predefined format criterion requires no reduction in direct passes versus
the matched control and an improvement of at least five parameter-name matches
out of 100. Meeting it does not meet the project's programming or agent-readiness
gates. Status and lineage are recorded in `runs/format-comparison-v1/status.json`.
Both training and evaluation artifacts are retained; no candidate replaces the
existing service automatically.

The matched result was **7/100 for each arm**. Parameter-name matches improved
from 58 to 66/100 with varied layouts, while syntactically valid outputs fell
from 91 to 83/100. Thus the narrow format criterion passed, but programming did
not improve. Both checkpoints were worse than their 17/100 parent, which remains
selected. [request-format-outcome.json](request-format-outcome.json) records this
distinction explicitly. Each arm consumed exactly 840,346 assistant target tokens.

Both short continuations reset the sampler to seed 42 on the same ordered answers,
replaying early samples already consumed by the parent. This was matched between
arms, so it does not invalidate the format comparison, but it limits the meaning
of “more training”: those exposures were not a fresh traversal of the curriculum.

### Larger-batch optimization pilot

`scripts/train_batched_pilot.sh` starts from the retained 17/100 parent, using the
varied layouts, microbatch eight / accumulation four, LR 1e-5, and a new fixed
sampling seed, 20261006. The larger batch averages more examples per update and
the lower rate makes smaller parameter updates. These are hypotheses about
stability, not established explanations for the earlier regressions. Batch size,
learning rate and sampling seed change together; no result will be attributed to
one of them in isolation.

The first gate is 2,000 updates, approximately 5.4M assistant-target exposures.
Extension requires at least 22/100 direct passes, 75/100 matching parameter lists,
90/100 syntactically valid outputs, and passes in at least five task families.
Only after passing that gate does the script continue to 5K/10K/20K total updates
with a three-hour continuation ceiling. A failed gate stops this recipe. Neither
outcome automatically promotes a model or counts as OpenCode readiness.

Manifest: `runs/batched-learning-experiment-v1.json`. Initial status:
`runs/batched-learning-pilot-v1/status.json`; decision:
`runs/batched-learning-pilot-decision.json`; conditional longer-run status:
`runs/batched-learning-continuation-v1/status.json`. Actual token exposures are
recorded at every checkpoint; the estimate above is not substituted for them.

The pilot finished at **15/100 direct passes**, with 93/100 syntactically valid
outputs, 81/100 matching parameter names, and passes in six families. It failed
the correctness gate, so the 5K/10K/20K continuation was **not launched**.
The 17/100 parent remains selected. See [batched-learning-outcome.json](batched-learning-outcome.json).

Actual exposure was 5,350,924 assistant tokens across 64,000 sampled episodes:
52,741 distinct episode indices and 19,093 source-program IDs, out of 19,915
available training source IDs. These counts reconstruct the fixed sampler and
match checkpoint token totals. They are not counts of independent semantic tasks.
Validation loss fell to 0.10185 while executable correctness remained weak.
For example, a requested product became `max(values)` and a requested sum of
negative numbers selected nonnegative numbers instead. Better interface copying
has not supplied the missing semantic reliability. This ends this optimization
recipe, not the overall project.

### Conditioning and inference diagnostic

A new fixed random sample of 80 validation episodes gave mean episode loss
0.34213 for the retained 17/100 parent and 0.25334 for the batched candidate.
Removing the problem statement but retaining the function signature raised those
losses to 0.47687 and 0.41661. Thus the request affects prediction, but this is not
proof of correctly understanding it. Function names remain informative, and
teacher forcing supplies earlier reference-answer tokens.

Math-versus-Flash loss differences were at most 0.00434 across these samples.
Reconstructing each original prepared input and label sequence matched exactly.
These checks found no substantial backend-loss discrepancy or preparation mismatch
in this sample; they do not prove every inference path correct.

The batched candidate passed 11/20 seen-training and 11/20 validation execution
probes from the existing frozen curriculum diagnostic. Together with 15/100 on
the broader task set, this indicates both imperfect curriculum learning and weak
transfer. The random-sample loss also differs substantially from the trainer's
small fixed validation sample. [conditioning-diagnostics.json](conditioning-diagnostics.json)
preserves the comparison. No acceptance tasks were used.

### Broader independently checked public code

The next data intervention uses the already-downloaded, pinned
[nvidia/OpenCodeInstruct](https://huggingface.co/datasets/nvidia/OpenCodeInstruct)
source. Its upstream scores are insufficient: in a 1,000-reference pilot, eight
answers failed our independent restricted-container run despite perfect upstream
scores. The other 992 passed, and all corresponding empty implementations failed.
[verified-code-pilot.json](verified-code-pilot.json) records the result.

`scripts/prepare_verified_code.py` decodes the source's occasionally doubled JSON
escaping exactly once, preserving Python string literals. It selects single-function
modules with supplied assertions, checks both original and NotImplementedError
implementations in networkless Docker, and retains only positive-pass/negative-fail
pairs. Identical prompts and function-name-normalized code ASTs are transitively
grouped for splitting. This is not semantic deduplication or independent verification
of the complete specification. It neither executes downloaded code on the host nor
imports development/acceptance answers into training.

The larger bounded selection scans the first 20,000 rows of each of 50 downloaded
shards (one million rows), retaining a reservoir of at most 100,000 distinct exact
ASTs before execution. New datasets and checks stay under the authorized NAS dataset
path. This increases source-program diversity instead of counting eight renamed
copies as eight new problems. The preparation pilot is inspection-only.

For the next data pilot, `scripts/train_verified_code.sh` retains the 17/100
parent, LR 1e-5, effective batch 32 and training seed 20261006. Microbatch four /
accumulation eight gives more memory headroom for longer public answers; a small
profile used 5.89 GB, versus 7.31 GB with eight/four. This does not establish a
universal speed advantage. The objective's microbatch equivalence is regression-tested.

The first bounded run evaluates executable code at 500, 1,000 and 2,000 updates,
so an early capability peak is visible. The 2K extension gate remains 22 direct
passes, 75 matching parameter lists, 90 valid Python outputs, and passes in five
families. This is a data intervention with equal update counts, not equal tokens
or compute; both will be recorded. A failed gate ends this data recipe rather
than triggering an unbounded extension. It does not end the overall project.

SFT now optionally accepts `--validation-seed`. This samples reproducibly within
source buckets using a separate RNG; the training sampler is unchanged. Omitting
it preserves legacy validation order and resume compatibility. New runs use 256
validation episodes with seed 20261007. The continuation controller carries that
setting through resume. Tests cover deterministic sampling, balanced sources,
unchanged bucket contents and legacy order. Execution-based model selection remains
necessary regardless of the reported loss.

### 239M inference capacity verification

The retained 239M checkpoint passed the existing cold/warm and changed-prefix cache
equivalence checks. Actual Flash kernels dispatched for 8,192-token and
32,752-token prompts, each followed by 16 generated tokens. Peak allocated memory
was 1.80 GB and 2.78 GB respectively. Instrumented elapsed times were 4.01 and 9.47
seconds; these include profiler overhead and are not serving latency guarantees.
[capacity-inference.json](capacity-inference.json) preserves the measurements.
Synthetic filler established allocation and dispatch, not useful 32K recall,
reasoning, or long-session programming. Those remain separate unmet gates.

The conditioning diagnostic is now runnable as `scripts/diagnose_conditioning.py`
with explicit checkpoint, prepared data and raw curriculum paths. A 32-example
sample from the new-source preparation pilot gave the retained parent a mean
reference loss of 1.04590 with full problems and 1.34559 with signatures only.
The maximum math/Flash difference was 0.00388. This is a different distribution,
including longer prose/docstrings, so the loss is not directly comparable to an
execution score or evidence that the new training will succeed.

One concrete failure caught by the larger negative-control check: a dataset answer
implemented `find_maximum`, but every supplied assertion called Python's built-in
`max` instead. Another used CPU-count-dependent assertions that returned `True`
without calling the implementation on this machine. Both original and empty
implementations therefore appeared to pass. Such rows are excluded. Merely finding
an `assert` AST node is insufficient; the test must distinguish an implementation
from an absent one. Even that is weaker than complete specification verification.

### Executed repair-data preparation

`scripts/prepare_executed_repairs.py` now supplies the missing evidence path for
future tool training. It changes one arithmetic/comparison operation or integer
literal in checked public code, executes the supplied tests, retains a mutation
only when a check actually fails, applies an exact-string correction, and reruns
the checks. An unrelated helper is present and checked before the supplied tests.
File contents and failure/success observations come from restricted Docker.

These are reference-policy demonstrations formatted for the existing OpenCode
fixture, **not model rollouts or runs of OpenCode itself**. The generator does not
claim the model discovered the correction or reasoned through it. Initial files
contain the bug; teaching recovery from the model's own failed action remains a
separate later extension. The source tests still do not prove full specification
correctness, and the context header remains a teaching fixture.

The preparation pilot requested 20 public examples: 19 supported these mutations,
and 18 yielded a verified failing mutation followed by a passing repair. All 18
passed protocol normalization and tokenization (2,401–3,263 tokens). Two authored
unit checks cover a boundary mutation and preservation of function interfaces and
docstrings. [executed-repair-pilot.json](executed-repair-pilot.json) records this
**data-preparation result only**. No tool SFT or agent-readiness gain is claimed;
that training still follows the direct-programming gate.

The larger corpus completed with **98,694 retained source programs**. Of 100,000
candidates, 1,297 reference implementations failed their checks and nine had tests
that also accepted the empty implementation. The retained data has 97,233 distinct
function-name-normalized ASTs and 98,065 original problem texts. Independent audit
found zero exact-prompt or normalized-AST overlap across its train/validation split;
this is not semantic or earlier-pretraining decontamination.

Tokenization retained every record: **93,742 training episodes / 14,417,356 stored
assistant tokens**, and **4,952 validation episodes / 766,576 assistant tokens**.
Stored tokens are not completed training exposures. Unlike the earlier curriculum,
there is one episode per source program, without eightfold name expansion.
The actual checked-source lineage and prepared hashes are frozen in
[verified-code-corpus.json](verified-code-corpus.json).

Preparing repairs from the first 1,000 entries of this completed corpus found
760 mutable programs and retained **737 executed failure→repair demonstrations**.
They preserve the parent's split groups. They are ready for later curriculum work,
but are not included in this direct-code pilot and have not trained an agent.

All 737 retained repair records matched their executed before/after file contents
and nonzero→zero check results, and preserved their source split groups.
Preparation kept 699 training episodes (334,601 assistant tokens) and 38 validation
episodes (18,796 assistant tokens), with no overlength drops at 8K. The full check
artifact hash and prepared manifest are in [executed-repair-corpus.json](executed-repair-corpus.json).
These data are prepared, not yet used for training.

### Broader long-session verification path

The existing horizon evaluator now accepts `--tasks-file` for the frozen programming
suite and `--no-handoff` for an uninterrupted OpenCode session. It samples distinct
named functions, rejects fixture name collisions, and cumulatively checks earlier
requirements after every edit. The existing transient-checker failure is retained.
Automatic OpenCode compaction stays enabled, but has not yet been shown reliable
for this candidate. A separate run can retain the halfway handoff.

The 32-task programming reference composition passed its restricted-Docker check,
and the empty starting implementation failed. The old arithmetic self-test also
passed. Unit checks ensure reference answers never enter user prompts or the state
file, reject colliding task definitions, and preserve old arithmetic prompts.
[programming-horizon-controls.json](programming-horizon-controls.json) is a **harness
control result**, not model performance. The harness writes user requirements into
TASK_STATE.md; this does not establish model-written memory or summarization.

For a qualified candidate, the broader session command is:

```sh
.venv/bin/python scripts/eval_horizon.py \
  --url http://127.0.0.1:8010/v1 --label 'exact candidate checkpoint' \
  --tasks-file evaluation/programming-v1/development.jsonl \
  --rounds 32 --seed 20261007 --no-handoff --effort none \
  --out runs/programming-session-candidate
```

Before choosing the next training change, compare the selected public-data candidate
on the already-frozen exact-signature diagnostic as well as the ordinary development
prompts. This diagnostic repeats interface information from the user request; it
does not supply answers. A gain of at least ten direct passes accompanied by at least
90 correct parameter interfaces would support a targeted interface-augmentation
experiment. Otherwise prioritize the remaining program-logic deficit. These are
next-experiment criteria, not relaxed readiness gates or headline model scores.

### Completed checked-public-data pilot

Direct scores were **24/100, 24/100, and 21/100** at 500, 1,000 and 2,000 updates.
The earlier 500-update checkpoint is retained among equal highest direct scores.
It consumed 2,454,569 assistant tokens; the final checkpoint consumed 9,852,956.
At 500, 13 previously failed cases passed while six earlier successes regressed.
This is a development-set gain over 17/100, not agent readiness or a statistical
claim based on 100 independent problems (the variants are correlated).

At 2K, all outputs parsed, but only 65/100 parameter interfaces matched. The
extension gate failed and no 5K/10K/20K continuation launched. Repeating the exact
signature in the selected checkpoint's prompts raised interface matches from
65 to 86/100, yet correctness stayed **24/100** and valid Python fell to 91/100.
This failed the predeclared interface-augmentation criterion: copying improves,
but the program-logic deficit remains. The results are frozen in
[verified-code-learning-outcome.json](verified-code-learning-outcome.json).

The next control adapts the same existing 1B-token scratch foundation directly on
the checked public data, omitting the earlier SFT chain. It keeps this pilot's LR,
effective batch, sampler seed, validation sample and 500/1K/2K evaluation points.
This tests prior-SFT ancestry before paying for another billion pretraining tokens
(roughly eleven hours at the observed 25.3K tokens/sec, not a guarantee). A further
extension requires at least 29 direct passes plus the existing interface/syntax/
family criteria. No inference is made that more pretraining must solve the problem.

Meanwhile, `prepare_large --source codeparrot --tokens 3000000000` prepares a larger
repository-grouped source stream from the already-downloaded pinned shards on the
NAS. It uses the existing license/length selection and exact-text deduplication.
This is a preparation ceiling, not a completed token count or training exposure.
Architecture, tokenizer and the served model are unchanged.

### Scoped edits and stochastic test gotchas

Repair preparation now optionally uses `--scoped-edits`: the smallest unique changed
line span replaces the bug, preserving other functions without copying the whole
file. The exact replacement is executed and must reproduce the checked reference
file. Tests cover repeated-line disambiguation and insertions that preserve prior
code. This matters for long sessions, where repeatedly rewriting an entire growing
file wastes context and risks regressions.

Repeating the repair checks yielded 740 instead of 737 retained examples. Three
changed retention status and one selected a different failing mutation; all four
references imported `random`. A single passing run had overstated the stability of
those labels. The filtered scoped corpus retains **727 examples** whose before/after
programs and fail/pass outcomes agreed in both executions, excluding another nine
references importing random/secrets. [scoped-repair-data.json](scoped-repair-data.json)
records both evidence hashes. This is not a proof of determinism or complete test
coverage. The generator's new `--exclude-random-imports` option conservatively checks
ordinary random/secrets imports in both references and tests for future runs.
The older corpora remain experiment history; no repair corpus has trained a model.

### Expanded foundation data and resume identity

The GitHub preparation exhausted its pinned 32 shards at **2,825,447,496 tokens**,
below the requested three-billion ceiling: 2,677,067,460 train and 148,380,036
validation tokens. `data/foundation-expanded-v1` combines that stream with the
original other four streams at unchanged 60/16/12/8/4 sampling weights. Its
3,057,260,620 training tokens are stored data, not new training exposure. Source
files are hardlinked on the NAS to avoid duplicate storage and must remain immutable.
Hash and old-prefix checks are recorded in `expanded-foundation-corpus.json`.

Pretraining now records its attention backend and rejects resume-time changes to
training settings, including the total cosine horizon, learning rate and batch
shape. `--init-from` intentionally begins a new optimizer/data stage. For a backend
change, or legacy checkpoints with no recorded backend, use the explicit
`--allow-attention-change` switch; this does not bypass schedule/data checks.
`--attention math` provides an explicit fallback. Twelve training tests pass,
including bitwise equality between interrupted/resumed and uninterrupted CPU
training with math attention, plus rejection of changed experiment settings.

A proposed expanded-data continuation keeps the 239M architecture and tokenizer.
Its configuration freezes a one-billion-token cosine horizon but the first run
will stop after 250M additional tokens. LR 3e-5 starts near the preceding run's
terminal rate; a short 200-update warmup accommodates the fresh optimizer. This
is a declared experiment, not a universally established continuation recipe:
[Ibrahim et al.](https://arxiv.org/abs/2403.08763) found rewarming/redecay useful in
their setting, whereas [NVIDIA's continuation study](https://arxiv.org/abs/2407.07263)
favored no warmup for their well-trained 15B model. Neither establishes the best
schedule for our undertrained 239M model. A short batch-two throughput/memory check
must precede the long run. No additional foundation training has begun at this entry.

The bounded launcher is `scripts/train_expanded_foundation.py --out runs/expanded-foundation-pilot-v1`.
It requires the clean control's three completed evaluations, records immutable
experiment identities, stops pretraining at 30,518 updates (250,003,456 new tokens)
or four training hours, then runs the same 250-update bootstrap and 500/1K/2K
adapter evaluations. Five additional direct passes over the best clean-control
checkpoint are required even to consider another tranche. That is a continuation
criterion, not readiness; the best prior recipe's 24/100 is also recorded for
comparison. Failed processes preserve their command and phase in `status.json`.
The full 54-test suite passed after the resume changes.
