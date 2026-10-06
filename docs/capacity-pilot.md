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
branch is now running under `runs/interface-diverse-continuation-v1`, with its
preregistered manifest in `runs/interface-diverse-experiment-v1.json` and bootstrap
output in `runs/interface-diverse-launch.log`.

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
Only `programming-interface-diverse-sft-v2` is eligible for training; the earlier
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
- `runs/interface-diverse-filtered-checks.json`: retained IDs and parent report hash.
- `data/programming-interface-diverse-sft-v2/manifest.json`: actual training split.

The filtered curriculum preserves the checked identifiers:

```sh
source scripts/dataset-env.sh
.venv/bin/python scripts/filter_interface_curriculum.py \
  --input data/programming-interface-diverse-v1.jsonl \
  --checks runs/interface-diverse-reference-checks.json \
  --source "$MICROGPT_DATA_ROOT/raw/opencoder-educational/educational_instruct/train-00000-of-00001.parquet" \
  --out data/programming-interface-diverse-checked-v1.jsonl \
  --out-checks runs/interface-diverse-filtered-checks.json
.venv/bin/python -m microgpt.trajectories \
  --input data/programming-interface-diverse-checked-v1.jsonl \
  --out data/programming-interface-diverse-sft-v2 --tokenizer data/large-v1 \
  --max-length 8192 --limit 200000 --pack-episodes --canonical-tool-ids
bash scripts/train_interface_diversity.sh
```

These commands refuse existing outputs. Do not rerun them over the recorded run.
