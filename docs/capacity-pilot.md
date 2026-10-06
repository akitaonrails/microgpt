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
