# Reasoning curriculum

This is supervised reasoning training of our own **90,969,600-parameter model**,
initialized from scratch. No Qwen or other pretrained weights are loaded. Public
teacher-generated text is supervision, so this is also trace distillation. It is
not reinforcement learning, and printing a thinking section does not demonstrate
correct reasoning.

The selected agent baseline uses **`none` effort**. Its held-out reasoning score
is **0/12 with none and 0/12 with high**, at a 256-token answer budget. Thinking
serialization is implemented; deep reasoning is not demonstrated. The curriculum
below records earlier experiments, not a required stage of the successful narrow
agent recipe. See [the current runbook](agentic-minimum.md).

## Data and filtering

The reproducible entry point is `bash scripts/prepare_coding_data.sh`. Downloads,
curated records and tokenized episodes live under the authorized NAS directory
`/mnt/gigachad/microgpt/dataset`. The new public shards add about 754 MB of raw data.

We pin [Mixture-of-Thoughts](https://huggingface.co/datasets/open-r1/Mixture-of-Thoughts)
to `e55fa28006c0d0ec60fb3547520f775dd42d02cd`, two code shards and two math shards.
Its authors report verified DeepSeek-R1 traces; we have not independently executed
the downloaded solutions. We retain complete user/assistant examples with both
thinking and final answer, at most 4,096 assistant tokens and 8,192 total tokens.
Oversized examples are rejected whole, never cut through a solution. Code answers
must contain a Python code fence. These filters leave 699 code and 6,000 math
examples. This bounded selection favors shorter solutions and is not representative
of the full source dataset.

The mixture card does not declare a single license. Preserve source attribution:
[Codeforces-CoTs](https://huggingface.co/datasets/open-r1/codeforces-cots) declares
CC-BY-4.0; [OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k)
declares Apache-2.0. Raw manifests retain revision and file hashes; curated records
retain upstream source names. Do not relabel the mixture as MIT.

Original data adds 6,000 direct/thinking records for 3,000 program-tracing problems,
with deterministic numeric oracles, plus 3,995 planning/tool variants of our
simulated practice and handoff episodes. These are constructed exercises, not
claims of real-world agent success. Paired answers and tool variants retain the
same problem group for a disjoint train/validation split.

`data/agent-sft-v4` combines the existing coding, conversation and tool curriculum
with these additions: **78,192 train / 4,343 validation episodes**, including
59,773,809 supervised training tokens. Complete episodes can reach 32K. The earlier
`agent-sft-v3` remains a historical artifact.

## Training

The 12-hour runner allocates up to 7.8 hours foundation, 1.8 hours instruction,
1.2 hours reasoning, and 1.2 hours long-horizon training. Existing foundation
progress is resumed with its optimizer and RNG. The reasoning stage balances the
four reasoning sources at up to 8K, with 20% general short-example replay. The final
stage balances long episode lengths through 32K and keeps 10% short replay. Its
tool episodes include thinking/action variants, recovery and handoff practice.
Budgets exclude data loading, checkpoint writes and evaluation; step limits may
finish a stage early.

```bash
.venv/bin/python scripts/train_stages.py --out runs/coding-v1 --hours 12 \
  --sft-data data/agent-sft-v4 --resume
```

Use `--resume` only for the existing run, and never start a second runner while one
is active. See [training-plan.md](training-plan.md) for a fresh run.

## Serving and OpenCode

Assistant targets use `<think>...</think>` followed by the answer or tool call.
The API exposes the section as `reasoning_content` in JSON and SSE. Only complete,
schema-valid tool calls **outside** thinking can become actions. Incomplete thinking
returns no action. Both thinking and answer consume the output-token budget.

`reasoning_effort` accepts `none`, `low`, `medium`, or `high`; these select learned
instructions, not guaranteed amounts of computation. OpenCode's example config
advertises reasoning and offers these variants through `reasoningEffort`, with
none as default. The server output ceiling is 4,096 tokens. SSE is buffered until
generation and tool validation finish. Exact-prefix KV caching remains supported;
clients that omit previous thinking can reduce cache reuse.

## What counts as improvement

The runner executes `scripts/eval_reasoning.py` on fresh program inputs, comparing
`none` and `high` at equal 256- and 1,024-token output caps. It reports exact final
integer accuracy separately for familiar procedural templates and a held-out
nested-loop template. It records errors, truncated answers, usage and latency.
Longer explanations earn no credit. This is a small diagnostic, not a broad
reasoning benchmark or a causal ablation of the training stage.

Basic OpenCode repairs, the 16-round recovery/handoff evaluation, and long-context
retrieval still run. Correct answers must coexist with usable tool calls and
successful tests. CPU tests validate masking, arithmetic oracles and the API
contract; a one-step GPU smoke validates training plumbing. Neither establishes
model capability. Read the completed run's evaluation files before claiming useful
reasoning. At 91M parameters, robust deep reasoning remains an ambitious target.
