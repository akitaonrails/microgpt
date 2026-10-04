# Larger scratch training and OpenCode capability gates

For the shortest validated route, use the [minimal agent runbook](agentic-minimum.md).
This document also retains the larger experimental curriculum and its failed stages.


The first 91M pilot failed all four repository repairs and produced no tool
calls. The next run keeps our randomly initialized architecture and o200k_base
token IDs; no external model weights are used. Its objective is basic coding
through OpenCode, followed by reliable longer tool runs. Neither objective can
be declared achieved from loss or a context-allocation test.

## Data and foundation stage

Prepare roughly one billion distinct token positions from pinned public shards
on `/mnt/gigachad/microgpt/dataset`. Sample by token proportions: 60% permissively
licensed GitHub Python (CodeParrot-clean), 16% OpenCoder coding instructions,
12% Cosmopedia educational text, 8% SmolTalk, 4% multilingual SmolTalk2. Exact
deduplication within each pretraining source and repository/problem grouping happen before partitioning.
These are bounded subsets, not downloads of every full dataset. Public teacher
outputs are training data; their teacher weights are never loaded.

Start with up to one billion token exposures, using 2K windows, Flash Attention,
BF16 computation and FP32 optimizer state. Continue our own pretraining pilot.
Token exposures and distinct prepared tokens are reported separately. Measure
the actual 5090 speed and use a finite wall-clock budget; H100 execution is an
optional deployment of the same code, not a prerequisite or a booked service.

The large o200k vocabulary consumes most parameters in this 91M model. The
Transformer itself is small. A billion tokens is a first experiment, not an
assurance of competent coding. For perspective, the original SmolLM 135M/360M
models used 600B training tokens ([training report](https://huggingface.co/blog/smollm)).

## Instruction and tool curriculum

1. Full-weight instruction training on complete episodes up to 8K: coding
   instructions, everyday/multilingual conversation, real commit-derived edits,
   and original, verifiable tool exercises.
2. Train complete trajectories up to 32K, including SWE-smith successful-source
   trajectories and constructed multi-file tasks. Supervise all assistant turns
   in one causal pass; mask user messages and tool observations. Never silently
   truncate a tool call or discard a failed attempt inside a successful episode.
3. Explicitly include planning, intermediate progress, failed-edit recovery,
   tests, changed requirements, and state reconstruction. Keep short examples
   in the broader mixture to reduce forgetting. A dedicated long-horizon stage
   can select episodes by actual tool-call count, independently of token length.

Source labels are not proof that imported code/tests were re-executed. Synthetic
exercises are labeled as constructed; only independently executed checks may
be called verified. Do not train on the four existing evaluation solutions.

## Harness and evidence

Use the native OpenAI-compatible server and OpenCode's read/edit/test tools.
Exact-prefix KV caching must preserve outputs when reused or invalidated.
Keep tool execution in the harness; parsing valid JSON must not invent a model
decision. Tool errors go back as observations. Bound steps, output and runtime.
Persistent task notes and compaction belong in the harness and its training
examples. Context length and action horizon are measured separately.

Minimum capability gates:

- Fresh held-out basic repairs: report task correctness, actual read/edit/test
  calls, valid-call rate, premature exits and loops. Keep the old four-task set
  as a regression, not as evidence of broad generalization.
- Longer tasks: report completed subtasks after 8, 16 and 32+ calls, recovery
  from a failed operation, and preservation of earlier requirements.
- Long context: retrieve/use task state at different positions in 8K/16K/32K
  histories. Allocation success alone is only a memory check.
- Resume a checkpoint with the same data/settings; record lineage and budgets.

If a gate fails, retain and report the failure. Do not label a checkpoint
OpenCode-capable just because its API connects. Subsequent changes should
target the observed failure: syntax, tool selection, editing, testing, memory,
or foundation knowledge.

## Sources

- [CodeParrot-clean](https://huggingface.co/datasets/codeparrot/codeparrot-clean)
- [OpenCoder stage 1](https://huggingface.co/datasets/OpenCoder-LLM/opc-sft-stage1):
  use `largescale_diverse_instruct`, not the subset whose authors flag poor outputs.
- [SmolLM-Corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus):
  Cosmopedia has text; Python-Edu contains IDs requiring separate content retrieval.
- [SmolTalk for small models](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk)
- [SmolTalk2](https://huggingface.co/datasets/HuggingFaceTB/smoltalk2)
- [SWE-smith trajectories](https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories)

See the source manifests for exact revisions, byte counts and SHA-256 hashes.

## Run it

Use the environment installation in [the scratch guide](finetuning.md), then:

```sh
# Roughly 12 GB of raw source shards; all dataset writes stay in the NAS root.
bash scripts/prepare_coding_data.sh

# Fresh random weights. Run in a persistent terminal for unattended training.
.venv/bin/python scripts/train_stages.py --out runs/my-coding --hours 12

# Resume an interrupted run; keep its original hours budget.
.venv/bin/python scripts/train_stages.py --out runs/my-coding --hours 12 --resume
```

If your project already has a different `data` directory, pass `--data
/mnt/gigachad/microgpt/dataset/project-data/large-v1 --sft-data
/mnt/gigachad/microgpt/dataset/project-data/agent-sft-v4` to the runner. The preparation
script only creates a `data` symlink when that project path does not exist.
Existing complete preparations are reused; incomplete outputs fail visibly.

The current `runs/coding-v1` run continues `runs/scratch-pretrain/last.pt`, which
was itself initialized randomly here. It does not restart from someone else's
model. Its planned budgets are up to 7.8 hours foundation, 1.8 hours instruction, 1.2 hours reasoning,
and 1.2 hours long-horizon training. A stage also stops at its configured step
limit. Preparation, checkpoint writes and evaluation add overhead to these
training budgets; this is not a hard operating-system wall-clock cutoff.

```sh
cat runs/coding-v1/status.json
tail -n 5 runs/coding-v1/foundation/metrics.jsonl
tail -f runs/coding-v1/foundation.log
```

`status.json` records the runner PID. Sending that process SIGTERM saves the
active trainer's checkpoint and stops before the next stage; SIGKILL cannot save
work. `--resume` restores optimizer and RNG and subtracts time already spent.
The runner stops visibly on errors and does not silently skip failed stages.

At completion, inspect `eval-basic/results.json`, `eval-horizon/results.json`
and `eval-context.json` under the run directory. The word `evaluated` is a run
status, not a model-quality claim. The long-horizon gate requires all 16 evolving
requirements, at least 32 actual tool calls, recovery from an injected transient
checker failure, and a fresh-session handoff using persistent requirements.
This is scaffolded procedural testing; it does not prove autonomous work on
unfamiliar large repositories. Automatic model-generated compaction quality is
also a separate, unproven capability.

After the run, serve its checkpoint:

```sh
.venv/bin/python -m microgpt.serve \
  --checkpoint runs/coding-v1/long-horizon/last.pt --attention flash
```

The same single-GPU runner works on an H100 with prepared data copied to its
local disk and explicit `--data`/`--sft-data` paths. No Runpod machine was rented.

## Prepared curriculum, 2026-10-03

| Source | Pretraining train tokens | Sampling share |
| --- | ---: | ---: |
| CodeParrot-clean, selected permissive licenses | 567,339,604 | 60% |
| OpenCoder diverse coding instructions | 151,911,977 | 16% |
| Cosmopedia v2 | 114,162,009 | 12% |
| SmolTalk for small models | 76,113,200 | 8% |
| SmolTalk2 multilingual | 38,005,974 | 4% |
| Total | 947,532,764 | 100% |

SFT has 62,366 training and 3,475 validation episodes, with 33,785,800 and
2,240,035 supervised assistant tokens. A complete episode is one training
example, with all assistant bodies supervised and user/tool bodies masked.
Of the training episodes, 3,492 have at least eight tool calls; 762 have at least
32 calls. There are 249 examples longer than 16K; the maximum is 32,698 tokens.
The corresponding held-out counts include 342 episodes with at least eight
calls and 70 with at least 32. Details are in `runs/curriculum-report.json`.

Imported SWE-smith examples retain the source success labels; their original
tests were not independently rerun. Original practice episodes verify exact
source transformations in a simulated environment, not executable code tests.
The OpenCode graders execute code independently inside restricted containers.

Limitations: exact deduplication does not remove semantic duplicates, source
datasets may overlap, and translations may share tasks without a common ID.
Within-source/repository splits are useful controls, not proof of complete
benchmark decontamination. Short greeting-based conversation groups also leave
the separate everyday-conversation source without retained validation examples;
SmolTalk and multilingual held-out episodes supply broader conversation coverage.

The native SWE-smith importer scans all 24,100 tool-split rows. It retains 866
resolved, structured-call episodes before length filtering. It rejects unresolved
runs, text/XML-only action formats, unsupported edits and source-only intermediate
`submit` actions. Original source system tool instructions are replaced with the
translated protocol instructions. 501 training and 184 held-out native episodes
fit 32K. The earlier broad preparations are superseded by `agent-sft-v4`.

The long stage samples source/length bins (up to 8K, 16K, 32K) uniformly and
requires at least eight tool calls for its main examples. It spends 10% of samples
replaying the broader short curriculum to reduce coding/conversation forgetting.
This deliberately oversamples long histories. Repeated exposure to
the limited long examples is not additional distinct training data.

The reasoning extension adds complete worked solutions, paired direct/thinking program traces, and thinking/tool episodes. See [reasoning.md](reasoning.md) for sources, filtering, controls and evaluation limits.
