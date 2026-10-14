# MicroGPT Lab

Train a GPT-style language model from random weights, on one consumer GPU, and
see every step: data, tokenizer, model, training loop, fine-tuning, serving,
and honest evaluation.

> **Status: experiment closed (October 14, 2026).** The final model is a
> 337M-parameter GPT trained from scratch on 10B tokens plus coding fine-tuning.
> It writes small Python functions (58/100 on a reserved holdout) but is **not**
> a working coding agent. This repository is for learning, not for use as an
> assistant. The [closeout report](docs/report/experiment-closeout.md) has the
> full story.

This is an independently written PyTorch project inspired by
[Karpathy's microgpt](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95).
No pretrained weights are downloaded: every model here starts from random
initialization and is trained by this code.

## Goals

The project started as a cleaner, modular counterpart to microgpt and grew into
a test of how far one person can take a scratch model on a desktop GPU:

1. Make the whole pipeline readable: tokenization, next-token prediction,
   attention, the training loop and sampling should be plain Python you can
   step through.
2. Scale the same code from a CPU toy to a few-hundred-million-parameter model.
3. Fine-tune that model into an assistant that writes code, uses tools and
   works inside [OpenCode](https://opencode.ai) over long sessions.
4. Measure capability with executed tests, not with loss curves or anecdotes.

Goals 1, 2 and 4 were met. Goal 3 was not: the model learned to write some
correct functions, but never learned to act as an agent in a real repository.

## What was achieved

| Final 337M model, one greedy attempt per task | Result |
|---|---:|
| Development programming suite (used to pick the checkpoint) | 68/100 |
| Reserved holdout, evaluated once after selection | 58/100 |
| Holdout answers that parse as Python | 100/100 |
| Real OpenCode repository repairs | 0/4, no tool calls |
| Exact program-tracing answers | 0/12 |
| Retrieval of a fact hidden in 8K/16K/32K prompts | 0/9 |

Along the way the project built and kept:

- a 337,167,360-parameter decoder (24 layers, width 1,024, RoPE, SwiGLU,
  RMSNorm) with a 32K local BPE tokenizer, trained on a curated 3.48B-token
  code and text corpus for 10B token exposures (about 136 hours on one RTX 5090);
- a coding learning curve across 2B, 5B and 10B tokens (39, 52 and 65 out of
  100 after short fine-tuning), showing that more foundation training helped;
- full-weight supervised fine-tuning that trains only on assistant tokens;
- an OpenAI-compatible server with KV/prefix caching, streaming, reasoning
  separation and schema-checked tool calls, wired to OpenCode;
- graders that run generated code only in a locked-down Docker container;
- an earlier 91M model that completed a 32-round, 130-tool-call OpenCode
  session on a narrow arithmetic task family, and failed broader tasks.

The scores are small local diagnostics, not HumanEval or SWE-bench. The
[closeout report](docs/report/experiment-closeout.md) explains their limits, the
failures, the training time and what a next experiment would cost.

## Quick start (CPU, about a minute)

The first lesson trains a tiny byte-level model on narrow Python patterns. It
needs no GPU and no downloads beyond PyTorch.

```sh
uv venv --python 3.12
uv pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
uv pip install -e .

.venv/bin/python -m microgpt.prepare --out data/demo     # build a tiny dataset
.venv/bin/python -m microgpt.train --device cpu          # ~600 steps, ~35 s
.venv/bin/python -m microgpt.sample --temperature 0      # continue a prompt
```

The sample continues `def add_42(x):\n    return`. Expect something like
`return x + 73`: the shape of the code is learned before its meaning, which is
the whole project in miniature. `prepare` refuses to overwrite existing output,
so pick a fresh `--out` (and pass it to `train --data`) when you repeat it.

## Run the tests

```sh
.venv/bin/python -m unittest discover -s tests -v
```

85 CPU tests (about 20 seconds) cover the model, tokenizer, data splits,
training resume, the chat protocol, tool parsing, graders and evaluation
controls. They do not need a GPU or the datasets.

## Explore the repository

```
microgpt/        the library: tokenizer, model, training, fine-tuning, serving
configs/         model and training sizes, from demo.toml to the 337M pivot
scripts/         data preparation, training runners and evaluators
tests/           CPU unit tests
evaluation/      frozen programming tasks (development and acceptance splits)
examples/        an OpenCode configuration that points at the local server
docs/            guides, the final report, experiment history and evidence
```

Read the library in this order; each module states its purpose in its first
line:

| Step | Files | What you learn |
|---|---|---|
| 1 | `tokenizer.py`, `prepare.py`, `data.py` | Text to token IDs, grouped train/validation splits, shifted targets |
| 2 | `model.py`, `attention.py` | RMSNorm, rotary positions, causal attention, SwiGLU, tied embeddings |
| 3 | `train.py` | Random initialization, AdamW, accumulation, validation, resumable checkpoints |
| 4 | `sample.py`, `inference.py` | Sampling, temperature, KV cache reuse |
| 5 | `protocol.py`, `trajectories.py`, `finetune.py` | Chat format and training on assistant tokens only |
| 6 | `corpus.py`, `prepare_large.py`, `mixture.py`, `episodes.py` | Large corpora without holding them in RAM |
| 7 | `chat.py`, `serve.py`, `constraints.py` | An HTTP server for OpenCode, with validated tool calls |

[Follow one prediction](docs/guides/walkthrough.md) traces a single token through
the code. PyTorch supplies autograd and attention kernels; the architecture and
loops are all here.

## The full pipeline (GPU)

The large runs used one RTX 5090 with PyTorch 2.7.1 and CUDA 12.8:

```sh
uv pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install -e '.[data,tokenizer,tune,serve]'
source scripts/dataset-env.sh    # sets the dataset root and every cache under it
```

`scripts/dataset-env.sh` is written for the original setup: it points
`MICROGPT_DATA_ROOT` at a NAS mounted on `/mnt/gigachad` and refuses to run when
that mount is missing, so no download silently fills your system disk. Edit
the path and the mount check for your machine before sourcing it.

The stages, each documented in the guides:

1. Download licensed code and conversation data with pinned revisions
   ([datasets](docs/guides/datasets.md)).
2. Fit a tokenizer and split by repository or conversation group.
3. Pretrain the randomly initialized model on next-token prediction.
4. Fine-tune all weights on assistant replies and tool calls
   ([training sequence](docs/guides/finetuning.md)).
5. Serve the checkpoint and grade real edits in OpenCode
   ([OpenCode setup](docs/guides/opencode.md)).

The final 337M run is `scripts/run_foundation_pivot.py` with
`configs/pivot-337m.toml`; its frozen plan is
[foundation-pivot.md](docs/history/foundation-pivot.md). Hardware notes and an
optional cloud route are in [hardware](docs/guides/hardware.md).

Serve a trained checkpoint locally (it never falls back to a remote model):

```sh
.venv/bin/python -m microgpt.serve --checkpoint runs/scratch-sft/last.pt --attention flash
```

The evaluators in `scripts/eval_*.py` and
`scripts/evaluate_repair_checkpoint.py` reproduce the published scores; the
exact commands are at the end of the
[closeout report](docs/report/experiment-closeout.md#reproduce-the-small-benchmark-and-inspect-the-model).

## Data and checkpoints are not in git

Datasets (about 191 GB) and checkpoints (a single full checkpoint is about
4 GB) live outside the repository. `data` is meant to be a symlink to your
dataset root and `runs/` holds local outputs; both are ignored, as are weight
and array formats (`*.pt`, `*.safetensors`, `*.npy` and so on). What git keeps
is small and reviewable: code, configs, frozen evaluation tasks, and JSON
evidence with hashes, counts and per-task results.

## Documentation

Start at the [documentation index](docs/README.md). The short version:

- [Experiment closeout](docs/report/experiment-closeout.md): results, model
  card, data, training time, failures and next steps.
- [Guides](docs/README.md#guides): how to run each part.
- [History](docs/README.md#history): the experiments in the order they
  happened, including the ones that failed.

## Credits

Inspired by Andrej Karpathy's microgpt, nanoGPT and nanochat, and by Sebastian
Raschka's *Build a Large Language Model (From Scratch)*. Training data comes from
public datasets listed with their licenses in the
[dataset guide](docs/guides/datasets.md); those licenses still apply to that
data.
