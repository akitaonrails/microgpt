# MicroGPT Lab — train our own model from scratch

**Current capability:** the scratch 239M candidate passes **17/100** development
variants at 10K updates on a broader checked-code curriculum. This is still far
below useful programming competence. The [controlled experiments](docs/capacity-pilot.md)
separate code correctness, request-format sensitivity and training-data defects.
The served 91M artifact below remains a narrow arithmetic lesson.


An independently written PyTorch GPT inspired by
[Karpathy's microgpt](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95).
Every model weight starts random. Pretraining, assistant/tool training and serving
use the same model implementation. There is no downloaded model or adapter path.

The project has two scales: a 0.9M byte-tokenizer lesson and a larger student model
with multilingual tokenization, a 32K context allocation limit, Flash Attention,
KV caching and an OpenCode-compatible server. A short pilot checks the engineering;
useful coding and conversation require substantial training and behavioral evaluation.
The selected **90,969,600-parameter scratch model** completed four 16-round
OpenCode sessions and one **32-round session with 130 tool calls**, including
checker-error recovery and a fresh-session handoff. It passes 8/8 narrow arithmetic
repairs, but 0/4 broader repairs and 0/12 reasoning problems in either tested mode.
This teaches a complete agent workflow for a small trained task family.

Start with the [minimal agent runbook](docs/agentic-minimum.md): one foundation
pretraining stage, one focused assistant-training stage, a portable export, and a
restricted OpenCode lesson. The selected export is `runs/agent-ready-v1/model.pt`.
The [committed evidence](docs/agentic-evidence.json) records hashes and measurements;
[verification](docs/verification.md) preserves the earlier failed experiments.

## Offline first lesson

```sh
uv venv --python 3.12
uv pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
uv pip install -e .
.venv/bin/python -m microgpt.prepare
.venv/bin/python -m microgpt.train
.venv/bin/python -m microgpt.sample --temperature 0
```

The lesson learns narrow Python function patterns. It does not train a general
assistant. Preparation refuses existing outputs: choose fresh paths for repeats.

## Scratch model to OpenCode

For the RTX 5090/H100 use the tested CUDA wheel, then follow
[the complete scratch training sequence](docs/finetuning.md):

```sh
uv pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install -e '.[data,tokenizer,tune,serve]'
source scripts/dataset-env.sh
```

The necessary stages are:

1. Download and curate licensed code plus conversational/multilingual data.
2. Fix a tokenizer and split by repository/conversation group.
3. Pretrain **our randomly initialized GPT** on next-token prediction.
4. Train its full weights on assistant replies and tool calls, masking user/tool text.
5. Continue training at longer windows and measure retrieval and tool recovery.
6. Serve that checkpoint through OpenCode and grade actual edits independently.

The current student has **90,969,600 parameters** with `o200k_base`.
`configs/student.toml` uses eight layers, width 384, six attention heads and a
32,768-token context ceiling. `o200k_base` is the default in the documented recipe;
`o200k_harmony`, local BPE and bytes are also available. Tokenizer vocabulary reuse
does not import learned model weights. Harmony vocabulary does not implement the
Harmony message format; our small text protocol is explicit in `protocol.py`.

```sh
.venv/bin/python -m microgpt.serve --checkpoint runs/scratch-sft/last.pt --attention flash
```

The server requires a local checkpoint. It never falls back to an external model.
OpenCode configuration and its limits are explained in [OpenCode setup](docs/opencode.md).

## Read the implementation

| File | Lesson |
| --- | --- |
| `tokenizer.py`, `prepare.py`, `corpus.py` | Text, token IDs, grouped splits and pretraining streams |
| `model.py`, `attention.py` | RMSNorm, RoPE, causal attention, SwiGLU and tied embeddings |
| `train.py` | Random initialization, AdamW, accumulation, validation and continuation |
| `protocol.py`, `trajectories.py` | Conversation serialization and assistant-only targets |
| `finetune.py` | Full-weight supervised training of our own checkpoint |
| `prepare_large.py`, `mixture.py`, `episodes.py` | Stream large corpora, mix sources and load episodes without keeping token lists in RAM |
| `practice.py`, `scripts/train_stages.py` | Construct tool curricula and run a bounded, resumable sequence |
| `inference.py`, `chat.py`, `serve.py` | KV reuse, validated tool calls and HTTP/SSE |

PyTorch provides differentiation and attention kernels; model architecture and
training loops remain visible. CPU tests check the important invariants:

```sh
.venv/bin/python -m unittest discover -s tests -v
```

[Datasets/NAS](docs/datasets.md) · [Training sequence](docs/finetuning.md) ·
[Hardware/Runpod](docs/hardware.md) · [Architecture walkthrough](docs/walkthrough.md) ·
[Research and Genesys inspiration](docs/research.md) · [Verification](docs/verification.md)

Reasoning training is included in the bounded curriculum: complete worked solutions, checked Python traces, and thinking/tool episodes. The API separates `reasoning_content` from answers and actions; OpenCode has reasoning-effort variants. These are training and protocol features, not demonstrated deep reasoning. See [the reasoning recipe](docs/reasoning.md).
