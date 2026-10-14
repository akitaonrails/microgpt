# Minimal scratch-to-assistant training sequence

For the shortest validated route, use the [minimal agent runbook](agentic-minimum.md).
This document also retains the larger experimental curriculum and its failed stages.


## Larger training run

For the current billion-token preparation, complete-episode curriculum and
automatic OpenCode/long-horizon evaluation, use [the training runbook](../history/training-plan.md).
The smaller commands below remain useful for learning each stage separately.

All weights are ours, initialized randomly by `GPT` in `model.py`. “Fine-tuning”
here means a later training stage of that same model, updating **all** its weights.
There are no imported model weights, frozen pretrained backbone or adapters.
Reuse of a public tokenizer and public teacher-generated datasets is explicit;
neither loads a teacher's model weights.

## 1. Install and select storage

```sh
uv venv --python 3.12
uv pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install -e '.[data,tokenizer,tune,serve]'
source scripts/dataset-env.sh
```

Dataset downloads, caches and prepared arrays use
`/mnt/gigachad/microgpt/dataset`. `data` points at its `project-data` subdirectory.
The environment script refuses an unmounted NAS. See [datasets](datasets.md) for
pinned download/curation commands. Checkpoints stay in `runs`; retain the tokenizer
beside each checkpoint when copying it. Use fresh output paths when repeating steps.

## 2. Build a mixed corpus and fix the tokenizer

The following inputs already exist from bounded, provenance-recorded downloads:
code files, coding questions, real commit pairs, complete everyday conversations,
and multilingual instructions. They are small initial samples, not enough data
for a broadly capable model.

```sh
.venv/bin/python -m microgpt.corpus --input \
  data/codeparrot.jsonl data/magicoder-sft.jsonl data/commits.jsonl \
  data/conversation-complete.jsonl data/multilingual-complete.jsonl \
  --out data/my-corpus.jsonl
.venv/bin/python -m microgpt.prepare --input data/my-corpus.jsonl \
  --out data/my-pretrain --encoding o200k_base --limit 10000
```

`o200k_base` has 200,019 vocabulary slots; `o200k_harmony` has 201,088. They share
useful multilingual segmentation, but vocabulary alone teaches no language.
Ordinary text cannot inject special token IDs through their spellings. For a
smaller embedding, omit `--encoding` and use `--vocab-size 32768` to learn BPE from
the training partition. With bytes, omit both. Never switch tokenizers after
learning weights: their token IDs define the embedding/output rows.

Code and conversation are combined as text for next-token pretraining. Repository
and conversation group IDs survive that conversion. Both preparation stages use
the same stable 5% group split, so using those sources again in SFT does not move
a known held-out group into training. Exact duplicate removal is not semantic
benchmark decontamination; translated prompts without parent IDs can still leak.

## 3. Pretrain from random initialization

```sh
.venv/bin/python -m microgpt.train --config configs/student.toml \
  --data data/my-pretrain --out runs/my-pretrain \
  --sequence-length 2048 --attention flash --steps 10000
```

The first invocation creates a new GPT and trains every weight. Eight layers,
width 384, six heads, tied embeddings, RMSNorm, RoPE and SwiGLU keep the model
inspectable. The context ceiling is 32K; the first stage uses 2K windows to reduce
cost. Batch 1 × accumulation 4 × 2,048 = 8,192 token exposures per update.
10,000 steps means 81.92M exposures, not distinct tokens and not a promise of quality.

Watch validation loss, repetition, held-out code completion and conversational
samples. Expand the diverse corpus before merely repeating the small sample.
A 100-step pilot checks throughput/memory; it is not an adequately pretrained
assistant. Choose a larger budget based on measured progress and compute limits.

Resume an interrupted stage with identical config/data/window:

```sh
.venv/bin/python -m microgpt.train --config configs/student.toml \
  --data data/my-pretrain --out runs/my-pretrain --sequence-length 2048 \
  --attention flash --steps 10000 --resume runs/my-pretrain/last.pt
```

The total `--steps` also determines the cosine schedule. Extending a finished
100-step pilot to 10,000 is therefore not identical to an uninterrupted planned
10,000-step run. Checkpoints restore model, optimizer and random states.
Resume now rejects changes to training settings or the recorded attention backend.
Use `--stop-at-step 100` with the full planned schedule for a resumable initial
segment. Use `--init-from` for an intentional new data/optimizer stage, and
`--allow-attention-change` only when explicitly changing a resume backend or
opening a legacy checkpoint whose backend was not recorded.

## 4. Train assistant behavior and tool syntax

Real trajectories preserve call IDs, tool schemas, arguments and observations.
A constructed commit exercise provides a smaller lesson in reading numbered
output and editing exact unnumbered source. It is labeled synthetic and makes
no claim that repository tests were executed.

```sh
.venv/bin/python -m microgpt.tool_examples --input data/commits.jsonl \
  --out data/my-tool-exercises.jsonl
.venv/bin/python -m microgpt.trajectories --input \
  data/magicoder-sft.jsonl data/my-tool-exercises.jsonl \
  data/conversation-complete.jsonl data/multilingual-complete.jsonl \
  data/swesmith-complete.jsonl --tokenizer runs/my-pretrain \
  --out data/my-sft --max-length 8192
.venv/bin/python -m microgpt.finetune --checkpoint runs/my-pretrain/last.pt \
  --data data/my-sft --out runs/my-sft --steps 1000 --attention flash
```

`protocol.py` defines both inference prompts and training targets. Each assistant
turn gets its preceding conversation as context. Loss covers only the current
assistant reply/call and its EOS; system, user, tool observations and prior turns
are masked. Encoding segments separately avoids ambiguous BPE loss boundaries.
Whole overlength episodes are rejected, never cut in the middle of a solution.

The full-weight SFT trainer uses BF16 computation, FP32 weights/AdamW state,
checkpointed activations, gradient accumulation and clipping. Only supervised
positions materialize vocabulary logits, in checkpointed 128-position chunks.
Pretraining also chunks its vocabulary loss for vocabularies larger than 32K.
This limits the output-head memory that Flash Attention alone cannot reduce. It validates the checkpoint tokenizer,
context and dataset hashes. Source-balanced validation sampling is bounded; inspect
per-language and per-task behavior separately. Example sampling is uniform over
assistant turns, so long trajectories contribute more examples.

To resume SFT, replace `--checkpoint` with `--resume runs/my-sft/last.pt`, keeping
all settings/data identical. Unlike pretraining, SFT uses a constant learning rate.

## 5. Increase training windows deliberately

Continue our pretraining weights with a fresh optimizer for a new window stage:

```sh
.venv/bin/python -m microgpt.train --config configs/student.toml \
  --data data/my-pretrain --init-from runs/my-pretrain/last.pt \
  --out runs/my-long-pretrain --sequence-length 8192 --steps 1000 --attention flash
# A later stage can use --sequence-length 32768 if data and memory permit.
```

Then repeat SFT from that new checkpoint, retaining full longer trajectories as
appropriate. Window stages are experiments, not an automatic RoPE extrapolation
claim. No sliding-window attention, sparse attention or GQA is implemented; this
model uses ordinary multi-head full attention through PyTorch's fused Flash path.
A large context allocation must be accompanied by long-context training and tests
of retrieval, constraint retention, compaction and recovery.

## 6. Serve and measure through OpenCode

```sh
.venv/bin/python -m microgpt.serve --checkpoint runs/my-sft/last.pt --attention flash
# Separate terminal:
.venv/bin/python scripts/eval_opencode.py --out runs/my-repair-eval --label 'my-sft exact checkpoint'
.venv/bin/python scripts/eval_dialogue.py --out runs/my-dialogue.json --label 'my-sft exact checkpoint'
.venv/bin/python scripts/check_inference.py --checkpoint runs/my-sft/last.pt
```

See [OpenCode setup](opencode.md) for configuration and the grader container.
“OpenCode-compatible” means the endpoint accepts messages/tools and serializes
valid model-generated tool calls. “OpenCode-usable” additionally requires the model
to choose tools, make correct edits, use test feedback and finish reliably.
Passing a wire-contract test or reducing loss proves neither coding nor reasoning.

Before calling a checkpoint usable, require successful fresh read/edit/test tasks,
valid calls without repeated failures, basic conversation in target languages,
and recovery under a bounded action budget. The four included repairs are a small
development regression, not enough to certify those abilities. Preserve test
repositories outside both pretraining and SFT. Train/test splits within a downloaded
corpus alone do not protect an external benchmark from contamination.

## Current stopping point

The shipped local pilot is `runs/scratch-sft/last.pt`: random initialization,
1,000 pretraining updates and 501 full-weight SFT updates. It connects through
OpenCode but passed **0/4** repair tasks and made no tool calls. It is not yet an
OpenCode-usable coding assistant. This guide supplies the minimum implemented
engineering sequence, not a proven minimum training budget for that capability.

At 2K windows the student recipe consumes 8,192 tokens/update. Budget subsequent
pretraining experiments explicitly: 10,000 updates = 81.92M exposures; 100,000 =
819.2M. Expand distinct training data alongside those budgets. Repeating the
current 7.85M-token corpus 100 times is not equivalent to learning from 785M
unique tokens. Measure held-out completions and repetition before advancing to
assistant/tool stages. After SFT, require actual successful tool runs before
calling the result usable; otherwise return to the failing stage rather than
changing the endpoint or substituting another model.
