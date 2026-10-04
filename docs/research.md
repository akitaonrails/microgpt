# Research notes — 2026-10-03

The design target is a student who can read Python, train something today, and
understand the difference between a working toy and a capable coding model.
Recommendations below are engineering judgments, not reproduced benchmark results.

## Projects to learn from

| Project | What it teaches | Implication for this lab |
| --- | --- | --- |
| [microgpt](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) | Scalar autograd, a small decoder, Adam, training and sampling in dependency-free Python | Preserve the complete visible pipeline, split by concept |
| [nanoGPT](https://github.com/karpathy/nanoGPT) | Compact PyTorch training and checkpointing, including GPT-2 experiments | Use simple modules and explicit configuration |
| [nanochat](https://github.com/karpathy/nanochat) | Tokenization through pretraining, post-training, evaluation and serving | A useful next destination after this smaller lab |
| [LLMs from Scratch](https://github.com/rasbt/LLMs-from-scratch) | Incremental explanations and implementations | Provide a reading order and small exercises |
| [SmolLM](https://github.com/huggingface/smollm) | Small-model recipes and data curation | Treat the data mixture as a first-class design choice |

The linked microgpt is GPT-2-inspired, not an exact GPT-2 replica: its comments
explicitly identify RMSNorm, no biases, and ReLU as differences. It uses learned
position embeddings. This lab uses RoPE and SwiGLU instead, plus a tied output
embedding. These are reasonable modern components; they do not by themselves
produce reasoning. External model weights and this lab's checkpoints are not
interchangeable.

## What the local Genesys assets establish

Read locally: `../llm-coding-benchmark/lua-forensics/README.md`, `FINDINGS.md`, and
the referenced probe inventory. The notes report o200k-like token counts and
behavioral probes. They explicitly leave weight provenance and scratch training
versus distillation unresolved. Some narrative conclusions are stronger than the
underlying black-box evidence supports: API token accounting need not expose the
internal tokenizer, and a tokenizer can be adapted without retraining every weight
from scratch; see [tokenizer transfer research](https://arxiv.org/abs/2405.07883).
Treat these observations as clues, not architecture specifications.

[LUA's public research page](https://lua.vision/en/research/) describes staged
training and confidence monitoring, while saying fuller materials require a
confidentiality agreement. That is inspiration for studying curriculum and
abstention, not a publicly reproducible replacement architecture or a validated
recipe for this lab. No Genesys weights or training corpus were used here.

The useful lesson is reproducibility: retain tokenizer files, dataset identities,
training configuration, loss curves, and held-out results. A public tokenizer is
neither trained model weights nor proof of a model's ancestry. This project makes
its random initialization and training path directly inspectable.

## Public datasets, with their actual roles

| Dataset | Best role | Access and practical limitation |
| --- | --- | --- |
| [CodeFeedback-Filtered-Instruction](https://huggingface.co/datasets/m-a-p/CodeFeedback-Filtered-Instruction) | Small code question/answer experiments | Apache-2.0 on the card; downloadable text; answers can contain errors. Included bounded downloader pins a revision |
| [The Stack v2](https://huggingface.co/datasets/bigcode/the-stack-v2) | Broad source-code pretraining | Gated access/terms; original file licenses and provenance matter; content retrieval adds work |
| [StarCoderData](https://huggingface.co/datasets/bigcode/starcoderdata) | An older large code-training corpus | Gated; inspect access conditions and underlying code licenses before building a mix |
| [SmolLM Python-Edu](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) | Educationally scored Python pretraining | The download contains IDs, paths and scores, not source text; the card describes separate Software Heritage S3 retrieval |
| [SmolLM Cosmopedia-v2 / FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) | Explanatory prose alongside code | Very large; stream bounded samples; corpus card lists ODC-BY and refers to source terms |
| [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) | Easier small-model language experiment | Synthetic stories, not programming; dataset card lists CDLA-Sharing-1.0 |
| [Open R1 Mixture-of-Thoughts, code](https://huggingface.co/datasets/open-r1/Mixture-of-Thoughts) | Supervised reasoning/distillation after a capable base | Roughly 83k code traces; long Python/C++ solutions; inspect source-specific terms rather than assuming a blanket license |
| [Open R1 Codeforces](https://huggingface.co/datasets/open-r1/codeforces) | Verifiable coding tasks and test cases | Card lists CC-BY-4.0; reserve problems by identity/date for evaluation; running submitted code needs isolation |

CodeFeedback is selected for an accessible pipeline exercise, not as a sufficient
pretraining corpus. This implementation learns the whole formatted question and
answer. It does not implement assistant-only SFT. For serious pretraining, use a
broader mixture of licensed source code and explanatory text; for instruction
training, preserve prompt/answer boundaries and mask prompt loss.

“Publicly available” does not mean every source has identical reuse conditions.
Keep dataset cards and source revisions with each experiment. For code corpora,
retain repository/license provenance and avoid mixing benchmark solutions into
training. Exact-text deduplication alone will not remove paraphrases or copied
repositories. Splitting by repository/problem is preferable to random rows.

## Reasoning and calibration are different

[DeepSeek-R1](https://arxiv.org/abs/2501.12948) describes reasoning post-training,
including reinforcement learning and distillation into smaller models. Public
reasoning traces are usable supervision, not a downloadable setting that makes
an untrained model reason. Long outputs and `<think>` markers alone do not measure
correctness. This lab's short contexts are particularly unsuitable for copying
whole competitive-programming reasoning traces.

[Temperature scaling](https://arxiv.org/abs/1706.04599) calibrates predicted
confidence on held-out labels. It does not add knowledge or change argmax decisions
for fixed logits. Sampling temperature is a generation control; tuning it is not
the same experiment as fitting a confidence calibrator. Quantization calibration
is a third meaning, concerned with approximation error, not reasoning training.

See [the next-stage plan](reasoning.md) for an affordable coding/reasoning route.
