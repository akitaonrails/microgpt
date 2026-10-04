# Programming improvement experiment — 2026-10-04

The target is varied programming requests, not replaying or repairing one user's
OpenCode session. The earlier 32-round arithmetic score did not establish general
simple-task competence. The original user session and the served checkpoint stay
unchanged while candidates train and undergo comparison.

## Evidence and acceptance

`scripts/eval_programming.py` defines eight independent programming tasks: greeting
output, bounds, initials, filtered squares, joining, lookup defaults, normalization,
and equality counting. Its direct mode executes generated code only in a restricted
Docker container. Agent mode sends the changing requests to an actual OpenCode
session and cumulatively checks prior functions. Passing requires correct code and
a clean finish, plus a successful observed checker call in agent mode. These are
small development gates, not proof of correctness for arbitrary prompts. The user's
exact free-form hello-world request is not a training example.

The existing 2250-step arithmetic checkpoint scored **0/8 direct**. Enabling grammar
constraints alone also scored **0/8**: preventing illegal tool JSON did not teach
programming or instruction following. Generated responses and grades are in
`runs/programming-baseline-direct` and `runs/programming-grammar-direct`.

The implementing agent owns three distinct checks: CPU tests for data/sampling and
schema masking; Docker behavioral grades for programming; real OpenCode gates for
mixed tasks and the prior long-horizon exercise. A candidate must improve behavioral
results before replacing the running service. Loss alone is not a deployment gate.

## Programming data

- [OpenCodeInstruct](https://huggingface.co/datasets/nvidia/OpenCodeInstruct), NVIDIA,
  CC-BY-4.0, revision `8f3ba5bafe4d6e8db46082cf7ae6741bc370604d`.
  Downloaded 50 shards. Scanned five million rows; 1,641,305 had upstream test score
  1.0. A seeded reservoir retains 100,000. Those scores are upstream annotations,
  not independently verified correctness; supplied tests can miss errors.
- [OpenCoder educational instructions](https://huggingface.co/datasets/OpenCoder-LLM/opc-sft-stage2),
  MIT, revision `7d28f40d579edd7c24402d17d0c7639f991e6f8d`: 118,278 records.
  A bounded short-reference subset is also checked in a networkless, read-only,
  resource-limited Docker container before creating simulated read/edit/check traces.
- [Code-Feedback](https://huggingface.co/datasets/m-a-p/Code-Feedback), Apache-2.0,
  revision `f411b16a97c910ac9acf8b0d0948e340aa77cc34`: 66,383 multi-turn programming
  conversations. These teach follow-up requests in addition to isolated answers.
- Existing Magicoder coding instructions, authored mixed-task tool demonstrations,
  and a small replay of the prior horizon curriculum. Only 500 public conversation
  records are retained for basic interaction; their sampling weight is 1%.

Raw files, curated records and token arrays reside under the authorized
`/mnt/gigachad/microgpt/dataset` path. Manifests pin revisions and SHA256 hashes.
Imported code is never executed on the host. All weights still descend from our
randomly initialized model; using public teacher-written text imports supervision,
not a pretrained model.

Stack-Edu was researched but not selected for this immediate SFT correction: its
public table lists repository/blob metadata, and resolving source contents is a
separate pretraining-data task. Larger raw-code volume alone does not teach the
assistant/tool interaction that failed here.

## Sampling and decoding

`finetune --source-weight SOURCE:WEIGHT` makes source probabilities explicit.
With length balancing, each source's weight is divided across its retained length
bins, so long-context coverage cannot silently increase that source's total share.
Weights refer to sampled examples, not tokens. Public coding answers supply most
updates; repeated arithmetic exercises supply only a small replay fraction.

The optional `grammar` extra uses pinned XGrammar 0.2.8. `serve --constrain-tools`
applies token masks to tool-call JSON using the offered names and argument schemas.
Ordinary text remains free-form; JSON validation and OpenCode permissions still
apply afterward. A valid call can still name the wrong file or contain wrong code.
Output-length limits can still interrupt a call, in which case it is not dispatched.
The implementation follows the [XGrammar structural-tag API](https://xgrammar.mlc.ai/docs/latest/api/python/grammar.html)
and [raw-token vocabulary interface](https://xgrammar.mlc.ai/docs/latest/api/python/tokenizer_info.html).

## Reproduce this run

```sh
uv pip install -e '.[data,tokenizer,tune,serve,grammar]'
bash scripts/prepare_programming_data.sh
bash scripts/train_programming.sh runs/coding-v1/foundation/last.pt runs/programming-v2 24000
# Optional comparison: same mixture, different random-origin parent.
bash scripts/train_programming.sh runs/agent-direct-v1/candidate-mid.pt runs/programming-continued-v2 24000
```

The prepared mixture contains **282,520 training / 14,883 validation episodes**, with
**96,768,432 assistant target tokens** in the training split. This is corpus size,
not the number of tokens a 6,000-step run has consumed. Record actual exposures from
`metrics.jsonl`. The tokenization manifest records source hashes, exclusions and
source counts. The earlier interrupted `general-sft-v1` preparation is unused and
has no completed manifest; no training used that conversation-heavy draft.

The reference-code sandbox retained **3,989/4,000** examples. Passing their source
tests is weaker than independently proving their specifications. Related direct and
tool variants use the same instruction-derived group ID, keeping those exact variants
on the same side of the split. Near-duplicate public tasks can remain; the small
behavioral gates are development tests, not a contamination-free benchmark.

The parser now reads each JSON object before recognizing its closing tool tag.
A code string containing literal protocol tags no longer cuts the object short.
Length-truncated responses never dispatch calls, including an earlier complete call
followed by a partial one. API tests now fail on broken project imports rather than
silently treating them as an absent optional dependency.

After training, serve a separate candidate port and grade both modes:

```sh
.venv/bin/python -m microgpt.serve --checkpoint runs/programming-v2/last.pt \
  --device cuda --attention flash --constrain-tools --port 8017
.venv/bin/python scripts/eval_programming.py --url http://127.0.0.1:8017/v1 \
  --mode direct --out runs/programming-direct-eval
.venv/bin/python scripts/eval_programming.py --url http://127.0.0.1:8017/v1 \
  --mode agent --out runs/programming-agent-eval
```

Keep the failed examples and report per-task results. Training completion, schema
validity and low loss must not be described as arbitrary programming competence.


### Prompt/interface correction

The 1,000-step and intermediate first-mixture snapshots still scored 0/8 direct.
One diagnostic produced a plausible greeting under a short system prompt, but used
the wrong function name. With the longer agent prompt it often emitted tool markup
without tools. This motivates a controlled second mixture: public direct answers
also receive the same general system prompt (or a short coding prompt, or neither),
and educational requests explicitly name the function signature parsed from their
reference code. No source code executes during signature extraction. Source group
IDs stay based on the original instruction to keep related variants together.

The final preparation script writes `programming-sft-v2`; first-mixture artifacts
remain separately under `programming-sft-v1`. A fresh foundation-to-SFT run allows
24,000 updates before evaluation rather than importing any pretrained weights.


Validation reporting now includes losses by source. The fixed 40-example validation
slice is a small diagnostic, not a full estimate over the 14K held-out conversations;
long easy traces can dominate its token-weighted aggregate. Final model selection
uses generated-code and real-tool behavior instead.

The first full 6,000-step mixture consumed 4,105,548 assistant-token exposures and
still scored **0/8** on the independent direct-code probes. Some functions had the
right requested name but wrong behavior (for example, returning matching items
instead of their count); other answers repeated text. This result is retained as a
failed experiment, not a deployment candidate. The longer second run uses the
corrected prompt/interface mixture and the same random-origin foundation.
