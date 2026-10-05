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

### Constrained-decoding integration gotcha

The initial grammar cache key sorted the full tool schema, including argument
properties. XGrammar's strict JSON mode follows schema property order. That forced
`newString` before `oldString`, while our training calls emit `oldString` first.
The 5,500-step tool probe consequently swapped the old/new text. Preserving schema
order allowed actual edits with the same frozen weights; behavioral correctness
still failed (0/2 cumulative tasks). A regression test checks the native edit order.
The artifact named `candidate-6k.pt` actually records step **5500**; the saved checkpoint
and evaluator health metadata are authoritative, not that filename.

JSON whitespace between fields is also bounded to one character in the current
grammar, preventing whitespace-only continuations from consuming the entire output
budget. Whitespace inside code strings is unaffected. The grammar compiler's own
unbounded cache is disabled; the wrapper retains at most eight schema compilations.


### Capacity accounting

The unchanged student has 90,969,600 parameters, of which 76,807,296 are the tied
200K-vocabulary embedding/output matrix and 14,162,304 are transformer/norm weights.
Thus its headline size overstates the transformer capacity compared with a model
using a smaller vocabulary. This experiment holds architecture and foundation fixed
to measure data/decoding changes. If broader programming remains weak after adequate
exposure, further pretraining or a larger transformer must be measured separately;
adding more repeated exercise templates is not evidence of generalization.


The second mixture's 9,500-step checkpoint scored **1/8** direct (equality counting).
A later snapshot froze at **14,000**, despite the last polled training metric being
13,500, and scored **0/8**. Always use the evaluator's captured server/checkpoint
metadata for the tested step. This fluctuation is a failed stability check; neither
checkpoint is promoted. The fixed tests remain unchanged.


### Final 24,000-step result and bounded continuation

The corrected mixture consumed **16,491,493 assistant-token exposures** in 1,638.8
training seconds. Final validation loss was 0.3153, but generated behavior failed:

| Development evaluation | Passed |
| --- | ---: |
| Direct programming requests | 0/8 |
| Mixed programming session, cumulative requirements | 0/8 |
| Current task independently correct in that session | 0/8 |
| Broader programming repairs | 0/4 |
| Previously trained arithmetic session | 16/16 |

This preserves the narrow exercise skill without fixing general programming or the
reported task-switch failure. These are development probes, not a contamination-free
benchmark. Grammar constraints improve structural validity; they cannot establish
that a function implements the request. The existing service was not replaced.

The frozen checkpoint is `runs/programming-v2/checkpoint-24000.pt`, SHA256
`b164a2a5ade8550c1536a9b3b939245f6d7ed586fc8e3cbe2263c3bb1928b6c9`.
Full evaluations and provenance are copied into `docs/programming-evidence.json`.
The 31 unit tests pass; model behavior is assessed separately above.

The corpus contains 96,768,432 assistant target tokens, so 16.5M sampled exposures
are limited coverage. Weighted sampling also repeats some sources disproportionately;
this is not a count of distinct tokens learned. More training may help, but is not
a demonstrated solution to the capacity or foundation limitations.

`scripts/continue_programming.py` resumes the same random-origin checkpoint,
optimizer and RNG, retaining all ten source weights. It evaluates at steps 48K,
80K, 120K and 160K, within a four-hour wall-clock budget checked before training
stages. An evaluation already underway may finish beyond that budget. It stops
after three checkpoints without a better direct-code score, or once all gates pass.
Direct probes always run; any direct success enables the mixed-session probe;
8/8 direct additionally enables broader repairs and a 32-round arithmetic session.
All four suites must pass to mark a candidate successful. Even that only establishes
the tested scope, not reliable responses to arbitrary prompts. No automatic promotion
occurs.

The continuation was launched with:

```sh
.venv/bin/python scripts/continue_programming.py \
  --out runs/programming-continuation --hours 4
```

It successfully resumed beyond step 26,000. Read current status from
`runs/programming-continuation/status.json`, training output from its `training.log`,
and evaluations from its `eval-STEP` directories. The status records the controller
PID; SIGTERM requests a graceful stop. Use a fresh output directory for a subsequent
invocation and milestones above the saved step. This runner continues while the
interactive coding session is idle; its status file, not this launch-time note,
is authoritative about whether it is still running.


### Continuation outcome and recovery curriculum (October 5)

The continuation finished all scheduled milestones, rather than reaching the goal:

| Step | Direct programming | Mixed agent session |
| --- | ---: | ---: |
| 48,000 | 1/8 | 1/8 |
| 80,000 | 4/8 | 0/8 |
| 120,000 | 2/8 | 0/8 |
| 160,000 | 2/8 | 0/8 |

The last checkpoint had 110,623,221 assistant-token exposures. In the 80K trace,
the model read the right file, applied a function, and called check, but produced a
literal interpolation placeholder and failed to repair it. Later turns inherited
the failure. The 160K session also encountered repeated tool cycles and eventually
exhausted its context allowance. Protocol compatibility is insufficient for task
completion. Longer training at the same mixture did not reliably improve behavior.

The next controlled change is `scripts/prepare_programming_recovery.py`: 11,967
simulated repair and task-switch episodes built from the 3,989 previously checked
OpenCoder references. Repair examples replace a function body with an explicit
NotImplementedError and teach reading, checking, restoring the reference, checking,
and stopping. Task switches span up to six files. Explicit signatures are extracted
with AST parsing, never execution. Every constituent of a combined episode belongs
to the same original hashed train/validation partition. This prevents this generator
from moving reference answers across partitions, but is not a semantic deduplication
claim about the public corpus.

Limitations: only original references were executed against upstream tests. Failure
observations and multi-file success observations are simulated. NotImplementedError
repair does not cover arbitrary semantic errors. These examples do not establish
long-session competence, autonomous memory management, or robust recovery. The
unchanged real-tool development evaluations remain the behavioral gate; their prompts
and expected answers are not imported into this curriculum.

Run the next stage with `bash scripts/train_programming_recovery.sh` after generating
`data/programming-recovery-v1.jsonl`. It prepares `data/programming-sft-v3` on the NAS,
starts from our random-origin 80K checkpoint with a **fresh optimizer**, lowers the
learning rate from 1e-4 to 3e-5, and uses 20% recovery plus 10% task-switch sampling.
Other sources supply coding replay and 1% conversation. This changes both curriculum
and optimizer settings; results cannot isolate either change individually. After
1,000 initial updates the bounded runner evaluates at 5K, 10K, 20K and 40K new-stage
steps. Stage steps reset; they are not total lifetime updates. Launch output lives in
`runs/programming-recovery-launch.log`; subsequent controller status lives in
`runs/programming-recovery-continuation/status.json`. No checkpoint is automatically
promoted to the user's server.
