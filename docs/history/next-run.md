> Current assessment and next steps: [October 5 revised programming plan](revised-programming-plan.md). Earlier run recommendations below are historical.

# Correcting the first full run

**Completed follow-up (2026-10-04):** a single focused SFT stage directly from our
scratch foundation produced the selected 2250-step export. It passed four 16-round
sessions and one 32-round session. Follow the [minimal agent runbook](../guides/agentic-minimum.md)
for the final recipe and limitations. The entries below preserve earlier experiments;
they are not prerequisites and are not currently running.

Baseline commit: `4d98c3a`. All four training stages finished, but the final model
passed 0/4 basic repairs, failed the first two horizon rounds, retrieved 0/9 context
needles, and answered 2/12 reasoning problems with thinking (1/12 directly).
Completing a curriculum is not a deployment gate.

The OpenCode event records show invented paths, rejected read calls, and repeated
text. The instruction checkpoint already invents paths on a direct read request;
the final long-horizon checkpoint degenerates into repeated path components.
A three-checkpoint diagnostic is saved in `runs/diagnosis/`. The same prompts and
cached/full-prefix logit comparisons make this reproducible. BF16 cached logits
differ slightly from full-prefix logits, so this does not establish exact numerical
equality or explain the behavioral failures by itself.

The next experiment starts from our own instruction checkpoint, preserving its
random-initialization ancestry. It trains copying the requested path, reading the
file, applying an exact edit, checking it, and retrying a transient checker error.
The 8,000 original episodes mix concise tool schemas with installed OpenCode
schemas, relative/absolute paths, simple arithmetic repairs, and direct/thinking
answers. These are simulated traces. Fixed original programs and exact edits are
checked independently by unit tests. The four basic evaluation tasks are excluded.

25% of samples replay coding, conversation, multilingual, reasoning and earlier
practice sources. This is an experiment against forgetting, not a guarantee.
`grounded-sft-v1` has 55,026 train / 2,967 validation episodes with disjoint groups.
The first trial is bounded to 2,000 updates or 30 minutes. It does not rent hardware.

```bash
.venv/bin/python scripts/capture_opencode_tools.py --out data/opencode-tools-v1.json
.venv/bin/python -m microgpt.grounding --tools data/opencode-tools-v1.json --out data/grounding-v1.jsonl
.venv/bin/python -m microgpt.trajectories --input data/grounding-v1.jsonl \
  data/reasoning-trace.jsonl data/smoltalk-large.jsonl data/multilingual-large.jsonl \
  data/magicoder-large.jsonl data/practice-v2.jsonl data/practice-handoff.jsonl \
  data/reasoning-tools.jsonl --out data/grounded-sft-v1 --tokenizer data/large-v1 \
  --max-length 32768 --limit 100000 --pack-episodes
.venv/bin/python -m microgpt.finetune --data data/grounded-sft-v1 --out runs/grounding-v1 \
  --checkpoint runs/coding-v1/instruction/last.pt --steps 2000 --max-seconds 1800 \
  --max-length 4096 --balanced-sources --sources grounding-read grounding-edit \
  --short-replay 0.25 --accumulation 4 --eval-every 250 --eval-examples 32 --lr 0.00005
```

Fresh output paths are required. All data stays under the existing authorized NAS
symlink. Training now reports cumulative supervised tokens and preserves `best.pt`
by held-out loss as well as `last.pt`. A lower loss is not a capability gate: compare
actual generated actions, repair tests, reasoning and context scores before choosing
which checkpoint advances to longer histories.

`eval_opencode.py --suite grounding` adds eight fresh arithmetic repair instances
from the trained template family. Passing them establishes execution on that narrow
family, not general coding or unseen-template reasoning. The original `--suite basic`
remains separate. Both use real OpenCode and isolated Docker graders; summaries now
include tool arguments and shortened error messages. Never loosen file permissions
to hide hallucinated paths.

The first measured comparison is **0/8 → 8/8** on the grounding suite, using the
same seeds and real OpenCode for both checkpoints. Results are in
`runs/diagnosis/instruction-grounding-v2/results.json` and
`runs/grounding-v1-eval/results.json`. The earlier unversioned diagnostic mistakenly
used a 1,024-output-token server against a 4,096-token client; it is an invalid
infrastructure run and is excluded. `candidate.pt` freezes the selected checkpoint
so subsequent updates cannot change evaluation provenance.

The follow-up `horizon_data` curriculum supplies 1,600 sessions split at a handoff
into 3,200 records, with 2–32 requested functions per session, cumulative source
checks and transient checker failures. Its seed range begins at 500000; evaluation
uses 849723. This is still the same procedural arithmetic domain. The second
experiment uses `horizon-sft-v1`, balances source/length bins up to 32K, replays 35%
of shorter examples, and starts from the successful grounding candidate. It is
bounded to 1,500 updates or 30 minutes. It must retain the grounding score and pass
actual multi-round execution before being called a successful long-horizon model.

Malformed model tool output now returns non-retryable HTTP 422 with error type
`invalid_model_output`. Previously a 502 made OpenCode repeatedly retry deterministic
invalid output. Calls remain undispatched; this does not improve model correctness.

The first horizon candidate retained 7/8 arithmetic repairs but failed the first
two cumulative rounds. It repeatedly read the same file. Installed OpenCode's
`steps` setting did not act as a hard action ceiling: one round made 104 reads.
The API now rejects three consecutive identical action/observation pairs with
`agent_loop_error` (HTTP 422), before further inference. Different observations,
different actions, and new user turns reset this narrow guard. It does not establish
long-horizon skill; it bounds an observed failure mode.

Counterfactual replay of the captured synthetic evaluation request showed that
removing OpenCode's read footer changed another read into an attempted edit, though
that edit was malformed. Renaming the function or using a familiar comment alone
did not resolve the repeated read. Training now includes actual read footer format,
absolute display paths, more comment patterns, and recovery from a redundant read.
A further experiment uses 10,000 identifier/word types extracted from the existing
Magicoder training source, addressing copying beyond a handful of fixed names.
No downloaded code executes during construction. The actual evaluation task
solutions are not incorporated into training.

The vocabulary-diverse recipe is:

```sh
.venv/bin/python -m microgpt.horizon_data --tools data/opencode-tools-v1.json \
  --out data/horizon-v3.jsonl --vocabulary-data data/magicoder-large.jsonl --count 4000
.venv/bin/python -m microgpt.trajectories --input data/horizon-v3.jsonl \
  data/grounding-v1.jsonl data/reasoning-trace.jsonl data/smoltalk-large.jsonl \
  data/multilingual-large.jsonl data/magicoder-large.jsonl --out data/horizon-sft-v3 \
  --tokenizer data/large-v1 --max-length 32768 --limit 100000 --pack-episodes
```

This data contains 55,053 train / 2,945 validation episodes. The current experiment
`runs/horizon-v4` continues our `runs/horizon-v3/best.pt`, with 2,500 updates maximum,
learning rate 5e-5, accumulation 2, a 30-minute ceiling, balanced source/length bins,
and 20% general replay. Main sources are `horizon-session`, `horizon-handoff`,
`grounding-read`, `grounding-edit`, and `reasoning-trace`. Earlier failed candidates
are retained for comparison; the 8/8 short-task candidate is not overwritten.

A further data audit found two confounds: original grounding examples used only
none/low reasoning effort, while horizon tasks used all four modes; original
program-trace effort labels were tied to the three exercise families. Neither
correlation reflects the requested behavior. Grounding v2 and trace v2 cover all
modes within each task family; regression tests check that coverage.

`recovery-sft-v1` combines 16,000 first-task episodes with those corrected examples,
full vocabulary-diverse sessions, and coding/conversation replay. It has 69,502
training and 3,696 validation episodes. Shared groups keep first-task/full-session
variants in the same split. Exact duplicate episodes are removed. First-task
supervision gives starting decisions weight independent of the number of easy
later turns in a long session.

The bounded follow-up is running with:

```sh
.venv/bin/python scripts/train_stages.py --curriculum recovery \
  --out runs/recovery-v1 --hours 1 --sft-data data/recovery-sft-v1 \
  --checkpoint runs/horizon-v4/best.pt
```

It stops at 16,000 updates or one hour of training, preserves the best-loss snapshot,
and automatically evaluates basic repairs, arithmetic grounding, 16-round recovery/
handoff, reasoning and long-context retrieval. Results and live state are written
under `runs/recovery-v1`. Evaluation failure is not treated as success or deployment.
The intermediate horizon-v4 candidate scored 2/8 grounding and failed the first
two horizon rounds; it is not the checkpoint served for use.

The working narrow candidate is served locally at `http://127.0.0.1:8000/v1`, using
`runs/grounding-v1/candidate.pt`; its process record is
`runs/grounding-v1-service.json`. The recovery runner uses port 8011 for isolated
final evaluation and stops that evaluation server afterward. No global OpenCode
configuration or cloud machine was changed.
