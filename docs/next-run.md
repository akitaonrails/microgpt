# Correcting the first full run

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
