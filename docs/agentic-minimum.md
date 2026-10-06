# The minimum agentic model: experiment notebook

**Current experiment:** broader programming remains weak (2/100 development variants).
The [controlled capacity pilot](capacity-pilot.md) follows diagnostic comparisons and tests a fresh
239M student against our original scratch foundation. The served 91M artifact below
remains a narrow arithmetic lesson.


The target is a small, scratch-trained model that can keep a real OpenCode session
going on simple tasks. This is not a claim of broad software-engineering skill.
We retain the 90,969,600-parameter model and o200k tokenizer. Its tied vocabulary
matrix accounts for 76,807,296 parameters, leaving 14,162,304 for the transformer
and normalization. The parameter headline alone overstates its reasoning capacity.

## Validated result

The final teaching artifact is `runs/agent-ready-v1/model.pt`: 363,898,165 bytes,
with the tokenizer and deployment manifest beside it. It contains our weights and
serving settings, with optimizer/RNG state removed. Tensor equality was checked
against the full training checkpoint. No pretrained model weights were imported.

The successful route is our scratch foundation followed by **one SFT stage of
2,250 updates**, not the chain of experimental checkpoints below. That stage used
13,063 training / 737 validation episodes and consumed 4,725,998 assistant-token
exposures. Its recorded wall time was 430.4 seconds on the RTX 5090, with concurrent
evaluation activity; this is an observed run time, not an isolated speed benchmark.

| Check on the frozen 2,250-step model | Result |
|---|---|
| Cumulative OpenCode runs, seeds 8675309, 42, 271828, 314159 | 16/16 each |
| Longer run, seed 20261004 | 32/32; 130 real tool calls |
| Short arithmetic repairs | 8/8 |
| Original broader repairs | 0/4 |
| Program tracing, 256-token budget | 0/12 direct; 0/12 thinking |
| Exact retrieval at 8K/16K/32K | 0/9 |
| Interactive OpenCode, unquoted user input | Two cumulative tasks passed |
| Flash Attention and cold/warm cache checks | Passed through the 32K allocation limit |

Every cumulative run includes a transient checker failure, a fresh-session handoff,
and independent tests of all earlier requirements. The largest observed prompt in
the 32-round run was **27,528 tokens**. The state file is maintained by the harness;
these results do not establish model-written compaction summaries or reliable
retrieval from arbitrary long text. Conversation probes also remain poor.

This is a working **narrow agent lesson**, not a general coding assistant or a
deep-reasoning model. “Minimum” describes the working implementation and training
sequence found here, not a proven lower bound on parameters or pretraining tokens.
Hashes, seeds and measurements are recorded in [the evidence file](agentic-evidence.json).

## Use the model here

The workspace service is running on port 8000. Its process record is
`runs/agent-ready-v1/service.json`. To start it again after stopping it:

```sh
.venv/bin/python -m microgpt.serve --checkpoint runs/agent-ready-v1/model.pt \
  --device cuda --attention flash --port 8000 --max-output 4096
```

Open the prepared interactive lesson, already advanced past the two tested tasks:

```sh
.venv/bin/python scripts/agent_lab.py run --out runs/opencode-lab
```

Keep that OpenCode session open. After it completes a task, run `agent_lab.py next
--out runs/opencode-lab` in another terminal and paste the printed next prompt.
`next` independently checks all current requirements before advancing. For a fresh
lesson, use `agent_lab.py init --out runs/my-lab`, then `run --out runs/my-lab`.
See [the OpenCode guide](opencode.md) for the restricted tools and state-file boundary.

## Reproduce the short successful route

Use the CUDA environment in [the scratch guide](finetuning.md), OpenCode 1.18.34
and the pinned Docker image used by the evaluator. Prepare just the required source
datasets and original tool exercises:

```sh
bash scripts/prepare_agent_data.sh
```

This reuses complete artifacts and keeps dataset writes under the authorized NAS
root. The foundation mixture is CodeParrot, OpenCoder, Cosmopedia, SmolTalk and
multilingual SmolTalk2. Magicoder supplies identifier vocabulary for the original
exercises. The SFT corpus contains 5,254,829 assistant target tokens; its 1,879
SmolTalk training conversations are replay data, not evidence of conversational skill.

The measured model uses our existing `runs/coding-v1/foundation/last.pt`, which
has random-initialization ancestry and includes an earlier small pretraining pilot.
For a new foundation starting directly from random weights, the command is:

```sh
.venv/bin/python -m microgpt.train --config configs/coding-foundation.toml \
  --data data/large-v1 --out runs/my-foundation --device cuda \
  --sequence-length 2048 --attention flash
```

That cold run is not a bit-for-bit reproduction of the pilot-initialized foundation;
rerun the behavioral gates. The prepared corpus has 947,532,764 training tokens,
and the foundation configuration requests about one billion token exposures.
No claim is made that this is the smallest sufficient pretraining budget.

For the measured SFT recipe, select a fresh output directory:

```sh
.venv/bin/python -m microgpt.finetune --data data/agent-related-v1 \
  --out runs/my-agent-sft --checkpoint runs/coding-v1/foundation/last.pt \
  --steps 2250 --accumulation 2 --lr 0.0001 --eval-every 250 --eval-examples 16 \
  --device cuda --attention flash --max-length 32768 --balanced-sources \
  --balanced-lengths --short-replay 0.2 \
  --sources horizon-session horizon-handoff grounding-read grounding-edit
.venv/bin/python scripts/export_agent.py --checkpoint runs/my-agent-sft/last.pt \
  --out runs/my-agent --canonical-tool-ids
.venv/bin/python -m microgpt.serve --checkpoint runs/my-agent/model.pt \
  --device cuda --attention flash --port 8010 --max-output 4096
```

Replace the foundation path when testing a new cold run. SFT uses a constant learning
rate, so stopping at update 2,250 reproduces the selected training point's schedule.
Export preserves weights and records internal tool-ID normalization as a serving
setting. The measured training corpus used numbered synthetic IDs; future
experiments can also pass `--canonical-tool-ids` during trajectory preparation.
That optional corpus change needs its own validation.

In another terminal, evaluate the model actually loaded on port 8010:

```sh
.venv/bin/python scripts/eval_horizon.py --url http://127.0.0.1:8010/v1 \
  --out runs/my-horizon --label 'my-agent export' --effort none --rounds 32 --seed 20261004
.venv/bin/python scripts/eval_opencode.py --url http://127.0.0.1:8010/v1 \
  --out runs/my-repairs --label 'same export' --effort none --suite grounding
```

Repeat with fresh seeds and keep failed results. The general four-task repair suite,
reasoning and context tests remain separate. Keep the full SFT checkpoint for
training continuation; the smaller exported model is for inference only. The same
model code and export can be used on an H100 via the [hardware runbook](hardware.md);
no rented machine was used for these results.

## Acceptance criteria

A successful demonstration must use the actual model and real OpenCode, with no
scripted model replies or injected solutions. For the narrow arithmetic-function
domain, require:

- At least 16 evolving requirements with cumulative independent Python tests.
- Correct real read/edit/check calls, not merely well-formed JSON.
- A deliberately failed checker call followed by successful recovery.
- A fresh session halfway through, recovering requirements from a state file.
- Earlier functions preserved through the final round.
- A clean end to each assistant turn, without an API error or output-limit loop.
- Repetition of the evaluation with fresh task seeds.
- A separate short-task regression suite retained after long-session training.

The state file in this evaluation is maintained by the harness. It is not evidence
that the model can write a faithful compaction summary itself. Graders run outside
the editable workspace in networkless containers. Training uses original simulated
tools, and the executable reference checks cover only our own bounded templates.
No downloaded or model-generated code executes directly on the host.

## Which component does what?

```mermaid
flowchart LR
  D[Code and conversation data] --> P[Pretrain randomly initialized GPT]
  P --> S[Assistant and tool-trajectory training]
  T[Original tool demonstrations] --> S
  S --> W[Our checkpoint]
  W --> A[Local API and KV cache]
  A <--> O[OpenCode]
  O <--> F[Editable source and persistent task state]
  O <--> C[Isolated correctness checker]
```

| Component | Minimum responsibility |
|---|---|
| Model | Choose an action, copy its arguments, produce the requested simple code, respond to observations |
| Training data | Show complete action/result histories in the actual interface format, with held-out task groups |
| Server | Preserve role/tool IDs, validate calls, enforce context/output limits, separate thinking from actions |
| Inference | Correct causal attention and KV caching; Flash Attention limits memory costs |
| OpenCode | Execute tools, provide file contents, manage sessions and context, display responses |
| Persistent task state | Recover requirements after history is replaced or a session restarts |
| Independent evaluator | Execute tests, detect lost requirements, record failures without trusting the model's claims |

Tool execution is outside the model. The server never executes generated Python or
shell commands. A context allocation limit is not evidence of effective context
use. Thinking text is not evidence of correct reasoning. Neither requirement is
hidden inside the label “agentic-ready.”

## Gotchas observed so far

### Teacher-forced loss does not establish agent success

Several checkpoints achieved very low supervised loss while failing actual tasks.
Packed long episodes contain many predictable tokens. A mistake in one identifier,
closing tag, or initial action can invalidate the whole rollout. We keep both loss
metrics and independent generated-action evaluations, and preserve a known working
checkpoint while experimenting.

### Check the reasoning mode before declaring all skills forgotten

The recovery checkpoint scored 0/8 arithmetic repairs using the default medium
reasoning mode. A raw wire capture showed an unclosed `<think>` followed by a
malformed tool delimiter. The parser correctly dispatched no action but previously
returned an empty-looking reply. With reasoning effort explicitly set to `none`,
the same weights scored 8/8 and completed the first cumulative-session task.

That is a protocol/mode failure, not evidence that the weights lost every repair
skill. Unclosed thinking that ends at EOS now reports a non-retryable error;
truncation at the output budget remains distinguishable. Direct action is the
baseline for the minimal agent. Optional thinking is a separate experiment.

### The real harness is part of the input distribution

Our first schema capture retained read/edit/check only. Real OpenCode also offered
write under its edit permission. Tool ordering, environment metadata, quoted CLI
messages, end-of-file annotations, and the checker's `PASS\nPASS\n` output differed
from simulated examples. These small textual differences mattered to this model.
The capture tool now retains all offered schemas; the wire-aligned curriculum
includes these observed formats with varying task values and workspace paths.

A counterfactual removal of the read footer changed a repeated read into an edit,
but the edit was still malformed. This isolates one contributor without claiming
that the footer was the only problem.

### Avoid accidental correlations in synthetic data

One early generator associated reasoning modes with exercise families. Another
used only none/low effort for short repairs while long tasks used all four modes.
Those shortcuts were not intended behavior. Tests now require mode coverage within
families. Pairs and handoff variants share a split group; exact duplicates are
removed. New task seeds alone do not prove generalization to new task families.

### Soft step limits are not a hard action budget

One OpenCode run configured for 12 steps made 104 repeated reads. The server now
stops three repetitions of an unchanged action/observation cycle of one to three
actions before further inference. This also catches alternating read/edit loops.
A changed observation or new user message breaks the match. This contains observed
loops; it does not teach recovery or detect every cycle.

A later run exposed a bypass: OpenCode appends a plain assistant message announcing
its step limit after the tool result. That hid the final action/result pair from
the first guard. A scripted client self-test reproduced this independently of the
model. The guard now skips plain assistant messages while scanning for repeated
actions, while a real user turn still resets it. Scripted guard tests are not model
capability results.

### A 5xx can make a bad generation much more expensive

Malformed tool output originally returned HTTP 502, encouraging the client to retry
the same deterministic failure. It now returns non-retryable HTTP 422 with a clear
error type. Invalid calls remain undispatched. API errors, wrong answers and failed
tests must be reported separately.

## Experiments leading to this result

`horizon-wire-v1` uses the captured four-tool interface, actual observation formatting,
quoted CLI requests, and direct actions. It keeps random task values and diverse
identifiers from the existing coding corpus. No evaluation solutions are imported.
The step-250 checkpoint completed 16/16 rounds (65 tool calls) with seed 849723,
including the transient checker failure and fresh-session handoff. Repeats exposed
remaining failures: seed 1337 passed 15/16; seed 20261004 passed 21/32, stopping
after two consecutive failures. It retained 8/8 short arithmetic repairs but
passed 0/4 broader repairs. These are development evaluations; improvements informed
by their failures need additional fresh seeds before a final capability claim.
The successful run reported 397,216 cached input tokens across 81 model responses,
13,821 uncached input tokens, and a largest prompt of 10,534 tokens. Those are
observed cache-use figures, not a measured wall-clock speedup or a 32K retrieval score.
Broader four-task repairs, reasoning tests and long-context
retrieval remain separate capability measurements, not prerequisites silently
redefined to make the narrow demonstration pass.

### Recovery examples must not reward the mistake

One failed rollout duplicated the previous function instead of adding the requested
one, then repeatedly attempted an ambiguous edit. The next curriculum includes
that error state and the correct repair. Deliberately wrong assistant actions carry
`supervise: false`: they remain visible as history, but their tokens receive no
training loss. Tool observations are always masked. This is supervised recovery
training, not reinforcement learning or evidence of general self-correction.

### Passing tests is not the same as finishing the turn

An early clean-recipe snapshot made a correct edit and called check successfully,
then kept editing and exhausted its output budget. The evaluator previously counted
only final code correctness and an observed passing check. It now also requires a
final `stop` event and no client error. Reinspection confirms that all 16 rounds of
the earlier successful wire-format run meet this stricter condition.

### Testing the minimum training sequence

`agent-minimal-v1` starts directly from our scratch-trained foundation, avoiding
the accumulated experimental SFT checkpoints. Its prepared corpus has 13,063 training
and 737 validation episodes, with 5,558,804 supervised training tokens. It combines
original wire-format arithmetic repairs, cumulative sessions and handoffs, masked
edit-error recovery, and 1,879 training conversations from SmolTalk. Two hundred
overlength/empty episodes were excluded rather than cut midway through a tool call.
The foundation already saw the mixed code/conversation/multilingual pretraining
corpus. This smaller SFT corpus does not establish conversational quality.

The tested route should ultimately require only random initialization → pretraining
→ one assistant/tool SFT stage → serving → independent OpenCode evaluation.
This first clean SFT recipe failed the complete session gate. The later
`agent-direct-v1` run used the corrected related-name curriculum and is the
successful one-stage SFT route documented above.

The following historical commands reproduce that earlier experiment, not the
recommended final recipe. They start after the foundation and source preparation in
[the training runbook](training-plan.md). Run from the repository root, with the
existing `data` symlink pointing inside the authorized NAS dataset directory.
All output paths below must be fresh:

```sh
.venv/bin/python scripts/capture_opencode_tools.py --out data/opencode-tools-v2.json
.venv/bin/python -m microgpt.grounding --tools data/opencode-tools-v2.json \
  --out data/grounding-wire-v1.jsonl --wire --effort none
.venv/bin/python -m microgpt.horizon_data --tools data/opencode-tools-v2.json \
  --out data/horizon-wire-v2.jsonl --vocabulary-data data/magicoder-large.jsonl \
  --count 2000 --wire --effort none --recover-errors
.venv/bin/python -m microgpt.trajectories --input data/horizon-wire-v2.jsonl \
  data/grounding-wire-v1.jsonl data/smoltalk-large.jsonl --out data/agent-minimal-v1 \
  --tokenizer data/large-v1 --max-length 32768 --limit 14000 --pack-episodes
.venv/bin/python -m microgpt.finetune --data data/agent-minimal-v1 \
  --out runs/agent-minimal-v1 --checkpoint runs/coding-v1/foundation/last.pt \
  --steps 6000 --accumulation 2 --lr 0.0001 --eval-every 250 --eval-examples 16 \
  --device cuda --attention flash --max-length 32768 --balanced-sources \
  --balanced-lengths --short-replay 0.2 \
  --sources horizon-session horizon-handoff grounding-read grounding-edit \
  --max-seconds 1800
```

Freeze the selected checkpoint before evaluating; training replaces `best.pt`
atomically, so a hard link preserves the exact evaluated file on this filesystem.
Record its step, parent, tokenizer fingerprint and SHA-256. Serve with a matching
4,096-token output budget, then run the actual client:

```sh
cp --reflink=auto runs/agent-minimal-v1/best.pt runs/agent-minimal-v1/candidate.pt
sha256sum runs/agent-minimal-v1/candidate.pt
.venv/bin/python -m microgpt.serve --checkpoint runs/agent-minimal-v1/candidate.pt \
  --device cuda --attention flash --port 8000 --max-output 4096
# In another terminal; select fresh result directories and repeat with new seeds.
.venv/bin/python scripts/eval_horizon.py --out runs/minimal-horizon-eval \
  --label 'frozen checkpoint identifier' --effort none --rounds 16 --seed 849723
.venv/bin/python scripts/eval_opencode.py --out runs/minimal-grounding-eval \
  --label 'same checkpoint identifier' --effort none --suite grounding
```

The horizon report includes cached-token counts and the largest observed prompt.
A 16-round success does not establish effective use of all 32K positions. Automatic
compaction, model-written handoff notes, unfamiliar tool sets, deep reasoning, and
general coding remain separate capabilities needing separate training and tests.

The direct recipe's early checkpoints did not yet reproduce the successful longer
lineage: step 750 failed after several rounds, step 1500 duplicated an identifier,
and the best-loss snapshot near step 3000 made malformed edit anchors. The recorded
failures are useful evidence against selecting checkpoints by loss alone.

An additional controlled variant adds a short visible plan immediately before each
correct edit: identify the new function and required expression, then preserve the
old code while building tool arguments. `horizon_data --action-plans` produces these
targets; the supplied requirements and tool results remain unchanged. This tests
whether explicit intermediate intent improves copying. It is not evidence of deep
reasoning, and its benefit must be established by complete generated rollouts.

That visible-plan variant failed both initial evaluation rounds even after 2,500
updates: it stated a plan and stopped without editing. It is retained as an
optional experiment, not enabled in the baseline. Likewise, replacing opaque
runtime tool IDs with training-shaped IDs did not fix the step-5000 failure.

The direct 6,000-step run completed the first eight rounds but failed after the
fresh-session boundary. Inspection exposed another input mismatch: synthetic
handoffs announced “Continue after a session handoff,” while actual fresh sessions
received an ordinary request plus the state file. Wire-format generation now uses
the same request/state-file formatter as the lesson harness, with independently
generated task instances. It no longer relies on that extra handoff announcement.

`agent-lines-v2` combines this correction with smaller edit targets: use a unique
final source line as the append anchor, falling back to the final function when
the line is ambiguous. Existing code need not be copied back in full. Recovery
examples still include duplicated-function failures. This changes the taught edit
strategy; the server does not repair model arguments or insert code for it.

The early smaller-edit candidate still failed at handoff by using the correct new
expression with an earlier function's identifier. The data audit found that most
training functions had distinct word prefixes; the evaluation's sibling functions
all share a prefix and differ in their numeric suffix. The `--related-names`
curriculum adds same-prefix siblings in half of sessions, balanced independently
of the error-recovery pattern. Prefixes still come from the existing vocabulary;
evaluation identifiers, values and solutions are not inserted into training.

## Portable deployment and the interactive client

The export bundles `model.pt`, `encoding.json`, `encoding.tiktoken`, and
`deployment.json`. Vocabulary ranks, regex and special tokens are preserved; loading
was checked with the tiktoken registry disabled. Copy this directory as a unit.
The optimizer is excluded, so keep the original checkpoint for continued training.
All 25 CPU checks pass, including offline tokenizer identity and corruption rejection.

The successful recipe combines related-identifier recovery examples with canonical
internal tool IDs. Earlier failures of ID normalization alone remain useful negative
results; the final combination passed the repeat seeds listed above. These internal
IDs do not change client-visible call/result association.

Actual OpenCode TUI runs passed the first two tasks, beyond the automated CLI tests.
The auxiliary title agent initially made an out-of-domain request that returned 422;
disabling it removed that error in the second run. The prepared local lesson now
starts at round three, with the first two model-written functions preserved.

## User-discovered task-switch failure (2026-10-04)

After successfully completing round three in the interactive lesson, the user asked:
“create a simple hello world function that takes a name as a parameter to print it
out in the message”. Session `ses_ef7683fbcffeqHLb9y5x8VJrrE` shows the model replaying
its previous arithmetic edit twice, changing the target to nonexistent `helpers.py`.
Each edit failed with file-not-found; intervening reads were denied by the lesson's
permissions. The next generation ended with malformed tool JSON (missing comma,
line 1 column 196), rejected as HTTP 422. The raw rejected generation is not in the
session export, so its exact malformed text has not been recovered.

This is an observed task-switch and instruction-following failure, followed by
failed recovery and invalid serialization. The server remained healthy. Relaxing
permissions or repairing JSON would not make the repeated arithmetic edit satisfy
the greeting request. The existing fixed grader covers only arithmetic requirements
and cannot validate a newly requested greeting. The session export is retained
locally at `runs/opencode-lab/user-failure-session.json`.

Consequently, the long-session scores establish repeated success within one trained
family, not general simple-task readiness. A future acceptance gate must include
changes of task family within the same session, independently grade printed output,
and test recovery from unavailable paths. Keep this exact interaction held out of
training. This failure is diagnosed, not fixed by the current checkpoint.
