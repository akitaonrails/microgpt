# Serve our scratch-trained model through OpenCode

Follow the [validated minimal recipe](agentic-minimum.md) first. The HTTP server
requires our checkpoint and never downloads a model:

```sh
.venv/bin/python -m microgpt.serve --checkpoint runs/agent-ready-v1/model.pt \
  --attention flash --host 127.0.0.1 --port 8000
```

Copy `examples/opencode/opencode.json` into the project where OpenCode should work,
or point `OPENCODE_CONFIG` at it. Match its context/output limits to the checkpoint
and server. The student configuration has a 32K ceiling; the teaching demo has 128
and cannot accommodate OpenCode's prompts. Keep the tool set small initially.
The provider uses `@ai-sdk/openai-compatible`, model ID `microgpt/microgpt`, and
base URL `http://127.0.0.1:8000/v1`. Tools execute in OpenCode, never in the server.

The server implements `/v1/models` and `/v1/chat/completions`, schema-checked tool
calls, tool-result round trips, usage/cached-token accounting, stop limits and
buffered SSE. It rejects context overflow and concurrent requests rather than
silently truncating or building an unbounded queue. Generated malformed calls
return an error; they are not silently dispatched. The same `protocol.py` builds
SFT examples and serving prompts. EOS ends an assistant response.

Inference uses chunked prefill, per-layer KV caches and one exact-prefix cache.
A cache hit crops to the common processed prefix; it is not a response cache.
The cache belongs to a single local trust domain. SSE is buffered rather than
live token streaming, and client disconnect does not immediately cancel generation.

## Independent checks

```sh
docker pull python@sha256:dddfd7e07f9d15aeeca61529320492139d21cac7f0070c00609243e51e4e0016
.venv/bin/python scripts/eval_opencode.py --out runs/my-eval --label 'scratch checkpoint identifier'
.venv/bin/python scripts/eval_dialogue.py --out runs/my-chat.json --label 'same checkpoint'
```

The repair evaluator uses Git, OpenCode and Docker. It checks tool availability
before running. The model can modify only the fixture's `solution.py`. Its fixed
`check` tool runs a restricted, networkless container. The independent final grader
checks behavior regardless of what the model claims. Results include tool failures,
patches and token/cache usage. No generated code executes directly on the host.
A failed model result remains a failed result; protocol compatibility is not a pass.

`smoke_opencode.py` checks a read/explanation round trip if the trained model can
perform it. `check_inference.py --checkpoint ...` checks cache equivalence and Flash
dispatch at up to 32K, using synthetic filler. That is an allocation test, not a
retrieval or reasoning benchmark.

## Longer runs

For the narrow cumulative-function lesson, create a persistent workspace using the
same interface as the evaluator:

```sh
.venv/bin/python scripts/agent_lab.py init --out runs/my-agent-lab
.venv/bin/python scripts/agent_lab.py run --out runs/my-agent-lab
```

The launcher opens the TUI with the current task prompt. In another terminal, run
`.venv/bin/python scripts/agent_lab.py next --out runs/my-agent-lab` after each
completed task. It independently checks every current requirement before advancing
and prints the next prompt. `show` repeats the prompt; `check` only grades the code.
Stay in the same OpenCode session, then try a fresh session halfway through.
The harness owns `TASK_STATE.md` and the grader; the model can edit only `solution.py`.
Round four deliberately fails one checker call. This lab checks cumulative code;
use `eval_horizon.py` for automated conversation/completion scoring.

This restricted lab profile is the measured teaching setup. The general example
configuration exposes a broader workflow whose reliability is not established by
the arithmetic exercise.

The example agent now allows 64 steps, asks for a task plan and `TASK_STATE.md`
handoff notes, and enables automatic compaction with an 8,192-token reserve.
Old tool outputs are not automatically pruned. These options follow the
[OpenCode configuration reference](https://opencode.ai/docs/config/#compaction).
Reliable note writing and summarization still require training; the config
alone does not confer those abilities.

```sh
.venv/bin/python scripts/eval_horizon.py --out runs/my-horizon --label 'checkpoint identifier'
.venv/bin/python scripts/eval_context.py --checkpoint runs/my-model/last.pt --out runs/my-context.json
```

The horizon evaluation maintains cumulative requirements outside the editable
file, injects one transient test-tool failure, and starts a fresh session halfway
through. It records actual calls and independently checks every earlier function
after each new requirement. Two failed rounds stop the run; unattempted rounds
remain unmet. This measures a scaffolded sequence, not unconstrained autonomous
software engineering. The context evaluation separately checks exact retrieval
near the beginning, middle and end of 8K/16K/32K inputs.

`scripts/train_stages.py` automatically runs both evaluations and the original
four repairs after training. See [the larger training runbook](../history/training-plan.md).

## Remote machine

The same checkpoint/server can run on one H100. Copy the entire checkpoint folder,
including tokenizer files; use SSH port forwarding to keep the server on loopback.
For a network bind the server requires `MICROGPT_API_KEY`; use the same key in
OpenCode, with private networking or a TLS proxy. It is not a public multi-user
serving system. See [hardware](hardware.md) for storage and rental considerations.

The example now enables reasoning with a 4,096-token output ceiling and
`none`/`low`/`medium`/`high` variants (`none` default for the measured agent baseline). Use `--variant high --thinking`
with `opencode run` to request more worked reasoning and display it. These are
learned instructions, not a guarantee of accuracy. The API returns thinking as
`reasoning_content`; tool-looking text inside thinking cannot dispatch an action.
See [reasoning.md](reasoning.md) for the data and evaluation recipe.

The selected export passes 8/8 narrow repairs, four 16-round sessions and one
32-round session. It still fails the broader four-task repair set. The exported
settings enable canonical internal tool IDs, preserving API result associations
while avoiding long random identifiers in model input. Its portable tokenizer
includes the vocabulary, regex, special tokens and checksums, so serving does not
need a tokenizer download. Resume training from the original full checkpoint.

The teaching profile disables OpenCode's auxiliary title agent: its unrelated
no-tools request caused HTTP 422 even while the actual coding task succeeded.
Two real TUI tasks passed independent grading; the second, with titles disabled,
produced no such error. Measured compatibility is with OpenCode 1.18.34.

Malformed generated calls return non-retryable HTTP 422 (`invalid_model_output`).
Three repetitions of the same one-to-three-action cycle with unchanged observations
return HTTP 422 (`agent_loop_error`); a new user turn or changed observation breaks
the match. Plain assistant reminders about OpenCode's step budget do not reset the
guard. This bounds repeated-read/edit loops that OpenCode's soft step limit did not
stop in testing. It does not substitute for learned recovery.

The [minimal-agent experiment notebook](agentic-minimum.md) records real cumulative
sessions, repeat failures, and the distinction between model behavior and harness
support. For comparable evaluations pass `--effort none` explicitly to both repair
and horizon evaluators; thinking modes need their own behavioral validation.
