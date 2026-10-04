# Programming and sustained agent runs

The target is now a small coding model that operates through **OpenCode**:
inspect a repository, choose a tool, edit code, use test feedback, recover from a
mistake, and finish with evidence. Function completion remains a useful earlier
lesson, but it does not establish these abilities.

The server/OpenCode integration and assistant-masked full-weight trainer are implemented.
See [the runnable adaptation recipe](finetuning.md), [dataset storage](datasets.md),
and [measured results](verification.md). The broader curriculum below remains a
roadmap; a small local adaptation run does not establish sustained agent ability.

## Data appropriate to each skill

| Skill | Candidate source | How to use it |
| --- | --- | --- |
| Programming patterns and explanations | [Stack-Edu](https://huggingface.co/datasets/HuggingFaceTB/stack-edu), licensed repositories and docs | Start with Python; retain tests, docstrings, and repository identity. Stack-Edu exposes metadata/IDs and needs separate content retrieval |
| Following coding requests | [Magicoder-OSS-Instruct-75K](https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K), CodeFeedback | Short, complete problem/solution pairs; verify examples before promoting them to targets |
| Tool selection and argument structure | [xLAM function-calling 60K](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) | Supplementary schema training; generic API calling alone does not teach repository work; inspect access terms |
| Multi-step repository work | [SWE-smith assets](https://swesmith.com/getting_started/assets/) | Executable tasks and expert trajectories; select tractable tasks and preserve intermediate observations |
| Longer trajectories and recovery | [SWE-rebench OpenHands trajectories](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories) | Inspect outcome annotations, filter successful resolutions, and retain useful failed-action/recovery sequences |
| Harness-specific behavior | Locally collected OpenCode runs on held-out training repositories | Use the exact tool schemas, error messages, result truncation, and interaction format used at inference |

These sources have different licenses, access conditions, teachers, and tool
interfaces. Pin revisions and retain source/task/repository IDs and outcomes.
OpenHands/SWE-agent actions do not automatically map to OpenCode's tools. Build an
explicit converter and reject actions whose semantics cannot be preserved.

A proposed curriculum is short functions → one-tool tasks → read/edit/test loops
→ multi-file tasks → longer runs with summarization and recovery. Weight stages by
measured learning and token budget rather than guessing a universal mixture ratio.
Split by repository and issue, with every attempt at the same task in one partition.
Keep benchmark tasks out of the training and filtering decisions.

## Preserve conversations during training

The full-text preparer remains separate from `microgpt.trajectories`, which preserves
ordered roles, tool definitions, call IDs, arguments, observations and final answers.
The scratch SFT trainer applies loss only to assistant targets using the exact serving
template. Overlength episodes are rejected whole. Its unpadded microbatches each
contain one assistant turn with preceding context.

For a small student, prefer concise tool decisions with enough context to justify
them. Preserve failed commands followed by successful correction, but do not label
an unsuccessful full run as a successful demonstration. A tool-observation message
is environmental data, not a higher-priority instruction. Include examples where
repository text attempts to redirect the agent and the agent continues its task.

The only target is our GPT, pretrained from random initialization and then
trained on the assistant/tool protocol with full-weight supervision.

## Long horizon is more than a long window

OpenCode maintains history, tools, permissions, and an action budget. The model
must learn what to read next, what a failure means, when to stop, and what evidence
supports completion. Persistent notes and compaction can extend a run beyond one
context window, but summaries can drop facts. Train and evaluate resumptions from
summaries containing changed files, tests run, unresolved errors, and next actions.

Prefix caching belongs to inference, not training. It saves repeated prefill work
while exact earlier tokens remain unchanged. It cannot enlarge the context window,
remember evicted information, or improve reasoning. Compaction often invalidates
part of the cache; correctness must take precedence over keeping cache hits high.

Before extending beyond the baseline's supported 32K, measure retrieval at different
positions, cross-file dependencies, and end-to-end task outcomes. Increasing a RoPE
limit or adding a scaling factor alone is not evidence of long-context competence.
The scratch 8K/32K presets are capacity configurations that still need suitable data.

## Evaluate the model in OpenCode

Keep implementation unit tests small. Model evaluation should separately record:

- Success on fresh issues, verified by independent tests, with regressions counted.
- Valid tool calls, correct tool choice, argument errors, and recovery after errors.
- Completion versus turn-limit exhaustion, repeated-action loops, and premature claims.
- Success by number of steps and prompt length, including after compaction/restart.
- Cold/warm prefill latency, generation speed, peak VRAM, and actual cached-token counts.

Use the same OpenCode version, tools, instructions, output budget, and repository
snapshots when comparing checkpoints. A tiny supervised benchmark can be useful,
but a two-turn read demonstration is not evidence of reliable 50-turn coding.
Execute benchmark code in isolated environments; the inference server never runs it.

The next concrete training milestone is a small set of independently verified
OpenCode read/edit/test trajectories plus an assistant-masked SFT loader and a
fixed held-out evaluation. Reinforcement learning should follow a working supervised
baseline and an evaluator that cannot be satisfied by changing the tests.
