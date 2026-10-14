# Foundation-first pivot — 2026-10-07

## Decision and authorization

The user authorized a fresh scratch run and full GPU use while away for the next
hours. We select an eight-hour full-duty window beginning at the GPU profile,
then automatically restore 75% cooperative pacing. This does not reserve display
VRAM. No paid machine is rented. All large new artifacts stay under
`/mnt/gigachad/microgpt/dataset`, via the repository's `data` symlink.

The user also requested an educational record of every important step. This file
records the predeclared experiment, not a retrospective success story. Runtime
status, exact commands, hashes, measurements and failures live in
`runs/pivot-v1/`; subsequent interpretation belongs in a separate outcome document
so the plan cannot be rewritten to fit the result.

## Why change direction?

The best 239M scratch candidate solves 24/100 development programming problems.
Agent fine-tuning improved tool usage but did not fix broad programming repairs.
The old 1K mixed-agent checkpoint solves 14/32 simple repairs, 0/32 structural
repairs and 0/32 recovery cases. Our interpretation is that weak programming
competence limits useful agent behavior. This is a hypothesis, not an established
architecture ceiling. See `repair-investigation.md` for the narrower successes
and the important limits of teacher-forced diagnostics.

The refined repair experiment is superseded, stopped by SIGTERM, and preserves
its latest optimizer/checkpoint and evidence. An interrupted experiment is not a
negative result. The existing user-facing server is not replaced.

## Architecture decision

| | Previous candidate | New scratch candidate |
|---|---:|---:|
| Total parameters | 238,568,448 | 337,167,360 |
| Tied embedding/output | 153,614,592 | 33,554,432 |
| Transformer/norm | 84,953,856 | 303,612,928 |
| Vocabulary | 200,019 | 32,768 |
| Layers / width / heads | 12 / 768 / 12 | 24 / 1,024 / 16 |

Byte-level BPE retains arbitrary UTF-8 coverage, including code, Portuguese and
other languages. Smaller vocabulary can require more tokens per document;
conversion records both old/new counts and UTF-8 bytes. Coverage is not a claim
of multilingual competence. OpenCode speaks our HTTP/tool protocol and does not
require o200k token IDs. Old weights cannot be transplanted into this tokenizer.

We retain RoPE, RMSNorm, SwiGLU, tied embeddings, full multi-head attention and
PyTorch Flash SDPA. The deeper model scales initial attention-output and MLP-down
weights by `1/sqrt(2*layers)`, following the residual initialization used in
[nanoGPT](https://github.com/karpathy/nanoGPT/blob/master/model.py).
This is an opt-in config field; old checkpoint configs default to the original
initialization. Loaded weights are unchanged. GQA is deferred: it can improve KV
memory, but changing attention now adds an independent implementation risk.

The model allocates 32K context, but initial foundation windows are **2K**. This
run does not establish 32K reasoning or long-session reliability.

## Corpus and tokenizer preparation

Run `scripts/prepare_pivot_corpus.py --out data/foundation-bpe32k-v1` with the
project Python. It fits the tokenizer on approximately 100MB of distributed,
training-only documents, weighted by the existing source mixture:

- 60% CodeParrot clean Python (license-filtered in the earlier preparation).
- 16% OpenCoder instruction/code text.
- 12% Cosmopedia educational text.
- 8% SmolTalk conversation.
- 4% multilingual SmolTalk2 text.

The frozen parent corpus has 3,057,260,620 training tokens under o200k. The new
counts will differ. Token budgets below mean **tokens consumed**, not unique new
data; repeated exposure is explicit. No new download is needed for this stage.

We decode complete EOS-delimited documents, never arbitrary token chunks that
could split UTF-8. We preserve train/validation membership and document order.
Each worker verifies the old binary checksum, checks every document's text
round-trip, verifies the document count, and hashes its new binary. The completed
manifest retains the parent manifests, pinned source revisions and checksums.
Workers publish progress files; `meta.json` appears only after all workers pass.
Tokenizer fitting never consumes validation documents.

Gotcha: literal `<|endoftext|>` in source text must remain ordinary text. The BPE
backend now encodes special-token spellings as text, matching our existing o200k
policy; actual document boundaries are appended by ID. Save/load and multilingual
round-trip tests cover this distinction.

Inherited repository/group splitting and exact deduplication do **not** prove
absence of near-duplicates or overlap with public benchmark families. The frozen
100-case development suite is a diagnostic reused for decisions, not a pristine
final acceptance set. Fresh repository tasks are required before claiming success.

## Training and decision gates, fixed before results

`configs/pivot-337m.toml` defines AdamW, LR 3e-4, 1,000 warmup updates, cosine decay
to 10% over 305,176 updates, BF16 and gradient clipping. Batch 2 × accumulation 8
× 2,048 positions = 32,768 tokens/update. A 30-update real-data Flash profile first
checks finite losses/gradients and memory, and measures throughput. Its weights
are discarded. The actual foundation starts independently from random seed
20261007; it uses neither `--init-from` nor any external checkpoint.

Checkpoints/evaluations are saved every 500 updates (~16.4M tokens), plus the
2B/5B/10B milestones rounded upward to complete updates. All milestones preserve
the same optimizer/RNG and full 10B learning-rate schedule. Interrupted resumes
must use `--resume`, not restart or reset the schedule.

At each milestone:

1. Preserve the foundation checkpoint and hash.
2. Evaluate raw completion on the same 100 development tasks in restricted Docker.
3. Starting from that foundation, apply the same independently checked public-code
   SFT dataset, retokenized with the new tokenizer. Use LR 1e-5, effective batch16,
   and measure at 500 and 1,000 updates. Each milestone gets a fresh adapter
   optimizer; SFT never modifies the continuing foundation optimizer.
4. Report both adapter scores and raw score, not just validation loss.

Continue 2B -> 5B if best adapter score is at least 10/100. This permissive gate
allows a foundation learning curve to develop. Continue 5B -> 10B only if best
score is at least 25/100 **and** at least three tasks above the 2B score. These
are engineering spending gates, not statistical significance claims. Failure
requires inspecting errors before extending the run. It does not establish that
scratch training cannot work. No checkpoint is automatically deployed.

We change vocabulary, architecture, initialization and training exposure together.
This is a practical recipe pivot, not an ablation identifying one causal factor.
Cross-tokenizer token loss is not directly comparable; correctness and byte-based
corpus accounting are the relevant comparison. Profile throughput excludes
checkpoint writes and behavioral evaluations; do not present it as a deadline.

## Capability sequence after foundation

Aim first for unfamiliar small Python repositories: inspect files, implement a
modest change, execute tests, recover and report honestly. Before broad agent SFT,
seek a materially stronger coding candidate (at least 30/100 development as an
entry gate, not readiness). Retain code replay during agent training to detect
forgetting. Use verified trajectories with localization, failed edits and useful
corrections; do not equate tool-call syntax or thinking tags with reasoning.

Then evaluate real OpenCode tasks with frozen fresh repository tests and no hidden
answer assistance. Add 4K/8K/32K training stages only after short-task competence.
Long sessions require retrieval, durable task state, compaction and regression
checks in addition to context allocation. Measure increasingly long sessions,
including earlier constraints and recovery, before any long-horizon claim.

The current unattended runner executes the **foundation comparison and coding
probes**, not automatic agent promotion or the later context curriculum. Those
stages depend on its measured outcomes.

## Evidence and recovery

- Plan owner: the primary coding agent; no delegated experiments.
- CPU verification: document boundaries including empty/cross-block cases,
  Unicode/control-token save/load, finite scaled-initialization gradients,
  old-config compatibility, staged optimizer/RNG equivalence, and pacing expiry.
- GPU verification: real corpus profile, then actual training loss/VRAM/throughput.
- Capability verification: sandboxed raw and fixed-adapter coding tests.
- Runtime record: `runs/pivot-v1/status.json`, `prepare.log`, `profile.json`,
  `foundation-2b.log`, and corresponding milestone evaluation directories.
  `runs/pivot-v1/journal.md` records phase starts/completions, exact commands,
  profile measurements, milestone scores and the final controller state.
- New weights: `data/experiments/foundation-pivot-v1/`.

The controller records exact commands and source/plan/data hashes. It refuses an
existing experiment directory. If interrupted, use the recorded foundation
command plus `--resume data/experiments/foundation-pivot-v1/foundation/last.pt`
with the same config/data and target step. Do not delete evidence to rerun the
controller. For later phases, inspect the saved step and existing evaluation
outputs before recovering them individually. SIGTERM saves an in-progress trainer;
SIGKILL does not. The prior served model remains available as a separate lesson.
