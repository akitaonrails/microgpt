# Execution notes — foundation pivot

2026-10-07. Read together with the frozen [experiment plan](foundation-pivot.md).
These are observations during preparation, not capability results.

1. Inspected the old run, model, tokenizer and trainer before changing the recipe.
   The user explicitly authorized random reinitialization and unrestricted GPU
   duty cycle for the next hours. Selected an eight-hour window with automatic
   return to the previous 75% pacing.
2. Created the 337,167,360-parameter configuration. Computed the parameter split
   directly from the model: 33,554,432 embedding/output; 303,612,928 transformer.
   Added optional depth-scaled residual initialization, retaining old defaults.
3. Trained a 32,768-entry byte BPE using only distributed training documents:
   60,012,040 CodeParrot bytes; 10,576,046 OpenCoder; 11,211,230 Cosmopedia;
   8,001,602 SmolTalk; 2,246,676 multilingual. The approximately 92MB actual sample
   undershot the 100MB target because the deterministic strides exhausted some
   smaller sources. This is recorded rather than described as exactly 100MB.
4. Started complete EOS-aware corpus conversion. Each split retains its original
   documents; input checksums, output text round-trips and document counts are
   checked. Final checksums/counts will appear in the completed corpus manifest.
5. Gracefully stopped the superseded refined agent experiment. Its last saved
   step is **829**; validation loss 3.32384. No behavioral evaluation of this
   interrupted candidate was completed, so this is neither a success nor a
   failure result for its curriculum. The original public server stays separate.
6. Ran a GPU preflight on an 8MB completed prefix of the retokenized OpenCoder
   training stream while the full corpus was still converting. The profile used
   the new architecture, Flash Attention, BF16, batch2 × accumulation8, 2K windows,
   three warmup updates and 30 measured updates. It reported **43,882 tokens/s**
   and **13.86GB peak allocated VRAM**, with finite losses/gradients. These profile
   weights are discarded. This short, single-source, cached-data measurement is
   not a sustained NAS throughput estimate or evidence of coding competence.
   The controller profiles the complete mixture again before foundation training.
7. All 78 then-existing tests passed. An additional corpus-conversion test now
   verifies document contents/counts and rejects a modified source. The expanded
   checkpoint-resume test also passes with the optional new config field removed
   from an old-format checkpoint; resumed weights match uninterrupted weights.

The controller writes an ongoing timestamped `runs/pivot-v1/journal.md`, including
commands, phase completion, profile measurements, coding scores and failures.
`runs/pivot-v1/status.json` is the live state. Do not describe a queued phase as
running GPU training; the controller waits for all corpus integrity checks first.
