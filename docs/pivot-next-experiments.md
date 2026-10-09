# Larger next steps after the foundation pivot

Authorized October 9, 2026. Preserve the in-progress 5B/10B foundation comparison.
Do not run competing GPU training jobs. Use 75% pacing; the temporary full-GPU
permission expired.

## Evidence driving the next experiment

The 2B foundation plus 1000 instruction updates scores 39/100 development tasks.
Of the remaining 61 cases, 2 have syntax errors, 12 have mismatched interfaces,
and 47 have valid syntax and matching parameter names but fail behavior or
completion checks. This automatic partition is not a complete semantic diagnosis.
Inspection found incorrect rotation, incorrect indexing, and a nonexistent
pathlib.splitext call. Do not put these exact evaluation solutions into training.

Source: runs/pivot-v1/coding-2b-recovery/eval-1000/direct/{analysis,results}.json
and saved generated Python files. These files are inspected as text, not executed
on the host. Five variants per family are correlated; 39/100 is not 100 independent
algorithmic successes. Loss and syntax alone remain inadequate capability gates.

## Executable sequence

`scripts/run_pivot_posttraining.py` waits for the foundation controller to finish
cleanly or reach its predefined spending gate. It refuses to proceed after a
pipeline failure. Then it selects the foundation whose fixed short instruction
training had the best development result, continues that foundation's saved SFT
optimizer from step1000, and measures at 2000/4000/8000 updates (eight-hour budget,
two-checkpoint plateau limit). It retains all earlier candidates, so longer
training cannot overwrite the best model. This is a separate experiment and does
not change the original 2B/5B/10B learning curve or gates.

After selection on development correctness, evaluate exactly one selected model
on the existing separately reserved 100-case acceptance set. A search of saved
run status/log files found no prior acceptance.jsonl evaluation. This is newly
used evaluation evidence, not a newly invented dataset; public-corpus novelty is
not guaranteed. Hash the holdout before training and prohibit its use for model
selection or training examples. Its different task families help test transfer,
but eventual repository acceptance requires additional real OpenCode tasks.

Verification: compile both queue/evaluator entry points; reuse existing tested
training, checkpoint and Docker graders; check controller liveness before queueing
GPU work; preserve all commands, selection evidence and checkpoint hash in
runs/pivot-posttraining-v1/status.json. Actual GPU serving and holdout execution
remain pending until the current foundation experiment releases the GPU.

## Subsequent capability work

If stronger code competence transfers to the holdout, prioritize verified small
Python repository trajectories: locate relevant code, implement changes, test,
and recover. Preserve code replay and measure forgetting. Test on fresh repository
fixtures rather than recycling function-level diagnostic answers. Then introduce
progressive longer-context/session training and state/compaction checks. Tool-call
format or longer thinking text is never sufficient evidence of correctness.

These later dataset and harness stages remain conditional on measured outcomes;
the queue executes longer instruction training and the reserved holdout only.
