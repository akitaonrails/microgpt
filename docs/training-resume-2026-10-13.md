# Training recovery — October 13, 2026

The foundation controller recorded an interruption at 12:55 São Paulo time.
The initiating source is unknown. The trainer recorded step 284725,
9,329,868,800 tokens, and validation loss 1.0833607. The post-training queue
then failed before issuing any commands because its upstream controller stopped.
The user explicitly authorized resuming.

## Recovery checks

The latest checkpoint loaded on CPU with 170 optimizer states, one CUDA RNG
state, and batch/CPU RNG states. Its checksum and size are recorded in
`resume-checkpoint-2026-10-13.json`. An exclusive write, fsync, and removal probe
in the checkpoint directory succeeded. Recovery rechecks the 14 corpus and SFT
files against the existing hashes before refreshing the controller's integrity
audit. NAS reads remain slow; this is separate from GPU duty-cycle pacing.

The queue restart guard previously accepted only `interrupted`, although an
upstream interruption produces `failed` in a waiting queue. It now also accepts
that specific upstream error, only with an empty command history and unchanged
holdout checksum. Other failures, active/completed queues, and queues that have
started work remain rejected. Three focused regression tests pass.

## Resume commands

Both controllers inherit `MICROGPT_FULL_GPU_WINDOW=03:00-09:00`,
`MICROGPT_GPU_TIMEZONE=America/Sao_Paulo`, and `MICROGPT_GPU_DUTY_CYCLE=.75`.

```sh
.venv/bin/python scripts/run_foundation_pivot.py --preparation-pid 1809748 --previous-run-pid 1500740 --resume-pause-record docs/resume-checkpoint-2026-10-13.json
.venv/bin/python scripts/run_pivot_posttraining.py --resume-waiting
```

Historical PID arguments are ignored in recovery mode. The expected-step guard
must restore step 284725, with the original optimizer and 10B learning-rate
schedule. Checkpoints continue in `foundation-resumed-5b`. The queue waits for
foundation training and its milestone evaluations before longer coding SFT and
the reserved holdout evaluation. No new capability result follows from resuming.
