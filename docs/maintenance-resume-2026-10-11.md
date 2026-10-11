# Maintenance recovery and recurring GPU schedule — October 11, 2026

The user authorized resuming training and requested full GPU duty from 03:00 to
09:00, returning to 75% outside that window. This is daily America/Sao_Paulo local
time, not UTC. GpuPacer checks the clock between microbatches; no optimizer reset
or training restart occurs at either boundary. This is cooperative pacing, not a
promise that nvidia-smi will continuously display exactly 75% or 100% utilization.

## Storage verification and lineage

The NAS mount changed from 192.168.0.21:/volume1/GIGACHAD to
192.168.0.126:/CHAD/GIGACHAD. The current mutable last.pt has an older timestamp
than the pause. We cannot establish the cause of the rollback from timestamps
alone. The complete original 5B training log and its frozen checkpoint remain.
Do not resume the regressed mutable checkpoint merely because it is named last.pt.

Verified SHA-256 checks for both manifests, all ten foundation token streams and
both instruction JSONL files. All 14 match the previously recorded values. The
report is runs/pivot-v1/maintenance-data-check.json. The frozen 5B checkpoint must
match its milestone SHA-256 and contain step152588, optimizer state and RNG state
before the recovery controller starts. Its metadata report is
runs/pivot-v1/maintenance-checkpoint-check.json.

The 5B candidate's recorded development scores are 49/100 after500 SFT updates
and 52/100 after1000. This passed the original spending gate. These scores do not
establish OpenCode agent or long-session readiness.

## Recovery implementation

Resume from the verified frozen checkpoint into a NEW NAS directory:
`data/experiments/foundation-pivot-v1/foundation-resumed-5b`.
Keep the old mutable checkpoint, logs and metrics untouched for investigation.
The controller archives its interrupted state, skips already completed stages,
and runs the original 10B milestone and coding evaluation. The trainer's new
`--expected-resume-step` guard rejects unexpected steps before optimizer restore.
Subsequent stage transitions use that guard too, preventing silent rollback.

The queued post-training controller has an explicit `--resume-waiting` option
that only accepts an interrupted queue with no executed commands and unchanged
holdout hash. It retains the original longer-SFT/holdout plan.

Commands (run from the project, with the original placeholder PIDs ignored by
this recovery mode):

```sh
MICROGPT_FULL_GPU_WINDOW=03:00-09:00 MICROGPT_GPU_TIMEZONE=America/Sao_Paulo MICROGPT_GPU_DUTY_CYCLE=.75 .venv/bin/python scripts/run_foundation_pivot.py --preparation-pid 1809748 --previous-run-pid 1500740 --resume-after-5b
MICROGPT_FULL_GPU_WINDOW=03:00-09:00 MICROGPT_GPU_TIMEZONE=America/Sao_Paulo MICROGPT_GPU_DUTY_CYCLE=.75 .venv/bin/python scripts/run_pivot_posttraining.py --resume-waiting
```

## Verification

17 pacing/training tests passed, plus the extended CPU resume regression test.
Checks cover local-time boundaries, midnight-crossing windows, invalid schedules,
live transitions, and rejection of stale saved steps. Existing staged-training
checks preserve exact CPU optimizer/RNG equivalence. Runtime verification must
show step152588 loaded, then a later saved training update; no checkpoint beyond
5B should be claimed until it is actually observed.

## Observed restart

The frozen checkpoint SHA-256 matched
`9434ae6027fb1f64f4229c8f954ce307f1c0d67c4878941989ff31d67954de95`.
It contains step152588, 170 optimizer-state entries, CUDA RNG state and random
initialization provenance. Tokenizer fingerprints match across foundation data,
instruction data, original foundation checkpoints and both adapter directories.
The mutable old last.pt instead contains step81500; it was not used.

The resumed trainer logged `step 152588: validation loss 1.2108`, matching the
preserved 5B result, and is active with the expected-step guard. The post-training
queue is active and waiting. New weights will save in foundation-resumed-5b.
The first later checkpoint is still pending at the time of this restart note.
