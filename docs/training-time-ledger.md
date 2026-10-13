# Training time ledger — 337M pivot

Snapshot: 2026-10-13T22:45:27.711997+00:00. The final interval is still running.

- Calendar time since the first foundation trainer launch: **147.73 hours** (6.16 days).
- Sum of foundation process intervals: **130.43 hours**.
- Time outside those intervals: **17.31 hours**. This combines maintenance, recovery, integrity checks, fine-tuning, evaluations, and other gaps; it is not all idle time.

## Recorded intervals

| Stage | Start (UTC) | End / snapshot (UTC) | Hours |
|---|---|---|---:|
| foundation-2b | 2026-10-07T19:01:29+00:00 | 2026-10-08T23:49:37+00:00 | 28.80 |
| foundation-5b | 2026-10-09T03:25:18+00:00 | 2026-10-11T02:43:28+00:00 | 47.30 |
| foundation-10b | 2026-10-11T03:28:49+00:00 | 2026-10-11T04:43:00+00:00 | 1.24 |
| foundation-10b | 2026-10-11T05:20:00+00:00 | 2026-10-11T22:55:47+00:00 | 17.60 |
| foundation-10b | 2026-10-12T00:44:33+00:00 | 2026-10-12T00:44:40+00:00 | 0.00 |
| foundation-10b | 2026-10-12T01:41:23+00:00 | 2026-10-12T20:20:46+00:00 | 18.66 |
| foundation-10b | 2026-10-13T00:56:06+00:00 | 2026-10-13T15:55:52+00:00 | 15.00 |
| foundation-10b | 2026-10-13T20:55:22+00:00 | 2026-10-13T22:45:27+00:00 | 1.83 |

## Interpretation

Source: `runs/pivot-v1/journal.md`, using each foundation start and its completion or interruption/failure entry. These are controller-observed process intervals, including startup, validation, checkpoint I/O, GPU pacing, and storage stalls. They are not GPU compute-hours. Graceful shutdown saving may extend beyond the controller interruption timestamp, so this is an approximate wall-time ledger.

The first 10B-stage attempt resumed a stale checkpoint and was discarded; its time is included as experiment cost. The six-second failed restart is also included. Earlier models, dataset preparation before the first launch, and coding fine-tuning/evaluation are not included in foundation process hours.

No continuous GPU utilization or energy ledger was collected, so full-utilization-equivalent GPU hours and electricity consumption cannot be reconstructed reliably. The recurring schedule is 100% duty from 03:00–09:00 America/Sao_Paulo and 75% otherwise; older intervals had different permissions.

The trainer records cumulative tokens/sec **since each process restart**, not an instantaneous speed or a per-checkpoint interval rate (`microgpt/train.py`). Consequently, previous status estimates based on that value are rough, particularly after NAS throughput changes. A completed-run article should report calendar time, foundation process wall time, fine-tuning/evaluation time, and recovery cost separately.

This snapshot does not change the running training process. Append a final measurement when the 10B phase completes.
