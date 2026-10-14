# Final training time ledger — 337M pivot

Closed October 14, 2026. This supersedes the October 13 in-progress snapshot.

- Calendar time from first foundation launch to completion: **153.98 hours** (6.42 days).
- Sum of foundation process intervals: **136.68 hours**.
- Outside those intervals: **17.31 hours**, combining maintenance, recovery, integrity checks, coding fine-tuning, evaluations and other gaps.
- Logged SFT training segments across the nine pivot checkpoints: **3.40 hours**, separately measured and not included in the foundation-process sum.

| Stage | Start (UTC) | End (UTC) | Hours |
|---|---|---|---:|
| foundation-2b | 2026-10-07T19:01:29.201975+00:00 | 2026-10-08T23:49:37.693736+00:00 | 28.80 |
| foundation-5b | 2026-10-09T03:25:18.554792+00:00 | 2026-10-11T02:43:28.014907+00:00 | 47.30 |
| foundation-10b | 2026-10-11T03:28:49.632855+00:00 | 2026-10-11T04:43:00.551768+00:00 | 1.24 |
| foundation-10b | 2026-10-11T05:20:00.615880+00:00 | 2026-10-11T22:55:47.524788+00:00 | 17.60 |
| foundation-10b | 2026-10-12T00:44:33.827776+00:00 | 2026-10-12T00:44:40.091853+00:00 | 0.00 |
| foundation-10b | 2026-10-12T01:41:23.760445+00:00 | 2026-10-12T20:20:46.884190+00:00 | 18.66 |
| foundation-10b | 2026-10-13T00:56:06.857193+00:00 | 2026-10-13T15:55:52.086534+00:00 | 15.00 |
| foundation-10b | 2026-10-13T20:55:22.491447+00:00 | 2026-10-14T05:00:27.761042+00:00 | 8.08 |

Source: `runs/pivot-v1/journal.md`; the committed interval records and SFT segment metrics are in [closeout/experiment-evidence.json](closeout/experiment-evidence.json).

Controller intervals include startup, validation, checkpoint I/O, GPU pacing and storage stalls. Graceful shutdown saving can extend beyond an interruption timestamp, so this is approximate wall-time accounting, not GPU compute-hours. The discarded stale-checkpoint attempt (1.24 hours) and failed six-second restart are included as experiment cost. Excluding those gives about 135.44 hours in the successful foundation lineage. The selected 4K coding branch adds 1.36 logged training hours; startup, final writes and evaluation are outside that SFT timer.

Earlier 91M/239M experiments, preparation before the first pivot launch, and all-project human effort are not included. There is no complete all-project compute or electricity ledger. Do not present these totals as the total cost of every experiment.

`tokens_per_second` is cumulative since each trainer process restart, not instantaneous throughput. It includes periodic validation and previous saves, but excludes initial validation and the current save. This limits the accuracy of chat ETAs based on that field.

See [the closeout report](experiment-closeout.md) for model quality, data, hardware, recovery details and future estimates.
