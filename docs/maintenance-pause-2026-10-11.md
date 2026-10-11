# Maintenance pause — October 11, 2026

The user explicitly requested a pause. Sent SIGTERM to the queued post-training
controller first, then the foundation controller, which forwarded it to the
trainer. All three processes exited. Both controller status files record
`interrupted`; no automatic restart is scheduled.

The trainer completed its current update, validated and saved `foundation/last.pt`
at step 83,304 (2,729,705,472 tokens recorded), validation loss 1.30404. The file
is under `data/experiments/foundation-pivot-v1/`, on the authorized NAS. No temporary
checkpoint remained after exit. The separate original inference server was not
stopped; it is not a training process and will exit normally on PC shutdown.

## Resume prerequisite

Do not blindly rerun the old controller or restart from last.pt. The earlier 5B
phase logged completion at step 152,588 and its coding adapter scored 52/100,
but the next phase loaded step 81,500. The reason has not been established.
Preserve last.pt, checkpoint-5b.pt, all adapter checkpoints, metrics and logs.
Inspect checkpoint metadata and provenance to choose the correct continuation.
A log filename claiming 10B is not proof of checkpoint lineage.

Once that discrepancy is resolved, resume from the verified foundation checkpoint
using the same config, corpus, optimizer/RNG and full learning-rate schedule,
with 75% GPU duty pacing. Restore the phase controller and waiting post-training
queue deliberately; their current command-line interfaces do not provide a
universal restart after this interruption. No training should restart until the
user says maintenance is finished and requests resumption.

Checkpoint metadata inspection via torch.load on the NAS stalled in filesystem
I/O. Sent SIGTERM to that read-only inspector rather than delay maintenance.
Full checkpoint integrity verification remains pending; successful trainer exit
and completed atomic checkpoint write were observed, but are not a substitute
for that verification.
