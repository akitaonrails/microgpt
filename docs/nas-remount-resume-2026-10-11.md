# Resume after second NAS maintenance — October 11, 2026

The user authorized checking the remount and resuming. The NAS is again mounted
from 192.168.0.21:/volume1/GIGACHAD. Verify the exact checkpoint recorded in
nas-pause-2026-10-11.json: step207508, 6,799,622,144 tokens, SHA-256
11cac3eb5d98313cabff027867bf6108d66bb50d852566fd3dc0a0722c654225.
Recheck all 14 corpus/SFT manifests and data files, plus tokenizer fingerprints,
before starting. Runtime audit: runs/pivot-v1/nas-remount-check.json.

The controller now accepts --resume-pause-record with that JSON path. It verifies
checkpoint ownership, checksum, step bounds and remount audit; preserves the
interrupted status under a unique archive name; and restores the same optimizer,
RNG and schedule with --expected-resume-step 207508. It continues writing to
foundation-resumed-5b. It does not repeat completed 2B/5B stages. The queued
post-training controller also archives each pause separately, so repeated
maintenance does not cause an archive filename collision.

Six pacing and exact-resume tests passed. Both controllers compile, and diff
whitespace checks pass. Final runtime proof is the trainer's restored step and
validation log. Recurring GPU policy is unchanged: America/Sao_Paulo 03:00–09:00
full duty, 75% otherwise. This is soft pacing, not fixed hardware utilization.

## Verification outcome and write-access blocker

Checkpoint checksum and step207508 verified. All 14 data/manifests matched, all
five tokenizer fingerprints matched, and earlier adapter checkpoints are present.
The first restart failed before any optimizer update: saving tokenizer.json
returned Permission denied. Exclusive temporary-file creation also failed at
both /mnt/gigachad/microgpt/dataset and the current checkpoint directory. No test
files were created. Client mount reports rw, and owner mode bits report writable,
so these observations do not identify the server-side ACL/export cause.

No trainer remains running; the queued controller was not restarted. The saved
checkpoint remains unchanged. Restore NAS write access for the client user
UID/GID1026 before retrying; do not change files outside the authorized dataset
path or move the full training pipeline onto local disk without considering space.
Preserve the failed controller record when retrying the verified pause checkpoint.

## NFS mapping correction

The user changed the NFS mapping to "no mapping". A fresh exclusive-file write
and removal in the checkpoint directory now succeeds. Retried from the same pause
record; the controller permits a failed pre-update restart to be retried, while
still requiring an unchanged checkpoint checksum and exact expected saved step.
