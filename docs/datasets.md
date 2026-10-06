# Datasets and storage

## Larger prepared selection

The new selection contains 947,532,764 pretraining train tokens from CodeParrot,
OpenCoder, Cosmopedia, SmolTalk and multilingual SmolTalk2. As of October 6, 2026,
the pinned raw-source manifests account for 20.40 GB (decimal), including 6.86 GB
of OpenCodeInstruct and 7.74 GB of CodeParrot. These are downloaded source bytes,
not all prepared or consumed training data.
See [the source proportions and reproducible commands](training-plan.md).
These are selected subsets; downloaded bytes, prepared tokens and tokens actually
consumed by training are different quantities. Manifests record each one.

The local `data` symlink points to
`/mnt/gigachad/microgpt/dataset/project-data`. Raw shards, dataset caches,
temporary files and prepared token arrays belong under that NAS directory.
Only the authorized `microgpt/dataset` subtree is used on the NAS. Check current
free space before each large preparation; earlier free-space snapshots are not
capacity guarantees.

Before downloads or preparation:

```sh
source scripts/dataset-env.sh
```

The script refuses a missing NAS mount, preventing an accidental local fallback.
It changes environment variables for the current shell only. The model-weight
cache remains separate. Data download programs use the NAS hub cache;
`download_corpus` downloads directly into its destination without a second hub
copy. Raw source shards have pinned revisions, recorded sizes and SHA-256 hashes.
The downloader checks available disk space and limits both file count and bytes.
Rerunning the same selection reuses verified downloads. Its manifest describes
that selection, not every old shard that may remain in the directory.

## Sources selected

| Source | What it contributes | Access / size considerations |
| --- | --- | --- |
| [CodeParrot-clean](https://huggingface.co/datasets/codeparrot/codeparrot-clean) | Deduplicated Python source from GitHub; raw pretraining | Card reports about 50 GB of code / 5.36M files; Hub lists about 12.8 GB compressed. The current manifest contains 32 downloaded shards totaling 7.74 GB. Per-file license retained and filtered. |
| [CommitPackFT](https://huggingface.co/datasets/bigcode/commitpackft) | Before/after files and commit instructions | Selected Python, Ruby and TypeScript shards total 350 MB. Dataset card MIT does not replace individual repository licenses. Ambiguous repository aliases and unselected licenses are excluded. |
| [Magicoder OSS-Instruct](https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K) | Short coding questions and solutions | MIT dataset card; preserve teacher/source restrictions and provenance. The initial teaching sample contains 400 Python pairs. |
| [SWE-smith trajectories](https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories) | Successful repository repair demonstrations | Keep source success labels, complete histories and observations. Labels have not been independently re-executed here. Most examples exceed an 8K budget. |
| [SmolTalk2](https://huggingface.co/datasets/HuggingFaceTB/smoltalk2) | Everyday conversations and multilingual instruction following | Downloaded the `no_think` everyday and multilingual subsets: 0.9 MB and 317 MB compressed. The multilingual source includes Portuguese, Spanish, French, Italian, German, Arabic, Russian and Chinese. Licenses depend on the component; keep the downloaded card and original-source attribution. |

The commands below reproduce the smaller initial download; they are not an
inventory of the current NAS selection. `curate` makes bounded seeded reservoir samples,
preserves repository groups and licenses, rejects incomplete conversations, and
writes provenance beside each output. License filtering is a dataset selection
rule, not a blanket legal conclusion. Original attribution requirements remain.

```sh
.venv/bin/python -m microgpt.download_corpus --source codeparrot --max-files 4 --max-gb 2
.venv/bin/python -m microgpt.download_corpus --source commitpack --max-files 3
.venv/bin/python -m microgpt.download_corpus --source conversation --max-files 1
.venv/bin/python -m microgpt.download_corpus --source multilingual --max-files 2

.venv/bin/python -m microgpt.curate --input "$MICROGPT_DATA_ROOT/raw/codeparrot" \
  --kind codeparrot --out data/code-new.jsonl --limit 5000 --scan-limit 50000
```

Increase shard and byte budgets deliberately for bigger experiments. A sample
from the first few shards is not a representative sample of the whole corpus.
The preparers remain teaching-scale implementations: bounded records are held
in RAM. Streaming downloads do not make their preparation stages out-of-core.

## Independently checked programming answers

The current data experiment selects up to 100,000 single-function answers from
[nvidia/OpenCodeInstruct](https://huggingface.co/datasets/nvidia/OpenCodeInstruct),
using its pinned local 50-shard download. It requires a successful restricted-Docker
reference run and a failing empty-implementation control. Upstream perfect scores
alone are insufficient: 8/1,000 pilot references failed independently. The pilot
retained 992; preparation of the larger selection is tracked in
`data/verified-opencodeinstruct-v1.source.json` when complete.

The dataset card uses CC-BY-4.0. Source revision, answer IDs, transformations,
checksums, and execution-image identity accompany the selection. Supplied tests
can be incomplete or share mistakes with the answer. See
[the experiment history](capacity-pilot.md) for those limits and training results.

## Larger alternatives

[Stack-Edu](https://huggingface.co/datasets/HuggingFaceTB/stack-edu) is the strongest
next pretraining candidate to investigate: educationally filtered code derived
from StarCoder2's corpus. Its card reports 21.8B Python tokens. The roughly 2.5 GB
Python download is **metadata**, not those source files. The official retrieval
recipe fetches content from Software Heritage's S3 bucket; access, transfer cost,
licenses and removal handling need to be established before scaling that route.

[StarCoderData](https://huggingface.co/datasets/bigcode/starcoderdata) and
[The Stack v2 dedup](https://huggingface.co/datasets/bigcode/the-stack-v2-dedup)
are also relevant. Both currently advertise gated access; unauthenticated source
retrieval was restricted when checked. This project does not bypass those gates.
Do not substitute an unverified mirror simply because it exposes the same name.

The [Python-Edu subset of SmolLM-Corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus#python-edu)
also distributes file identifiers, repository paths and educational scores; its
card retrieves source contents separately from Software Heritage S3. It is not a
direct-content shortcut around the Stack-Edu retrieval requirement.

Large raw-code collections teach code distribution. They do not replace
conversation examples, schema-conditioned tool calls, repair trajectories, or
independent executable evaluations. Start with measured adaptation; storing a
terabyte is easier than getting useful learning from it on one GPU.

For multilingual conversation, reservoir sampling covers both full selected
SmolTalk2 shards rather than taking their first few rows (which begin in Portuguese).
The tiny training sample does not guarantee equal language representation. Use
explicit language labels/quotas and per-language evaluations before claiming
balanced multilingual quality. Rejecting unfinished conversations preserves whole
assistant answers; it does not certify factual correctness of the teacher text.
