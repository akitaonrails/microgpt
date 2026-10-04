"""Download a bounded sample of public coding instructions with pinned provenance."""
import argparse
import hashlib
import json
import os
from pathlib import Path

if os.environ.get("MICROGPT_DATA_HUB_CACHE"):
    os.environ["HF_HUB_CACHE"] = os.environ["MICROGPT_DATA_HUB_CACHE"]
from datasets import load_dataset
from huggingface_hub import HfApi

DATASET = "m-a-p/CodeFeedback-Filtered-Instruction"
REVISION = "a08c213a9748c66c15d0225814be80a2e77adf4a"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/codefeedback.jsonl"))
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--revision", default=REVISION)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("limit must be positive")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        parser.error("Output exists; choose a new path")
    revision = HfApi().dataset_info(DATASET, revision=args.revision).sha
    records = load_dataset(DATASET, split="train", revision=revision, streaming=True)
    count = 0
    checksum = hashlib.sha256()
    temporary = args.out.with_suffix(".partial")
    with temporary.open("w") as target:
        for row in records:
            query, answer = row["query"], row["answer"]
            if not query.strip() or not answer.strip():
                continue
            record = {"text": f"### Question\n{query}\n\n### Answer\n{answer}\n",
                      "group": hashlib.sha256(query.strip().encode()).hexdigest()}
            line = json.dumps(record, ensure_ascii=False) + "\n"
            target.write(line)
            checksum.update(line.encode())
            count += 1
            if count >= args.limit:
                break
    if not count:
        raise ValueError("Dataset yielded no usable records")
    temporary.replace(args.out)
    metadata = {"dataset": DATASET, "revision": revision, "split": "train", "documents": count,
                "sha256": checksum.hexdigest(), "license": "apache-2.0 (dataset card)",
                "format": "Full question + answer language modeling; not assistant-only SFT"}
    args.out.with_suffix(".source.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
