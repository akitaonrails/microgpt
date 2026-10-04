"""Stream verified source shards into token files; keep text off the Python heap.

Each source gets its own train/val stream. Training samples these streams with
explicit token proportions instead of letting the largest download win.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
from itertools import islice
import json
from pathlib import Path

import numpy as np

from .curate import convert, rows
from .protocol import body, tokenizer_identity
from .tokenizer import Tokenizer


def documents(directory, kind, counts):
    manifest = json.loads((directory / "manifest.json").read_text())
    for entry in manifest["files"]:
        path = (directory / entry["path"]).resolve()
        if not path.is_relative_to(directory.resolve()):
            raise ValueError("Source path escapes its directory")
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != entry["sha256"]:
                raise ValueError(f"Changed source: {path}")
        for row in rows(path):
            counts["scanned"] += 1
            record, reason = convert(row, kind)
            if record is None:
                counts[reason] += 1
                continue
            text = record.get("text")
            if text is None:
                text = "\n".join(f"{m['role']}:\n{body(m)}" for m in record["messages"])
            yield text, str(record["group"])


def prepare_source(directory, kind, output, max_tokens, encoding="o200k_base", workers=8):
    output.mkdir(parents=True, exist_ok=False)
    tokenizer = Tokenizer(encoding=encoding)
    tokenizer.save(output)
    counts, seen = Counter(), set()
    totals = {name: Counter() for name in ("train", "val")}
    hashes = {name: hashlib.sha256() for name in totals}
    streams = {name: (output / f"{name}.bin").open("wb") for name in totals}
    def encode(item):
        text, group = item
        return hashlib.sha256(text.encode()).digest(), group, tokenizer.encode(text) + [tokenizer.eos]
    try:
        iterator = documents(directory, kind, counts)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            while totals["train"]["tokens"] + totals["val"]["tokens"] < max_tokens:
                batch = list(islice(iterator, 256))
                if not batch:
                    break
                for identity, group, tokens in pool.map(encode, batch):
                    if identity in seen:
                        counts["duplicates"] += 1
                        continue
                    seen.add(identity)
                    split = "val" if int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 20 == 0 else "train"
                    raw = np.asarray(tokens, dtype="<u4").tobytes()
                    streams[split].write(raw)
                    hashes[split].update(raw)
                    totals[split].update(tokens=len(tokens), documents=1)
                if len(seen) % 25600 == 0:
                    print(json.dumps({"source": kind, "documents": len(seen), "tokens": sum(x["tokens"] for x in totals.values())}), flush=True)
        if any(x["tokens"] < 32769 for x in totals.values()):
            raise ValueError("Both splits need enough tokens for the long-context stage")
    finally:
        for stream in streams.values():
            stream.close()
    metadata = {"vocab_size": tokenizer.vocab_size, "eos": tokenizer.eos, "tokenizer": encoding,
                "tokenizer_fingerprint": tokenizer_identity(tokenizer), "source": kind,
                "source_manifest": json.loads((directory / "manifest.json").read_text()),
                "counts": dict(counts), "requested_tokens": max_tokens, "splits": {
                    name: {**totals[name], "sha256": hashes[name].hexdigest()} for name in totals}}
    (output / "meta.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"source": kind, "splits": metadata["splits"]}), flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("/mnt/gigachad/microgpt/dataset/raw"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source", choices=["codeparrot", "cosmopedia", "fineweb", "opencoder", "smoltalk", "multilingual"], required=True)
    parser.add_argument("--tokens", type=int, required=True, help="Approximate source budget; finishes the last 256-document batch")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if min(args.tokens, args.workers) < 1:
        parser.error("Counts must be positive")
    prepare_source(args.raw / args.source, args.source, args.out, args.tokens, workers=args.workers)


if __name__ == "__main__":
    main()
