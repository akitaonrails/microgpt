"""Prepare bounded JSONL documents, split before tokenization, write token arrays."""
import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np

from .tokenizer import Tokenizer, train_bpe
from .protocol import tokenizer_identity


def demo_documents():
    """Original, deliberately narrow Python patterns; no downloaded data required."""
    for operation, symbol in (("add", "+"), ("subtract", "-"), ("multiply", "*")):
        for number in range(100):
            for variable in ("x", "value", "number", "n"):
                yield {"text": f"def {operation}_{number}({variable}):\n    return {variable} {symbol} {number}\n"}


def read_documents(path):
    with Path(path).open() as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def prepare(documents, output, vocab_size=0, limit=20000, encoding=None):
    if encoding and vocab_size:
        raise ValueError("Choose --encoding or --vocab-size")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    splits = {"train": [], "val": []}
    seen = set()
    for index, record in enumerate(documents):
        if index >= limit:
            break
        text = record["text"]
        if not isinstance(text, str) or not text.strip():
            continue
        digest = hashlib.sha256(text.encode()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        # A repo/problem id can keep related documents in the same partition.
        group = str(record.get("group", digest))
        partition = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 20
        splits["val" if partition == 0 else "train"].append(text)
    if any(not texts for texts in splits.values()):
        raise ValueError("Need documents in both partitions; use a larger input")
    tokenizer = train_bpe(splits["train"], vocab_size) if vocab_size else Tokenizer(encoding=encoding)
    tokenizer.save(output)
    metadata = {"vocab_size": tokenizer.vocab_size, "eos": tokenizer.eos,
                "tokenizer_fingerprint": tokenizer_identity(tokenizer),
                "tokenizer": encoding or ("bpe" if vocab_size else "bytes"), "splits": {}}
    for name, texts in splits.items():
        random.Random(42).shuffle(texts)
        count = 0
        hasher = hashlib.sha256()
        with (output / f"{name}.bin").open("wb") as target:
            for text in texts:
                encoded = np.asarray(tokenizer.encode(text) + [tokenizer.eos], dtype="<u4").tobytes()
                target.write(encoded)
                hasher.update(encoded)
                count += len(encoded) // 4
        metadata["splits"][name] = {"documents": len(texts), "tokens": count,
                                    "sha256": hasher.hexdigest()}
    (output / "meta.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help='JSONL with {"text": "...", "group": "optional repo/problem id"}')
    parser.add_argument("--out", type=Path, default=Path("data/demo"))
    parser.add_argument("--vocab-size", type=int, default=0, help="0: UTF-8 bytes; >=257: train BPE on train split")
    parser.add_argument("--limit", type=int, default=20000)
    parser.add_argument("--encoding", choices=["o200k_base", "o200k_harmony"], help="Scratch models only; vocabulary choice does not enable Harmony chat serialization")
    args = parser.parse_args()
    if args.limit < 1 or (args.vocab_size and args.vocab_size < 257):
        parser.error("limit must be positive; BPE vocab must be >=257")
    prepare(read_documents(args.input) if args.input else demo_documents(),
            args.out, args.vocab_size, args.limit, args.encoding)


if __name__ == "__main__":
    main()
