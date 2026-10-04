"""Prepare complete assistant turns using the exact serving template and frozen tokenizer."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from .tokenizer import Tokenizer
from .chat import ChatRequest, normalize
from .protocol import digest, prompt_ids, body, tokenizer_identity, VERSION


def assistant_examples(record, tokenizer, max_length, pack=False):
    messages, tools = normalize(ChatRequest(model="training", messages=record["messages"], tools=record.get("tools", [])))
    if messages[-1]["role"] != "assistant" or messages[-1].get("tool_calls"):
        raise ValueError("A trajectory must end with a final assistant answer")
    examples = []
    if pack:
        # Match prompt_ids segment boundaries exactly, but supervise every
        # assistant body in one causal pass through the complete episode.
        ids = prompt_ids(tokenizer, [], tools)
        ids = ids[:-len(tokenizer.encode('assistant:\n'))]
        labels = [-100] * len(ids)
        for message in messages:
            header = tokenizer.encode(f"{message['role']}:\n")
            target = tokenizer.encode(body(message)) + [tokenizer.eos]
            ids += header + target
            labels += [-100] * len(header) + (target if message['role'] == 'assistant' and message.get('supervise', True) else [-100] * len(target))
        if len(ids) > max_length:
            return []
        return [{"input_ids": ids, "labels": labels}]
    for index, message in enumerate(messages):
        if message["role"] != "assistant" or not message.get('supervise', True):
            continue
        if not index:
            raise ValueError("An assistant turn needs context")
        prefix = prompt_ids(tokenizer, messages[:index], tools)
        target = tokenizer.encode(body(message)) + [tokenizer.eos]
        if len(prefix) + len(target) > max_length:
            return []
        examples.append({"input_ids": prefix + target, "labels": [-100] * len(prefix) + target})
    return examples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True, help="Prepared pretraining data or checkpoint directory")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--limit", type=int, default=2000, help="Maximum input records across all sources")
    parser.add_argument("--pack-episodes", action="store_true", help="Supervise all assistant turns in one causal pass; no repeated prefixes")
    args = parser.parse_args()
    if args.max_length < 2 or args.limit < 1:
        parser.error("Limits must be positive, with max-length >= 2")
    if args.out.exists():
        parser.error("Output exists; choose a fresh directory")
    tokenizer = Tokenizer.load(args.tokenizer)
    args.out.mkdir(parents=True)
    (args.out / "tokenizer").mkdir()
    tokenizer.save(args.out / "tokenizer")
    seen = set()
    counts = Counter()
    split_groups = {"train": set(), "val": set()}
    totals = {name: Counter() for name in ("train", "val")}
    by_source = {name: Counter() for name in totals}
    handles = {name: (args.out / f"{name}.jsonl").open("w") for name in totals}
    sources = []
    for path in args.input:
        with path.open('rb') as source:
            sources.append({"file": str(path), "sha256": hashlib.file_digest(source, 'sha256').hexdigest()})
        with path.open() as source:
            for line in source:
                if counts["scanned"] >= args.limit:
                    break
                counts["scanned"] += 1
                record = json.loads(line)
                identity = digest({"messages": record["messages"], "tools": record.get("tools", [])})
                if identity in seen:
                    counts["duplicates"] += 1
                    continue
                seen.add(identity)
                group = str(record.get("group") or record.get("id") or identity)
                split = "val" if int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 20 == 0 else "train"
                try:
                    examples = assistant_examples(record, tokenizer, args.max_length, pack=args.pack_episodes)
                except (ValueError, KeyError, TypeError) as error:
                    raise ValueError(f"Invalid record {record.get('id', identity)}: {error}") from error
                if not examples:
                    counts["overlength_or_empty"] += 1
                    continue
                split_groups[split].add(group)
                counts[split + "_episodes"] += 1
                for example in examples:
                    source_name = record.get("source", "local")
                    item = {**example, "group": group, "episode": identity, "source": source_name,
                            "tool_calls": sum(len(m.get('tool_calls', [])) for m in record['messages'])}
                    handles[split].write(json.dumps(item) + '\n')
                    totals[split].update(examples=1, assistant_tokens=sum(x != -100 for x in example['labels']))
                    by_source[split][source_name] += 1
    for handle in handles.values():
        handle.close()
    if not all(x['examples'] for x in totals.values()):
        raise ValueError("Need retained examples in both splits; increase input diversity/length budget")
    manifest = {"protocol": VERSION, "max_length": args.max_length, "packed_episodes": args.pack_episodes,
                "tokenizer_fingerprint": tokenizer_identity(tokenizer), "sources": sources,
                "counts": dict(counts), "splits": {}}
    for split in totals:
        path = args.out / f"{split}.jsonl"
        with path.open('rb') as source:
            checksum = hashlib.file_digest(source, 'sha256').hexdigest()
        manifest["splits"][split] = {**totals[split], "groups": len(split_groups[split]),
            "examples_by_source": dict(by_source[split]), "sha256": checksum}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
