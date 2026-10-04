"""Turn verified NAS shards into bounded, provenance-preserving training JSONL."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import random

PERMISSIVE = {"mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "isc", "cc0-1.0", "unlicense"}


def rows(path):
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        for batch in pq.ParquetFile(path).iter_batches(batch_size=128):
            yield from batch.to_pylist()
    else:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt") as source:
            for line in source:
                yield json.loads(line)


def convert(row, kind):
    if kind == "opencodeinstruct":
        try:
            score = float(row["average_test_score"])
        except (TypeError, ValueError):
            return None, "missing_score"
        if score != 1.0:
            return None, "upstream_tests_failed"
        prompt, answer = row["input"], row["output"]
        return {"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}],
                "tools": [], "source": kind, "group": hashlib.sha256(prompt.encode()).hexdigest(),
                "license": "CC-BY-4.0", "upstream_id": row["id"], "upstream_test_score": score,
                "verification": "Upstream score only; downloaded code/tests never executed here."}, None
    if kind == "code-feedback":
        messages = row["messages"]
        if not messages or messages[-1]["role"] != "assistant" or any(
                m["role"] not in ("user", "assistant") or not isinstance(m["content"], str) for m in messages):
            return None, "unsupported_conversation"
        return {"messages": messages, "tools": [], "source": kind, "license": "Apache-2.0",
                "group": hashlib.sha256(messages[0]["content"].encode()).hexdigest(), "upstream_id": row["id"]}, None
    if kind == "opencoder-educational":
        prompt = row["instruction"]
        return {"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": row["output"]}],
                "tools": [], "source": kind, "license": "MIT", "upstream_id": row["seq_id"],
                "group": hashlib.sha256(prompt.encode()).hexdigest()}, None
    if kind in ("cosmopedia", "fineweb"):
        text = row["text"]
        if not 100 <= len(text) <= 100000:
            return None, "length"
        # seed_data is a category such as "fineweb", NOT a document identifier.
        seed = row.get("prompt") or row.get("id") or text
        return {"text": text, "group": hashlib.sha256(str(seed).encode()).hexdigest(), "source": kind}, None
    if kind == "opencoder":
        return {"messages": [{"role": "user", "content": row["instruction"]},
                             {"role": "assistant", "content": row["output"]}], "tools": [],
                "source": kind, "group": hashlib.sha256(row["instruction"].encode()).hexdigest()}, None
    if kind in ("conversation", "multilingual", "smoltalk"):
        kwargs = row.get("chat_template_kwargs") or {}
        if kwargs.get("python_tools") or kwargs.get("xml_tools") or kwargs.get("enable_thinking"):
            return None, "unsupported_template"
        messages = row["messages"]
        if not messages or messages[-1]["role"] != "assistant":
            return None, "unfinished_conversation"
        if any(m["role"] not in ("system", "user", "assistant") or not isinstance(m["content"], str) for m in messages):
            return None, "unsupported_message"
        instruction = kwargs.get("custom_instructions")
        if instruction:
            messages = [{"role": "system", "content": instruction}, *messages]
        first = next(m["content"] for m in messages if m["role"] == "user")
        return {"messages": messages, "tools": [], "source": kind,
                "group": hashlib.sha256(first.encode()).hexdigest()}, None
    license = row.get("license", "").lower()
    if license not in PERMISSIVE:
        return None, "license_not_selected"
    if kind == "codeparrot":
        if not 100 <= len(row["content"]) <= 32000:
            return None, "length"
        return {"text": row["content"], "group": row["repo_name"], "path": row["path"], "license": license,
                "source": kind}, None
    # Preserve commit provenance; the original edit is a target, not a claim that
    # its tests passed. Multiple repository aliases are conservatively excluded.
    if "," in row["repos"] or not row["repos"]:
        return None, "ambiguous_repository"
    if not row["old_contents"] or not row["new_contents"]:
        return None, "empty_file"
    return {"id": row["commit"], "group": row["repos"], "license": license, "source": kind, "tools": [],
            "messages": [{"role": "user", "content": f"Update {row['old_file']}: {row['subject']}\nReturn the complete updated file.\n\n{row['old_contents']}"},
                         {"role": "assistant", "content": row["new_contents"]}]}, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Directory produced by download_corpus")
    parser.add_argument("--kind", choices=["opencodeinstruct", "code-feedback", "opencoder-educational", "codeparrot", "commitpack", "conversation", "multilingual", "smoltalk", "opencoder", "cosmopedia", "fineweb"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--scan-limit", type=int, default=300000)
    args = parser.parse_args()
    if args.out.exists() or min(args.limit, args.scan_limit) < 1:
        parser.error("Use a fresh output path and positive limits")
    manifest = json.loads((args.input / "manifest.json").read_text())
    counts = Counter()
    reservoir = []
    rng = random.Random(42)
    for entry in manifest["files"]:
        path = (args.input / entry["path"]).resolve()
        if not path.is_relative_to(args.input.resolve()):
            raise ValueError("Shard escapes the dataset directory")
        with path.open("rb") as source:
            if hashlib.file_digest(source, "sha256").hexdigest() != entry["sha256"]:
                raise ValueError(f"Changed source shard: {path}")
        for row in rows(path):
            if counts["scanned"] >= args.scan_limit:
                break
            counts["scanned"] += 1
            record, reason = convert(row, args.kind)
            if record is None:
                counts[reason] += 1
                continue
            counts["eligible"] += 1
            record["dataset"] = manifest["dataset"]
            record["revision"] = manifest["revision"]
            if len(reservoir) < args.limit:
                reservoir.append(record)
            else:
                index = rng.randrange(counts["eligible"])
                if index < args.limit:
                    reservoir[index] = record
        if counts["scanned"] >= args.scan_limit:
            break
    if not reservoir:
        raise ValueError(f"No retained records: {dict(counts)}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as output:
        for record in reservoir:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    with args.out.open('rb') as stream:
        checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
    report = {"source_manifest": manifest, "counts": dict(counts), "retained": len(reservoir), "seed": 42,
              "sha256": checksum}
    args.out.with_suffix(".source.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"counts": dict(counts), "retained": len(reservoir)}, indent=2))


if __name__ == "__main__":
    main()
