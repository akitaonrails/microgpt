"""Bounded public Python instruction and successful SWE-smith trajectory import."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

# Configure the dataset process before Hugging Face modules bind cache defaults.
if os.environ.get("MICROGPT_DATA_HUB_CACHE"):
    os.environ["HF_HUB_CACHE"] = os.environ["MICROGPT_DATA_HUB_CACHE"]

SOURCES = {
    "magicoder": ("ise-uiuc/Magicoder-OSS-Instruct-75K", "5f839b1f368a76b161028bb9edff055db34022b2", "train"),
    "swesmith": ("SWE-bench/SWE-smith-trajectories", "08e109b4a59eaeebf80e4675cd125d42e7ac99a4", "tool"),
}


def function(name, properties, required):
    return {"type": "function", "function": {"name": name, "description": f"{name} a repository file or command",
            "parameters": {"type": "object", "properties": properties, "required": required, "additionalProperties": False}}}


STRING = {"type": "string"}
TOOLS = [
    function("read", {"filePath": STRING, "offset": {"type": "integer", "minimum": 1},
                      "limit": {"type": "integer", "minimum": 1}}, ["filePath"]),
    function("edit", {"filePath": STRING, "oldString": STRING, "newString": STRING}, ["filePath", "oldString", "newString"]),
    function("write", {"filePath": STRING, "content": STRING}, ["filePath", "content"]),
    function("bash", {"command": STRING, "description": STRING}, ["command", "description"]),
]


def text_content(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list) and all(part.get("type") == "text" for part in content):
        return "\n".join(part["text"] for part in content)
    raise ValueError("Non-text content")


def convert_swesmith(row):
    if row["resolved"] is not True:
        return None, "unresolved"
    raw = json.loads(row["messages"])
    if not any(message.get('tool_calls') for message in raw):
        return None, 'no_structured_tool_calls'
    messages = []
    used_tools = set()
    for index, original in enumerate(raw):
        role = original["role"]
        message = {"role": role, "content": text_content(original.get("content"))}
        if role == 'system':
            # Source system prompts describe str_replace_editor/XML formatting,
            # which conflicts with the translated native read/edit/write tools.
            message['content'] = ('You are a coding assistant. Use only the supplied tools. '
                'Inspect the repository, make the requested change, and run appropriate tests. '
                'Use exact file contents for edits; displayed line numbers are annotations. '
                'Recover from tool failures. Do not change tests to hide failures. '
                'Report the actual result without claiming unperformed checks.')
        if role == "tool":
            ids = original.get("tool_call_ids", [])
            call_id = original.get("tool_call_id") or (ids[0] if len(ids) == 1 else None)
            if not call_id:
                return None, "ambiguous_result"
            message["tool_call_id"] = call_id
        calls = original.get("tool_calls") or []
        converted = []
        for call in calls:
            name = call["function"]["name"]
            args = call["function"]["arguments"]
            args = json.loads(args) if isinstance(args, str) else args
            if name == "submit":
                if args or len(calls) != 1:
                    return None, "unsupported_submit"
                if index == len(raw) - 1:
                    message["content"] = message["content"] or "Changes are ready."
                    continue
                return None, 'intermediate_source_submit'
            elif name == "bash" and set(args) == {"command"}:
                arguments = {**args, "description": "Run repository command"}
            elif name == "str_replace_editor":
                command = args.get("command")
                if command == "view" and set(args) <= {"command", "path", "view_range"}:
                    name, arguments = "read", {"filePath": args["path"]}
                    span = args.get("view_range")
                    if span:
                        if len(span) != 2 or span[0] < 1 or span[1] < span[0]:
                            return None, "unsupported_view_range"
                        arguments.update(offset=span[0], limit=span[1] - span[0] + 1)
                elif command == "str_replace" and set(args) == {"command", "path", "old_str", "new_str"}:
                    name, arguments = "edit", {"filePath": args["path"], "oldString": args["old_str"], "newString": args["new_str"]}
                elif command == "create" and set(args) == {"command", "path", "file_text"}:
                    name, arguments = "write", {"filePath": args["path"], "content": args["file_text"]}
                else:
                    return None, "unsupported_editor_action"
            else:
                return None, "unsupported_tool"
            used_tools.add(name)
            converted.append({"id": call["id"], "type": "function", "function": {"name": name, "arguments": arguments}})
        if converted:
            message["tool_calls"] = converted
        messages.append(message)
    if not used_tools:
        return None, 'no_native_actions'
    return {"id": row["traj_id"], "group": row["instance_id"].split(".")[0],
            "source": "swesmith", "messages": messages,
            "tools": [tool for tool in TOOLS if tool["function"]["name"] in used_tools]}, None


def convert_magicoder(row):
    if row["lang"].lower() not in ("python", "py"):
        return None, "other_language"
    group = hashlib.sha256(row["seed"].encode()).hexdigest()
    return {"id": str(row["index"]), "group": group, "source": "magicoder", "messages": [
        {"role": "user", "content": row["problem"]}, {"role": "assistant", "content": row["solution"]}], "tools": []}, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=SOURCES, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=500, help="Maximum accepted records")
    parser.add_argument("--scan-limit", type=int, default=5000, help="Maximum examined source records")
    parser.add_argument("--local", type=Path, help="Verified local shards from microgpt.download_corpus")
    args = parser.parse_args()
    if min(args.limit, args.scan_limit) < 1:
        parser.error("Limits must be positive")
    if args.out.exists():
        parser.error("Output exists")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    name, revision, split = SOURCES[args.source]
    options = {}
    if args.source == "swesmith":
        from pyarrow.dataset import ParquetFragmentScanOptions
        options["fragment_scan_options"] = ParquetFragmentScanOptions(pre_buffer=False)
        options["batch_size"] = 16
    if args.local:
        from .curate import rows
        manifest = json.loads((args.local / 'manifest.json').read_text())
        if (manifest['dataset'], manifest['revision']) != (name, revision):
            raise ValueError('Local source revision differs')
        def local_rows():
            for entry in manifest['files']:
                path = (args.local / entry['path']).resolve()
                if not path.is_relative_to(args.local.resolve()):
                    raise ValueError('Source path escapes dataset directory')
                with path.open('rb') as stream:
                    if hashlib.file_digest(stream, 'sha256').hexdigest() != entry['sha256']:
                        raise ValueError('Changed source shard')
                yield from rows(path)
        records = local_rows()
    else:
        from datasets import load_dataset
        records = load_dataset(name, revision=revision, split=split, streaming=True, **options)
    convert = convert_magicoder if args.source == "magicoder" else convert_swesmith
    counts = Counter()
    temporary = args.out.with_suffix(".partial")
    iterator = iter(records)
    with temporary.open("w") as output:
        for row in iterator:
            counts["scanned"] += 1
            record, reason = convert(row)
            if record is None:
                counts[reason] += 1
            else:
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                counts["accepted"] += 1
            if counts["scanned"] >= args.scan_limit or counts["accepted"] >= args.limit:
                break
    iterator.close()
    if not counts["accepted"]:
        raise ValueError(f"No compatible records: {dict(counts)}")
    temporary.replace(args.out)
    with args.out.open('rb') as stream:
        checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
    manifest = {"dataset": name, "revision": revision, "split": split, "license": "MIT (dataset card)",
                "counts": dict(counts), "sha256": checksum,
                "note": "Teacher labels are not independently re-executed; source tool observation formatting is retained"}
    args.out.with_suffix(".source.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
