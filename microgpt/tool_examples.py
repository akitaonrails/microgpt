"""Derive labeled, one-file OpenCode protocol exercises from real commit pairs.

These are constructed read/edit demonstrations, not captured agent runs. The
replacement is checked against the source target; no test-success claim is made.
"""
import argparse
import json
from pathlib import Path, PurePosixPath


def convert(record):
    user, assistant = record["messages"]
    header, old = user["content"].split("\nReturn the complete updated file.\n\n", 1)
    new = assistant["content"]
    if not old or old == new:
        return None
    relative, instruction = header.removeprefix("Update ").split(": ", 1)
    if PurePosixPath(relative).is_absolute() or ".." in PurePosixPath(relative).parts:
        return None
    path = "/workspace/" + relative
    # Check the exact operation our target call will ask the harness to perform.
    if old.replace(old, new, 1) != new:
        raise ValueError("Replacement does not reproduce the target")
    properties = {name: {"type": "string"} for name in ("filePath", "oldString", "newString")}
    tools = [{"type": "function", "function": {"name": name, "description": description,
        "parameters": {"type": "object", "properties": {key: properties[key] for key in fields},
                       "required": fields, "additionalProperties": False}}}
        for name, description, fields in [
            ("read", "Read a file. Displayed line numbers are not file content.", ["filePath"]),
            ("edit", "Replace an exact source string in a file; omit displayed line numbers.", ["filePath", "oldString", "newString"])]]
    def call(name, arguments, id):
        return {"role": "assistant", "content": "", "tool_calls": [
            {"id": id, "type": "function", "function": {"name": name, "arguments": arguments}}]}
    numbered = "\n".join(f"{i}: {line}" for i, line in enumerate(old.splitlines(), 1))
    messages = [
        {"role": "user", "content": f"Update {path}: {instruction}"},
        call("read", {"filePath": path}, "read_file"),
        {"role": "tool", "tool_call_id": "read_file", "content": f"<path>{path}</path>\n<type>file</type>\n<content>\n{numbered}\n</content>"},
        call("edit", {"filePath": path, "oldString": old, "newString": new}, "edit_file"),
        {"role": "tool", "tool_call_id": "edit_file", "content": "Edit applied successfully."},
        {"role": "assistant", "content": "Updated the file. Tests have not been run."},
    ]
    return {**record, "source": "commitpack-tools", "messages": messages, "tools": tools,
            "construction": "Synthetic protocol exercise from a real commit; exact replacement reproduces target, no execution validation"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    count = 0
    with args.input.open() as source, args.out.open("x") as output:
        for line in source:
            record = convert(json.loads(line))
            if record:
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
    print(f"Constructed {count} commit-based protocol exercises")


if __name__ == "__main__":
    main()
