"""Check a real OpenCode read-tool round trip against an already running local server."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "examples/opencode/opencode.json").read_text())
    config["provider"]["microgpt"]["options"]["baseURL"] = args.url
    config["enabled_providers"] = ["microgpt"]
    config["permission"] = {"*": "deny", "read": "allow"}
    config["agent"]["microgpt"]["steps"] = 4
    with tempfile.TemporaryDirectory(prefix="microgpt-opencode-") as directory:
        path = Path(directory)
        (path / "hello.py").write_text("def add(a, b):\n    return a - b\n")
        (path / "opencode.json").write_text(json.dumps(config))
        environment = {**os.environ, "OPENCODE_CONFIG": str(path / "opencode.json")}
        result = subprocess.run([
            "opencode", "run", "--pure", "--dir", directory, "--agent", "microgpt",
            "-m", "microgpt/microgpt", "--format", "json",
            "Use the read tool to read hello.py, then explain the bug in one sentence. Do not edit any files.",
        ], env=environment, capture_output=True, text=True, timeout=120)
        events = []
        for line in result.stdout.splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        reads = [event for event in events if event.get("type") == "tool_use"
                 and event["part"].get("tool") == "read"
                 and event["part"]["state"]["status"] == "completed"]
        replies = [event["part"]["text"] for event in events if event.get("type") == "text"]
        steps = [event["part"].get("tokens", {}) for event in events if event.get("type") == "step_finish"]
        if result.returncode or not reads or not replies:
            raise SystemExit(f"OpenCode smoke failed (exit {result.returncode}):\n{result.stdout}\n{result.stderr}")
        print(json.dumps({"completed_read_calls": len(reads), "replies": replies, "steps": steps}, indent=2))
        if "ERR_STREAM_DESTROYED" in result.stderr:
            print("Note: OpenCode logged ERR_STREAM_DESTROYED during shutdown after completing the round trip.")


if __name__ == "__main__":
    main()
