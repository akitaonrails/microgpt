"""Tiny repository-repair evaluation through real OpenCode and containerized graders.

This is a regression set, not a benchmark of general coding or long-horizon skill.
The model can edit only solution.py; it cannot change tools, config, or graders.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import uuid
from microgpt.opencode_format import AGENT_PROMPT
from microgpt.programming_data import GENERAL_PROMPT

IMAGE = "python@sha256:dddfd7e07f9d15aeeca61529320492139d21cac7f0070c00609243e51e4e0016"
CHECK_SUCCESS = 73


def completion_evidence(events, code_correct, returncode):
    """A correct file alone does not establish a completed, verified agent task."""
    last_edit = -1
    last_check = None
    for index, event in enumerate(events):
        if event.get('type') != 'tool_use':
            continue
        part = event['part']
        if part.get('tool') in ('edit', 'write', 'apply_patch'):
            last_edit = index
        elif part.get('tool') == 'check':
            last_check = (index, part.get('state', {}))
    checked = bool(last_check and last_check[0] > last_edit
                   and last_check[1].get('status') == 'completed'
                   and str(last_check[1].get('output', '')).startswith('PASS'))
    finishes = [event['part'].get('reason') for event in events
                if event.get('type') == 'step_finish']
    completed = bool(returncode == 0 and finishes and finishes[-1] == 'stop'
                     and not any(event.get('type') == 'error' for event in events))
    return {'code_correct': code_correct, 'checked': checked, 'completed': completed,
            'passed': code_correct and checked and completed}


TASKS = [
    {"id": "mean", "prompt": "Fix mean(values) in solution.py. Return the arithmetic mean; raise ValueError for an empty input. Support negative numbers and floats.",
     "code": "def mean(values):\n    return sum(values) // len(values)\n",
     "tests": "assert s.mean([1, 2]) == 1.5\nassert s.mean([-3, 2]) == -0.5\nassert s.mean([1.5, 2.5]) == 2.0\ntry:\n    s.mean([])\nexcept ValueError:\n    pass\nelse:\n    raise AssertionError('empty input must raise ValueError')\n"},
    {"id": "chunks", "prompt": "Fix chunks(values, size) in solution.py. Return consecutive chunks as a list of lists, including a shorter final chunk. Empty input returns []; nonpositive size raises ValueError.",
     "code": "def chunks(values, size):\n    return [values[i:i + size] for i in range(0, len(values) - size + 1, size)]\n",
     "tests": "assert s.chunks([1,2,3,4,5], 2) == [[1,2],[3,4],[5]]\nassert s.chunks([], 2) == []\nassert s.chunks([1], 4) == [[1]]\nfor n in [0, -1]:\n    try:\n        s.chunks([1,2], n)\n    except ValueError:\n        pass\n    else:\n        raise AssertionError('invalid size must raise ValueError')\n"},
    {"id": "unique", "prompt": "Fix unique(values) in solution.py. Return distinct items in first-occurrence order. Support unhashable items such as lists. Do not mutate the input.",
     "code": "def unique(values):\n    return list(set(values))\n",
     "tests": "assert s.unique([3,1,3,2,1]) == [3,1,2]\nassert s.unique([[1],[2],[1]]) == [[1],[2]]\nassert s.unique([]) == []\nx = [2,1,2]\nassert s.unique(x) == [2,1]\nassert x == [2,1,2]\n"},
    {"id": "counts", "prompt": "Fix count_words(text) in solution.py. Split on whitespace and count words case-insensitively using Unicode casefold. Preserve punctuation as part of each word. Empty or whitespace-only text returns {}.",
     "code": "def count_words(text):\n    return {word: 1 for word in text.split(' ')}\n",
     "tests": "assert s.count_words('One one TWO') == {'one':2,'two':1}\nassert s.count_words('Straße STRASSE') == {'strasse':2}\nassert s.count_words('a\\tb\\na') == {'a':2,'b':1}\nassert s.count_words('  ') == {}\nassert s.count_words('Hi! HI! hi') == {'hi!':2,'hi':1}\n"},
]


def grounding_tasks():
    """Fresh instances of trained arithmetic templates; not broad coding skill."""
    from microgpt.grounding import episode
    tasks = []
    for index in range(8):
        fixture = episode(700000 + index, [])['fixture']
        tests = ''
        for value in (-7, 0, 3, 2.5):
            expected = eval(fixture['expression'], {}, {'value': value})
            tests += f"assert s.{fixture['name']}({value!r}) == {expected!r}\n"
        tasks.append({'id': f'grounding-{index}', 'code': fixture['before'], 'tests': tests,
                      'prompt': f"Fix {fixture['name']}(value) in solution.py. It must return {fixture['expression']}."})
    return tasks


def docker_command(work, grader, name):
    return ["docker", "run", "--rm", "--name", name, "--network", "none", "--read-only",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--pids-limit", "64",
            "--memory", "256m", "--cpus", "1", "--user", "65534:65534", "--tmpfs", "/tmp:rw,noexec,size=16m",
            "--mount", f"type=bind,src={work},dst=/work,readonly",
            "--mount", f"type=bind,src={grader},dst=/grader.py,readonly", IMAGE,
            "python", "-I", "-B", "-c",
            "import runpy; runpy.run_path('/grader.py', run_name='__main__'); "
            f"raise SystemExit({CHECK_SUCCESS})"]


def grade(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=20)
        return {"passed": result.returncode == CHECK_SUCCESS, "output": (result.stdout + result.stderr)[-4000:]}
    except subprocess.TimeoutExpired:
        return {"passed": False, "output": "Grader timed out"}
    finally:
        subprocess.run(["docker", "rm", "-f", command[command.index("--name") + 1]], capture_output=True)


def configure(work, grader, url, steps, state_file=False, reasoning_effort=None, agent_prompt=AGENT_PROMPT):
    """Expose only the exercise files and a fixed, isolated checker to OpenCode."""
    root = Path(__file__).resolve().parents[1]
    name = 'microgpt-eval-' + uuid.uuid4().hex
    command = docker_command(work, grader, name)
    tools = work / '.opencode' / 'tools'
    tools.mkdir(parents=True)
    fault = grader.with_suffix('.fault')
    tool = '''import { tool } from "@opencode-ai/plugin";
import { execFile } from "node:child_process";
import { existsSync, unlinkSync } from "node:fs";
export default tool({
  description: "Run the fixed Python correctness tests in an isolated container. Call after editing solution.py.",
  args: {},
  async execute() {
    if (existsSync(FAULT)) { unlinkSync(FAULT); return "TRANSIENT_ERROR: Test runner temporarily unavailable. Retry check."; }
    return await new Promise((resolve) => {
      execFile("docker", COMMAND, {timeout: 20000, maxBuffer: 65536}, (error, stdout, stderr) => {
        execFile("docker", ["rm", "-f", NAME], () => {});
        const passed = error?.code === EXPECTED_STATUS;
        resolve((passed ? "PASS\\n" : "FAIL\\n") + (stdout + stderr).slice(-4000));
      });
    });
  }
});
'''.replace('COMMAND', json.dumps(command[1:])).replace('NAME', json.dumps(name)).replace('FAULT', json.dumps(str(fault))).replace('EXPECTED_STATUS', str(CHECK_SUCCESS))
    (tools / 'check.ts').write_text(tool)
    config = json.loads((root / 'examples/opencode/opencode.json').read_text())
    config['provider']['microgpt']['options']['baseURL'] = url
    config['provider']['microgpt']['models']['microgpt']['options']['temperature'] = 0
    if reasoning_effort is not None:
        config['provider']['microgpt']['models']['microgpt']['options']['reasoningEffort'] = reasoning_effort
    config['enabled_providers'] = ['microgpt']
    config['lsp'], config['formatter'] = False, False
    allowed_read = {'*': 'deny', 'solution.py': 'allow', str(work / 'solution.py'): 'allow'}
    if state_file:
        allowed_read.update({'TASK_STATE.md': 'allow', str(work / 'TASK_STATE.md'): 'allow'})
    config['permission'] = {'*': 'deny', 'read': allowed_read,
        'edit': {'*': 'deny', 'solution.py': 'allow', str(work / 'solution.py'): 'allow'}, 'check': 'allow'}
    agent = config['agent']['microgpt']
    agent['steps'], agent['permission'] = steps, config['permission']
    agent['prompt'] = agent_prompt
    (work / 'opencode.json').write_text(json.dumps(config))
    environment = {**os.environ, 'OPENCODE_CONFIG': str(work / 'opencode.json')}
    probe = subprocess.run(['opencode', 'debug', 'agent', 'microgpt', '--pure'], cwd=work, env=environment,
                           capture_output=True, text=True, timeout=30)
    exposed = json.loads(probe.stdout[probe.stdout.index('{'):])['tools']
    if probe.returncode or not all(exposed.get(tool) for tool in ('read', 'edit', 'check')):
        raise RuntimeError(f'Required tools not exposed: {exposed}')
    return command, environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--label", required=True, help="Exact scratch checkpoint provenance for this run")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument('--suite', choices=['basic', 'grounding'], default='basic')
    parser.add_argument('--general-prompt', action='store_true', help='Use the general assistant instruction shared with the public repair curriculum')
    parser.add_argument('--effort', choices=['none', 'low', 'medium', 'high'])
    args = parser.parse_args()
    if args.out.exists() or args.steps < 1:
        parser.error("Use a fresh output directory and positive steps")
    subprocess.run(["docker", "image", "inspect", IMAGE], check=True, stdout=subprocess.DEVNULL)
    root = Path(__file__).resolve().parents[1]
    args.out.mkdir(parents=True)
    tasks = TASKS if args.suite == 'basic' else grounding_tasks()
    summary = {"label": args.label, "image": IMAGE, "steps": args.steps, "suite": args.suite, 'effort': args.effort,
               'agent_prompt': 'general' if args.general_prompt else 'repair_lesson',
               "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "tasks_sha256": hashlib.sha256(json.dumps(tasks, sort_keys=True).encode()).hexdigest(),
               "opencode": subprocess.check_output(["opencode", "--version"], text=True).strip(), "tasks": []}
    for task in tasks:
        with tempfile.TemporaryDirectory(prefix="microgpt-eval-") as directory:
            base = Path(directory)
            work = base / "work"
            work.mkdir(mode=0o755)
            subprocess.run(["git", "init", "-q", str(work)], check=True)
            (work / "solution.py").write_text(task["code"])
            grader = base / "grader.py"
            grader.write_text("import sys\nsys.path.insert(0, '/work')\nimport solution as s\n" + task["tests"] + "print('PASS')\n")
            grader.chmod(0o644)
            command, environment = configure(work, grader, args.url, args.steps, reasoning_effort=args.effort,
                                             agent_prompt=GENERAL_PROMPT if args.general_prompt else AGENT_PROMPT)
            baseline = grade(command)
            if baseline["passed"]:
                raise RuntimeError(f"Broken fixture already passes: {task['id']}")
            started = time.monotonic()
            try:
                result = subprocess.run(["opencode", "run", "--pure", "--title", "MicroGPT repair evaluation", "--dir", str(work), "--agent", "microgpt",
                    "-m", "microgpt/microgpt", "--format", "json", task["prompt"] + " Use read, edit, and check tools."],
                    env=environment, capture_output=True, text=True, timeout=180)
                stdout, stderr, returncode = result.stdout, result.stderr, result.returncode
            except subprocess.TimeoutExpired as error:
                stdout = (error.stdout or b"").decode()
                stderr = (error.stderr or b"").decode() + "\nOpenCode timed out"
                returncode = -1
            events = []
            for line in stdout.splitlines():
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
            calls = [e["part"] for e in events if e.get("type") == "tool_use"]
            final = grade(command)
            row = {"id": task["id"], **completion_evidence(events, final['passed'], returncode), "grade": final["output"],
                   "returncode": returncode, "seconds": round(time.monotonic() - started, 1),
                   "errors": [e.get('error') for e in events if e.get('type') == 'error'],
                   "tools": [{"name": c["tool"], "status": c["state"]["status"],
                              "input": c["state"].get("input"), "error": c["state"].get("error", "")[:500]} for c in calls],
                   "tokens": [e["part"].get("tokens") for e in events if e.get("type") == "step_finish"]}
            summary["tasks"].append(row)
            destination = args.out / task["id"]
            destination.mkdir()
            (destination / "solution.py").write_text((work / "solution.py").read_text())
            (destination / "events.jsonl").write_text(stdout)
            (destination / "stderr.txt").write_text(stderr)
            (args.out / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(json.dumps(row), flush=True)
    print(f"Passed {sum(row['passed'] for row in summary['tasks'])}/{len(tasks)}", flush=True)


if __name__ == "__main__":
    main()
