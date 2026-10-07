"""Independent programming probes, direct and cumulative real-OpenCode modes.

Independent from training generators. Model code executes only in the restricted
Docker grader. These small tests measure behavior, not arbitrary-prompt competence.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
import uuid

import httpx
from eval_opencode import configure, grade, docker_command, completion_evidence, IMAGE
from microgpt.programming_data import GENERAL_PROMPT

TASKS = [
    ('greeting', 'Write greet(name) in Python. Print Hello, NAME! followed by a newline. Return None.',
     "import io, contextlib\nb = io.StringIO()\nwith contextlib.redirect_stdout(b):\n    result = s.greet('Zoë')\nassert result is None\nassert b.getvalue() == 'Hello, Zoë!\\n'\n"),
    ('clamp', 'Implement clamp(value, low, high). Return low below the range, high above it, otherwise value. Assume low <= high.',
     'assert s.clamp(-4, 0, 8) == 0\nassert s.clamp(20, 0, 8) == 8\nassert s.clamp(3, 0, 8) == 3\n'),
    ('initials', 'Implement initials(name): take the first letter of every whitespace-separated word and return them uppercase, without separators. Empty input returns an empty string.',
     "assert s.initials('  ada   lovelace ') == 'AL'\nassert s.initials('') == ''\nassert s.initials('grace') == 'G'\n"),
    ('squares', 'Implement square_evens(values). Return the squares of just the even integers, preserving their order. Do not change the input.',
     'v = [3, -2, 0, 4]\nassert s.square_evens(v) == [4, 0, 16]\nassert v == [3, -2, 0, 4]\nassert s.square_evens([]) == []\n'),
    ('join', 'Implement join_names(names): return the strings joined with a comma and space. Empty input returns an empty string.',
     "assert s.join_names(['a', 'b']) == 'a, b'\nassert s.join_names([]) == ''\nassert s.join_names(['Zoë']) == 'Zoë'\n"),
    ('lookup', 'Implement lookup(mapping, key, default=None). Return the stored value when key exists, otherwise default. A stored None must remain None.',
     "assert s.lookup({'x': None}, 'x', 4) is None\nassert s.lookup({}, 'x', 9) == 9\nassert s.lookup({}, 'x') is None\n"),
    ('strip_lower', 'Implement normalize_label(text): remove whitespace at both ends, then lowercase the result. Keep interior whitespace.',
     "assert s.normalize_label('  Ab C  ') == 'ab c'\nassert s.normalize_label(' ') == ''\n"),
    ('count', 'Implement count_matches(values, target). Count items equal to target. Support lists as items.',
     'assert s.count_matches([[1], [2], [1]], [1]) == 2\nassert s.count_matches([], 3) == 0\n'),
]


# Assisted diagnostics only. Never import these reference answers into training.
REFERENCES = {
    'greeting': 'def greet(name):\n    print(f"Hello, {name}!")\n',
    'clamp': 'def clamp(value, low, high):\n    return min(high, max(low, value))\n',
    'initials': 'def initials(name):\n    return "".join(word[0].upper() for word in name.split())\n',
    'squares': 'def square_evens(values):\n    return [v * v for v in values if v % 2 == 0]\n',
    'join': 'def join_names(names):\n    return ", ".join(names)\n',
    'lookup': 'def lookup(mapping, key, default=None):\n    return mapping.get(key, default)\n',
    'strip_lower': 'def normalize_label(text):\n    return text.strip().lower()\n',
    'count': 'def count_matches(values, target):\n    return sum(v == target for v in values)\n',
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url', default='http://127.0.0.1:8000/v1')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--mode', choices=['direct', 'agent', 'fresh', 'oracle'], default='direct')
    p.add_argument('--limit', type=int, default=len(TASKS))
    p.add_argument('--tasks-file', type=Path, help='Frozen JSONL task set; references are used only in assisted controls')
    p.add_argument('--exact-prompt', action='store_true', help='Diagnostic: do not append the direct-code formatting instruction')
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    records = [json.loads(line) for line in args.tasks_file.open()] if args.tasks_file else None
    all_tasks = [(r['id'], r['prompt'], r['tests']) for r in records] if records else TASKS
    tasks = all_tasks[:args.limit]
    references = {r['id']: r['reference'] for r in records} if records else REFERENCES
    families = {r['id']: r['family'] for r in records} if records else {}
    health = httpx.get(args.url.removesuffix('/v1') + '/health', timeout=10)
    health.raise_for_status()
    summary = {'server': health.json(), 'harness_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'mode': args.mode, 'exact_prompt': args.exact_prompt, 'image': IMAGE, 'tasks_sha256': hashlib.sha256(json.dumps(tasks).encode()).hexdigest(), 'tasks': []}
    with tempfile.TemporaryDirectory(prefix='microgpt-programming-') as folder:
        base = Path(folder); work = base / 'work'; work.mkdir(mode=0o755)
        subprocess.run(['git', 'init', '-q', str(work)], check=True)
        solution = work / 'solution.py'; solution.write_text('# Python utilities\n')
        grader = base / 'grader.py'; grader.write_text('')
        command, env = configure(work, grader, args.url, 12, reasoning_effort='none', agent_prompt=GENERAL_PROMPT)
        session = None
        cumulative = ''
        for name, prompt, tests in tasks:
            if args.mode in ('fresh', 'oracle'):
                session = None
                solution.write_text('# Python utilities\n')
            cumulative = cumulative + tests if args.mode == 'agent' else tests
            grader.write_text("import sys\nsys.path.insert(0, '/work')\nimport solution as s\n" + cumulative + "print('PASS')\n")
            completed, checked, errors = False, False, []
            if args.mode == 'direct':
                response = httpx.post(args.url + '/chat/completions', json={'model': 'microgpt', 'messages': [
                    {'role': 'system', 'content': GENERAL_PROMPT}, {'role': 'user', 'content': prompt if args.exact_prompt else prompt + ' Return only the Python code in a fenced block.'}],
                    'max_tokens': 768, 'temperature': 0, 'reasoning_effort': 'none'}, timeout=90)
                raw = response.text
                solution.write_text('')
                if response.is_success:
                    choice = response.json()['choices'][0]
                    answer = choice['message'].get('content') or ''
                    code = re.search(r'```(?:python)?\s*\n(.*?)```', answer, re.S)
                    solution.write_text(code.group(1) if code else answer)
                    completed = choice['finish_reason'] == 'stop' and not choice['message'].get('tool_calls')
                else:
                    errors.append(raw)
            else:
                cli = ['opencode', 'run', '--pure', '--dir', str(work), '--agent', 'microgpt', '-m', 'microgpt/microgpt', '--format', 'json']
                if session:
                    cli += ['--session', session]
                if args.mode == 'oracle':
                    prompt += '\nApply this supplied reference implementation exactly:\n```python\n' + references[name] + '```'
                try:
                    run = subprocess.run(cli + [prompt + ' Add it to solution.py; preserve existing functions and run check.'],
                        env=env, capture_output=True, text=True, timeout=90)
                    raw = run.stdout
                    cli_returncode = run.returncode
                except subprocess.TimeoutExpired as error:
                    raw = (error.stdout or b'').decode(); errors.append('timeout')
                    cli_returncode = -1
                events = []
                for line in raw.splitlines():
                    try: events.append(json.loads(line))
                    except json.JSONDecodeError: pass
                session = next((e['sessionID'] for e in events if e.get('sessionID')), session)
                errors += [e.get('error') for e in events if e.get('type') == 'error']
            result = grade(command)
            if args.mode != 'direct':
                evidence = completion_evidence(events, result['passed'], cli_returncode)
                completed, checked = evidence['completed'], evidence['checked']
            current = result
            if args.mode == 'agent':
                current_grader = base / 'current.py'
                current_grader.write_text("import sys\nsys.path.insert(0, '/work')\nimport solution as s\n" + tests + "print('PASS')\n")
                current = grade(docker_command(work, current_grader, 'microgpt-current-' + uuid.uuid4().hex))
            row = {'id': name, 'current_task_correct': current['passed'], 'passed': result['passed'] and completed and (checked or args.mode == 'direct'),
                   'code_correct': result['passed'], 'completed': completed, 'checked': checked, 'errors': errors, 'grade': result['output']}
            row['family'] = families.get(name, name)
            summary['tasks'].append(row)
            groups = {}
            for item in summary['tasks']:
                group = groups.setdefault(item['family'], {'passed': 0, 'total': 0})
                group['passed'] += int(item['passed'])
                group['total'] += 1
            summary['families'] = groups
            (args.out / (name + '.txt')).write_text(raw)
            (args.out / (name + '.py')).write_text(solution.read_text())
            (args.out / 'results.json').write_text(json.dumps(summary, indent=2) + '\n')
            print(json.dumps(row), flush=True)
    print(f"Passed {sum(t['passed'] for t in summary['tasks'])}/{len(tasks)}")


if __name__ == '__main__':
    main()
