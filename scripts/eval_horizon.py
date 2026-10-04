"""Real OpenCode across evolving requirements, a tool failure and session handoff.

Procedural integration test, not a general software-engineering benchmark.
Graders run outside the editable repository in a networkless container.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import tempfile
import time

from eval_opencode import IMAGE, configure, docker_command, grade


def requirements(seed, rounds):
    rng = random.Random(seed)
    return [{'name': f'calibrate_{rng.randrange(10000, 99999)}', 'factor': rng.randrange(2, 20), 'offset': rng.randrange(-50, 50)} for _ in range(rounds)]


def grader_text(items):
    text = "import sys\nsys.path.insert(0, '/work')\nimport solution as s\n"
    for item in items:
        for value in [-13, 0, 1, 2.5, 97]:
            text += f"assert s.{item['name']}({value}) == {value * item['factor'] + item['offset']!r}\n"
    return text + "print('PASS')\n"


def state_text(items):
    notes = '\n'.join(f"- {t['name']}(value) must return value * {t['factor']} + {t['offset']}." for t in items)
    return '# Required behavior\n' + notes + '\nPreserve and test every requirement above.\n'


def task_prompt(task):
    return f"In solution.py add {task['name']}(value), returning value * {task['factor']} + {task['offset']}. Keep all earlier functions correct. Read TASK_STATE.md for the requirements, read the code, edit it, and call check. Retry transient tool failures."


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8000/v1')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--steps', type=int, default=12, help='Per-round OpenCode step bound')
    parser.add_argument('--rounds', type=int, default=16)
    parser.add_argument('--seed', type=int, default=849723, help='Held-out task seed, never used by practice generator')
    parser.add_argument('--self-test', action='store_true', help='Verify isolated graders with broken and trusted reference code; no model score')
    parser.add_argument('--effort', choices=['none', 'low', 'medium', 'high'])
    args = parser.parse_args()
    if args.out.exists() or min(args.rounds, args.steps) < 1:
        parser.error('Fresh output and positive counts required')
    args.out.mkdir(parents=True)
    tasks = requirements(args.seed, args.rounds)
    summary = {'label': args.label, 'kind': 'harness-self-test' if args.self_test else 'model-evaluation',
        'seed': args.seed, 'requested_rounds': args.rounds, 'rounds': [], 'image': IMAGE, 'effort': args.effort,
        'harness_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'tasks_sha256': hashlib.sha256(json.dumps(tasks, sort_keys=True).encode()).hexdigest(),
        'session_handoff': 'Fresh session halfway through; trusted harness preserves requirements in TASK_STATE.md',
        'tool_failure': 'One transient checker failure on round 4; recovery requires another check call'}
    with tempfile.TemporaryDirectory(prefix='microgpt-horizon-') as directory:
        base = Path(directory)
        work = base / 'work'
        work.mkdir(mode=0o755)
        subprocess.run(['git', 'init', '-q', str(work)], check=True)
        (work / 'solution.py').write_text('# Keep earlier calibration functions when adding new ones.\n')
        grader = base / 'grader.py'
        grader.write_text(grader_text(tasks[:1]))
        grader.chmod(0o644)
        if args.self_test:
            command = docker_command(work, grader, 'microgpt-horizon-selftest')
            if grade(command)['passed']:
                raise RuntimeError('Broken fixture unexpectedly passed')
            (work / 'solution.py').write_text(''.join(f"def {t['name']}(value):\n    return value * {t['factor']} + {t['offset']}\n\n" for t in tasks))
            grader.write_text(grader_text(tasks))
            reference = grade(command)
            if not reference['passed']:
                raise RuntimeError(reference['output'])
            summary['reference_passed'] = True
        else:
            command, environment = configure(work, grader, args.url, args.steps, state_file=True, reasoning_effort=args.effort)
            session = None
            consecutive_failures = 0
            for index, task in enumerate(tasks):
                grader.write_text(grader_text(tasks[:index + 1]))
                (work / 'TASK_STATE.md').write_text(state_text(tasks[:index + 1]))
                if index == args.rounds // 2:
                    session = None
                if index == 3:
                    grader.with_suffix('.fault').write_text('fail one check')
                prompt = task_prompt(task)
                cli = ['opencode', 'run', '--pure', '--title', 'MicroGPT horizon evaluation', '--dir', str(work), '--agent', 'microgpt', '-m', 'microgpt/microgpt', '--format', 'json']
                if session:
                    cli += ['--session', session]
                started = time.monotonic()
                try:
                    run = subprocess.run(cli + [prompt], env=environment, capture_output=True, text=True, timeout=90)
                    stdout, stderr = run.stdout, run.stderr
                except subprocess.TimeoutExpired as error:
                    stdout, stderr = (error.stdout or b'').decode(), 'OpenCode timed out'
                events = []
                for line in stdout.splitlines():
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
                if events:
                    session = next((e.get('sessionID') for e in events if e.get('sessionID')), session)
                calls = [e['part'] for e in events if e.get('type') == 'tool_use']
                usage = [e['part']['tokens'] for e in events if e.get('type') == 'step_finish' and 'tokens' in e.get('part', {})]
                final = grade(command)
                check_outputs = [str(c['state'].get('output', '')) for c in calls if c.get('tool') == 'check']
                tested = any(output.startswith('PASS') for output in check_outputs)
                finishes = [e['part'].get('reason') for e in events if e.get('type') == 'step_finish']
                completed = bool(finishes and finishes[-1] == 'stop') and not any(e.get('type') == 'error' for e in events)
                row = {'round': index + 1, 'correct': final['passed'], 'checked': tested,
                    'completed': completed, 'passed': final['passed'] and tested and completed,
                    'tool_calls': len(calls), 'session': session,
                    'tools': [{'name': c['tool'], 'status': c['state']['status']} for c in calls],
                    'recovered_transient_error': any('TRANSIENT_ERROR' in s for s in check_outputs) and tested,
                    'cached_tokens': sum(t.get('cache', {}).get('read', 0) for t in usage),
                    'uncached_input_tokens': sum(t.get('input', 0) for t in usage),
                    'output_tokens': sum(t.get('output', 0) for t in usage),
                    'peak_prompt_tokens': max((t.get('input', 0) + t.get('cache', {}).get('read', 0) for t in usage), default=0),
                    'seconds': round(time.monotonic() - started, 1), 'grade': final['output']}
                summary['rounds'].append(row)
                folder = args.out / f'round-{index + 1:02d}'
                folder.mkdir()
                (folder / 'events.jsonl').write_text(stdout)
                (folder / 'stderr.txt').write_text(stderr)
                (folder / 'solution.py').write_text((work / 'solution.py').read_text())
                (args.out / 'results.json').write_text(json.dumps(summary, indent=2) + '\n')
                print(json.dumps(row), flush=True)
                consecutive_failures = 0 if row['passed'] else consecutive_failures + 1
                if consecutive_failures >= 2:
                    summary['stopped_early'] = 'Two failed rounds; remaining rounds count as unmet, not passed.'
                    break
            total_calls = sum(r['tool_calls'] for r in summary['rounds'])
            summary['total_tool_calls'] = total_calls
            summary['passed'] = (len(summary['rounds']) == args.rounds and all(r['passed'] for r in summary['rounds'])
                                 and total_calls >= 2 * args.rounds
                                 and (args.rounds < 4 or any(r['recovered_transient_error'] for r in summary['rounds'])))
    (args.out / 'results.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'rounds'}), flush=True)


if __name__ == '__main__':
    main()
