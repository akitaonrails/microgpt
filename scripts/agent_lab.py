"""A persistent, restricted OpenCode workspace for the cumulative-function lesson."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import uuid

from eval_horizon import grader_text, requirements, state_text, task_prompt
from eval_opencode import IMAGE, configure, docker_command, grade


def prepare_round(base, state):
    items = state['tasks'][:state['round']]
    (base / 'grader.py').write_text(grader_text(items))
    (base / 'work' / 'TASK_STATE.md').write_text(state_text(items))
    if state['round'] == 4:
        (base / 'grader.fault').write_text('fail one check')
    (base / 'state.json').write_text(json.dumps(state, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['init', 'show', 'check', 'next'])
    parser.add_argument('--out', type=Path, default=Path('runs/agent-lab'))
    parser.add_argument('--url', default='http://127.0.0.1:8000/v1')
    parser.add_argument('--seed', type=int, default=314159)
    parser.add_argument('--rounds', type=int, default=16)
    args = parser.parse_args()
    base = args.out.resolve()
    work = base / 'work'
    if args.action == 'init':
        if base.exists() or args.rounds < 1:
            parser.error('Initialization needs a fresh directory and positive rounds')
        subprocess.run(['docker', 'image', 'inspect', IMAGE], check=True, stdout=subprocess.DEVNULL)
        work.mkdir(parents=True)
        subprocess.run(['git', 'init', '-q', str(work)], check=True)
        (work / 'solution.py').write_text('# Keep earlier calibration functions when adding new ones.\n')
        state = {'round': 1, 'seed': args.seed, 'tasks': requirements(args.seed, args.rounds)}
        prepare_round(base, state)
        configure(work, base / 'grader.py', args.url, 12, state_file=True, reasoning_effort='none')
    else:
        state = json.loads((base / 'state.json').read_text())
    if args.action in ('check', 'next'):
        result = grade(docker_command(work, base / 'grader.py', 'microgpt-lab-' + uuid.uuid4().hex))
        print(result['output'], end='')
        if not result['passed']:
            raise SystemExit(1)
        if args.action == 'check':
            return
        if state['round'] == len(state['tasks']):
            print('All cumulative requirements passed. This checks code, not the quality of the conversation.')
            return
        state['round'] += 1
        prepare_round(base, state)
    print(f"Round {state['round']}/{len(state['tasks'])}. Send this prompt in OpenCode:\n")
    print(task_prompt(state['tasks'][state['round'] - 1]))
    print('\nOpen the workspace:')
    print(shlex.join(['opencode', str(work), '--pure', '--agent', 'microgpt', '-m', 'microgpt/microgpt']))
    print('Keep the same session between rounds; halfway through, try a fresh session using TASK_STATE.md.')


if __name__ == '__main__':
    main()
