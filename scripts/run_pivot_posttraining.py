"""Queue a fixed longer SFT comparison and one reserved-holdout evaluation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume-waiting', action='store_true')
    args = parser.parse_args()
    root = Path('runs/pivot-posttraining-v1'); root.mkdir(exist_ok=args.resume_waiting)
    parent = Path('runs/pivot-v1/status.json')
    holdout = Path('evaluation/programming-v1/acceptance.jsonl')
    state = {'status': 'waiting_for_foundation', 'pid': os.getpid(), 'holdout_sha256': sha(holdout),
             'selection': 'Development correctness only; holdout evaluated once after selection',
             'automatic_promotion': False, 'commands': []}
    if args.resume_waiting:
        previous = json.loads((root / 'status.json').read_text())
        if previous['status'] != 'interrupted' or previous['commands']:
            raise ValueError('Only an interrupted waiting queue can resume here')
        if previous['holdout_sha256'] != state['holdout_sha256']: raise ValueError('Holdout changed')
        with (root / 'status-before-maintenance.json').open('x') as f: json.dump(previous, f, indent=2)
    child = None
    stopping = False
    def save():
        temp = root / 'status.tmp';temp.write_text(json.dumps(state, indent=2)+'\n');temp.replace(root/'status.json')
    def stop(*_):
        nonlocal stopping
        stopping = True
        if child is not None and child.poll() is None: child.terminate()
    for sig in (signal.SIGTERM, signal.SIGINT): signal.signal(sig, stop)
    def run(name, command):
        nonlocal child
        if stopping: raise InterruptedError('Stopped')
        state['status'] = name; state['commands'].append(command);save()
        env = dict(os.environ, MICROGPT_GPU_DUTY_CYCLE='0.75')
        env.pop('MICROGPT_FULL_GPU_UNTIL', None)
        with (root / f'{name}.log').open('w') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
            code = child.wait()
        child = None
        if stopping: raise InterruptedError('Stopped')
        if code: raise RuntimeError(f'{name} failed with {code}')
    save()
    try:
        while True:
            if stopping: raise InterruptedError('Stopped')
            upstream = json.loads(parent.read_text())
            if upstream['status'] != 'running': break
            os.kill(upstream['pid'], 0)
            time.sleep(30)
        if upstream['status'] not in ('foundation_gate_unmet', 'foundation_comparison_complete'):
            raise RuntimeError('Foundation pipeline requires recovery before further GPU work')
        candidates = []
        for milestone in upstream['milestones']:
            name = str(round(milestone['tokens'] / 1e9)) + 'b'
            path = Path(milestone.get('recovery_evidence', f'runs/pivot-v1/coding-{name}/status.json'))
            evidence = json.loads(path.read_text())
            for row in evidence['stages']:
                if row['status'] == 'evaluated':
                    candidates.append({'score': row['direct']['passed'], 'checkpoint': row['checkpoint'], 'name': name})
        best = max(candidates, key=lambda r: r['score'])
        state['foundation_choice'] = best;save()
        # Continue that foundation's SFT optimizer from step 1000. Preserve all
        # earlier candidates in the final selection pool, including step 500.
        adapter = Path(f'data/experiments/foundation-pivot-v1/adapter-{best["name"]}')
        run('longer_sft', [sys.executable, 'scripts/continue_programming.py', '--run', str(adapter),
            '--data', 'data/pivot-code-sft-v1', '--out', str(root/'longer-sft'),
            '--milestones', '2000', '4000', '8000', '--hours', '8', '--port', '8036',
            '--tasks-file', 'evaluation/programming-v1/development.jsonl', '--direct-only', '--patience', '2'])
        results = json.loads((root/'longer-sft/status.json').read_text())
        if results['status'] not in ('milestones_exhausted', 'stopped_no_improvement', 'budget_exhausted'):
            raise ValueError('Longer SFT did not complete cleanly')
        for row in results['stages']:
            if row['status'] == 'evaluated':
                candidates.append({'score': row['direct']['passed'], 'checkpoint': row['checkpoint']})
        selected = max(candidates, key=lambda r: r['score'])
        state['selected'] = selected; state['selected_sha256'] = sha(selected['checkpoint']);save()
        if sha(holdout) != state['holdout_sha256']: raise ValueError('Holdout changed')
        run('holdout', [sys.executable, 'scripts/evaluate_repair_checkpoint.py', '--programming',
            '--checkpoint', selected['checkpoint'], '--tasks-file', str(holdout),
            '--out', str(root/'holdout'), '--port', '8036'])
        result = json.loads((root/'holdout/direct/results.json').read_text())
        if len(result['tasks']) != 100: raise ValueError('Incomplete holdout')
        state['holdout'] = {'passed': sum(t['passed'] for t in result['tasks']), 'total': 100}
        state['status'] = 'complete_no_agent_readiness_claim'
    except Exception as error:
        state.update(status='interrupted' if stopping else 'failed', error=str(error))
        if not stopping: raise
    finally: save()


if __name__ == '__main__': main()
