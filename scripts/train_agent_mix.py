"""Longer verified code/repair/conversation SFT from our best scratch checkpoint.

Checkpoints live on the authorized NAS path. Behavioral evaluations use a private
server and never replace the user's service. See docs/agent-mix-plan.md.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parent = Path('runs/verified-code-learning-v1/checkpoint-500.pt')
    data = Path('data/programming-agent-mix-v1')
    run = Path('data/experiments/agent-mix-v1')
    out = Path('runs/agent-mix-v1')
    if run.exists() or out.exists():
        raise ValueError('Use a fresh experiment; resume existing checkpoints explicitly')
    if not run.resolve().is_relative_to(Path('/mnt/gigachad/microgpt/dataset')):
        raise ValueError('Training checkpoints must remain inside the authorized NAS path')
    baseline = json.loads(Path('runs/agent-best24-repair-baseline-v1/results.json').read_text())
    if baseline['label'] != str(parent) or baseline['agent_prompt'] != 'general':
        raise ValueError('Baseline checkpoint/prompt mismatch')
    weights = {'verified-opencodeinstruct': .5, 'executed-public-repair': .4,
               'smoltalk': .08, 'multilingual': .02}
    out.mkdir()
    state = {'status': 'running', 'pid': os.getpid(), 'started': datetime.now(timezone.utc).isoformat(),
             'parent': str(parent), 'parent_sha256': sha(parent),
             'data_manifest_sha256': sha(data / 'manifest.json'),
             'plan_sha256': sha('docs/agent-mix-plan.md'),
             'baseline': {'direct': 24, 'repair': sum(row['passed'] for row in baseline['tasks']),
                          'tasks_using_tools': sum(bool(row['tools']) for row in baseline['tasks'])},
             'source_weights': weights, 'lr': 1e-5, 'microbatch': 1, 'accumulation': 16,
             'attention': 'math', 'max_length': 8192, 'seed': 20261008,
             'milestones': [250, 1000, 3000, 6000, 12000], 'hours': 6,
             'gpu_duty_cycle': os.environ.get('MICROGPT_GPU_DUTY_CYCLE', '0.75'),
             'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
             'acceptance_used': False, 'automatic_promotion': False, 'commands': []}
    child = None
    stopping = False
    def save():
        temp = out / 'experiment.tmp'
        temp.write_text(json.dumps(state, indent=2) + '\n')
        temp.replace(out / 'experiment.json')
    def stop(*_):
        nonlocal stopping
        stopping = True
        if child is not None and child.poll() is None:
            child.terminate()
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, stop)
    def execute(command, log):
        nonlocal child
        if stopping:
            raise InterruptedError('Interrupted')
        state['commands'].append(command); save()
        with (out / log).open('x') as output:
            child = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
            code = child.wait()
        child = None
        if stopping:
            raise InterruptedError('Interrupted')
        if code:
            raise RuntimeError(f'Command failed with {code}: {out / log}')
    started = time.monotonic()
    try:
        command = [sys.executable, '-m', 'microgpt.finetune', '--checkpoint', str(parent),
                   '--data', str(data), '--out', str(run), '--steps', '250', '--lr', '0.00001',
                   '--seed', '20261008', '--microbatch', '1', '--accumulation', '16',
                   '--eval-every', '250', '--eval-examples', '256', '--validation-seed', '20261007',
                   '--max-length', '8192', '--device', 'cuda', '--attention', 'math']
        for source, weight in weights.items():
            command += ['--source-weight', f'{source}:{weight}']
        execute(command, 'bootstrap.log')
        remaining = 6 - (time.monotonic() - started) / 3600
        if remaining <= 0:
            raise RuntimeError('Bootstrap exhausted the six-hour budget')
        execute([sys.executable, 'scripts/continue_programming.py', '--run', str(run),
                 '--data', str(data), '--out', str(out / 'pilot'), '--milestones',
                 '250', '1000', '3000', '6000', '12000', '--hours', str(remaining),
                 '--port', '8034', '--tasks-file', 'evaluation/programming-v1/development.jsonl',
                 '--direct-only', '--repair-eval', '--patience', '0', '--evaluate-current'], 'controller.log')
        state['status'] = json.loads((out / 'pilot/status.json').read_text())['status']
    except Exception as error:
        state.update(status='interrupted' if stopping else 'failed', error=str(error))
        if not stopping:
            raise
    finally:
        state['finished'] = datetime.now(timezone.utc).isoformat(); save()


if __name__ == '__main__':
    main()
