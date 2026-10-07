"""Run the predeclared higher-rate SFT control only after memorization succeeds.

Uses our original random-origin foundation and the unchanged low-rate control's
curriculum. Does not use diagnostic weights, acceptance tasks, or promote models.
"""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from datetime import datetime, timezone


def sha(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def main():
    diagnostic = json.loads(Path('runs/learnability-pilot-v1/status.json').read_text())
    if diagnostic['status'] != 'milestones_exhausted':
        raise RuntimeError('Finish the bounded learnability diagnostic first')
    seen = []
    for stage in diagnostic['stages']:
        report = json.loads(Path(f"runs/learnability-pilot-v1/eval-{stage['metrics']['step']}/direct/results.json").read_text())
        seen.append(report['families']['seen_train']['passed'])
    if max(seen) < 15:
        raise RuntimeError('Memorization gate unmet; inspect the pipeline before broader SFT')
    parent = 'runs/capacity-pilot-v1/foundation/last.pt'
    data = 'data/verified-opencodeinstruct-sft-v1'
    run = 'data/experiments/optimization-lr1e4-v1'
    out = Path('runs/optimization-lr1e4-v1')
    if Path(run).exists():
        raise RuntimeError('Use a fresh experiment; never overwrite training')
    out.mkdir(exist_ok=False)
    control = json.loads(Path('runs/clean-foundation-adapter-pilot-v1/status.json').read_text())
    if control['status'] != 'milestones_exhausted':
        raise RuntimeError('The low-rate control must be complete')
    baseline = max(s['direct']['passed'] for s in control['stages'])
    experiment = {
        'started': datetime.now(timezone.utc).isoformat(), 'status': 'running',
        'parent': parent, 'parent_sha256': sha(parent),
        'data_manifest_sha256': sha(Path(data) / 'manifest.json'),
        'tasks_sha256': sha('evaluation/programming-v1/development.jsonl'),
        'control_sha256': sha('runs/clean-foundation-adapter-pilot-v1/status.json'),
        'diagnostic_sha256': sha('runs/learnability-pilot-v1/status.json'),
        'plan_sha256': sha('docs/learnability-plan.md'),
        'lr': 1e-4, 'control_lr': 1e-5, 'control_best': baseline,
        'seed': 20261006, 'validation_seed': 20261007, 'microbatch': 4, 'accumulation': 8,
        'milestones': [500, 1000, 2000], 'extension_milestones': [4000, 6000],
        'extension_minimum_gain': 5, 'gpu_duty_cycle': os.environ.get('MICROGPT_GPU_DUTY_CYCLE', '0.75'),
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'acceptance_used': False, 'automatic_promotion': False, 'commands': [],
    }
    frozen_control = json.loads(Path('docs/clean-foundation-adapter-outcome.json').read_text())['experiment']
    for key in ('parent_sha256', 'data_manifest_sha256', 'microbatch', 'accumulation', 'seed', 'validation_seed'):
        if experiment[key] != frozen_control[key]:
            raise RuntimeError(f'Unmatched control: {key}')
    def save():
        (out / 'experiment.json').write_text(json.dumps(experiment, indent=2) + '\n')
    child = None
    stopping = False
    def stop(*_):
        nonlocal stopping
        stopping = True
        if child is not None and child.poll() is None:
            child.terminate()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, stop)
    def execute(command, log):
        nonlocal child
        if stopping:
            raise InterruptedError('Interrupted before next stage')
        experiment['commands'].append(command)
        save()
        with (out / log).open('x') as output:
            child = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
            code = child.wait()
        child = None
        if stopping:
            raise InterruptedError('Interrupted')
        if code:
            raise RuntimeError(f'Command failed with {code}: {out / log}')
    try:
        execute([sys.executable, '-m', 'microgpt.finetune', '--checkpoint', parent,
                 '--data', data, '--out', run, '--steps', '250', '--lr', '0.0001',
                 '--seed', '20261006', '--microbatch', '4', '--accumulation', '8',
                 '--eval-every', '250', '--eval-examples', '256', '--validation-seed', '20261007',
                 '--max-length', '8192', '--device', 'cuda', '--attention', 'math',
                 '--max-seconds', '1200'], 'bootstrap.log')
        def continuation(name, milestones, hours):
            destination = out / name
            execute([sys.executable, 'scripts/continue_programming.py', '--run', run,
                     '--data', data, '--out', str(destination), '--milestones', *map(str, milestones),
                     '--hours', str(hours), '--port', '8032', '--tasks-file',
                     'evaluation/programming-v1/development.jsonl', '--direct-only'], name + '.log')
            state = json.loads((destination / 'status.json').read_text())
            if state['status'] != 'milestones_exhausted':
                raise RuntimeError(f'Incomplete comparison: {state["status"]}')
            return [s['direct']['passed'] for s in state['stages']]
        scores = continuation('pilot', [500, 1000, 2000], 2)
        experiment['scores'] = scores
        experiment['extend'] = max(scores) >= baseline + 5
        save()
        if experiment['extend']:
            experiment['extension_scores'] = continuation('extension', [4000, 6000], 3)
        experiment['status'] = 'complete'
    except BaseException as error:
        experiment.update(status='failed_or_interrupted', error=str(error))
        raise
    finally:
        experiment['updated'] = datetime.now(timezone.utc).isoformat()
        save()


if __name__ == '__main__':
    main()
