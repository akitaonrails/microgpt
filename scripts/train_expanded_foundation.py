"""Bounded +250M-token foundation trial followed by the frozen coding adapter.

Uses only our scratch-origin weights. A one-billion-token schedule is fixed, but
there is no automatic extension or promotion. Inspect status.json after failures;
pretraining can be resumed with the exact recorded command and --resume.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8031)
    args = parser.parse_args()
    baseline_path = Path('runs/clean-foundation-adapter-pilot-v1/status.json')
    baseline = json.loads(baseline_path.read_text())
    if baseline['status'] != 'milestones_exhausted' or len(baseline['stages']) != 3:
        parser.error('Finish the clean foundation adapter control first')
    if any(s['status'] != 'evaluated' for s in baseline['stages']):
        parser.error('All three control evaluations must finish')
    parent = Path('runs/capacity-pilot-v1/foundation/last.pt')
    config = Path('configs/foundation-continuation-239m.toml')
    data = Path('data/foundation-expanded-v1')
    adapter_data = Path('data/verified-opencodeinstruct-sft-v1')
    tasks = Path('evaluation/programming-v1/development.jsonl')
    audit = json.loads(Path('docs/expanded-foundation-corpus.json').read_text())
    if sha(data / 'meta.json') != audit['manifest_sha256']:
        parser.error('Expanded data differs from the verified corpus')
    args.out.mkdir(parents=True, exist_ok=False)
    best_baseline = max(s['direct']['passed'] for s in baseline['stages'])
    state = {
        'status': 'running', 'pid': os.getpid(), 'started': datetime.now(timezone.utc).isoformat(),
        'hypothesis': 'Additional broad code exposure improves the same short coding adapter.',
        'parent': str(parent), 'parent_sha256': sha(parent),
        'config_sha256': sha(config), 'data_manifest_sha256': audit['manifest_sha256'],
        'adapter_manifest_sha256': sha(adapter_data / 'manifest.json'),
        'tasks_sha256': sha(tasks), 'control_status_sha256': sha(baseline_path),
        'control_best': best_baseline, 'previous_best_across_recipes': 24,
        'minimum_gain_to_consider_more_pretraining': 5,
        'new_token_ceiling': 30518 * 8192, 'full_schedule_steps': 122071,
        'training_seconds_limit': 4 * 3600, 'adapter_hours_limit': 2,
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'automatic_promotion': False, 'acceptance_used': False, 'commands': [],
        'limitations': 'A new data/optimizer stage, not an isolated causal test of dataset size. No capability inferred from validation loss.',
    }
    child = None
    stopping = False

    def save():
        state['updated'] = datetime.now(timezone.utc).isoformat()
        temp = args.out / 'status.tmp'
        temp.write_text(json.dumps(state, indent=2) + '\n')
        temp.replace(args.out / 'status.json')

    def stop(*_):
        nonlocal stopping
        stopping = True
        if child is not None and child.poll() is None:
            child.terminate()

    def run(phase, command, timeout):
        nonlocal child
        if stopping:
            raise InterruptedError('Stopped')
        state['phase'] = phase
        state['commands'].append(command)
        save()
        started = time.monotonic()
        with (args.out / f'{phase}.log').open('a') as output:
            child = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
            try:
                code = child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                child.terminate()
                try:
                    child.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    child.kill(); child.wait()
                raise
        child = None
        state[phase + '_seconds'] = time.monotonic() - started
        save()
        if stopping or code:
            raise RuntimeError(f'{phase} stopped with exit {code}; inspect its log')

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    foundation = args.out / 'foundation'
    adapter = args.out / 'adapter'
    save()
    try:
        run('pretraining', [sys.executable, '-m', 'microgpt.train', '--config', str(config),
            '--data', str(data), '--init-from', str(parent), '--out', str(foundation),
            '--device', 'cuda', '--attention', 'flash', '--sequence-length', '2048',
            '--stop-at-step', '30518', '--max-seconds', '14400'], 14700)
        metric = json.loads((foundation / 'metrics.jsonl').read_text().splitlines()[-1])
        state['pretraining_metric'] = metric
        initial = float(re.search(r'step 0: validation loss ([\d.]+)',
                                  (args.out / 'pretraining.log').read_text())[1])
        state['initial_validation_loss'] = initial
        if metric['step'] != 30518:
            state['status'] = 'training_budget_exhausted'
            return
        if not math.isfinite(metric['val_loss']) or metric['val_loss'] > initial + 0.2:
            state['status'] = 'rejected_loss'
            return
        run('adapter_bootstrap', [sys.executable, '-m', 'microgpt.finetune',
            '--checkpoint', str(foundation / 'last.pt'), '--data', str(adapter_data),
            '--out', str(adapter), '--steps', '250', '--lr', '0.00001', '--seed', '20261006',
            '--microbatch', '4', '--accumulation', '8', '--eval-every', '250',
            '--eval-examples', '256', '--validation-seed', '20261007', '--max-length', '8192',
            '--device', 'cuda', '--attention', 'math', '--max-seconds', '1200'], 1500)
        run('adapter_evaluation', [sys.executable, 'scripts/continue_programming.py',
            '--run', str(adapter), '--data', str(adapter_data), '--out', str(args.out / 'adapter-pilot'),
            '--milestones', '500', '1000', '2000', '--hours', '2', '--port', str(args.port),
            '--tasks-file', str(tasks), '--direct-only'], 9600)
        result = json.loads((args.out / 'adapter-pilot/status.json').read_text())
        if result['status'] != 'milestones_exhausted':
            raise RuntimeError('Adapter comparison incomplete')
        state['adapter_scores'] = [s['direct']['passed'] for s in result['stages']]
        state['best_direct'] = max(state['adapter_scores'])
        state['gain_over_clean_control'] = state['best_direct'] - best_baseline
        state['consider_more_pretraining'] = state['gain_over_clean_control'] >= 5
        state['status'] = 'comparison_complete'
    except Exception as error:
        state['status'] = 'interrupted' if stopping else 'failed'
        state['error'] = str(error)
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
