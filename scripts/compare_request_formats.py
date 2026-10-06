"""Matched continuation: original versus varied public request formats.

Both arms start from the best completed broader-data development checkpoint,
use a fresh identical optimizer, and consume the same ordered answer targets.
The only intended data change is user-request formatting. No automatic serving.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys


def sha(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    prior = json.loads(Path('runs/interface-diverse-continuation-v2/status.json').read_text())
    if prior['status'] != 'milestones_exhausted':
        parser.error('Finish the broader-data comparison first')
    parent = max(prior['stages'], key=lambda stage: stage['direct']['passed'])
    profiles = json.loads(Path('runs/sft-profile-effective2.json').read_text())['results']
    speed = {row['microbatch']: row for row in profiles if 'error' not in row}
    microbatch = 2 if (2 in speed and speed[2]['target_tokens_per_second'] > 1.1 * speed[1]['target_tokens_per_second']
                      and speed[2]['peak_allocated_gb'] < 24) else 1
    arms = {'control': 'data/programming-interface-diverse-sft-v3',
            'varied': 'data/programming-format-variety-sft-v1'}
    manifests = {name: json.loads((Path(data) / 'manifest.json').read_text()) for name, data in arms.items()}
    for split in ('train', 'val'):
        for key in ('examples', 'assistant_tokens', 'groups'):
            if manifests['control']['splits'][split][key] != manifests['varied']['splits'][split][key]:
                raise ValueError(f'The format intervention changed {split} {key}')
    audit = json.loads(Path('runs/programming-format-variety-audit.json').read_text())
    if not audit['assistant_code_and_split_groups_unchanged']:
        raise ValueError('Unverified format-only intervention')
    args.out.mkdir(exist_ok=False)
    state = {'status': 'running', 'started': datetime.now(timezone.utc).isoformat(),
             'hypothesis': 'Varied request layouts improve signature copying and code execution on ordinary prompts.',
             'parent': parent['checkpoint'], 'parent_sha256': sha(parent['checkpoint']),
             'parent_direct': parent['direct'], 'microbatch': microbatch, 'accumulation': 2 // microbatch,
             'steps_per_arm': 5000, 'learning_rate': 3e-5, 'seed': 42, 'attention': 'math',
             'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
             'data_manifests': {name: sha(Path(data) / 'manifest.json') for name, data in arms.items()},
             'selection': 'Adopt varied format only if direct passes do not regress versus control and parameter-name matches improve by at least 5/100. Never automatically promote to serving.',
             'phases': [], 'acceptance_used': False}
    child = None
    stopping = False
    def save():
        temporary = args.out / 'status.tmp'
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(args.out / 'status.json')
    def stop(*_):
        nonlocal stopping
        stopping = True
        if child is not None and child.poll() is None:
            child.terminate()
    def run(command, log):
        nonlocal child
        if stopping:
            raise InterruptedError('Interrupted before next stage')
        with log.open('w') as output:
            child = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
            code = child.wait()
        child = None
        if stopping:
            raise InterruptedError('Interrupted')
        if code:
            raise RuntimeError(f'Command exited {code}; see {log}')
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, stop)
    save()
    try:
        for name, data in arms.items():
            phase = {'arm': name, 'status': 'training'}
            state['phases'].append(phase)
            save()
            training = args.out / name
            evaluation = args.out / f'{name}-evaluation'
            run([sys.executable, '-m', 'microgpt.finetune', '--checkpoint', parent['checkpoint'],
                 '--data', data, '--out', str(training), '--steps', '1000', '--lr', '0.00003',
                 '--microbatch', str(microbatch), '--accumulation', str(2 // microbatch),
                 '--eval-every', '500', '--eval-examples', '80', '--max-length', '8192',
                 '--device', 'cuda', '--attention', 'math', '--max-seconds', '600'], args.out / f'{name}-bootstrap.log')
            run([sys.executable, 'scripts/continue_programming.py', '--run', str(training), '--data', data,
                 '--out', str(evaluation), '--milestones', '5000', '--hours', '0.5', '--port', '8026',
                 '--tasks-file', 'evaluation/programming-v1/development.jsonl', '--direct-only'],
                args.out / f'{name}-continuation.log')
            outcome = json.loads((evaluation / 'status.json').read_text())
            if outcome['status'] != 'milestones_exhausted' or outcome['stages'][-1]['metrics']['step'] != 5000:
                raise RuntimeError(f'{name} did not complete the matched update budget')
            phase.update(status='evaluated', result=outcome['stages'][-1])
            save()
        control, varied = [phase['result']['code_diagnostics'] for phase in state['phases']]
        state['format_criterion_passed'] = (varied['passed'] >= control['passed'] and
                                           varied['parameter_names_match'] >= control['parameter_names_match'] + 5)
        state['status'] = 'completed'
    except Exception as error:
        state.update(status='interrupted' if stopping else 'failed', error=str(error))
        if not stopping:
            raise
    finally:
        save()


if __name__ == '__main__':
    main()
