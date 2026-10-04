"""Run a bounded local curriculum, checkpoint each stage, then evaluate OpenCode.

Launch with nohup for an unattended run. SIGTERM forwards a graceful checkpoint
request to the active trainer and stops the curriculum. No cloud services used.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--hours', type=float, default=12)
    parser.add_argument('--data', type=Path, default=Path('data/large-v1'))
    parser.add_argument('--sft-data', type=Path, default=Path('data/agent-sft-v4'))
    parser.add_argument('--checkpoint', type=Path, help='Continue our own checkpoint; omit for fresh random initialization')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.hours <= 0 or (args.out.exists() and not args.resume):
        parser.error('Use positive hours and a fresh output directory, or --resume')
    args.out.mkdir(parents=True, exist_ok=True)
    state_path = args.out / 'status.json'
    state = json.loads(state_path.read_text()) if args.resume else {
        'started': datetime.now(timezone.utc).isoformat(), 'hours': args.hours,
        'initial_checkpoint': str(args.checkpoint) if args.checkpoint else None, 'stages': {}, 'status': 'running',
        'note': 'Training budget, excluding data load, checkpoint writes and evaluations. Capability requires passing evaluations.'}
    state['data'] = str(args.data)
    state['sft_data'] = str(args.sft_data)
    state['planned_stages'] = ['foundation: up to 1B tokens at 2K', 'instruction: complete episodes up to 8K',
                               'reasoning: checked traces, worked solutions and tool planning at 8K',
                               'long-horizon: complete episodes up to 32K with at least 8 tool calls',
                               'evaluate: basic repairs, 16-round task with handoff/recovery, long-context retrieval']
    if state['hours'] != args.hours or (args.checkpoint and state['initial_checkpoint'] != str(args.checkpoint)):
        parser.error('Resume the same budget and initial checkpoint')
    state['status'] = 'running'
    child = None
    stopping = False
    def stop(*_):
        nonlocal stopping
        stopping = True
        if child is not None and child.poll() is None:
            child.terminate()
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, stop)
    def save():
        state['updated'] = datetime.now(timezone.utc).isoformat()
        state['pid'] = os.getpid()
        temporary = state_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(state_path)
    parent = Path(state['initial_checkpoint']) if state['initial_checkpoint'] else None
    try:
        for name, fraction, steps, maximum in [('foundation', .65, 122071, 2048), ('instruction', .15, 16000, 8192), ('reasoning', .10, 12000, 8192), ('long-horizon', .10, 2000, 32768)]:
            destination = args.out / name
            info = state['stages'].setdefault(name, {'status': 'pending', 'seconds_used': 0})
            if info['status'] == 'complete':
                parent = destination / 'last.pt'
                continue
            remaining = args.hours * 3600 * fraction - info['seconds_used']
            checkpoint = destination / 'last.pt'
            if remaining <= 0 and checkpoint.exists():
                info['status'] = 'complete'
                parent = checkpoint
                save()
                continue
            if remaining <= 0:
                raise RuntimeError('Stage budget exhausted without a checkpoint')
            if name == 'foundation':
                command = [sys.executable, '-m', 'microgpt.train', '--config', 'configs/coding-foundation.toml',
                           '--data', str(args.data), '--out', str(destination), '--sequence-length', str(maximum),
                           '--attention', 'flash', '--steps', str(steps), '--max-seconds', str(remaining)]
                if checkpoint.exists():
                    command += ['--resume', str(checkpoint)]
                elif parent is not None:
                    command += ['--init-from', str(parent)]
            else:
                command = [sys.executable, '-m', 'microgpt.finetune', '--data', str(args.sft_data), '--out', str(destination),
                           '--steps', str(steps), '--max-seconds', str(remaining), '--max-length', str(maximum),
                           '--balanced-sources', '--accumulation', '2', '--eval-every', '250', '--eval-examples', '12',
                           '--lr', '0.0001' if name == 'instruction' else '0.00003',
                           '--resume' if checkpoint.exists() else '--checkpoint', str(checkpoint if checkpoint.exists() else parent)]
                if name == 'reasoning':
                    command += ['--sources', 'reasoning-code', 'reasoning-math', 'reasoning-trace', 'reasoning-tools', '--short-replay', '0.2']
                if name == 'long-horizon':
                    command += ['--min-tool-calls', '8', '--balanced-lengths', '--short-replay', '0.1']
            info.update(status='running', command=command)
            state['active_stage'] = name
            save()
            started = time.monotonic()
            with (args.out / f'{name}.log').open('a') as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                code = child.wait()
            info['seconds_used'] += time.monotonic() - started
            if stopping:
                state['status'] = 'interrupted'
                info['status'] = 'interrupted'
                save()
                return
            if code or not checkpoint.exists():
                raise RuntimeError(f'{name} failed (exit {code}); see {name}.log')
            info['status'] = 'complete'
            parent = checkpoint
            save()
        state['status'] = 'evaluating'
        state['checkpoint'] = str(parent)
        save()
        with (args.out / 'server.log').open('a') as log:
            child = subprocess.Popen([sys.executable, '-m', 'microgpt.serve', '--checkpoint', str(parent),
                                      '--device', 'cuda', '--attention', 'flash', '--port', '8011', '--max-output', '4096'], stdout=log, stderr=subprocess.STDOUT)
            try:
                for _ in range(120):
                    if child.poll() is not None:
                        raise RuntimeError('Evaluation server exited')
                    try:
                        with urllib.request.urlopen('http://127.0.0.1:8011/health', timeout=1) as response:
                            health = json.load(response)
                        if health['source'].get('checkpoint') != str(parent):
                            raise RuntimeError('Another checkpoint owns the evaluation port')
                        break
                    except OSError:
                        time.sleep(1)
                else:
                    raise RuntimeError('Evaluation server did not become ready')
                for script, folder, steps in [('eval_opencode.py', 'eval-basic', '12'), ('eval_horizon.py', 'eval-horizon', '64')]:
                    with (args.out / f'{folder}.log').open('a') as evaluation_log:
                        subprocess.run([sys.executable, f'scripts/{script}', '--url', 'http://127.0.0.1:8011/v1',
                                        '--out', str(args.out / folder), '--label', str(parent), '--steps', steps],
                                       stdout=evaluation_log, stderr=subprocess.STDOUT, check=True, timeout=1800)
                with (args.out / 'eval-reasoning.log').open('a') as evaluation_log:
                    subprocess.run([sys.executable, 'scripts/eval_reasoning.py', '--url', 'http://127.0.0.1:8011/v1',
                                    '--out', str(args.out / 'eval-reasoning.json'), '--label', str(parent)],
                                   stdout=evaluation_log, stderr=subprocess.STDOUT, check=True, timeout=3600)
                with (args.out / 'eval-context.log').open('a') as evaluation_log:
                    subprocess.run([sys.executable, 'scripts/eval_context.py', '--checkpoint', str(parent),
                                    '--url', 'http://127.0.0.1:8011/v1', '--out', str(args.out / 'eval-context.json')],
                                   stdout=evaluation_log, stderr=subprocess.STDOUT, check=True, timeout=1800)
            finally:
                child.terminate()
                child.wait(timeout=30)
        state['status'] = 'evaluated'
        state['note'] = 'Read eval-basic/results.json and eval-horizon/results.json; evaluated does not mean passed.'
        save()
    except Exception as error:
        state['status'] = 'failed'
        state['error'] = str(error)
        save()
        raise


if __name__ == '__main__':
    main()
