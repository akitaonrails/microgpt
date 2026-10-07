"""Bounded unattended programming training with behavioral checkpoints.

Resume our own optimizer/RNG; never load an external model. Stop on passed gates,
a training budget (allowing evaluations to finish), or three evaluation checkpoints without a better direct score.
No candidate automatically replaces the user's running server.
"""
import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

import torch


def resume_command(run, data, settings, target, seconds):
    command = [sys.executable, '-m', 'microgpt.finetune', '--resume', str(run / 'last.pt'),
               '--out', str(run), '--data', str(data), '--steps', str(target), '--max-seconds', str(seconds),
               '--eval-every', '1000']
    for key in ('lr', 'accumulation', 'eval_examples', 'seed', 'device', 'attention', 'max_length', 'min_tool_calls', 'short_replay'):
        command += ['--' + key.replace('_', '-'), str(settings[key])]
    command += ['--microbatch', str(settings.get('microbatch', 1))]
    if settings.get('validation_seed') is not None:
        command += ['--validation-seed', str(settings['validation_seed'])]
    for key in ('balanced_sources', 'balanced_lengths'):
        if settings[key]: command.append('--' + key.replace('_', '-'))
    if settings.get('sources'): command += ['--sources', *settings['sources']]
    for name, value in settings.get('source_weights', {}).items():
        command += ['--source-weight', f'{name}:{value}']
    return command


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, default=Path('runs/programming-v2'))
    p.add_argument('--data', type=Path, default=Path('data/programming-sft-v2'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--milestones', type=int, nargs='+', default=[48000, 80000, 120000, 160000])
    p.add_argument('--hours', type=float, default=4)
    p.add_argument('--port', type=int, default=8021)
    p.add_argument('--tasks-file', type=Path, help='Frozen broader evaluation set')
    p.add_argument('--direct-only', action='store_true', help='Diagnose instruction training without claiming agent readiness')
    p.add_argument('--repair-eval', action='store_true', help='Also measure the four real OpenCode repairs with the general assistant prompt')
    p.add_argument('--repair-tasks-file', type=Path, help='Also evaluate a frozen, stratified repair diagnostic')
    p.add_argument('--patience', type=int, default=3, help='Stop after this many direct-score plateaus; 0 disables for a fixed mixed-agent curriculum')
    p.add_argument('--exact-prompt', action='store_true', help='Use the frozen task prompt without adding formatting instructions')
    p.add_argument('--evaluate-current', action='store_true', help='First milestone must equal the saved step; evaluate it without another optimizer update')
    p.add_argument('--server-startup-seconds', type=int, default=300, help='Allow cold NAS checkpoint loads before evaluation')
    args = p.parse_args()
    if args.hours <= 0 or args.patience < 0 or args.server_startup_seconds < 1 or args.milestones != sorted(set(args.milestones)):
        p.error('Positive hours and strictly increasing milestones are required')
    args.out.mkdir(parents=True, exist_ok=False)
    saved = torch.load(args.run / 'last.pt', map_location='cpu', weights_only=True)
    settings, start = saved['settings'], saved['step']
    if saved.get('origin', {}).get('initialization') != 'random':
        p.error('This runner requires our random-origin checkpoint')
    del saved; gc.collect()
    if args.evaluate_current:
        if args.milestones[0] != start: p.error('Evaluation recovery must start at the saved step')
    elif args.milestones[0] <= start:
        p.error('Milestones must exceed the saved step')
    state = {'status': 'running', 'pid': os.getpid(), 'started': datetime.now(timezone.utc).isoformat(),
             'initial_step': start, 'milestones': args.milestones, 'hours': args.hours, 'exact_prompt': args.exact_prompt,
             'evaluate_current': args.evaluate_current, 'server_startup_seconds': args.server_startup_seconds, 'stages': [],
             'repair_eval': args.repair_eval, 'patience': args.patience,
             'repair_tasks_file': str(args.repair_tasks_file) if args.repair_tasks_file else None,
             'note': 'Continuation, not a claim of model readiness. The existing service is not replaced.'}
    child = None; stopping = False
    def save():
        state['updated'] = datetime.now(timezone.utc).isoformat()
        temp = args.out / 'status.tmp'; temp.write_text(json.dumps(state, indent=2) + '\n'); temp.replace(args.out / 'status.json')
    def stop(*_):
        nonlocal stopping
        stopping = True
        if child is not None and child.poll() is None: child.terminate()
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, stop)
    def execute(command, log, timeout=None):
        nonlocal child
        with log.open('a') as output:
            child = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
            try: code = child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                child.terminate()
                try: child.wait(timeout=15)
                except subprocess.TimeoutExpired: child.kill(); child.wait()
                raise
        child = None
        if code: raise RuntimeError(f'Command failed with {code}: {log}')
    started = time.monotonic(); best = 0; stalled = 0
    try:
        for target in args.milestones:
            remaining = args.hours * 3600 - (time.monotonic() - started)
            if remaining <= 0 or stopping: break
            stage = {'target': target, 'status': 'training'}; state['stages'].append(stage); save()
            recovery = args.evaluate_current and target == start
            if not recovery:
                execute(resume_command(args.run, args.data, settings, target, remaining), args.out / 'training.log')
            if stopping: break
            metric = json.loads((args.run / 'metrics.jsonl').read_text().splitlines()[-1])
            stage['metrics'] = metric; step = metric['step']
            if recovery and step != start:
                raise RuntimeError('Saved checkpoint and training metrics disagree')
            frozen = args.run / f'checkpoint-{step}.pt'
            if frozen.exists():
                if not recovery or not os.path.samefile(args.run / 'last.pt', frozen):
                    raise RuntimeError(f'Refusing to overwrite or evaluate a different checkpoint: {frozen}')
            else:
                os.link(args.run / 'last.pt', frozen)
            stage.update(status='evaluating', checkpoint=str(frozen)); save()
            destination = args.out / f'eval-{step}'; destination.mkdir()
            with (destination / 'server.log').open('w') as log:
                server = subprocess.Popen([sys.executable, '-m', 'microgpt.serve', '--checkpoint', str(frozen),
                    '--device', 'cuda', '--attention', 'flash', '--constrain-tools', '--port', str(args.port)], stdout=log, stderr=subprocess.STDOUT)
                try:
                    url = f'http://127.0.0.1:{args.port}'
                    deadline = time.monotonic() + args.server_startup_seconds
                    while time.monotonic() < deadline:
                        if stopping: raise InterruptedError('Interrupted')
                        if server.poll() is not None: raise RuntimeError('Evaluation server exited')
                        try:
                            with urllib.request.urlopen(url + '/health', timeout=2) as response: health = json.load(response)
                            if health['source']['checkpoint'] != str(frozen): raise RuntimeError('Evaluation port belongs to another server')
                            break
                        except OSError: time.sleep(1)
                    else: raise RuntimeError('Evaluation server did not start')
                    evaluation_flags = ['--tasks-file', str(args.tasks_file), '--limit', '100'] if args.tasks_file else []
                    if args.exact_prompt:
                        evaluation_flags.append('--exact-prompt')
                    checks = [('direct', 'scripts/eval_programming.py', ['--mode', 'direct', *evaluation_flags])]
                    if args.repair_eval:
                        checks.append(('repair', 'scripts/eval_opencode.py', ['--suite', 'basic', '--general-prompt', '--effort', 'none', '--label', str(frozen)]))
                    if args.repair_tasks_file:
                        checks.append(('diagnostic', 'scripts/eval_opencode.py', ['--tasks-file', str(args.repair_tasks_file), '--general-prompt', '--effort', 'none', '--label', str(frozen)]))
                    passed = True; checked = set()
                    for name, script, flags in checks:
                        execute([sys.executable, script, *flags, '--url', url + '/v1', '--out', str(destination / name)], destination / f'{name}.log', 7200 if name == 'diagnostic' else 1800)
                        result = json.loads((destination / name / 'results.json').read_text())
                        checked.add(name)
                        if name == 'horizon':
                            stage[name] = {'passed': result['passed'], 'artifact': str(destination / name / 'results.json')}
                            passed = passed and result['passed']
                        else:
                            score = sum(row['passed'] for row in result['tasks'])
                            stage[name] = {'passed': score, 'total': len(result['tasks'])}
                            if name in ('repair', 'diagnostic'):
                                stage[name]['tasks_using_tools'] = sum(bool(row['tools']) for row in result['tasks'])
                                stage[name]['unsupported_pass_claim_flags'] = sum(row.get('unsupported_pass_claim_flag', False) for row in result['tasks'])
                                stage[name]['families'] = {family: {'passed': sum(row['passed'] for row in result['tasks'] if row.get('family') == family),
                                    'total': sum(row.get('family') == family for row in result['tasks'])}
                                    for family in sorted({row.get('family', 'unknown') for row in result['tasks']})}
                            passed = passed and score == len(result['tasks'])
                        if name == 'direct':
                            if args.tasks_file:
                                analysis_path = destination / name / 'analysis.json'
                                execute([sys.executable, 'scripts/analyze_programming.py',
                                         '--results', str(destination / name / 'results.json'),
                                         '--tasks-file', str(args.tasks_file), '--out', str(analysis_path)],
                                        destination / 'analysis.log', 60)
                                stage['code_diagnostics'] = json.loads(analysis_path.read_text())['counts']
                            if score > best: best, stalled = score, 0
                            else: stalled += 1
                            if score and not args.direct_only:
                                checks.append(('agent', 'scripts/eval_programming.py', ['--mode', 'agent', *evaluation_flags]))
                            if score == len(result['tasks']) and not args.direct_only:
                                checks += [('basic', 'scripts/eval_opencode.py', ['--suite', 'basic', '--effort', 'none', '--label', str(frozen)]),
                                           ('horizon', 'scripts/eval_horizon.py', ['--rounds', '32', '--seed', '20261004', '--effort', 'none', '--label', str(frozen)])]
                        save()
                        if stopping: break
                    required = {'direct', 'agent', 'basic', 'horizon'} | ({'repair'} if args.repair_eval else set())
                    if args.repair_tasks_file: required.add('diagnostic')
                    stage['passed_gates'] = passed and not stopping and checked == required
                finally:
                    server.terminate()
                    try: server.wait(timeout=15)
                    except subprocess.TimeoutExpired: server.kill(); server.wait()
            stage['status'] = 'evaluated'; save()
            if stage['passed_gates']:
                state['status'] = 'candidate_passed_gates'; state['candidate'] = str(frozen); break
            if step < target:
                state['status'] = 'budget_exhausted'; break
            if args.patience and stalled >= args.patience:
                state['status'] = 'stopped_no_improvement'; break
        else:
            state['status'] = 'milestones_exhausted'
        if stopping: state['status'] = 'interrupted'
        elif state['status'] == 'running': state['status'] = 'budget_exhausted'
    except Exception as error:
        state['status'] = 'interrupted' if stopping else 'failed'; state['error'] = str(error)
        if not stopping: raise
    finally:
        save()


if __name__ == '__main__':
    main()
