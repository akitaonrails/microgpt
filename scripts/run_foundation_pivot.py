"""Scratch foundation pivot: prepare -> profile -> 2B/5B/10B + fixed coding probes.

No pretrained weights, cloud rental, or automatic deployment. Failures leave logs
and resumable last.pt files. Large artifacts remain in the authorized NAS path.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preparation-pid', type=int, required=True)
    p.add_argument('--previous-run-pid', type=int, required=True)
    p.add_argument('--resume-after-2b', action='store_true')
    p.add_argument('--resume-after-5b', action='store_true')
    args = p.parse_args()
    if args.resume_after_2b and args.resume_after_5b: p.error('Choose one recovery mode')
    root = Path('runs/pivot-v1')
    corpus = Path('data/foundation-bpe32k-v1')
    base = Path('data/experiments/foundation-pivot-v1')
    adapter_data = Path('data/pivot-code-sft-v1')
    config = Path('configs/pivot-337m.toml')
    tasks = Path('evaluation/programming-v1/development.jsonl')
    for path in (corpus, base, adapter_data):
        if not path.resolve().is_relative_to(Path('/mnt/gigachad/microgpt/dataset')):
            raise ValueError('Artifacts must stay within the authorized NAS path')
    if not (args.resume_after_2b or args.resume_after_5b) and ((root / 'status.json').exists() or base.exists()):
        raise ValueError('Existing experiment; use its recorded resume commands, never overwrite')
    base.mkdir(parents=True, exist_ok=args.resume_after_2b or args.resume_after_5b)
    state = {'status': 'running', 'phase': 'waiting_for_corpus', 'pid': os.getpid(),
             'started': datetime.now(timezone.utc).isoformat(), 'initialization': 'random',
             'config_sha256': sha(config), 'plan_sha256': sha('docs/foundation-pivot.md'),
             'tasks_sha256': sha(tasks), 'automatic_promotion': False, 'acceptance_used': False,
             'commands': [], 'milestones': [], 'full_gpu_seconds': 8 * 3600,
             'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()}
    if args.resume_after_2b:
        previous = json.loads((root / 'status.json').read_text())
        recovery = json.loads((root / 'coding-2b-recovery/status.json').read_text())
        if previous['status'] != 'failed' or previous['phase'] != 'coding-2b' or len(previous['milestones']) != 1:
            raise ValueError('Recovery requires the failed 2B coding phase')
        for key in ('config_sha256', 'plan_sha256', 'tasks_sha256'):
            if previous[key] != state[key]: raise ValueError(f'Changed experiment: {key}')
        for key, path in [('corpus_sha256', corpus / 'meta.json'),
                          ('adapter_manifest_sha256', adapter_data / 'manifest.json')]:
            if previous[key] != sha(path): raise ValueError(f'Changed data: {key}')
        if recovery['status'] != 'milestones_exhausted' or [r['target'] for r in recovery['stages']] != [500, 1000]:
            raise ValueError('Incomplete recovery evaluation')
        if any(r['status'] != 'evaluated' or r['direct']['total'] != 100 for r in recovery['stages']):
            raise ValueError('Incomplete coding scores')
        scores = [r['direct']['passed'] for r in recovery['stages']]
        if max(scores) < 10: raise ValueError('Original 2B spending gate not met')
        checkpoint = base / 'foundation/last.pt'
        if sha(checkpoint) != previous['milestones'][0]['checkpoint_sha256']:
            raise ValueError('Foundation checkpoint changed since 2B milestone')
        archive = root / 'status-before-2b-recovery.json'
        with archive.open('x') as handle: json.dump(previous, handle, indent=2)
        previous['milestones'][0].update(direct_scores=scores, best_direct=max(scores), raw_passed=0,
            recovery_evidence=str(root / 'coding-2b-recovery/status.json'))
        previous.update(status='running', pid=os.getpid(), recovery_commit=state['git_commit'])
        previous.pop('error', None)
        state = previous
    if args.resume_after_5b:
        previous = json.loads((root / 'status.json').read_text())
        if previous['status'] != 'interrupted' or previous['phase'] != 'foundation-10b' or len(previous['milestones']) != 2:
            raise ValueError('Recovery requires interrupted post-5B training')
        for key in ('config_sha256', 'plan_sha256', 'tasks_sha256'):
            if previous[key] != state[key]: raise ValueError(f'Changed experiment: {key}')
        for key, path in [('corpus_sha256', corpus / 'meta.json'), ('adapter_manifest_sha256', adapter_data / 'manifest.json')]:
            if previous[key] != sha(path): raise ValueError(f'Changed data: {key}')
        milestone = previous['milestones'][-1]
        if milestone['best_direct'] < max(25, previous['milestones'][0]['best_direct'] + 3):
            raise ValueError('5B spending gate failed')
        if sha(milestone['checkpoint']) != milestone['checkpoint_sha256']:
            raise ValueError('Frozen 5B checkpoint checksum mismatch')
        checkpoint_audit = json.loads((root / 'maintenance-checkpoint-check.json').read_text())
        if checkpoint_audit['step'] != 152588 or checkpoint_audit['sha256'] != milestone['checkpoint_sha256']:
            raise ValueError('Verified checkpoint metadata does not match milestone')
        audit = json.loads((root / 'maintenance-data-check.json').read_text())
        if len(audit['files']) != 14 or not all(r['matches'] for r in audit['files']):
            raise ValueError('Complete corpus and SFT integrity checks first')
        if (base / 'foundation-resumed-5b').exists(): raise ValueError('Recovery output already exists')
        with (root / 'status-before-maintenance-recovery.json').open('x') as f:
            json.dump(previous, f, indent=2)
        previous.update(status='running', pid=os.getpid(), maintenance_recovery_commit=state['git_commit'],
                        foundation_output=str(base / 'foundation-resumed-5b'))
        previous.pop('error', None)
        state = previous
    state['gpu_schedule'] = {'window': os.environ.get('MICROGPT_FULL_GPU_WINDOW'),
                             'timezone': os.environ.get('MICROGPT_GPU_TIMEZONE', 'America/Sao_Paulo'),
                             'outside_fraction': 0.75}
    child = None
    stopping = False
    env = dict(os.environ, TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4')
    def journal(message):
        with (root / 'journal.md').open('a') as log:
            log.write(f'\n{datetime.now(timezone.utc).isoformat()} — {message}\n')

    def save():
        state['updated'] = datetime.now(timezone.utc).isoformat()
        temp = root / 'status.tmp'
        temp.write_text(json.dumps(state, indent=2) + '\n')
        temp.replace(root / 'status.json')
    def stop(*_):
        nonlocal stopping
        stopping = True
        if child is not None and child.poll() is None: child.terminate()
    for sig in (signal.SIGTERM, signal.SIGINT): signal.signal(sig, stop)
    def run(name, command):
        nonlocal child
        if stopping: raise InterruptedError('Interrupted')
        state['phase'] = name
        state['commands'].append({'phase': name, 'command': command})
        save()
        journal(f'Started **{name}**. Command: `{subprocess.list2cmdline(command)}`')
        with (root / f'{name}.log').open('a') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
            code = child.wait()
        child = None
        if stopping: raise InterruptedError('Interrupted')
        if code: raise RuntimeError(f'{name} exited {code}; inspect its log')
        journal(f'Completed **{name}**.')
    py = sys.executable
    save()
    try:
        if not (args.resume_after_2b or args.resume_after_5b):
            deadline = time.monotonic() + 6 * 3600
            while not (corpus / 'meta.json').exists():
                if stopping: raise InterruptedError('Interrupted')
                os.kill(args.preparation_pid, 0)
                if time.monotonic() > deadline: raise TimeoutError('Corpus preparation exceeded six hours')
                time.sleep(15)
            state['corpus_sha256'] = sha(corpus / 'meta.json')
            # End the superseded experiment gracefully, including its child trainer.
            # The checked PID belongs to the explicitly supplied old runner.
            previous = Path(f'/proc/{args.previous_run_pid}/cmdline')
            if previous.exists():
                if b'scripts/run_refined_agent.py' not in previous.read_bytes():
                    raise ValueError('Previous PID no longer belongs to the old experiment')
                os.kill(args.previous_run_pid, signal.SIGTERM)
                until = time.monotonic() + 600
                while previous.exists():
                    if stopping: raise InterruptedError('Interrupted')
                    if time.monotonic() > until: raise TimeoutError('Old run did not exit gracefully')
                    time.sleep(2)
            state['previous_run'] = json.loads(Path('runs/refined-agent-v1/status.json').read_text())
            run('prepare_adapter', [py, '-m', 'microgpt.trajectories', '--input',
                'data/public-code-multiseed-v1/code.jsonl', '--out', str(adapter_data),
                '--tokenizer', str(corpus), '--max-length', '8192', '--limit', '100000', '--pack-episodes'])
            state['adapter_manifest_sha256'] = sha(adapter_data / 'manifest.json')
            env['MICROGPT_GPU_DUTY_CYCLE'] = '1'
            env['MICROGPT_FULL_GPU_UNTIL'] = str(time.time() + state['full_gpu_seconds'])
            state['full_gpu_until'] = float(env['MICROGPT_FULL_GPU_UNTIL']); save()
            run('profile', [py, 'scripts/profile_pivot.py', '--data', str(corpus), '--config', str(config),
                '--out', str(root / 'profile.json')])
            profile = json.loads((root / 'profile.json').read_text())
            state['profile'] = profile; save()
            journal(f'GPU profile: {profile["tokens_per_second"]:.0f} tokens/s, {profile["peak_allocated_gb"]:.2f} GB allocated. Short profile, not a deadline.')
            if not math.isfinite(profile['tokens_per_second']) or profile['tokens_per_second'] <= 0:
                raise ValueError('Invalid throughput')
        else:
            env['MICROGPT_GPU_DUTY_CYCLE'] = '0.75'
            env.pop('MICROGPT_FULL_GPU_UNTIL', None)
            journal('Resuming after verified 5B checkpoint following maintenance.' if args.resume_after_5b else 'Recovered 2B evaluation; spending gate passed.')
        foundation = base / ('foundation-resumed-5b' if args.resume_after_5b else 'foundation')
        for tokens in ((10_000_000_000,) if args.resume_after_5b else (5_000_000_000, 10_000_000_000) if args.resume_after_2b else (2_000_000_000, 5_000_000_000, 10_000_000_000)):
            target = math.ceil(tokens / 32768)
            name = f'{tokens // 1_000_000_000}b'
            command = [py, '-m', 'microgpt.train', '--config', str(config), '--data', str(corpus),
                '--out', str(foundation), '--device', 'cuda', '--attention', 'flash',
                '--sequence-length', '2048', '--stop-at-step', str(target)]
            if args.resume_after_5b:
                command += ['--resume', state['milestones'][-1]['checkpoint'], '--expected-resume-step', '152588']
            elif (foundation / 'last.pt').exists():
                command += ['--resume', str(foundation / 'last.pt'), '--expected-resume-step', str(state['milestones'][-1]['metric']['step'])]
            run(f'foundation-{name}', command)
            metric = json.loads((foundation / 'metrics.jsonl').read_text().splitlines()[-1])
            if metric['step'] != target or not math.isfinite(metric['val_loss']):
                raise ValueError('Foundation stopped before milestone or became nonfinite')
            frozen = foundation / f'checkpoint-{name}.pt'
            os.link(foundation / 'last.pt', frozen)
            row = {'tokens': target * 32768, 'metric': metric, 'checkpoint': str(frozen),
                   'checkpoint_sha256': sha(frozen)}
            state['milestones'].append(row); save()
            run(f'raw-{name}', [py, 'scripts/eval_raw_programming.py', '--checkpoint', str(frozen),
                '--tasks-file', str(tasks), '--out', str(root / f'raw-{name}')])
            adapter = base / f'adapter-{name}'
            run(f'adapter-{name}', [py, '-m', 'microgpt.finetune', '--checkpoint', str(frozen),
                '--data', str(adapter_data), '--out', str(adapter), '--steps', '500', '--lr', '0.00001',
                '--microbatch', '1', '--accumulation', '16', '--eval-every', '500', '--eval-examples', '256',
                '--validation-seed', '20261007', '--seed', '20261007', '--max-length', '8192',
                '--device', 'cuda', '--attention', 'flash'])
            run(f'coding-{name}', [py, 'scripts/continue_programming.py', '--run', str(adapter),
                '--data', str(adapter_data), '--out', str(root / f'coding-{name}'),
                '--milestones', '500', '1000', '--evaluate-current', '--hours', '6', '--port', '8035',
                '--tasks-file', str(tasks), '--direct-only', '--patience', '0'])
            result = json.loads((root / f'coding-{name}/status.json').read_text())
            if result['status'] != 'milestones_exhausted' or len(result['stages']) != 2:
                raise ValueError('Incomplete fixed adapter comparison')
            row['direct_scores'] = [s['direct']['passed'] for s in result['stages']]
            row['best_direct'] = max(row['direct_scores'])
            raw = json.loads((root / f'raw-{name}/results.json').read_text())
            row['raw_passed'] = sum(t['passed'] for t in raw['tasks']); save()
            journal(f'{name}: raw {row["raw_passed"]}/100; fixed adapter scores {row["direct_scores"]}/100. No agent-readiness claim.')
            if name == '2b' and row['best_direct'] < 10:
                state['status'] = 'foundation_gate_unmet'; break
            if name == '5b' and row['best_direct'] < max(25, state['milestones'][0]['best_direct'] + 3):
                state['status'] = 'foundation_gate_unmet'; break
        else:
            state['status'] = 'foundation_comparison_complete'
        state['next_stage'] = ('Verified agent curriculum and fresh repository evaluations; no readiness claim. '
            'If a gate failed, inspect learning curves and errors before spending more compute.')
    except Exception as error:
        state['status'] = 'interrupted' if stopping else 'failed'
        state['error'] = str(error)
        if not stopping: raise
    finally:
        save()
        journal(f'Controller state: **{state["status"]}**. {state.get("error", state.get("next_stage", ""))}')


if __name__ == '__main__': main()
