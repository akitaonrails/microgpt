"""Execute the frozen repair-data comparison, with guarded longer continuation.

Prerequisite: the 64 actual training-source rollouts have finished. Uses only our
scratch weights, private evaluation ports and NAS checkpoints; never promotes.
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
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    out = Path('runs/refined-agent-v1'); out.mkdir(exist_ok=False)
    data = Path('data/refined-agent-sft-v1'); run = Path('data/experiments/refined-agent-v1')
    parent = Path('runs/verified-code-learning-v1/checkpoint-500.pt')
    diagnostic = Path('data/repair-diagnostic-v1/evaluation.jsonl')
    if not run.resolve().is_relative_to(Path('/mnt/gigachad/microgpt/dataset')):
        raise ValueError('Checkpoints must be inside the authorized NAS dataset path')
    rollout = json.loads(Path('data/repair-training-rollouts-v1/evaluation/results.json').read_text())
    if len(rollout['tasks']) != 64: raise ValueError('Finish the training-source rollouts first')
    state = {'status': 'running', 'pid': os.getpid(), 'started': datetime.now(timezone.utc).isoformat(),
        'parent': str(parent), 'parent_sha256': sha(parent), 'diagnostic_sha256': sha(diagnostic),
        'plan_sha256': sha('docs/history/refined-agent-execution.md'),
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'phases': [], 'automatic_promotion': False, 'acceptance_used': False}
    child = None; stopping = False
    def save():
        state['updated'] = datetime.now(timezone.utc).isoformat()
        temp = out / 'status.tmp'; temp.write_text(json.dumps(state, indent=2) + '\n'); temp.replace(out / 'status.json')
    def stop(*_):
        nonlocal stopping
        stopping = True
        if child is not None and child.poll() is None: child.terminate()
    for sig in (signal.SIGTERM, signal.SIGINT): signal.signal(sig, stop)
    def execute(name, command):
        nonlocal child
        if stopping: raise InterruptedError('Interrupted')
        phase = {'name': name, 'command': command, 'status': 'running'}; state['phases'].append(phase); save()
        with (out / f'{name}.log').open('x') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            code = child.wait()
        child = None
        if stopping: raise InterruptedError('Interrupted')
        if code: raise RuntimeError(f'{name} exited {code}; inspect its log')
        phase['status'] = 'complete'; save()
    py = sys.executable
    try:
        execute('corrections', [py, 'scripts/prepare_rollout_corrections.py', '--captures', 'data/repair-training-rollouts-v1/requests.jsonl',
            '--tasks-file', 'data/repair-diagnostic-v1/training-rollouts.jsonl', '--out', 'data/rollout-corrections-v1'])
        execute('assemble', [py, 'scripts/assemble_refined_mix.py', '--out', 'data/refined-repairs-v1.jsonl'])
        execute('tokenize', [py, '-m', 'microgpt.trajectories', '--input', 'data/public-code-multiseed-v1/code.jsonl',
            'data/refined-repairs-v1.jsonl', 'data/smoltalk-large.jsonl', 'data/multilingual-large.jsonl', '--out', str(data),
            '--tokenizer', 'data/verified-opencodeinstruct-sft-v1/tokenizer', '--max-length', '8192', '--limit', '150000',
            '--pack-episodes', '--canonical-tool-ids'])
        from microgpt.finetune import load_data
        from microgpt.protocol import digest
        manifest, splits = load_data(data)
        correction_ids = {digest({'messages': r['messages'], 'tools': r['tools']}) for line in
            Path('data/rollout-corrections-v1/corrections.jsonl').open() if (r := json.loads(line))}
        retained = sum(json.loads(line)['episode'] in correction_ids for line in (data / 'train.jsonl').open())
        if not retained: raise ValueError('No actual model-failure corrections survived packing')
        state['data'] = {'manifest_sha256': sha(data / 'manifest.json'), 'counts': manifest['counts'],
                         'verified_model_corrections_retained': retained}
        del splits; save()
        for step in (1000, 3000):
            execute(f'control-{step}', [py, 'scripts/evaluate_repair_checkpoint.py', '--checkpoint',
                f'data/experiments/agent-mix-v1/checkpoint-{step}.pt', '--tasks-file', str(diagnostic),
                '--out', str(out / f'control-{step}'), '--port', '8034'])
        training_started = time.monotonic()
        weights = {'verified-opencodeinstruct': .5, 'executed-public-repair': .4, 'smoltalk': .08, 'multilingual': .02}
        command = [py, '-m', 'microgpt.finetune', '--checkpoint', str(parent), '--data', str(data), '--out', str(run),
            '--steps', '250', '--lr', '0.00001', '--seed', '20261008', '--microbatch', '1', '--accumulation', '16',
            '--eval-every', '250', '--eval-examples', '256', '--validation-seed', '20261007', '--max-length', '8192',
            '--device', 'cuda', '--attention', 'math']
        for source, weight in weights.items(): command += ['--source-weight', f'{source}:{weight}']
        execute('bootstrap', command)
        def continuation(name, milestones):
            remaining = 6 - (time.monotonic() - training_started) / 3600
            if remaining <= 0: return False
            execute(name, [py, 'scripts/continue_programming.py', '--run', str(run), '--data', str(data),
                '--out', str(out / name), '--milestones', *map(str, milestones), '--hours', str(remaining),
                '--port', '8034', '--tasks-file', 'evaluation/programming-v1/development.jsonl', '--direct-only',
                '--repair-eval', '--repair-tasks-file', str(diagnostic), '--patience', '0'])
            return True
        continuation('pilot', [1000, 3000])
        pilot = json.loads((out / 'pilot/status.json').read_text())
        decisions = []
        for stage in pilot['stages']:
            step = stage.get('metrics', {}).get('step')
            if step not in (1000, 3000) or 'diagnostic' not in stage: continue
            control = json.loads((out / f'control-{step}/repairs/results.json').read_text())
            control_score = sum(row['passed'] for row in control['tasks'])
            d = stage['diagnostic']; families = d['families']
            passed = (stage['direct']['passed'] >= 24 and d['passed'] >= control_score + 12
                and all(families.get(f, {}).get('passed', 0) >= 16 for f in ('structural', 'recovery'))
                and d['unsupported_pass_claim_flags'] == 0)
            decisions.append({'step': step, 'passed': passed, 'control': control_score,
                              'direct': stage['direct'], 'diagnostic': d})
        state['decisions'] = decisions; save()
        # Extend only the current 3K state; an earlier passing snapshot does not
        # license extending a later regressed optimizer state.
        if decisions and decisions[-1]['step'] == 3000 and decisions[-1]['passed']:
            continuation('extension', [6000, 12000])
            state['status'] = 'extended_evaluated_no_promotion'
        else:
            state['status'] = 'diagnostic_gate_unmet'
    except Exception as error:
        state.update(status='interrupted' if stopping else 'failed', error=str(error))
        if not stopping: raise
    finally: save()


if __name__ == '__main__': main()
