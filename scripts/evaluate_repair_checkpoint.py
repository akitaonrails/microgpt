"""Evaluate a frozen scratch checkpoint on fixed repairs using a private server."""
import argparse
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--tasks-file', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--port', type=int, default=8034)
    p.add_argument('--capture', action='store_true', help='Capture actual training-fixture requests and file snapshots')
    args = p.parse_args(); args.out.mkdir(parents=True, exist_ok=False)
    child = None
    def stop(*_):
        if child is not None and child.poll() is None: child.terminate()
        raise InterruptedError('Evaluation interrupted')
    for sig in (signal.SIGTERM, signal.SIGINT): signal.signal(sig, stop)
    with (args.out / 'server.log').open('w') as log:
        server = subprocess.Popen([sys.executable, '-m', 'microgpt.serve', '--checkpoint', str(args.checkpoint),
            '--device', 'cuda', '--attention', 'flash', '--constrain-tools', '--port', str(args.port)], stdout=log, stderr=subprocess.STDOUT)
        try:
            url = f'http://127.0.0.1:{args.port}'
            deadline = time.monotonic() + 300
            while time.monotonic() < deadline:
                if server.poll() is not None: raise RuntimeError('Private server exited')
                try:
                    with urllib.request.urlopen(url + '/health', timeout=2) as response: health = json.load(response)
                    if health['source']['checkpoint'] != str(args.checkpoint): raise RuntimeError('Wrong checkpoint on evaluation port')
                    break
                except OSError: time.sleep(1)
            else: raise TimeoutError('Checkpoint did not load in five minutes')
            with (args.out / 'evaluation.log').open('w') as output:
                command = ([sys.executable, 'scripts/capture_repair_rollouts.py', '--backend', url + '/v1',
                            '--out', str(args.out / 'rollouts')] if args.capture else
                           [sys.executable, 'scripts/eval_opencode.py', '--url', url + '/v1',
                            '--general-prompt', '--effort', 'none', '--out', str(args.out / 'repairs')])
                command += ['--tasks-file', str(args.tasks_file), '--label', str(args.checkpoint)]
                child = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
                if child.wait(): raise RuntimeError('Repair evaluation failed')
        finally:
            server.terminate()
            try: server.wait(timeout=15)
            except subprocess.TimeoutExpired: server.kill(); server.wait()


if __name__ == '__main__': main()
