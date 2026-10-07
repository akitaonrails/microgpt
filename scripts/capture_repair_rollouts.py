"""Capture only this run's public-fixture requests and pre-action file snapshots.

A loopback proxy forwards unchanged requests to our private model server. The
snapshot is restricted to the evaluator's temporary solution.py files.
"""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import subprocess
import sys
import threading
import urllib.request


def snapshot(request):
    for message in request['messages']:
        if message['role'] != 'system': continue
        content = message.get('content', '')
        if not isinstance(content, str): continue
        found = re.search(r'Working directory: (/tmp/microgpt-eval-[A-Za-z0-9_-]+/work)\s', content)
        if found:
            work = Path(found[1]); file = work / 'solution.py'
            if file.is_symlink() or file.resolve().parent != work or not file.is_file():
                raise ValueError('Unexpected snapshot path')
            return {'workdir': str(work), 'code': file.read_text()}
    raise ValueError('Request is not from our isolated repair fixture')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tasks-file', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--backend', default='http://127.0.0.1:8034/v1')
    p.add_argument('--label', required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    lock = threading.Lock()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_): pass
        def do_POST(self):
            raw = self.rfile.read(int(self.headers['Content-Length']))
            request = json.loads(raw)
            record = {'request': request, **snapshot(request)}
            with lock, (args.out / 'requests.jsonl').open('a') as stream:
                stream.write(json.dumps(record) + '\n')
            upstream = urllib.request.Request(args.backend + '/chat/completions', data=raw,
                headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(upstream, timeout=180) as response:
                data = response.read()
                self.send_response(response.status)
                self.send_header('Content-Type', response.headers.get('Content-Type', 'text/event-stream'))
                self.send_header('Content-Length', str(len(data)))
                self.end_headers(); self.wfile.write(data)
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        subprocess.run([sys.executable, 'scripts/eval_opencode.py', '--url', f'http://127.0.0.1:{server.server_port}/v1',
            '--out', str(args.out / 'evaluation'), '--label', args.label, '--tasks-file', str(args.tasks_file),
            '--general-prompt', '--effort', 'none'], check=True)
    finally:
        server.shutdown(); server.server_close()


if __name__ == '__main__': main()
