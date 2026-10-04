"""Capture installed OpenCode's public tool schemas using a scripted local reply.

Stores schemas only, never user/system messages or credentials. No model or cloud
service is involved. The fixture never requests tool execution.
"""
import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import subprocess
import tempfile
import threading

from eval_opencode import configure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Choose a fresh output file')
    captured = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            captured.extend(request.get('tools', []))
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.end_headers()
            base = {'id': 'fixture', 'object': 'chat.completion.chunk', 'created': 0, 'model': 'microgpt'}
            for delta, reason in [({'role': 'assistant', 'content': 'Ready.'}, None), ({}, 'stop')]:
                self.wfile.write(('data: ' + json.dumps({**base, 'choices': [{'index': 0, 'delta': delta, 'finish_reason': reason}]}) + '\n\n').encode())
            self.wfile.write(b'data: [DONE]\n\n')
    server = HTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with tempfile.TemporaryDirectory(prefix='microgpt-schema-') as directory:
            base = Path(directory)
            work = base / 'work'
            work.mkdir()
            grader = base / 'grader.py'
            grader.write_text("print('fixture')\n")
            _, environment = configure(work, grader, f'http://127.0.0.1:{server.server_port}/v1', 1)
            subprocess.run(['opencode', 'run', '--pure', '--title', 'Tool schema capture', '--dir', str(work),
                            '--agent', 'microgpt', '-m', 'microgpt/microgpt', '--format', 'json', 'Say ready.'],
                           env=environment, check=True, capture_output=True, timeout=45)
    finally:
        server.shutdown()
    tools = {t['function']['name']: t for t in captured}
    if not all(name in tools for name in ('read', 'edit', 'check')):
        raise RuntimeError(f'Missing tools: {list(tools)}')
    args.out.write_text(json.dumps([tools[n] for n in ('read', 'edit', 'check')], indent=2) + '\n')
    print(json.dumps({'tools': list(tools), 'out': str(args.out)}))


if __name__ == '__main__':
    main()
