"""A rejected reference invalidates every episode of its cumulative session."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.filter_public_horizon import main


class PublicHorizonFilterTests(unittest.TestCase):
    def test_rejected_source_removes_whole_session(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checked = root / 'checked'
            checked.mkdir()
            code = checked / 'code.jsonl'
            code.write_text(json.dumps({'upstream_id': 'good'}) + '\n')
            (checked / 'manifest.json').write_text(json.dumps({
                'code_sha256': hashlib.sha256(code.read_bytes()).hexdigest()}))
            source = root / 'sessions.jsonl'
            rows = [{'group': 'broken-session', 'upstream_ids': ['good']},
                    {'group': 'broken-session', 'upstream_ids': ['good', 'bad']},
                    {'group': 'retained-session', 'upstream_ids': ['good']}]
            source.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            output = root / 'filtered.jsonl'
            with patch('sys.argv', ['filter', '--checked', str(checked), '--input', str(source), '--out', str(output)]):
                main()
            self.assertEqual([json.loads(line) for line in output.read_text().splitlines()], rows[2:])
            report = json.loads(output.with_suffix('.filter.json').read_text())
            self.assertEqual(report['retained_sessions'], 1)
            self.assertEqual(report['rejected_episodes'], 2)
