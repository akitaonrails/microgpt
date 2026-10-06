"""Cumulative public lessons keep split boundaries and withhold reference answers."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from prepare_public_horizon import episodes, fit_context, select_session, split_for
from prepare_verified_code import candidate, checked_references
from microgpt.tokenizer import Tokenizer


class PublicHorizonTests(unittest.TestCase):
    def test_context_handoffs_preserve_every_complete_turn(self):
        messages = [{'role': 'system', 'content': 'Instructions.'}]
        for index in range(3):
            messages.extend([{'role': 'user', 'content': str(index) + 'x' * 80},
                             {'role': 'assistant', 'content': 'y' * 80}])
        result = fit_context({'messages': messages, 'tools': [], 'group': 'same'}, Tokenizer(), 350)
        self.assertEqual(len(result), 3)
        self.assertEqual([m for r in result for m in r['messages'] if m['role'] == 'user'],
                         [m for m in messages if m['role'] == 'user'])
        self.assertTrue(all(r['messages'][-1]['role'] == 'assistant' and r['group'] == 'same' for r in result))
        with self.assertRaisesRegex(ValueError, 'single task'):
            fit_context({'messages': messages, 'tools': []}, Tokenizer(), 32)

    def test_composition_preserves_original_split_and_rejects_name_collisions(self):
        records = [{'seq_id': str(i), 'group': f'source-{i}',
                    'code': f'def fn_{i // 2}(value):\n    return value + {i}\n'} for i in range(400)]
        for seed in range(10):
            session = select_session(records, seed, 16)
            names = [ast.parse(r['code']).body[0].name for r in session['records']]
            self.assertEqual(len(set(names)), 16)
            self.assertTrue(all(split_for(r['group']) == split_for(session['group']) for r in session['records']))

    def test_handoff_retains_requirements_without_putting_answers_in_user_state(self):
        records = [{'seq_id': str(i), 'group': 'group', 'prompt': f'Implement fn_{i}(x).',
                    'signature': f'fn_{i}(x)'} for i in range(3)]
        checked = {'root': '/tmp/example', 'rounds': [
            {'before': 'VISIBLE_SOURCE', 'old_string': 'VISIBLE_SOURCE', 'new_string': 'REFERENCE_ANSWER',
             'positive': {'output': ''}} for _ in records]}
        output = episodes({'group': 'session', 'records': records}, checked, [], 2)
        self.assertEqual(len(output), 2)
        last = output[-1]['messages']
        users = [m['content'] for m in last if m['role'] == 'user']
        self.assertEqual(len(users), 1)
        self.assertIn('fn_2', users[0])
        state = last[4]['content']
        self.assertTrue(all(f'fn_{i}' in state for i in range(3)))
        self.assertNotIn('REFERENCE_ANSWER', state + ''.join(users))

    def test_reference_recovery_checks_source_hash_and_answer_identity(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        raw = {'id': 'one', 'average_test_score': 1.0,
               'input': 'Implement increment(value), returning the integer input plus one.',
               'output': '```python\ndef increment(value):\n    return value + 1\n```',
               'unit_tests': ['assert increment(3) == 4']}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard = root / 'data.parquet'
            pq.write_table(pa.Table.from_pylist([raw]), shard)
            manifest = {'files': [{'path': shard.name, 'sha256': hashlib.sha256(shard.read_bytes()).hexdigest()}]}
            (root / 'manifest.json').write_text(json.dumps(manifest))
            curriculum = root / 'checked.jsonl'
            record = {'upstream_id': 'one', 'group': 'original-group',
                      'messages': [{'content': '```python\n' + candidate(raw)['code'] + '```'}]}
            curriculum.write_text(json.dumps(record) + '\n')
            result = list(checked_references(root, curriculum, 1))
            self.assertEqual(result[0]['group'], 'original-group')
            record['messages'][0]['content'] = 'changed answer'
            curriculum.write_text(json.dumps(record) + '\n')
            with self.assertRaisesRegex(ValueError, 'Reference differs'):
                list(checked_references(root, curriculum, 1))
            manifest['files'][0]['sha256'] = 'bad'
            (root / 'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'Source hash mismatch'):
                list(checked_references(root, curriculum, 1))


if __name__ == '__main__':
    unittest.main()
