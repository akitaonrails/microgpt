import ast
import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from prepare_structural_repairs import structural_mutation, apply_exact, recovery_episode
from microgpt.tokenizer import Tokenizer
from microgpt.trajectories import assistant_examples


class StructuralRepairTests(unittest.TestCase):
    def test_mutation_always_changes_source_even_return_none(self):
        code = 'def f(x):\n    if x:\n        pass\n    return None\n'
        canonical = ast.unparse(ast.parse(code)) + '\n'
        for seed in range(32):
            mutated, family = structural_mutation(code, seed)
            self.assertNotEqual(mutated, canonical)
            ast.parse(mutated)

    def test_rejected_edits_and_exact_replacement(self):
        before = 'return 2\n'
        for old, new in [(before, before), ('missing', 'return 1\n')]:
            with self.assertRaises(ValueError): apply_exact(before, old, new)
        self.assertEqual(apply_exact(before, before, 'return 1\n'), 'return 1\n')

    def test_bad_action_is_context_not_supervised_target(self):
        record = {'group': 'train', 'seq_id': 'fixture', 'prompt': 'Return one.', 'signature': 'f()'}
        check = {'root': '/tmp/fixture', 'before': 'def f():\n    return 2\n',
                 'old_string': '    return 2\n', 'new_string': '    return 1\n',
                 'mutation': 'integer', 'failed': {'output': 'AssertionError'},
                 'repaired': {'output': 'PASS'}, 'errors': {'noop': 'No changes'},
                 'transient': {'output': 'TRANSIENT_ERROR'}}
        result = recovery_episode(record, check, [], 'noop')
        bad = next(m for m in result['messages'] if m.get('supervise') is False)
        self.assertEqual(bad['tool_calls'][0]['function']['arguments']['oldString'],
                         bad['tool_calls'][0]['function']['arguments']['newString'])
        tokenizer = Tokenizer()
        masked = assistant_examples(result, tokenizer, 8192, pack=True)[0]
        unmasked = copy.deepcopy(result)
        for message in unmasked['messages']: message.pop('supervise', None)
        plain = assistant_examples(unmasked, tokenizer, 8192, pack=True)[0]
        self.assertEqual(masked['input_ids'], plain['input_ids'])
        self.assertLess(sum(x != -100 for x in masked['labels']), sum(x != -100 for x in plain['labels']))


class SnapshotTests(unittest.TestCase):
    def test_snapshot_cannot_read_arbitrary_system_paths(self):
        from capture_repair_rollouts import snapshot
        with self.assertRaises(ValueError):
            snapshot({'messages': [{'role': 'system', 'content': 'Working directory: /home/user\n'}]})

    def test_snapshot_matches_isolated_fixture(self):
        from capture_repair_rollouts import snapshot
        import tempfile
        with tempfile.TemporaryDirectory(prefix='microgpt-eval-') as directory:
            work = Path(directory) / 'work'; work.mkdir()
            (work / 'solution.py').write_text('def f(): return 1\n')
            result = snapshot({'messages': [{'role': 'system', 'content': f'Working directory: {work}\n'}]})
            self.assertEqual(result['code'], 'def f(): return 1\n')

class CapturedPromptTests(unittest.TestCase):
    def test_cli_quotes_preserve_newlines_and_backslashes(self):
        from microgpt.opencode_format import cli_user_content
        text = 'Fix f()\nReturn "hello" and literal \\n.'
        result = cli_user_content(text)
        self.assertEqual(result, '"Fix f()\nReturn \\"hello\\" and literal \\n."')


if __name__ == '__main__': unittest.main()
