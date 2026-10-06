"""The public-source decoder must preserve Python escapes and real test controls."""
import ast
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from prepare_verified_code import candidate, parse_tests, split_groups


class VerifiedCodeTests(unittest.TestCase):
    def test_split_groups_link_code_and_prompts_transitively(self):
        items = [
            {'seq_id': 'a', 'prompt': 'same request', 'code': 'def a(x): return x'},
            {'seq_id': 'b', 'prompt': 'same request', 'code': 'def b(x): return x + 1'},
            {'seq_id': 'c', 'prompt': 'another request', 'code': 'def c(x): return x + 1'},
            {'seq_id': 'd', 'prompt': 'independent', 'code': 'def d(x): return x - 1'},
        ]
        groups = split_groups(items)
        self.assertEqual(groups['a'], groups['c'])
        self.assertEqual(groups['a'], groups['b'])
        self.assertNotEqual(groups['a'], groups['d'])

    def test_plain_and_extra_json_escaped_tests_preserve_literals(self):
        plain = 'assert identity("a\\nb") == "a\\nb"\n'
        expected = ast.dump(ast.parse(plain))
        for value in (plain, json.dumps(plain)[1:-1]):
            self.assertEqual(ast.dump(ast.parse('\n'.join(parse_tests(json.dumps([value]))))), expected)

    def test_rejects_assertion_free_tests(self):
        with self.assertRaises(ValueError):
            parse_tests('["identity(2)"]')

    def test_empty_control_keeps_signature_and_raises(self):
        record = candidate({'average_test_score': 1.0, 'id': 'authored',
            'input': 'Please implement a function that returns its input integer unchanged.',
            'output': '```python\ndef identity(value):\n    return value\n```',
            'unit_tests': '["assert identity(4) == 4"]'})
        # Authored fixtures only; downloaded code is never executed by this test.
        scope = {}; exec(record['code'], scope); self.assertEqual(scope['identity'](4), 4)
        exec(record['empty'], scope)
        with self.assertRaises(NotImplementedError):
            scope['identity'](4)


if __name__ == '__main__':
    unittest.main()
