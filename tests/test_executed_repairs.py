"""Authored semantic mutations must stay valid Python and change behavior."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from prepare_executed_repairs import mutations


class ExecutedRepairTests(unittest.TestCase):
    def test_boundary_mutation_is_killed_by_an_edge_case(self):
        code = 'def below(value, bound):\n    return value < bound\n'
        variants = mutations(code, seed=4)
        self.assertEqual(len(variants), 1)
        reference, changed = {}, {}
        exec(code, reference)
        exec(variants[0]['code'], changed)
        self.assertFalse(reference['below'](3, 3))
        self.assertTrue(changed['below'](3, 3))
        self.assertEqual(variants, mutations(code, seed=4))

    def test_literal_mutation_preserves_docstring_and_function_signature(self):
        code = 'def increment(value):\n    "Increase the input."\n    return value + 1\n'
        variants = mutations(code, seed=1, limit=20)
        self.assertEqual(len(variants), 2)
        for variant in variants:
            scope = {}; exec(variant['code'], scope)
            self.assertNotEqual(scope['increment'](4), 5)
            self.assertEqual(scope['increment'].__doc__, 'Increase the input.')


if __name__ == '__main__':
    unittest.main()
