"""Authored semantic mutations must stay valid Python and change behavior."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from prepare_executed_repairs import mutations, scoped_edit, random_imports


class ExecutedRepairTests(unittest.TestCase):
    def test_random_import_filter_handles_aliases_without_matching_prose(self):
        self.assertEqual(random_imports('from random import randint as choose\nimport secrets as hidden'), {'random', 'secrets'})
        self.assertEqual(random_imports('"Use a random-looking name."\nimport math'), set())

    def test_scoped_edit_preserves_other_functions_and_disambiguates_repeated_lines(self):
        before = 'def first():\n    return 1\n\ndef second():\n    return 1\n'
        after = 'def first():\n    return 1\n\ndef second():\n    return 2\n'
        old, new = scoped_edit(before, after)
        self.assertEqual(before.count(old), 1)
        self.assertEqual(before.replace(old, new, 1), after)
        self.assertNotIn('def first', old + new)

    def test_scoped_insert_preserves_existing_code(self):
        before = 'def first():\n    return 1\n'
        after = before + '\ndef second():\n    return 2\n'
        old, new = scoped_edit(before, after)
        self.assertTrue(old)
        self.assertEqual(before.replace(old, new, 1), after)

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
