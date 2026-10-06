"""Check authored renaming semantics; never execute downloaded code here."""
import ast
import unittest
from scripts.prepare_interface_curriculum import rename


class InterfaceTests(unittest.TestCase):
    def test_recursive_name_and_callers_are_renamed(self):
        source = 'def factorial(n):\n    return 1 if n < 2 else n * factorial(n-1)\n'
        tree = ast.parse(source)
        changed = rename(tree, 'factorial', 'calculate_abc123')
        scope = {}
        exec(changed, scope)
        self.assertEqual(scope['calculate_abc123'](5), 120)
        self.assertEqual(tree.body[0].name, 'factorial')
        self.assertNotIn('factorial(', changed)

    def test_attribute_names_and_string_values_are_not_rewritten(self):
        source = 'def run(obj):\n    return obj.run("run")\n'
        changed = rename(ast.parse(source), 'run', 'handle_123abc')
        self.assertIn('obj.run(', changed)
        self.assertIn("'run'", changed)
