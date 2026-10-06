"""Only authored code executes here; public references are checked in Docker."""
import unittest

from microgpt.curate import normalize_reference_tests


class ReferenceTestNormalization(unittest.TestCase):
    def test_indented_assertion_cannot_hide_after_return(self):
        reference = 'def answer():\n    return 0\n'
        tests = normalize_reference_tests(['    assert answer() == 1'])
        with self.assertRaises(AssertionError):
            exec(reference + '\n' + '\n'.join(tests), {})

    def test_multiline_relative_indentation_is_preserved(self):
        tests = normalize_reference_tests(['    for n in range(3):\n        assert n < 3'])
        exec('\n'.join(tests), {})
        with self.assertRaises(ValueError):
            normalize_reference_tests(['# no actual test'])
