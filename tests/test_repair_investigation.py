"""Check that the contrastive probe isolates a real edit without changing input."""
import copy
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from investigate_repairs import edit_pair
from microgpt.tokenizer import Tokenizer


class RepairInvestigationTests(unittest.TestCase):
    def test_counterfactual_shares_history_and_does_not_mutate_record(self):
        record = {'tools': [], 'messages': [
            {'role': 'user', 'content': 'Fix the boundary.'},
            {'role': 'assistant', 'content': '', 'tool_calls': [{
                'id': 'original-id', 'type': 'function', 'function': {'name': 'edit',
                'arguments': {'filePath': 'solution.py', 'oldString': 'return x <= 3\n',
                              'newString': 'return x < 3\n'}}}]},
            {'role': 'tool', 'tool_call_id': 'original-id', 'content': 'Edit applied.'},
            {'role': 'assistant', 'content': 'Done.'}]}
        before = copy.deepcopy(record)
        tokenizer = Tokenizer()
        prefix, wanted, copied, first, changed = edit_pair(record, tokenizer)
        self.assertEqual(record, before)
        self.assertEqual(wanted[:first], copied[:first])
        self.assertNotEqual(wanted[first], copied[first])
        self.assertIn(first, changed)
        self.assertTrue(all(0 <= index < len(wanted) for index in changed))
        self.assertIn('Fix the boundary.', tokenizer.decode(prefix))
        self.assertIn('return x < 3', tokenizer.decode(wanted))
        self.assertEqual(tokenizer.decode(copied).count('return x <= 3'), 2)


if __name__ == '__main__':
    unittest.main()
