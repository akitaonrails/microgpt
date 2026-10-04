"""Check original trace oracles and training targets, not model intelligence."""
import contextlib
import io
import unittest

from microgpt.reasoning_data import record, trace_problem
from microgpt.tokenizer import Tokenizer
from microgpt.trajectories import assistant_examples


class ReasoningTests(unittest.TestCase):
    def test_original_program_oracles_and_paired_groups(self):
        for family in range(4):
            for seed in range(40):
                problem = trace_problem(seed, family)
                output = io.StringIO()
                # Only our fixed, bounded templates run here, never downloaded code.
                with contextlib.redirect_stdout(output):
                    exec(problem['code'], {})
                self.assertEqual(output.getvalue().strip(), problem['answer'])
        direct = record('question', 'work', '42', 'test', 'none')
        thinking = record('question', 'work', '42', 'test', 'high')
        self.assertEqual(direct['group'], thinking['group'])
        tokenizer = Tokenizer(encoding='o200k_base')
        example = assistant_examples(thinking, tokenizer, 1024, pack=True)[0]
        target = tokenizer.decode([x for x in example['labels'] if x != -100 and x != tokenizer.eos])
        self.assertEqual(target, '<think>\nwork\n</think>\n\n42')
        self.assertNotIn('question', target)
