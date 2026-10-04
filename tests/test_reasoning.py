"""Check original trace oracles and training targets, not model intelligence."""
import contextlib
import io
import unittest

from microgpt.reasoning_data import record, trace_problem, trace_record
from microgpt.tokenizer import Tokenizer
from microgpt.trajectories import assistant_examples


class ReasoningTests(unittest.TestCase):
    def test_mistakes_remain_context_but_are_not_loss_targets(self):
        tokenizer = Tokenizer(encoding='o200k_base')
        item = {'messages': [{'role': 'user', 'content': 'Try the task.'},
                             {'role': 'assistant', 'content': 'WRONG_TARGET', 'supervise': False},
                             {'role': 'user', 'content': 'That failed; correct it.'},
                             {'role': 'assistant', 'content': 'Corrected.'}]}
        for packed in (False, True):
            examples = assistant_examples(item, tokenizer, 1024, pack=packed)
            self.assertEqual(len(examples), 1)
            example = examples[0]
            self.assertIn('WRONG_TARGET', tokenizer.decode(example['input_ids']))
            target = tokenizer.decode([x for x in example['labels'] if x != -100 and x != tokenizer.eos])
            self.assertEqual(target, 'Corrected.')

    def test_each_family_has_every_effort_instruction(self):
        coverage = {family: set() for family in range(3)}
        for index in range(180):
            item = trace_record(index)
            coverage[item['family']].add(item['messages'][0]['content'])
        self.assertEqual([len(modes) for modes in coverage.values()], [4, 4, 4])

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
