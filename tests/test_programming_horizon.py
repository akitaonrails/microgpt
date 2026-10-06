"""Long-session fixtures preserve requirements without giving answers to the model."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from eval_horizon import notes_for, programming_requirements, prompt_for, requirements
from microgpt.opencode_format import state_text, task_prompt


class ProgrammingHorizonTests(unittest.TestCase):
    def test_requirements_and_prompts_exclude_reference_answers(self):
        task = {'id': 'example', 'prompt': 'Implement echo(value).',
                'reference': 'def echo(value): return "REFERENCE_ONLY"', 'tests': 'assert s.echo(0) == "REFERENCE_ONLY"'}
        self.assertIn(task['prompt'], notes_for([task]))
        self.assertIn(task['prompt'], prompt_for(task))
        self.assertNotIn('REFERENCE_ONLY', notes_for([task]) + prompt_for(task))

    def test_duplicate_definitions_cannot_silently_overwrite_prior_tasks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'tasks.jsonl'
            path.write_text('\n'.join(json.dumps({'id': str(i), 'reference': 'def same(x): return x'}) for i in range(2)))
            with self.assertRaises(ValueError):
                programming_requirements(path, 1, 2)

    def test_legacy_arithmetic_prompts_and_notes_are_unchanged(self):
        tasks = requirements(42, 3)
        self.assertEqual(notes_for(tasks), state_text(tasks))
        self.assertEqual(prompt_for(tasks[0]), task_prompt(tasks[0]))


if __name__ == '__main__':
    unittest.main()
