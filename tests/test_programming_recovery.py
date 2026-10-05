"""Check that repair demonstrations preserve reference code and do not execute it."""
import json
import unittest

from scripts.prepare_programming_recovery import episode, reference
from microgpt.chat import ChatRequest, normalize


class RecoveryTests(unittest.TestCase):
    def test_repair_restores_exact_reference_and_switches_files(self):
        code = '# Utilities\ndef double(value):\n    return value * 2\n'
        record = {'group': 'reference-group', 'tools': [], 'messages': [
            {'role': 'user', 'content': json.dumps('Double the input. Implement this in solution.py and run check.')},
            {'role': 'assistant', 'tool_calls': [{'function': {'name': 'edit', 'arguments': {'newString': code}}}]}]}
        prompt, restored, broken = reference(record)
        self.assertIn('double(value)', prompt)
        self.assertEqual(restored, code)
        self.assertIn('raise NotImplementedError()', broken)
        result = episode([record, record], 42, True)
        normalize(ChatRequest(model='test', messages=result['messages']))
        files = set()
        for message in result['messages']:
            for call in message.get('tool_calls', []):
                function = call['function']
                if function['name'] == 'edit':
                    args = function['arguments']
                    self.assertEqual(args['oldString'], broken)
                    self.assertEqual(args['newString'], code)
                    files.add(args['filePath'])
        self.assertEqual(len(files), 2)

    def test_public_code_is_parsed_without_execution(self):
        record = {'messages': [
            {'role': 'user', 'content': json.dumps('Return one.')},
            {'tool_calls': [{'function': {'name': 'edit', 'arguments': {
                'newString': "raise RuntimeError('MUST NOT RUN')\ndef one():\n    return 1\n"}}}]}]}
        self.assertIn('one()', reference(record)[0])
