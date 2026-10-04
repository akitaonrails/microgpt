"""Checks for the new programming curriculum and optional tool grammar."""
import contextlib
import importlib.util
import io
import json
import unittest

from microgpt.programming_data import episode, task
from microgpt.chat import ChatRequest, normalize, parse_reply
from microgpt.finetune import bucket_weights
from microgpt.curate import convert
from microgpt.tokenizer import Tokenizer


class ProgrammingTests(unittest.TestCase):
    def test_authored_code_and_cumulative_edits(self):
        # Only our bounded, authored templates execute here; never public data.
        families = set()
        for seed in range(910000, 910100):
            record = episode(seed, [])
            normalize(ChatRequest(model='test', messages=record['messages']))
            content = '# Python utilities\n'
            for message in record['messages']:
                for call in message.get('tool_calls', []):
                    f = call['function']
                    if f['name'] == 'edit':
                        args = f['arguments']
                        self.assertEqual(content.count(args['oldString']), 1)
                        content = content.replace(args['oldString'], args['newString'], 1)
            for fixture in record['fixtures']:
                families.add(fixture['family'])
                scope = {}
                exec(fixture['code'], scope)
                function = scope[fixture['name']]
                for value, expected in fixture['cases']:
                    self.assertEqual(function(value), expected)
                if fixture['family'] == 16:
                    output = io.StringIO()
                    with contextlib.redirect_stdout(output):
                        self.assertIsNone(function('Nina'))
                    self.assertEqual(output.getvalue(), 'Welcome, Nina!\n')
        self.assertEqual(len(families), 18)

    def test_tool_json_can_contain_literal_protocol_tags(self):
        tool = {'type': 'function', 'function': {'name': 'edit', 'parameters': {'type': 'object'}}}
        arguments = {'newString': 'text = "</tool_call><tool_call><think></think>"'}
        text = '<tool_call>' + json.dumps({'name': 'edit', 'arguments': arguments}) + '</tool_call>'
        message, reason = parse_reply(text, [tool], 'stop')
        self.assertEqual(reason, 'tool_calls')
        self.assertEqual(json.loads(message['tool_calls'][0]['function']['arguments']), arguments)
        with self.assertRaises(ValueError):
            parse_reply(text + '<tool_call>{bad}', [tool], 'stop')
        self.assertNotIn('tool_calls', parse_reply(text + '<tool_call>{bad}', [tool], 'length')[0])

    def test_source_probability_does_not_increase_with_length_bins(self):
        buckets = {'code@8192': [1], 'code@32768': [2], 'chat@8192': [3]}
        self.assertEqual(bucket_weights(buckets, {'code': 0.98, 'chat': 0.02}), [0.49, 0.49, 0.02])
        with self.assertRaises(ValueError):
            bucket_weights(buckets, {'code': 1})

    def test_upstream_test_filter_is_not_our_verification(self):
        row = {'input': 'Task', 'output': 'Answer', 'id': 'sample', 'average_test_score': '0.9'}
        self.assertIsNone(convert(row, 'opencodeinstruct')[0])
        row['average_test_score'] = '1.0'
        record, error = convert(row, 'opencodeinstruct')
        self.assertIsNone(error)
        self.assertIn('never executed', record['verification'])

    def test_public_signature_keeps_original_split_group(self):
        import hashlib
        row = {'instruction': 'Return the input doubled.', 'code': 'def double(value):\n    return value * 2\n',
               'output': 'def double(value):\n    return value * 2\n', 'entry_point': 'double', 'seq_id': 1}
        record, error = convert(row, 'opencoder-educational')
        self.assertIsNone(error)
        self.assertIn('double(value)', record['messages'][0]['content'])
        self.assertEqual(record['group'], hashlib.sha256(row['instruction'].encode()).hexdigest())

    @unittest.skipUnless(importlib.util.find_spec('xgrammar'), 'Install the grammar extra')
    def test_grammar_rejects_malformed_unknown_and_missing_arguments(self):
        from microgpt.constraints import ToolGrammar
        grammar = ToolGrammar(Tokenizer())
        tools = [{'type': 'function', 'function': {'name': 'read', 'parameters': {
            'type': 'object', 'properties': {'filePath': {'type': 'string'}},
            'required': ['filePath'], 'additionalProperties': False}}}]
        valid = '<tool_call>{"name":"read","arguments":{"filePath":"src/ação.py"}}</tool_call>'
        matcher = grammar.matcher(tools)
        for token in valid.encode():
            self.assertTrue(matcher.accept_token(token))
        self.assertTrue(matcher.accept_token(256))
        for invalid in [valid.replace('read', 'delete'), valid.replace(',"arguments"', '"arguments"'),
                        '<tool_call>{"name":"read","arguments":{}}</tool_call>']:
            self.assertFalse(grammar.matcher(tools).accept_string(invalid))
        self.assertFalse(grammar.matcher([]).accept_string(valid))
        self.assertTrue(grammar.matcher([]).accept_string('Here is Python code.'))
        message, reason = parse_reply('<think>' + valid + '</think>Done.', tools, 'stop')
        self.assertNotIn('tool_calls', message)
        # Valid syntax does not authorize execution of a length-truncated response.
        self.assertNotIn('tool_calls', parse_reply(valid, tools, 'length')[0])
