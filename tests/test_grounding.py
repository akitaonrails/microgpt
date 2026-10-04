"""Verify constructed edits and their oracles before using them as supervision."""
import unittest
import json
from microgpt.grounding import episode
from microgpt.practice import TOOLS
from microgpt.chat import ChatRequest, normalize, parse_reply
from microgpt.protocol import body


class GroundingTests(unittest.TestCase):
    def test_edits_paths_and_recovery(self):
        recovered = False
        for seed in range(40):
            item = episode(seed, TOOLS[:3])
            messages, tools = normalize(ChatRequest(model='test', messages=item['messages'], tools=item['tools']))
            fixture = item['fixture']
            source = fixture['before']
            for message in messages:
                if message.get('tool_calls'):
                    reply, _ = parse_reply(body(message), tools, 'stop')
                    function = reply['tool_calls'][0]['function']
                    arguments = json.loads(function['arguments'])
                    if function['name'] in ('read', 'edit'):
                        self.assertEqual(arguments['filePath'], fixture['path'])
                    if function['name'] == 'edit':
                        self.assertEqual(source.count(arguments['oldString']), 1)
                        source = source.replace(arguments['oldString'], arguments['newString'])
                recovered |= 'TRANSIENT_ERROR' in (message.get('content') or '')
            self.assertEqual(source, fixture['after'])
            # Only fixed original templates execute, never external/model output.
            namespace = {}
            exec(source, namespace)
            for value in (-9, 0, 3, 12):
                self.assertEqual(namespace[fixture['name']](value), eval(fixture['expression'], {}, {'value': value}))
        self.assertTrue(recovered)

    def test_handoff_preserves_cumulative_functions(self):
        from microgpt.horizon_data import sessions
        for seed in range(4):
            first, second = sessions(seed, TOOLS[:3], rounds=8)
            self.assertEqual(first['group'], second['group'])
            self.assertEqual(len(first['fixture']['requirements']), 4)
            self.assertEqual(len(second['fixture']['requirements']), 8)
            self.assertIn('session handoff', second['messages'][2]['content'])
            for item in (first, second):
                normalize(ChatRequest(model='test', messages=item['messages'], tools=item['tools']))
                namespace = {}
                exec(item['fixture']['source'], namespace)
                for requirement in item['fixture']['requirements']:
                    for value in (-3, 0, 2.5):
                        self.assertEqual(namespace[requirement['name']](value), value * requirement['factor'] + requirement['offset'])
