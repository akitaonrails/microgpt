"""Wire contract checks; these scripted replies are not model capability tests."""
import json
import unittest

try:
    from fastapi.testclient import TestClient
    from microgpt.chat import ChatRequest, normalize, parse_reply
    from microgpt.inference import Completion
    from microgpt.serve import create_app
    SERVER_AVAILABLE = True
except ImportError:
    SERVER_AVAILABLE = False


@unittest.skipUnless(SERVER_AVAILABLE, "Install .[serve] for API checks")
class ChatTests(unittest.TestCase):
    def setUp(self):
        self.tool = {"type": "function", "function": {"name": "read", "parameters": {
            "type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}

    def test_tool_round_trip_and_invalid_calls(self):
        message, reason = parse_reply('<tool_call>{"name":"read","arguments":{"path":"a.py"}}</tool_call>', [self.tool], "stop")
        self.assertEqual(reason, "tool_calls")
        call = message["tool_calls"][0]
        request = ChatRequest(model="microgpt", messages=[{"role": "user", "content": "Read a.py"}, message,
                              {"role": "tool", "tool_call_id": call["id"], "content": "print(1)"}])
        messages, _ = normalize(request)
        self.assertEqual(messages[1]["tool_calls"][0]["function"]["arguments"], {"path": "a.py"})
        for text in ['<tool_call>{"name":"delete","arguments":{}}</tool_call>',
                     '<tool_call>{"name":"read","arguments":{"path":12}}</tool_call>']:
            with self.assertRaises(ValueError):
                parse_reply(text, [self.tool], "stop")
        incomplete, reason = parse_reply('<tool_call>{"name":"read"', [self.tool], "length")
        self.assertNotIn("tool_calls", incomplete)
        self.assertEqual(reason, "length")
        with self.assertRaises(ValueError):
            normalize(ChatRequest(model="microgpt", messages=[{"role": "tool", "tool_call_id": "wrong", "content": "x"}]))

    def test_canonical_ids_preserve_associations_and_training_parity(self):
        import copy
        from microgpt.protocol import canonical_messages, prompt_ids
        from microgpt.tokenizer import Tokenizer
        from microgpt.trajectories import assistant_examples
        tokenizer = Tokenizer(encoding='o200k_base')
        messages = [{'role': 'user', 'content': 'Read both files.'},
                    {'role': 'assistant', 'tool_calls': [
                        {'id': name, 'type': 'function', 'function': {'name': 'read', 'arguments': {'path': path}}}
                        for name, path in [('opaque-a', 'a.py'), ('opaque-b', 'b.py')]]},
                    {'role': 'tool', 'tool_call_id': 'opaque-b', 'content': 'B'},
                    {'role': 'tool', 'tool_call_id': 'opaque-a', 'content': 'A'},
                    {'role': 'assistant', 'content': 'Read both.'}]
        canonical = canonical_messages(messages)
        self.assertEqual([m['tool_call_id'] for m in canonical if m['role'] == 'tool'], ['call_2', 'call_1'])
        self.assertEqual(messages[1]['tool_calls'][0]['id'], 'opaque-a')
        changed = copy.deepcopy(messages)
        for call in changed[1]['tool_calls']:
            call['id'] += '-different-uuid'
        for message in changed[2:4]:
            message['tool_call_id'] += '-different-uuid'
        for packed in (False, True):
            first = assistant_examples({'messages': messages, 'tools': [self.tool]}, tokenizer, 4096,
                                       pack=packed, canonical_tool_ids=True)
            second = assistant_examples({'messages': changed, 'tools': [self.tool]}, tokenizer, 4096,
                                        pack=packed, canonical_tool_ids=True)
            self.assertEqual(first, second)
            prefix = prompt_ids(tokenizer, messages[:-1], [self.tool], canonical_tool_ids=True)
            self.assertEqual(first[-1]['input_ids'][:len(prefix)], prefix)

    def test_http_stream_auth_usage_and_context(self):
        class Backend:
            context = 100
            source = {"test": True}
            def encode(self, messages, tools):
                return [1, 2, 3]
            def complete(self, ids, count, temperature, stop, seed):
                return Completion("hello", len(ids), 1, 2)
        client = TestClient(create_app(Backend(), api_key="test", max_output=200))
        self.assertEqual(client.get("/v1/models").status_code, 401)
        headers = {"Authorization": "Bearer test"}
        payload = {"model": "microgpt", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 10}
        response = client.post("/v1/chat/completions", json=payload, headers=headers)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["usage"]["prompt_tokens_details"]["cached_tokens"], 2)
        response = client.post("/v1/chat/completions", json={**payload, "stream": True, "stream_options": {"include_usage": True}}, headers=headers)
        events = [line[6:] for line in response.text.splitlines() if line.startswith("data: ")]
        self.assertEqual(events[-1], "[DONE]")
        self.assertEqual(json.loads(events[-2])["usage"]["total_tokens"], 4)
        self.assertEqual(client.post("/v1/chat/completions", json={**payload, "max_tokens": 100}, headers=headers).status_code, 400)
        self.assertEqual(client.post("/v1/chat/completions", json={**payload, "model": "wrong"}, headers=headers).status_code, 404)

    def test_thinking_is_separate_from_actions_and_survives_streaming(self):
        from microgpt.chat import sse_events
        from microgpt.protocol import body
        call = '<tool_call>{"name":"read","arguments":{"path":"a.py"}}</tool_call>'
        message, reason = parse_reply('<think>Consider ' + call + '</think>42', [self.tool], 'stop')
        self.assertNotIn('tool_calls', message)
        self.assertEqual(message['content'], '42')
        unfinished, _ = parse_reply('<think>' + call, [self.tool], 'length')
        self.assertNotIn('tool_calls', unfinished)
        with self.assertRaisesRegex(ValueError, 'unclosed thinking'):
            parse_reply('<think>' + call, [self.tool], 'stop')
        message, reason = parse_reply('<think>Inspect first.</think>' + call, [self.tool], 'stop')
        self.assertEqual(reason, 'tool_calls')
        request = ChatRequest(model='microgpt', reasoning_effort='high', messages=[
            {'role': 'user', 'content': 'Read a.py'}, message,
            {'role': 'tool', 'tool_call_id': message['tool_calls'][0]['id'], 'content': '42'}])
        messages, _ = normalize(request)
        self.assertIn('edge cases', messages[0]['content'])
        self.assertEqual(body(messages[2]), body(message))
        reparsed, _ = parse_reply(body(messages[2]), [self.tool], 'stop')
        self.assertEqual(reparsed['reasoning_content'], 'Inspect first.')
        self.assertEqual(reparsed['tool_calls'][0]['function']['arguments'], message['tool_calls'][0]['function']['arguments'])
        response = {'id': 'test', 'created': 0, 'model': 'microgpt', 'choices': [{'message': message, 'finish_reason': reason}]}
        events = list(sse_events(response))
        self.assertIn('reasoning_content', events[1])
        self.assertIn('tool_calls', events[2])
        with self.assertRaises(ValueError):
            parse_reply('ambiguous <think>' + call + '</think>', [self.tool], 'stop')

    def test_invalid_generation_is_not_a_retryable_server_failure(self):
        class Backend:
            context = 100
            source = {'test': True}
            def encode(self, messages, tools):
                return [1]
            def complete(self, *args):
                return Completion('<tool_call>{"name":"unknown","arguments":{}}</tool_call>', 1, 12)
        client = TestClient(create_app(Backend()))
        response = client.post('/v1/chat/completions', json={'model': 'microgpt', 'max_tokens': 20,
                              'messages': [{'role': 'user', 'content': 'Read a.py'}], 'tools': [self.tool]})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()['error']['type'], 'invalid_model_output')

    def test_unchanged_tool_loop_stops_but_changed_results_and_user_turns_reset(self):
        from microgpt.chat import repeated_tool_cycle
        history = [{'role': 'user', 'content': 'Read a.py'}]
        for index in range(3):
            history.extend([{'role': 'assistant', 'tool_calls': [{'id': str(index), 'type': 'function',
                'function': {'name': 'read', 'arguments': {'filePath': 'a.py'}}}]},
                {'role': 'tool', 'tool_call_id': str(index), 'content': 'unchanged'}])
        self.assertTrue(repeated_tool_cycle(history))
        reminder = {'role': 'assistant', 'content': 'CRITICAL - MAXIMUM STEPS REACHED\nRespond with text only.'}
        self.assertTrue(repeated_tool_cycle(history + [reminder]))
        # The HTTP boundary must stop the loop before touching the backend.
        response = TestClient(create_app(object())).post('/v1/chat/completions', json={
            'model': 'microgpt', 'max_tokens': 20, 'messages': history + [reminder]})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()['error']['type'], 'agent_loop_error')
        self.assertFalse(repeated_tool_cycle(history[:-2]))
        self.assertFalse(repeated_tool_cycle(history + [{'role': 'user', 'content': 'Read it again.'}]))
        history[-1]['content'] = 'changed'
        self.assertFalse(repeated_tool_cycle(history))
        history = [{'role': 'user', 'content': 'Fix a.py'}]
        for index in range(6):
            name = 'read' if index % 2 == 0 else 'edit'
            history.extend([{'role': 'assistant', 'tool_calls': [{'function': {'name': name, 'arguments': {}}}]},
                            {'role': 'tool', 'content': 'same source' if name == 'read' else 'ambiguous edit'}])
        self.assertTrue(repeated_tool_cycle(history))
        history[-1]['content'] = 'Edit succeeded'
        self.assertFalse(repeated_tool_cycle(history))


if __name__ == "__main__":
    unittest.main()
