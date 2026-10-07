"""A few checks for data boundaries; capability is evaluated separately."""
import tempfile
import unittest

try:
    from microgpt.trajectories import assistant_examples
    from microgpt.tokenizer import Tokenizer
    AVAILABLE = True
except ImportError:
    AVAILABLE = False


@unittest.skipUnless(AVAILABLE, "Install .[tune] for training checks")
class TrainingTests(unittest.TestCase):
    def test_validation_sample_is_reproducible_and_does_not_mutate_training_buckets(self):
        from microgpt.finetune import validation_sample
        buckets = {'b': [10, 11, 12], 'a': [0, 1, 2, 3, 4, 5]}
        self.assertEqual(validation_sample(buckets, 5), [0, 10, 1, 11, 2])
        sampled = validation_sample(buckets, 6, seed=17)
        self.assertEqual(sampled, validation_sample(buckets, 6, seed=17))
        self.assertEqual(len(set(sampled)), 6)
        self.assertEqual(sum(index >= 10 for index in sampled), 3)
        self.assertEqual(buckets, {'b': [10, 11, 12], 'a': [0, 1, 2, 3, 4, 5]})

    def test_padded_microbatch_preserves_episode_mean_loss_and_gradients(self):
        import torch
        from microgpt.model import GPT, ModelConfig
        from microgpt.finetune import loss_for, loss_for_batch
        torch.manual_seed(19)
        model = GPT(ModelConfig(vocab_size=32, context=256, width=32, layers=1, heads=2)).train()
        model.gradient_checkpointing = True
        long_ids = [i % 31 + 1 for i in range(170)]
        examples = [
            {'input_ids': [1, 2, 3, 4, 5], 'labels': [-100, -100, 3, 4, 5]},
            {'input_ids': [2, 7, 9, 10, 11, 12, 13, 14], 'labels': [-100, -100, 9, -100, 11, -100, 13, 14]},
            {'input_ids': long_ids, 'labels': [-100] * 12 + long_ids[12:]},
        ]
        device = torch.device('cpu')
        losses = [loss_for(model, example, device) for example in examples]
        expected = torch.stack([loss for loss, _ in losses]).mean()
        expected.backward()
        gradients = {name: parameter.grad.clone() for name, parameter in model.named_parameters()}
        model.zero_grad(set_to_none=True)
        observed, count = loss_for_batch(model, examples, device)
        observed.backward()
        self.assertEqual(count, sum(count for _, count in losses))
        torch.testing.assert_close(observed, expected, atol=1e-6, rtol=1e-5)
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter.grad, gradients[name], atol=1e-6, rtol=1e-5)

    def test_selected_logits_match_full_masked_loss_and_gradients(self):
        import torch
        import torch.nn.functional as F
        from microgpt.model import GPT, ModelConfig
        from microgpt.finetune import loss_for
        torch.manual_seed(7)
        model = GPT(ModelConfig(vocab_size=32, width=32, layers=1, heads=2)).train()
        example = {"input_ids": [1, 2, 3, 4, 5, 6], "labels": [-100, -100, -100, 4, 5, 6]}
        ids = torch.tensor([example["input_ids"]])
        full = model(ids)[0]
        reference = F.cross_entropy(full[:, :-1].reshape(-1, 32), torch.tensor(example["labels"][1:]))
        reference.backward()
        gradients = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
        model.zero_grad(set_to_none=True)
        selected, count = loss_for(model, example, torch.device("cpu"))
        selected.backward()
        self.assertEqual(count, 3)
        torch.testing.assert_close(selected, reference)
        for name, p in model.named_parameters():
            if name in gradients:
                torch.testing.assert_close(p.grad, gradients[name], atol=1e-6, rtol=1e-5)
        model.zero_grad(set_to_none=True)
        _, chunked = model(ids[:, :-1], torch.tensor([example['labels'][1:]]), loss_chunk=2)
        chunked.backward()
        torch.testing.assert_close(chunked, reference)
        for name, p in model.named_parameters():
            if name in gradients:
                torch.testing.assert_close(p.grad, gradients[name], atol=1e-6, rtol=1e-5)

    def test_assistant_mask_preserves_tool_observations_as_context(self):
        tokenizer = Tokenizer()
        record = {"messages": [
            {"role": "user", "content": "Read a.py"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "a", "type": "function", "function": {"name": "read", "arguments": {"filePath": "a.py"}}}]},
            {"role": "tool", "tool_call_id": "a", "content": "UNTRUSTED OBSERVATION"},
            {"role": "assistant", "content": "Finished reading."}],
            "tools": [{"type": "function", "function": {"name": "read", "parameters": {"type": "object", "properties": {"filePath": {"type": "string"}}, "required": ["filePath"]}}}]}
        examples = assistant_examples(record, tokenizer, 4096)
        self.assertEqual(len(examples), 2)
        from microgpt.protocol import prompt_ids
        from microgpt.chat import ChatRequest, normalize
        messages, tools = normalize(ChatRequest(model='training', **record))
        prefix = prompt_ids(tokenizer, messages[:1], tools)
        self.assertEqual(examples[0]['input_ids'][:len(prefix)], prefix)
        self.assertTrue(all(label == -100 for label in examples[0]['labels'][:len(prefix)]))
        for ex in examples:
            targets = tokenizer.decode([x for x in ex["labels"] if x != -100])
            self.assertNotIn("UNTRUSTED OBSERVATION", targets)
            self.assertNotIn("Read a.py", targets)
        self.assertIn("<tool_call>", tokenizer.decode([x for x in examples[0]["labels"] if x != -100]))
        self.assertIn("UNTRUSTED OBSERVATION", tokenizer.decode(examples[1]["input_ids"]))
        self.assertEqual(assistant_examples(record, tokenizer, 5), [])
        packed = assistant_examples(record, tokenizer, 4096, pack=True)[0]
        self.assertEqual(packed['input_ids'], examples[-1]['input_ids'])
        expected_targets = [x for ex in examples for x in ex['labels'] if x != -100]
        self.assertEqual([x for x in packed['labels'] if x != -100], expected_targets)
        self.assertEqual(assistant_examples(record, tokenizer, 5, pack=True), [])

    def test_pretraining_resume_rejects_changed_experiment(self):
        from microgpt.train import validate_resume_settings
        settings = dict(steps=100, learning_rate=0.001, batch_size=2,
                        accumulation=4, sequence_length=32, attention='sdpa')
        validate_resume_settings(settings, settings, 32)
        for field, value in [('steps', 200), ('learning_rate', 0.002),
                             ('batch_size', 4), ('accumulation', 8)]:
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'identical training settings'):
                validate_resume_settings(settings, {**settings, field: value}, 32, True)
        for previous in [settings, {k: v for k, v in settings.items() if k != 'attention'}]:
            with self.assertRaisesRegex(ValueError, 'attention'):
                validate_resume_settings(previous, {**settings, 'attention': 'math'}, 32)
            validate_resume_settings(previous, {**settings, 'attention': 'math'}, 32, True)

    def test_staged_pretraining_preserves_schedule_and_weights(self):
        import subprocess
        import sys
        from pathlib import Path
        import torch
        from microgpt.prepare import demo_documents, prepare
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            prepare(list(demo_documents()), base / 'data')
            config = base / 'model.toml'
            config.write_text('[model]\ncontext=32\nlayers=1\nheads=2\nwidth=16\n'
                              '[train]\nsteps=4\nbatch_size=1\naccumulation=1\n'
                              'learning_rate=0.001\nwarmup=1\neval_every=2\neval_batches=1\nseed=42\n')
            command = [sys.executable, '-m', 'microgpt.train', '--config', str(config),
                       '--data', str(base / 'data'), '--device', 'cpu', '--attention', 'math']
            subprocess.run(command + ['--out', str(base / 'full')], check=True, capture_output=True)
            subprocess.run(command + ['--out', str(base / 'staged'), '--stop-at-step', '2'], check=True, capture_output=True)
            midpoint = torch.load(base / 'staged/last.pt', weights_only=True)
            self.assertEqual(midpoint['step'], 2)
            self.assertEqual(midpoint['settings']['steps'], 4)
            # Old checkpoints without this optional field still resume exactly.
            midpoint["model_config"].pop("residual_scale")
            torch.save(midpoint, base / "staged/last.pt")
            rejected = subprocess.run(command + ['--out', str(base / 'staged'),
                '--resume', str(base / 'staged/last.pt'), '--steps', '8'], capture_output=True, text=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn('identical training settings', rejected.stderr)
            self.assertEqual(torch.load(base / 'staged/last.pt', weights_only=True)['step'], 2)
            subprocess.run(command + ['--out', str(base / 'staged'), '--resume', str(base / 'staged/last.pt')], check=True, capture_output=True)
            full = torch.load(base / 'full/last.pt', weights_only=True)
            staged = torch.load(base / 'staged/last.pt', weights_only=True)
            for name in full['model']:
                torch.testing.assert_close(full['model'][name], staged['model'][name], rtol=0, atol=0)

    def test_packed_and_individual_turn_losses_agree(self):
        import torch
        from microgpt.model import GPT, ModelConfig
        from microgpt.finetune import loss_for
        record = {'messages': [
            {'role': 'user', 'content': 'Double 3.'},
            {'role': 'assistant', 'content': '6'},
            {'role': 'user', 'content': 'Now double 4.'},
            {'role': 'assistant', 'content': '8'}]}
        tokenizer = Tokenizer()
        torch.manual_seed(19)
        model = GPT(ModelConfig(context=1024, width=32, heads=2, layers=1)).eval()
        packed = assistant_examples(record, tokenizer, 1024, pack=True)[0]
        separate = assistant_examples(record, tokenizer, 1024)
        with torch.no_grad():
            observed, count = loss_for(model, packed, torch.device('cpu'))
            losses = [loss_for(model, item, torch.device('cpu')) for item in separate]
        self.assertEqual(count, sum(n for _, n in losses))
        expected = sum(loss * n for loss, n in losses) / count
        torch.testing.assert_close(observed, expected)

    def test_indexed_curriculum_keeps_whole_episodes_and_checks_hash(self):
        import hashlib
        import json
        from pathlib import Path
        from microgpt.episodes import Episodes
        records = [{'source': 'a', 'group': 'one', 'input_ids': [1, 2], 'tool_calls': 2},
                   {'source': 'b', 'group': 'two', 'input_ids': list(range(20)), 'tool_calls': 16}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'train.jsonl'
            path.write_text(''.join(json.dumps(item) + '\n' for item in records))
            dataset = Episodes(path, hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(dataset[1], records[1])
            self.assertEqual(dataset.buckets(10), {'a': [0]})
            self.assertEqual(dataset.buckets(32, min_calls=8), {'b': [1]})
            with self.assertRaises(ValueError):
                Episodes(path, 'wrong hash')

    def test_practice_recovery_and_handoff_are_real_context(self):
        from microgpt.practice import episode
        record = episode(100000, 8, handoff=True)
        self.assertEqual(record['recovered_errors'], 1)
        self.assertGreaterEqual(record['tool_calls'], 20)
        calls = [m['tool_calls'][0]['function'] for m in record['messages'] if m.get('tool_calls')]
        self.assertEqual(calls[0]['name'], 'read')
        self.assertTrue(calls[0]['arguments']['filePath'].endswith('TASK_STATE.md'))
        tool_results = [m['content'] for m in record['messages'] if m['role'] == 'tool']
        self.assertTrue(any(s.startswith('Error:') for s in tool_results))
        self.assertTrue(any(s.startswith('PASS:') for s in tool_results))

    def test_mixture_replays_rng_and_never_joins_different_sources(self):
        from pathlib import Path
        import numpy as np
        import torch
        from microgpt.data import MixtureStream
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, token in [('code', 10), ('text', 20)]:
                (root / name).mkdir()
                np.full(64, token, dtype='<u4').tofile(root / name / 'train.bin')
            stream = MixtureStream(root, 'train', 8, [{'path': 'code', 'weight': .6}, {'path': 'text', 'weight': .4}])
            generator = torch.Generator().manual_seed(9)
            saved = generator.get_state()
            expected = [stream.batch(2, generator, 'cpu')[0] for _ in range(8)]
            generator.set_state(saved)
            for batch in expected:
                torch.testing.assert_close(batch, stream.batch(2, generator, 'cpu')[0])
                self.assertEqual(len(batch.unique()), 1)

    def test_swesmith_import_rejects_text_only_actions_and_translates_tools(self):
        import json
        from microgpt.download_sft import convert_swesmith
        row = {'resolved': True, 'traj_id': 'example', 'instance_id': 'project.task',
               'messages': json.dumps([{'role': 'assistant', 'content': '<function=bash>text action</function>'}])}
        self.assertEqual(convert_swesmith(row), (None, 'no_structured_tool_calls'))
        row['messages'] = json.dumps([
            {'role': 'system', 'content': 'Use str_replace_editor with XML syntax.'},
            {'role': 'user', 'content': 'Read the source.'},
            {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'read1', 'function': {
                'name': 'str_replace_editor', 'arguments': {'command': 'view', 'path': '/testbed/a.py'}}}]},
            {'role': 'tool', 'tool_call_id': 'read1', 'content': 'source'},
            {'role': 'assistant', 'content': 'Read it.'}])
        record, reason = convert_swesmith(row)
        self.assertIsNone(reason)
        self.assertNotIn('str_replace_editor', record['messages'][0]['content'])
        self.assertEqual(record['messages'][2]['tool_calls'][0]['function'], {'name': 'read', 'arguments': {'filePath': '/testbed/a.py'}})

    def test_o200k_unicode_roundtrip_and_literal_control_tokens(self):
        try:
            import tiktoken
        except ImportError:
            self.skipTest("Install .[tokenizer]")
        from microgpt.tokenizer import Tokenizer
        for name in ("o200k_base", "o200k_harmony"):
            tokenizer = Tokenizer(encoding=name)
            text = 'Olá! 你好 مرحبا\n<|endoftext|> <|im_end|>'
            ids = tokenizer.encode(text)
            self.assertNotIn(tokenizer.eos, ids)
            self.assertEqual(tokenizer.decode(ids), text)
            with tempfile.TemporaryDirectory() as directory:
                tokenizer.save(directory)
                self.assertEqual(Tokenizer.load(directory).encode(text), ids)


if __name__ == "__main__":
    unittest.main()
