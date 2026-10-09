import tempfile
import unittest
from pathlib import Path
import numpy as np
from scripts.prepare_pivot_corpus import documents
from microgpt.tokenizer import Tokenizer, train_bpe


class PivotCorpusTests(unittest.TestCase):
    def test_documents_cross_blocks_without_losing_tokens(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'tokens.bin'
            np.asarray([1, 2, 3, 99, 4, 99, 99], dtype='<u4').tofile(path)
            self.assertEqual(list(documents(path, 99, block=2)), [[1, 2, 3], [4], []])
            np.asarray([1, 2], dtype='<u4').tofile(path)
            with self.assertRaises(ValueError): list(documents(path, 99, block=2))

    def test_code_and_multilingual_roundtrip(self):
        texts = ['def f(x):\n\treturn "ação 😀 日本語"\n', 'مرحبا мир 中文 <|endoftext|>', 'a\\b\n  x\r\n']
        tokenizer = train_bpe(texts, 300)
        for text in texts:
            self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)
            self.assertNotIn(tokenizer.eos, tokenizer.encode(text))

    def test_tokenizer_reload_and_scaled_initialization(self):
        import torch
        from microgpt.model import GPT, ModelConfig
        tokenizer = train_bpe(['print("<|endoftext|> ação")'], 300)
        with tempfile.TemporaryDirectory() as folder:
            tokenizer.save(folder)
            restored = Tokenizer.load(folder)
            text = '<|endoftext|> ação'
            self.assertEqual(restored.decode(restored.encode(text)), text)
        torch.manual_seed(42)
        model = GPT(ModelConfig(width=128, heads=4, layers=4, residual_scale=True))
        expected = .02 / (2 * 4) ** .5
        self.assertAlmostEqual(model.blocks[0].down.weight.std().item(), expected, delta=.0003)
        ids = torch.randint(0, 257, (1, 16))
        _, loss = model(ids, ids)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_conversion_preserves_documents_and_rejects_modified_source(self):
        import hashlib
        from scripts.prepare_pivot_corpus import convert
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, target = root / 'old' / 'code', root / 'new' / 'code'
            source.mkdir(parents=True)
            target.parent.mkdir()
            texts = ['def f():\n    return "ação"\n', '', '<|endoftext|> 日本語']
            old = Tokenizer()
            new = train_bpe(texts, 300)
            new.save(target.parent)
            tokens = [token for text in texts for token in old.encode(text) + [old.eos]]
            path = source / 'train.bin'
            np.asarray(tokens, dtype='<u4').tofile(path)
            expected = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'documents': 3}
            _, _, counts = convert((str(source), str(target), 'train', expected))
            actual = [new.decode(ids) for ids in documents(target / 'train.bin', new.eos)]
            self.assertEqual(actual, texts)
            self.assertEqual(counts['documents'], 3)
            with self.assertRaisesRegex(ValueError, 'checksum'):
                convert((str(source), str(target), 'train', {**expected, 'sha256': 'wrong'}))

    def test_bpe_tool_grammar_accepts_unicode_and_rejects_unknown_tool(self):
        from microgpt.constraints import ToolGrammar
        text = '<tool_call>{"name":"read","arguments":{"filePath":"ação/日本語.py"}}</tool_call>'
        tokenizer = train_bpe([text], 300)
        grammar = ToolGrammar(tokenizer)
        tools = [{'type': 'function', 'function': {'name': 'read', 'parameters': {
            'type': 'object', 'properties': {'filePath': {'type': 'string'}},
            'required': ['filePath'], 'additionalProperties': False}}}]
        matcher = grammar.matcher(tools)
        for token in tokenizer.encode(text):
            self.assertTrue(matcher.accept_token(token))
        self.assertTrue(matcher.accept_token(tokenizer.eos))
        self.assertFalse(grammar.matcher(tools).accept_string(text.replace('read', 'delete')))
