"""Optional schema-guided tool decoding. Structure is constrained, meaning is not."""
from functools import lru_cache
import json


class ToolGrammar:
    def __init__(self, tokenizer):
        import xgrammar as xgr
        self.xgr = xgr
        if tokenizer.encoding:
            vocab = [b''] * tokenizer.vocab_size
            for value, index in tokenizer.encoding._mergeable_ranks.items():
                vocab[index] = value
            for value, index in tokenizer.encoding._special_tokens.items():
                vocab[index] = value.encode()
        elif tokenizer.backend is None:
            vocab = [bytes([i]) for i in range(256)] + [b'']
        else:
            raise ValueError('Tool grammar currently supports bytes and o200k tokenizers')
        info = xgr.TokenizerInfo(vocab, vocab_type=xgr.VocabType.RAW,
                                 vocab_size=tokenizer.vocab_size, stop_token_ids=[tokenizer.eos])
        self.compiler = xgr.GrammarCompiler(info, max_threads=2, cache_enabled=False)
        self.vocab_size = tokenizer.vocab_size

    @lru_cache(maxsize=8)
    def compile(self, tools_json):
        tools = json.loads(tools_json)
        if not tools:
            format = {'type': 'any_text', 'excludes': ['<tool_call>', '</tool_call>']}
        else:
            schemas = [{'type': 'object', 'properties': {
                'name': {'const': tool['function']['name']},
                'arguments': tool['function'].get('parameters', {'type': 'object'})},
                'required': ['name', 'arguments'], 'additionalProperties': False} for tool in tools]
            format = {'type': 'triggered_tags', 'triggers': ['<tool_call>'], 'tags': [
                {'type': 'tag', 'begin': '<tool_call>', 'end': '</tool_call>',
                 'content': {'type': 'json_schema', 'json_schema': {'anyOf': schemas}, 'max_whitespace_cnt': 1}}]}
        return self.compiler.compile_structural_tag({'type': 'structural_tag', 'format': format})

    def matcher(self, tools):
        compiled = self.compile(json.dumps(tools))
        return self.xgr.GrammarMatcher(compiled)

    def mask(self, logits, matcher, bitmask):
        if matcher.fill_next_token_bitmask(bitmask):
            self.xgr.apply_token_bitmask_inplace(logits, bitmask.to(logits.device))
