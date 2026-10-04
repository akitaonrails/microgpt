"""Bytes for the first lesson; a locally trained BPE for longer experiments."""
from pathlib import Path
import json


class Tokenizer:
    def __init__(self, path=None, encoding=None):
        self.backend = None
        self.encoding = None
        if encoding:
            if path is not None or encoding not in ("o200k_base", "o200k_harmony"):
                raise ValueError("Choose a BPE file or a supported tiktoken encoding")
            import tiktoken
            self.encoding = tiktoken.get_encoding(encoding)
            self.eos = self.encoding.eot_token
            self.vocab_size = self.encoding.n_vocab
            return
        if path is not None:
            from tokenizers import Tokenizer as BPE
            self.backend = BPE.from_file(str(path))
        self.eos = self.backend.token_to_id("<|endoftext|>") if self.backend else 256
        self.vocab_size = self.backend.get_vocab_size() if self.backend else 257
        if self.eos is None:
            raise ValueError("BPE tokenizer needs an <|endoftext|> token")

    def encode(self, text):
        if self.encoding:
            # Dataset text containing control-token spellings stays ordinary text.
            return self.encoding.encode_ordinary(text)
        return self.backend.encode(text).ids if self.backend else list(text.encode("utf-8"))

    def decode(self, ids):
        if self.encoding:
            # o200k has unused vocabulary slots; a scratch model can sample them.
            chunks = []
            for token in ids:
                if token == self.eos:
                    continue
                try:
                    chunks.append(self.encoding.decode_single_token_bytes(token))
                except KeyError:
                    continue
            return b"".join(chunks).decode("utf-8", errors="replace")
        if self.backend:
            return self.backend.decode(ids)
        return bytes(i for i in ids if i < 256).decode("utf-8", errors="replace")

    def save(self, directory):
        if self.encoding:
            (Path(directory) / "encoding.json").write_text(json.dumps({"encoding": self.encoding.name}) + "\n")
        if self.backend:
            self.backend.save(str(Path(directory) / "tokenizer.json"))

    @classmethod
    def load(cls, directory):
        descriptor = Path(directory) / "encoding.json"
        if descriptor.exists():
            return cls(encoding=json.loads(descriptor.read_text())["encoding"])
        path = Path(directory) / "tokenizer.json"
        return cls(path if path.exists() else None)


def train_bpe(texts, vocab_size):
    from tokenizers import Tokenizer as BPE, models, pre_tokenizers, decoders, trainers
    backend = BPE(models.BPE())
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.train_from_iterator(texts, trainers.BpeTrainer(
        vocab_size=vocab_size, special_tokens=["<|endoftext|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False,
    ))
    tokenizer = Tokenizer()
    tokenizer.backend = backend
    tokenizer.eos = backend.token_to_id("<|endoftext|>")
    tokenizer.vocab_size = backend.get_vocab_size()
    return tokenizer
