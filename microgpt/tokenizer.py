"""Bytes for the first lesson; local BPE or o200k for longer experiments."""
from pathlib import Path
import base64
import hashlib
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

    def save(self, directory, portable=False):
        if self.encoding:
            descriptor = {"encoding": self.encoding.name}
            if portable:
                # Snapshot the pinned tiktoken encoding so deployment needs no
                # network fetch or pre-existing global tokenizer cache.
                ranks = b''.join(base64.b64encode(token) + b' ' + str(rank).encode() + b'\n'
                                 for token, rank in sorted(self.encoding._mergeable_ranks.items(), key=lambda pair: pair[1]))
                (Path(directory) / 'encoding.tiktoken').write_bytes(ranks)
                descriptor.update(pat_str=self.encoding._pat_str, special_tokens=self.encoding._special_tokens,
                                  ranks_sha256=hashlib.sha256(ranks).hexdigest())
            (Path(directory) / "encoding.json").write_text(json.dumps(descriptor) + "\n")
        if self.backend:
            self.backend.save(str(Path(directory) / "tokenizer.json"))

    @classmethod
    def load(cls, directory):
        descriptor = Path(directory) / "encoding.json"
        if descriptor.exists():
            config = json.loads(descriptor.read_text())
            if 'ranks_sha256' in config:
                import tiktoken
                raw = (Path(directory) / 'encoding.tiktoken').read_bytes()
                if hashlib.sha256(raw).hexdigest() != config['ranks_sha256']:
                    raise ValueError('Portable tokenizer checksum mismatch')
                ranks = {base64.b64decode(token): int(rank) for token, rank in (line.split() for line in raw.splitlines())}
                tokenizer = cls()
                tokenizer.encoding = tiktoken.Encoding(name=config['encoding'], pat_str=config['pat_str'],
                    mergeable_ranks=ranks, special_tokens=config['special_tokens'])
                tokenizer.eos = tokenizer.encoding.eot_token
                tokenizer.vocab_size = tokenizer.encoding.n_vocab
                return tokenizer
            return cls(encoding=config["encoding"])
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
