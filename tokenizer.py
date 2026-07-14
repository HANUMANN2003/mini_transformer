"""
tokenizer.py
------------
A minimal character-level tokenizer. This keeps the project dependency-free
and easy to understand end-to-end. Swap this out for a BPE tokenizer
(e.g. tiktoken or a HuggingFace tokenizer) once you understand the basics.
"""

import json
import os


class CharTokenizer:
    def __init__(self, text: str = None, vocab_path: str = None):
        if vocab_path and os.path.exists(vocab_path):
            self.load(vocab_path)
        elif text is not None:
            chars = sorted(list(set(text)))
            self.stoi = {ch: i for i, ch in enumerate(chars)}
            self.itos = {i: ch for i, ch in enumerate(chars)}
            self.vocab_size = len(chars)
        else:
            raise ValueError("Provide either `text` to build a new vocab or `vocab_path` to load one.")

    def encode(self, s: str):
        """string -> list[int]"""
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        """list[int] -> string"""
        return "".join(self.itos[i] for i in ids)

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"stoi": self.stoi}, f, ensure_ascii=False)

    def load(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.stoi = data["stoi"]
        self.itos = {int(v): k for k, v in self.stoi.items()}
        self.vocab_size = len(self.stoi)
