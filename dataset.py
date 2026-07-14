"""
dataset.py
----------
Loads the text corpus, tokenizes it, and provides random minibatches
of (input, target) sequences for next-token prediction training.
"""

import os
import torch
import config
from tokenizer import CharTokenizer


def load_data():
    if not os.path.exists(config.DATA_PATH):
        raise FileNotFoundError(
            f"No training file found at {config.DATA_PATH}.\n"
            f"Put any plain-text corpus there (e.g. a book, articles, code, chat logs).\n"
            f"A good free starter corpus: tiny-shakespeare — "
            f"https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        )

    with open(config.DATA_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = CharTokenizer(text=text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    n = int(config.TRAIN_SPLIT * len(data))
    train_data = data[:n]
    val_data = data[n:]
    return train_data, val_data, tokenizer


def get_batch(split, train_data, val_data):
    """Randomly sample a batch of contiguous chunks from the data."""
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - config.BLOCK_SIZE, (config.BATCH_SIZE,))
    x = torch.stack([data[i: i + config.BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i + 1: i + 1 + config.BLOCK_SIZE] for i in ix])  # shifted by 1 = next-token target
    x, y = x.to(config.DEVICE), y.to(config.DEVICE)
    return x, y
