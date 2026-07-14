"""
generate.py
-----------
Loads a trained checkpoint and generates text from a prompt.

Usage:
    python generate.py --prompt "Once upon a time"
    python generate.py --prompt "def add(a, b):" --max_new_tokens 300 --temperature 0.7
"""

import argparse
import os
import torch

import config
from model import MiniGPT
from tokenizer import CharTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="\n")
    parser.add_argument("--max_new_tokens", type=int, default=config.MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=config.TEMPERATURE)
    parser.add_argument("--top_k", type=int, default=config.TOP_K)
    parser.add_argument("--checkpoint", type=str, default=config.CHECKPOINT_PATH)
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(
            f"No checkpoint found at {args.checkpoint}. Run `python train.py` first."
        )

    vocab_path = os.path.join(os.path.dirname(args.checkpoint), "vocab.json")
    tokenizer = CharTokenizer(vocab_path=vocab_path)

    ckpt = torch.load(args.checkpoint, map_location=config.DEVICE)
    model_cfg = ckpt["config"]

    model = MiniGPT(**model_cfg).to(config.DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"Loaded model ({model.num_params() / 1e6:.2f}M params) from {args.checkpoint}\n")

    context = torch.tensor(
        [tokenizer.encode(args.prompt)], dtype=torch.long, device=config.DEVICE
    )

    out = model.generate(
        context,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    text = tokenizer.decode(out[0].tolist())

    print("=" * 60)
    print(text)
    print("=" * 60)


if __name__ == "__main__":
    main()
