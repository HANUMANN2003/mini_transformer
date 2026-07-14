"""
train.py
--------
Trains the MiniGPT model on your text corpus.

Usage:
    python train.py
"""

import os
import time
import torch

import config
from model import MiniGPT
from dataset import load_data, get_batch


@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    """Average loss over several batches for a less noisy estimate."""
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(config.EVAL_ITERS)
        for k in range(config.EVAL_ITERS):
            X, Y = get_batch(split, train_data, val_data)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    torch.manual_seed(config.SEED)

    train_data, val_data, tokenizer = load_data()
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Train tokens: {len(train_data):,} | Val tokens: {len(val_data):,}")

    model = MiniGPT(
        vocab_size=tokenizer.vocab_size,
        block_size=config.BLOCK_SIZE,
        n_embed=config.N_EMBED,
        n_head=config.N_HEAD,
        n_layer=config.N_LAYER,
        dropout=config.DROPOUT,
    ).to(config.DEVICE)

    print(f"Model parameters: {model.num_params() / 1e6:.2f}M")
    print(f"Device: {config.DEVICE}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    os.makedirs(os.path.dirname(config.CHECKPOINT_PATH), exist_ok=True)
    tokenizer.save(os.path.join(os.path.dirname(config.CHECKPOINT_PATH), "vocab.json"))

    start = time.time()
    best_val_loss = float("inf")

    for it in range(config.MAX_ITERS):
        if it % config.EVAL_INTERVAL == 0 or it == config.MAX_ITERS - 1:
            losses = estimate_loss(model, train_data, val_data)
            elapsed = time.time() - start
            print(
                f"step {it:5d} | train loss {losses['train']:.4f} | "
                f"val loss {losses['val']:.4f} | {elapsed:.1f}s elapsed"
            )
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": {
                            "vocab_size": tokenizer.vocab_size,
                            "block_size": config.BLOCK_SIZE,
                            "n_embed": config.N_EMBED,
                            "n_head": config.N_HEAD,
                            "n_layer": config.N_LAYER,
                            "dropout": config.DROPOUT,
                        },
                    },
                    config.CHECKPOINT_PATH,
                )

        xb, yb = get_batch("train", train_data, val_data)
        logits, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
        optimizer.step()

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint saved to: {config.CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
