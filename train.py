"""
train.py
--------
Trains the MiniGPT model on your text corpus.

Usage:
    python train.py                      # fresh training run
    python train.py --resume             # continue from checkpoints/model.pt
    python train.py --resume --iters 2000  # continue for 2000 more steps
"""

import argparse
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="continue training from checkpoints/model.pt")
    parser.add_argument("--iters", type=int, default=config.MAX_ITERS, help="number of additional/total steps to run")
    args = parser.parse_args()

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

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    start_step = 0
    best_val_loss = float("inf")

    if args.resume and os.path.exists(config.CHECKPOINT_PATH):
        print(f"Resuming from {config.CHECKPOINT_PATH} ...")
        ckpt = torch.load(config.CHECKPOINT_PATH, map_location=config.DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = ckpt.get("step", 0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed at step {start_step}, best val loss so far: {best_val_loss:.4f}")
    elif args.resume:
        print("No existing checkpoint found — starting fresh instead.")

    print(f"Model parameters: {model.num_params() / 1e6:.2f}M")
    print(f"Device: {config.DEVICE}")

    os.makedirs(os.path.dirname(config.CHECKPOINT_PATH), exist_ok=True)
    tokenizer.save(os.path.join(os.path.dirname(config.CHECKPOINT_PATH), "vocab.json"))

    start = time.time()
    end_step = start_step + args.iters

    def save_checkpoint(step, val_loss):
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "step": step,
                "best_val_loss": val_loss,
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

    for it in range(start_step, end_step):
        if it % config.EVAL_INTERVAL == 0 or it == end_step - 1:
            losses = estimate_loss(model, train_data, val_data)
            elapsed = time.time() - start
            print(
                f"step {it:5d} | train loss {losses['train']:.4f} | "
                f"val loss {losses['val']:.4f} | {elapsed:.1f}s elapsed"
            )
            # Only checkpoint when val loss actually improves. Saving on every
            # eval step (regardless of whether it's better) would overwrite a
            # good checkpoint with a later, possibly overfit one.
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                save_checkpoint(it + 1, best_val_loss)
                print(f"  -> new best, checkpoint saved")

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