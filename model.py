"""
model.py
--------
A GPT-style decoder-only Transformer, built from scratch in PyTorch.

Architecture (per block):
    x -> LayerNorm -> Multi-Head Causal Self-Attention -> residual add
      -> LayerNorm -> Feed-Forward (MLP)               -> residual add

This is a "pre-norm" transformer (norm before the sublayer), which is
what GPT-2/GPT-3 and most modern LLMs use because it trains more stably
than the original "post-norm" design from "Attention Is All You Need".
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    """
    Multi-head self-attention with a causal mask, so token i can only
    attend to tokens <= i. This is what makes the model autoregressive
    (it can only use the past to predict the future).
    """

    def __init__(self, n_embed, n_head, block_size, dropout):
        super().__init__()
        assert n_embed % n_head == 0, "n_embed must be divisible by n_head"
        self.n_head = n_head
        self.head_dim = n_embed // n_head

        # One linear layer produces Q, K, V all at once (more efficient than 3 separate layers)
        self.qkv_proj = nn.Linear(n_embed, 3 * n_embed, bias=False)
        self.out_proj = nn.Linear(n_embed, n_embed, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Causal mask: lower-triangular matrix of 1s. Registered as a buffer
        # so it moves with .to(device) but isn't treated as a learnable param.
        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("mask", mask.view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.shape  # batch, time (sequence length), channels (n_embed)

        qkv = self.qkv_proj(x)  # (B, T, 3*C)
        q, k, v = qkv.split(C, dim=2)

        # reshape to (B, n_head, T, head_dim) so each head attends independently
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # scaled dot-product attention: softmax(QK^T / sqrt(d_k)) V
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))  # (B, nh, T, T)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        out = att @ v  # (B, nh, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # merge heads back
        out = self.resid_dropout(self.out_proj(out))
        return out


class FeedForward(nn.Module):
    """
    Position-wise MLP applied to every token independently.
    Standard practice: expand to 4x n_embed, apply nonlinearity, project back.
    """

    def __init__(self, n_embed, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.GELU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """One transformer decoder block: attention + feed-forward, each with
    a residual connection and pre-layernorm."""

    def __init__(self, n_embed, n_head, block_size, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embed)
        self.attn = CausalSelfAttention(n_embed, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embed)
        self.ffwd = FeedForward(n_embed, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # residual around attention
        x = x + self.ffwd(self.ln2(x))   # residual around feed-forward
        return x


class MiniGPT(nn.Module):
    """
    The full model: token + positional embeddings -> N transformer blocks
    -> final layernorm -> linear head projecting to vocabulary logits.
    """

    def __init__(self, vocab_size, block_size, n_embed, n_head, n_layer, dropout):
        super().__init__()
        self.block_size = block_size

        self.token_embedding = nn.Embedding(vocab_size, n_embed)
        self.position_embedding = nn.Embedding(block_size, n_embed)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.Sequential(
            *[TransformerBlock(n_embed, n_head, block_size, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size, bias=False)

        # Weight tying: sharing weights between input embedding and output
        # projection is a well-known trick (used in GPT-2) that reduces
        # parameter count and often improves quality.
        self.token_embedding.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.block_size, f"sequence length {T} exceeds block_size {self.block_size}"

        tok_emb = self.token_embedding(idx)                                   # (B, T, C)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))  # (T, C)
        x = self.dropout(tok_emb + pos_emb)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Autoregressively generate `max_new_tokens` tokens given a starting
        context `idx` (B, T) of token ids.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]  # crop context to block_size
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)  # last time step only

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
