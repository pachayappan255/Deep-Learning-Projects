"""Decoder-only Transformer LM (pre-LN, tied embeddings) and a tied-weight LSTM LM."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.h, self.p = n_heads, dropout
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.h, C // self.h).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.p if self.training else 0.0, is_causal=True)
        return self.resid_drop(self.proj(y.transpose(1, 2).contiguous().view(B, T, C)))


class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.mlp = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model), nn.Dropout(dropout))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_layers=4, n_heads=4, d_ff=1024, dropout=0.2, max_len=48):
        super().__init__()
        self.max_len = max_len
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(Block(d_model, n_heads, d_ff, dropout) for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.apply(self._init)
        for name, p in self.named_parameters():  # GPT-2 style residual scaling
            if name.endswith("attn.proj.weight") or name.endswith("mlp.2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layers))
        self.head.weight = self.tok.weight  # weight tying

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        B, T = x.shape
        assert T <= self.max_len, f"sequence length {T} > max_len {self.max_len}"
        h = self.drop(self.tok(x) + self.pos(torch.arange(T, device=x.device)))
        for blk in self.blocks:
            h = blk(h)
        return self.head(self.ln_f(h))


class LockedDropout(nn.Module):
    """Variational dropout: one mask shared across all time steps."""

    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        mask = x.new_empty(x.size(0), 1, x.size(2)).bernoulli_(1 - self.p) / (1 - self.p)
        return x * mask


class LSTMLM(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_layers=2, dropout=0.3):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.drop_in, self.drop_out = LockedDropout(dropout), LockedDropout(dropout)
        self.rnn = nn.LSTM(d_model, d_model, n_layers, batch_first=True, dropout=dropout if n_layers > 1 else 0.0)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        nn.init.uniform_(self.emb.weight, -0.1, 0.1)
        self.head.weight = self.emb.weight  # weight tying

    def forward(self, x):
        out, _ = self.rnn(self.drop_in(self.emb(x)))
        return self.head(self.drop_out(out))


def build_model(cfg, vocab_size: int) -> nn.Module:
    if cfg.arch == "transformer":
        return TransformerLM(vocab_size, cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.d_ff, cfg.dropout, cfg.max_len)
    if cfg.arch == "lstm":
        return LSTMLM(vocab_size, cfg.d_model, cfg.n_layers, cfg.dropout)
    raise ValueError(f"unknown arch: {cfg.arch}")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
