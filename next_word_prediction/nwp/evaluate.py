"""Metrics: perplexity + top-k accuracy overall and per token type (stem/choice/ending/eos)."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import config_from_dict
from .data import SentenceDataset, make_collate
from .model import build_model
from .text import Vocab, load_corpus, group_split, entropy_floor


@torch.no_grad()
def evaluate_model(model, dataset, vocab: Vocab, device, batch_size: int = 256) -> dict:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=make_collate(vocab.pad_id))
    acc = defaultdict(lambda: [0.0, 0, 0, 0, 0])  # nll, n, top1, top3, top5
    for x, y, b in loader:
        x, y, b = x.to(device), y.to(device), b.to(device)
        logits = model(x).float()
        V = logits.size(-1)
        nll = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1), ignore_index=-100, reduction="none").view_as(y)
        top5 = logits.topk(min(5, V), dim=-1).indices
        hit = top5 == y.unsqueeze(-1)
        c1, c3, c5 = hit[..., :1].any(-1), hit[..., :3].any(-1), hit.any(-1)
        pos = torch.arange(y.size(1), device=device).unsqueeze(0).expand_as(y)
        valid = y != -100
        eos = valid & (y == vocab.eos_id)
        known = (b >= 0).unsqueeze(1)
        word = valid & ~eos
        masks = {
            "overall": valid,
            "stem": word & known & (pos < b.unsqueeze(1)),
            "choice": word & known & (pos == b.unsqueeze(1)),
            "ending": word & known & (pos > b.unsqueeze(1)),
            "eos": eos,
        }
        for name, m in masks.items():
            a = acc[name]
            a[0] += nll[m].sum().item(); a[1] += int(m.sum().item())
            a[2] += int(c1[m].sum().item()); a[3] += int(c3[m].sum().item()); a[4] += int(c5[m].sum().item())
    out = {}
    for name, (s, n, t1, t3, t5) in acc.items():
        if n:
            out[name] = {"ppl": math.exp(s / n), "top1": t1 / n, "top3": t3 / n, "top5": t5 / n, "n": n}
    return out


def dataset_for(df, split, vocab, max_len):
    sub = df[df.split == split]
    return SentenceDataset(sub["tokens"].tolist(), sub["boundary"].tolist(), vocab, max_len)


def main():
    ap = argparse.ArgumentParser(description="Evaluate a saved checkpoint on the held-out stems")
    ap.add_argument("--ckpt", default="runs/transformer/model.pt")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=True)
    cfg, vocab = config_from_dict(ck["config"]), Vocab(ck["vocab"])
    model = build_model(cfg, len(vocab)); model.load_state_dict(ck["model"]); model.to(dev)
    df = group_split(load_corpus(cfg.data), cfg.val_frac, cfg.test_frac, cfg.seed, cfg.group_words)
    res = evaluate_model(model, dataset_for(df, a.split, vocab, cfg.max_len), vocab, dev)
    floor = entropy_floor(df, df[df.split == a.split])
    print(json.dumps({"model": res, "oracle_floor_ppl": {k: v["ppl"] for k, v in floor.items()}}, indent=1))


if __name__ == "__main__":
    main()
