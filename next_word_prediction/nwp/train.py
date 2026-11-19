"""Training entry point:  python -m nwp.train --config config.yaml [--set key=value ...]"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .baseline_ngram import NGramLM, perplexity
from .config import load_config, config_to_dict
from .data import SentenceDataset, make_collate, word_dropout
from .evaluate import evaluate_model, dataset_for
from .model import build_model, count_params
from .text import Vocab, load_corpus, group_split, leakage_report, entropy_floor


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


class EMA:
    """Exponential moving average of weights; evaluation uses the averaged copy."""

    def __init__(self, model, decay):
        self.decay, self.model = decay, copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        src = dict(model.named_parameters())
        for name, p in self.model.named_parameters():
            p.lerp_(src[name].detach(), 1.0 - self.decay)


def make_scheduler(opt, warmup, total):
    def f(step):
        if step < warmup:
            return (step + 1) / warmup
        prog = (step - warmup) / max(1, total - warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog))  # cosine to 10 % of peak
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def plot_history(hist: pd.DataFrame, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(hist.epoch, hist.train_loss); ax[0].set_title("train loss (with regularisation)"); ax[0].set_xlabel("epoch")
    ax[1].plot(hist.epoch, hist.val_ppl, label="val ppl (EMA)")
    best = hist.val_ppl.idxmin(); ax[1].axvline(hist.epoch[best], ls="--", c="gray", label="best epoch")
    ax[1].set_yscale("log"); ax[1].set_title("held-out-stem perplexity"); ax[1].set_xlabel("epoch"); ax[1].legend()
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    seed_everything(cfg.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(cfg.out_dir); out.mkdir(parents=True, exist_ok=True)

    # ---------------- data ----------------
    df = group_split(load_corpus(cfg.data), cfg.val_frac, cfg.test_frac, cfg.seed, cfg.group_words)
    leak = leakage_report(df); print("leakage report:", json.dumps(leak))
    assert all(v["stem_overlap_with_train"] == 0 for v in leak.values()), "train/eval stems overlap!"
    df[["sentence", "category", "group", "split"]].to_csv(out / "splits.csv", index=False)
    train_tokens = df[df.split == "train"]["tokens"].tolist()
    if cfg.use_lexicon:
        lex = pd.read_csv(cfg.lexicon)["word"].astype(str).str.lower().tolist()
        train_tokens = train_tokens + [[w] for w in lex]
    vocab = Vocab.build(train_tokens, cfg.min_freq); vocab.save(out / "vocab.json")
    print(f"vocab={len(vocab)}  train={sum(df.split=='train')}  val={sum(df.split=='val')}  test={sum(df.split=='test')}")

    tr = df[df.split == "train"]
    train_ds = SentenceDataset(tr["tokens"].tolist(), tr["boundary"].tolist(), vocab, cfg.max_len)
    val_ds, test_ds = dataset_for(df, "val", vocab, cfg.max_len), dataset_for(df, "test", vocab, cfg.max_len)
    loader = DataLoader(train_ds, cfg.batch_size, shuffle=True, drop_last=True,
                        collate_fn=make_collate(vocab.pad_id), num_workers=cfg.num_workers)

    # ---------------- model / optimiser ----------------
    model = build_model(cfg, len(vocab)).to(dev)
    print(f"arch={cfg.arch} params={count_params(model)/1e6:.2f}M device={dev}")
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=cfg.lr, betas=(0.9, 0.95))
    total_steps = cfg.epochs * len(loader)
    sched = make_scheduler(opt, cfg.warmup_steps, total_steps)
    use_amp = cfg.amp and dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ema = EMA(model, cfg.ema_decay)

    # ---------------- training loop ----------------
    hist, best, bad, t0 = [], float("inf"), 0, time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train(); run, nb = 0.0, 0
        for x, y, _ in loader:
            x, y = x.to(dev), y.to(dev)
            x = word_dropout(x, cfg.word_dropout, vocab.unk_id)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
                logits = model(x)
            loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1),
                                   ignore_index=-100, label_smoothing=cfg.label_smoothing)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt); scaler.update(); sched.step(); ema.update(model)
            run += loss.item(); nb += 1
        val = evaluate_model(ema.model, val_ds, vocab, dev)
        vp = val["overall"]["ppl"]
        hist.append({"epoch": epoch, "train_loss": run / nb, "val_ppl": vp, "val_top1": val["overall"]["top1"],
                     "val_choice_ppl": val.get("choice", {}).get("ppl", float("nan")),
                     "val_ending_ppl": val.get("ending", {}).get("ppl", float("nan")),
                     "lr": sched.get_last_lr()[0]})
        flag = ""
        if vp < best - 1e-4:
            best, bad, flag = vp, 0, "  *best*"
            torch.save({"model": ema.model.state_dict(), "config": config_to_dict(cfg),
                        "vocab": vocab.itos, "epoch": epoch, "val_ppl": vp}, out / "model.pt")
        else:
            bad += 1
        print(f"epoch {epoch:3d} | loss {run/nb:.3f} | val ppl {vp:.3f} | top1 {val['overall']['top1']:.3f} | "
              f"{time.time()-t0:.0f}s{flag}")
        if bad >= cfg.patience:
            print(f"early stopping (no val improvement for {cfg.patience} epochs)"); break

    # ---------------- final report ----------------
    hist = pd.DataFrame(hist); hist.to_csv(out / "history.csv", index=False); plot_history(hist, out / "curves.png")
    ck = torch.load(out / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(ck["model"]); model.to(dev)
    res = {"best_epoch": ck["epoch"], "params": count_params(model),
           "val": evaluate_model(model, val_ds, vocab, dev), "test": evaluate_model(model, test_ds, vocab, dev)}
    mapt = lambda t: [w if w in vocab.stoi else "<unk>" for w in t]
    ng = NGramLM(3).fit([mapt(t) for t in train_tokens if len(t) > 1], len(vocab))
    res["reference"] = {
        "ngram3_test_ppl": perplexity(ng, [mapt(t) for t in df[df.split == "test"]["tokens"]]),
        "oracle_floor_test": {k: v["ppl"] for k, v in entropy_floor(df, df[df.split == "test"]).items()},
    }
    (out / "metrics.json").write_text(json.dumps(res, indent=1))
    t = res["test"]
    print("\n=== TEST (unseen stems) ===")
    for k in ("overall", "stem", "choice", "ending", "eos"):
        if k in t:
            fl = res["reference"]["oracle_floor_test"].get(k, float("nan"))
            print(f"{k:8s} ppl {t[k]['ppl']:7.3f} (oracle floor {fl:6.3f}) | top1 {t[k]['top1']:.3f} top3 {t[k]['top3']:.3f} top5 {t[k]['top5']:.3f}")
    print(f"3-gram baseline test ppl: {res['reference']['ngram3_test_ppl']:.3f}")
    print(f"artifacts in {out}/  (model.pt, vocab.json, splits.csv, history.csv, curves.png, metrics.json)")


if __name__ == "__main__":
    main()
