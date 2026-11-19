"""Inference: next-word suggestions, sentence completion, interactive prompt."""
from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from .config import config_from_dict
from .model import build_model
from .text import Vocab, tokenize

END = "<end-of-sentence>"


class Predictor:
    def __init__(self, ckpt_path: str, device: str | None = None):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        self.cfg, self.vocab = config_from_dict(ck["config"]), Vocab(ck["vocab"])
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = build_model(self.cfg, len(self.vocab))
        self.model.load_state_dict(ck["model"]); self.model.to(self.device).eval()

    @torch.no_grad()
    def _next_logits(self, ids):
        x = torch.tensor([ids[-self.cfg.max_len:]], device=self.device)
        return self.model(x)[0, -1].float()

    def _prompt_ids(self, text):
        toks = tokenize(text)
        return self.vocab.encode(toks, add_bos=True), [t for t in toks if t not in self.vocab.stoi]

    def _probs(self, ids, temperature=1.0, allow_end=True):
        logits = self._next_logits(ids) / max(temperature, 1e-6)
        banned = [self.vocab.pad_id, self.vocab.unk_id, self.vocab.bos_id] + ([] if allow_end else [self.vocab.eos_id])
        logits[banned] = float("-inf")
        return F.softmax(logits, dim=-1)

    def next_words(self, text: str, k: int = 5, temperature: float = 1.0):
        """Top-k (word, probability) continuations of `text`."""
        ids, _ = self._prompt_ids(text)
        p = self._probs(ids, temperature)
        top = p.topk(min(k, len(p)))
        return [(END if i == self.vocab.eos_id else self.vocab.itos[i], float(v)) for v, i in zip(top.values, top.indices)]

    def unknown_words(self, text: str):
        return self._prompt_ids(text)[1]

    def complete(self, text: str, max_new: int = 15, temperature: float = 0.8, top_k: int = 10,
                 top_p: float = 0.95, greedy: bool = False, seed: int | None = None) -> str:
        gen = torch.Generator(device="cpu")
        if seed is not None:
            gen.manual_seed(seed)
        ids, _ = self._prompt_ids(text)
        words = []
        for _ in range(max_new):
            p = self._probs(ids, 1.0 if greedy else temperature, allow_end=len(words) > 0 or len(ids) > 3).cpu()
            if greedy:
                nxt = int(p.argmax())
            else:
                v, i = p.topk(min(top_k, len(p)))
                keep = (v.cumsum(0) - v) < top_p          # nucleus filter (always keeps the best token)
                v, i = v[keep], i[keep]
                nxt = int(i[torch.multinomial(v / v.sum(), 1, generator=gen)])
            if nxt == self.vocab.eos_id:
                break
            ids.append(nxt); words.append(self.vocab.itos[nxt])
        return (text.strip() + " " + " ".join(words)).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/transformer/model.pt")
    ap.add_argument("--text", help="prompt; omit for interactive mode")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--complete", action="store_true", help="also generate a full continuation")
    ap.add_argument("--temperature", type=float, default=0.8)
    a = ap.parse_args()
    pr = Predictor(a.ckpt)

    def show(text):
        unk = pr.unknown_words(text)
        if unk:
            print(f"  (unknown to the model, treated as <unk>: {unk})")
        for w, p in pr.next_words(text, a.k):
            print(f"  {p:6.1%}  {w}")
        if a.complete:
            print("  ->", pr.complete(text, temperature=a.temperature))

    if a.text:
        show(a.text)
    else:
        print("Type the beginning of a sentence (empty line to quit).")
        while True:
            t = input("> ").strip()
            if not t:
                break
            show(t)


if __name__ == "__main__":
    main()
