"""Interpolated absolute-discount n-gram baseline (pure python, no deep learning).

Gives an honest reference point: a deep model must beat this on *held-out stems*.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict

from .text import EOS, BOS, Vocab, load_corpus, group_split


class NGramLM:
    def __init__(self, order=4, discount=0.75):
        self.n, self.d = order, discount
        self.counts = [defaultdict(Counter) for _ in range(order)]
        self.V = 1

    def fit(self, seqs, vocab_size):
        self.V = vocab_size
        for s in seqs:
            s = [BOS] * (self.n - 1) + s + [EOS]
            for i in range(self.n - 1, len(s)):
                for k in range(self.n):
                    self.counts[k][tuple(s[i - k:i])][s[i]] += 1
        return self

    def prob(self, ctx, w):
        p = 1.0 / self.V
        for k in range(self.n):
            if k > len(ctx):
                break
            c = tuple(ctx[len(ctx) - k:]) if k else ()
            cnt = self.counts[k].get(c)
            if not cnt:
                break
            tot = sum(cnt.values())
            p = max(cnt.get(w, 0) - self.d, 0) / tot + (self.d * len(cnt) / tot) * p
        return p


def perplexity(lm, seqs):
    nll, n = 0.0, 0
    for s in seqs:
        s = [BOS] * (lm.n - 1) + s + [EOS]
        for i in range(lm.n - 1, len(s)):
            nll -= math.log(lm.prob(s[max(0, i - lm.n + 1):i], s[i]))
            n += 1
    return math.exp(nll / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/next_word_prediction_10000_sentences.csv")
    ap.add_argument("--order", type=int, default=4)
    a = ap.parse_args()
    df = group_split(load_corpus(a.data))
    vocab = Vocab.build(df[df.split == "train"]["tokens"])
    mapt = lambda toks: [t if t in vocab.stoi else "<unk>" for t in toks]
    lm = NGramLM(a.order).fit([mapt(t) for t in df[df.split == "train"]["tokens"]], len(vocab))
    res = {sp: perplexity(lm, [mapt(t) for t in df[df.split == sp]["tokens"]]) for sp in ("val", "test")}
    print(json.dumps({"order": a.order, "perplexity": res}, indent=1))


if __name__ == "__main__":
    main()
