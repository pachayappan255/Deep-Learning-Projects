"""Framework-free text utilities: tokenizer, vocabulary, leakage-safe splitting.

Nothing in this module imports torch, so it can be unit-tested and used by the
n-gram baseline / data audit on any machine.
"""
from __future__ import annotations

import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

PAD, UNK, BOS, EOS = "<pad>", "<unk>", "<bos>", "<eos>"
SPECIALS = [PAD, UNK, BOS, EOS]
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def tokenize(sentence: str) -> list[str]:
    """Lower-case word tokenizer (keeps possessives such as "person's")."""
    return _TOKEN_RE.findall(str(sentence).lower())


class Vocab:
    def __init__(self, itos: list[str]):
        self.itos = list(itos)
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.pad_id, self.unk_id = self.stoi[PAD], self.stoi[UNK]
        self.bos_id, self.eos_id = self.stoi[BOS], self.stoi[EOS]

    def __len__(self) -> int:
        return len(self.itos)

    @classmethod
    def build(cls, token_lists, min_freq: int = 1) -> "Vocab":
        counts = Counter(t for toks in token_lists for t in toks)
        words = sorted(w for w, c in counts.items() if c >= min_freq)
        return cls(SPECIALS + words)

    def encode(self, tokens, add_bos=False, add_eos=False) -> list[int]:
        ids = [self.stoi.get(t, self.unk_id) for t in tokens]
        return ([self.bos_id] if add_bos else []) + ids + ([self.eos_id] if add_eos else [])

    def decode(self, ids) -> list[str]:
        return [self.itos[i] for i in ids]

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.itos, indent=0))

    @classmethod
    def load(cls, path) -> "Vocab":
        return cls(json.loads(Path(path).read_text()))


# --------------------------------------------------------------------------- #
# Corpus loading + leakage-safe splitting
# --------------------------------------------------------------------------- #
def load_corpus(path, sentence_col="sentence", category_col="category") -> pd.DataFrame:
    """Load a .csv (needs a sentence column) or .txt (one sentence per line)."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        if sentence_col not in df.columns:
            raise ValueError(f"{path} has no '{sentence_col}' column: {list(df.columns)}")
        out = pd.DataFrame({"sentence": df[sentence_col].astype(str)})
        out["category"] = df[category_col].astype(str) if category_col in df.columns else "all"
    else:
        lines = [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        out = pd.DataFrame({"sentence": lines, "category": "all"})
    out["tokens"] = out["sentence"].map(tokenize)
    out = out[out["tokens"].map(len) >= 2].drop_duplicates("sentence").reset_index(drop=True)
    return out


def add_groups(df: pd.DataFrame, group_words: int = 6) -> pd.DataFrame:
    """A *group* = sentences sharing their first `group_words` words (same stem).

    All members of a group always land in the same split, so validation/test
    sentences never share a stem with training sentences.
    Also stores `boundary` = length (in words) of the stem shared by the group,
    used to break metrics down into "stem" vs "ending" tokens (-1 if unknown).
    """
    df = df.copy()
    df["group"] = df["tokens"].map(lambda t: " ".join(t[:group_words]))
    boundary = {}
    for g, toks in df.groupby("group")["tokens"]:
        toks = list(toks)
        if len(toks) < 2:
            boundary[g] = -1
            continue
        k = 0
        while all(len(t) > k for t in toks) and len({t[k] for t in toks}) == 1:
            k += 1
        boundary[g] = k
    df["boundary"] = df["group"].map(boundary)
    return df


def group_split(df, val_frac=0.1, test_frac=0.1, seed=42, group_words=6) -> pd.DataFrame:
    """Stratified (by category) split at the *group* level."""
    df = add_groups(df, group_words)
    rng = random.Random(seed)
    split_of = {}
    for _, sub in df.groupby("category"):
        groups = sorted(sub["group"].unique())
        rng.shuffle(groups)
        n = len(groups)
        n_val, n_test = int(round(n * val_frac)), int(round(n * test_frac))
        for i, g in enumerate(groups):
            split_of[g] = "val" if i < n_val else "test" if i < n_val + n_test else "train"
    df["split"] = df["group"].map(split_of)
    return df


def leakage_report(df: pd.DataFrame) -> dict:
    """Quantify train->eval leakage so the split can be audited."""
    train_groups = set(df.loc[df.split == "train", "group"])
    train_sents = set(df.loc[df.split == "train", "sentence"])
    rep = {}
    for sp in ("val", "test"):
        sub = df[df.split == sp]
        rep[sp] = {
            "n": int(len(sub)),
            "stem_overlap_with_train": float(sub["group"].isin(train_groups).mean()) if len(sub) else 0.0,
            "exact_duplicate_in_train": float(sub["sentence"].isin(train_sents).mean()) if len(sub) else 0.0,
        }
    return rep


# --------------------------------------------------------------------------- #
# Irreducible-entropy floor (data-only; no model)
# --------------------------------------------------------------------------- #
def entropy_floor(df_all: pd.DataFrame, df_eval: pd.DataFrame) -> dict:
    """Perplexity of an *oracle* that knows P(next | full prefix) from `df_all`.

    No model that only sees the prefix can beat this on `df_eval`, because the
    continuation after a shared prefix is genuinely random in the data.
    Returns overall and per-token-type (stem / choice / ending / eos) values.
    """
    ctx = defaultdict(Counter)
    for toks in df_all["tokens"]:
        seq = list(toks) + [EOS]
        for i, w in enumerate(seq):
            ctx[tuple(seq[:i])][w] += 1
    buckets = defaultdict(list)
    for toks, b in zip(df_eval["tokens"], df_eval["boundary"]):
        seq = list(toks) + [EOS]
        for i, w in enumerate(seq):
            c = ctx[tuple(seq[:i])]
            nll = -math.log(c[w] / sum(c.values()))
            kind = "eos" if i == len(seq) - 1 else "all" if b < 0 else (
                "stem" if i < b else "choice" if i == b else "ending")
            buckets[kind].append(nll)
            buckets["overall"].append(nll)
    return {k: {"ppl": math.exp(sum(v) / len(v)), "n": len(v)} for k, v in buckets.items()}
