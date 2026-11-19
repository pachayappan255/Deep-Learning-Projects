"""Data audit: why this dataset needs a grouped split, and what the vocab file contributes."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from .text import load_corpus, group_split, leakage_report, entropy_floor, Vocab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/next_word_prediction_10000_sentences.csv")
    ap.add_argument("--lexicon", default="data/lstm_vocabulary_10000.csv")
    ap.add_argument("--out", default="reports/data_audit.json")
    a = ap.parse_args()
    df = group_split(load_corpus(a.data))
    words = Counter(w for t in df.tokens for w in t)
    lex = pd.read_csv(a.lexicon)
    used = set(lex.word) & set(words)
    rep = {
        "n_sentences": len(df),
        "sentence_length_words": df.tokens.map(len).describe().round(2).to_dict(),
        "distinct_words_in_sentences": len(words),
        "distinct_stems(first_6_words)": int(df.group.nunique()),
        "sentences_per_stem": int(df.groupby("group").size().median()),
        "distinct_endings_per_category": {
            c: int(len({tuple(t[b:]) for t, b in zip(sub.tokens, sub.boundary)}))
            for c, sub in df.groupby("category")},
        "lexicon_words": len(lex),
        "lexicon_words_used_in_sentences": len(used),
        "sentence_words_missing_from_lexicon": len(set(words) - set(lex.word)),
        "random_split_stem_overlap": float(
            df.sample(frac=1, random_state=0).pipe(lambda r: r.group[int(.9 * len(r)):].isin(set(r.group[:int(.9 * len(r))])).mean())),
        "grouped_split_leakage": leakage_report(df),
        "oracle_floor_ppl_val": {k: round(v["ppl"], 3) for k, v in entropy_floor(df, df[df.split == "val"]).items()},
        "interpretation": [
            "Corpus is synthetic/templated: 1000 stems x 10 category-specific endings.",
            "A random split puts ~100% of eval stems in train -> memorisation looks like generalisation.",
            "The lexicon is a rare-word list; only a handful of its words occur in the sentences, so it is not used as the output vocabulary by default.",
            "The first ending word ('choice' token) is ~uniform over ~10 options: perplexity ~9.7 there is optimal, not a failure.",
        ],
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps(rep, indent=1, default=str))


if __name__ == "__main__":
    main()
