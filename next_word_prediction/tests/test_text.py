from nwp.text import Vocab, tokenize, load_corpus, group_split, leakage_report

DATA = "data/next_word_prediction_10000_sentences.csv"


def test_tokenize():
    assert tokenize("The person's plan, today.") == ["the", "person's", "plan", "today"]


def test_vocab_roundtrip():
    v = Vocab.build([["a", "b"], ["b", "c"]])
    ids = v.encode(["a", "zzz"], add_bos=True, add_eos=True)
    assert v.decode(ids) == ["<bos>", "a", "<unk>", "<eos>"]


def test_split_has_no_leakage():
    df = group_split(load_corpus(DATA))
    rep = leakage_report(df)
    for sp in ("val", "test"):
        assert rep[sp]["stem_overlap_with_train"] == 0.0
        assert rep[sp]["exact_duplicate_in_train"] == 0.0
    assert set(df.split) == {"train", "val", "test"}
    assert (df.split == "train").mean() > 0.75


def test_split_is_deterministic():
    a = group_split(load_corpus(DATA), seed=1).split.tolist()
    b = group_split(load_corpus(DATA), seed=1).split.tolist()
    assert a == b
