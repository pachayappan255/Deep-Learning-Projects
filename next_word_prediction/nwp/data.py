"""Torch datasets / collate / augmentation."""
from __future__ import annotations

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .text import Vocab

N_SPECIAL = 4  # <pad>, <unk>, <bos>, <eos> occupy ids 0..3


class SentenceDataset(Dataset):
    """Each item: ids = [<bos>, w1 .. wn, <eos>] and the stem boundary (word count)."""

    def __init__(self, token_lists, boundaries, vocab: Vocab, max_len: int):
        self.items = []
        for toks, b in zip(token_lists, boundaries):
            ids = vocab.encode(toks, add_bos=True, add_eos=True)[:max_len]
            self.items.append((torch.tensor(ids, dtype=torch.long), int(b)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def make_collate(pad_id: int):
    def collate(batch):
        seqs = pad_sequence([s for s, _ in batch], batch_first=True, padding_value=pad_id)
        x, y = seqs[:, :-1].contiguous(), seqs[:, 1:].clone()
        y[y == pad_id] = -100  # ignored by the loss / metrics
        b = torch.tensor([b for _, b in batch], dtype=torch.long)
        return x, y, b
    return collate


def word_dropout(x: torch.Tensor, p: float, unk_id: int) -> torch.Tensor:
    """Replace a fraction p of *word* tokens in the input by <unk> (regulariser)."""
    if p <= 0:
        return x
    mask = (torch.rand(x.shape, device=x.device) < p) & (x >= N_SPECIAL)
    return x.masked_fill(mask, unk_id)
