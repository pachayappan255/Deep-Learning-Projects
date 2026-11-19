"""Single source of truth for hyper-parameters (YAML + `--set key=value` overrides)."""
from __future__ import annotations

from dataclasses import dataclass, fields, asdict
from pathlib import Path

import yaml


@dataclass
class Config:
    # data
    data: str = "data/next_word_prediction_10000_sentences.csv"
    lexicon: str = "data/lstm_vocabulary_10000.csv"
    use_lexicon: bool = False      # add the 10k word list to the output vocab (see README: not recommended)
    out_dir: str = "runs/transformer"
    val_frac: float = 0.1
    test_frac: float = 0.1
    group_words: int = 6           # sentences sharing their first N words are split together
    min_freq: int = 1
    max_len: int = 48
    # model
    arch: str = "transformer"      # transformer | lstm
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 1024
    dropout: float = 0.2
    # optimisation / regularisation
    epochs: int = 80
    batch_size: int = 64
    lr: float = 6e-4
    weight_decay: float = 0.1
    warmup_steps: int = 200
    grad_clip: float = 1.0
    label_smoothing: float = 0.05
    word_dropout: float = 0.10     # randomly replace input words by <unk>
    ema_decay: float = 0.995
    patience: int = 10
    seed: int = 42
    num_workers: int = 0
    amp: bool = True


def _cast(value: str, ref):
    if isinstance(ref, bool):
        return value.lower() in {"1", "true", "yes", "y"}
    return type(ref)(value)


def load_config(path: str | None = None, overrides: list[str] | None = None) -> Config:
    cfg = Config()
    if path and Path(path).exists():
        for k, v in (yaml.safe_load(Path(path).read_text()) or {}).items():
            if k not in {f.name for f in fields(Config)}:
                raise KeyError(f"Unknown config key in {path}: {k}")
            setattr(cfg, k, v)
    for item in overrides or []:
        k, v = item.split("=", 1)
        if not hasattr(cfg, k):
            raise KeyError(f"Unknown config key: {k}")
        setattr(cfg, k, _cast(v, getattr(cfg, k)))
    return cfg


def config_from_dict(d: dict) -> Config:
    valid = {f.name for f in fields(Config)}
    return Config(**{k: v for k, v in d.items() if k in valid})


def config_to_dict(cfg: Config) -> dict:
    return asdict(cfg)
