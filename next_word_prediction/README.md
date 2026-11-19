# AI-Based Next-Word Prediction — advanced, leakage-safe, generalization-first

Decoder-only Transformer (plus a tied-weight LSTM for comparison) trained on
`next_word_prediction_10000_sentences.csv`, evaluated on **sentence stems the model has never seen**.

## Quick start
```bash
# NVIDIA GPU (e.g. RTX 2050): install the CUDA build of torch first, https://pytorch.org/get-started/locally/
pip install -r requirements.txt
python -m pytest -q tests                                   # sanity tests (causality, weight tying, no-leak split)
python -m nwp.train --config config.yaml                    # Transformer  -> runs/transformer/
python -m nwp.train --config config_lstm.yaml               # LSTM         -> runs/lstm/
python -m nwp.evaluate --ckpt runs/transformer/model.pt     # per-token-type metrics on the test stems
python -m nwp.predict  --ckpt runs/transformer/model.pt --text "The student starts the day with" --complete
python -m nwp.predict  --ckpt runs/transformer/model.pt     # interactive
```
Or everything at once: `bash scripts/run_all.sh`. Override anything: `--set d_model=384 dropout=0.3 epochs=120`.

## What the data audit found (`python -m nwp.audit`, output in `reports/`)
| Finding | Consequence |
|---|---|
| 10,000 sentences = **1,000 stems × 10 endings**; each category has exactly 10 endings; only 598 distinct words | The corpus is synthetic/templated, not natural text |
| A **random** split puts 100 % of eval stems in train | Memorisation would look like generalisation |
| The 10k "LSTM vocabulary" is a rare-word list; only **57** of its words occur in the sentences | Using it as the output vocab adds ~9.4k classes the model never sees. Off by default (`use_lexicon: true` to enable) |
| After a stem, the next word is ~uniform over ~10 endings | Perplexity ≈ 9.7 at that position is *optimal*, not a failure |

## Design for generalization
* **Stem-grouped, category-stratified split** (80/10/10): all 10 sentences sharing a stem stay together, asserted at train time (0 % stem overlap, 0 % duplicates). Vocabulary is built from **train only**.
* **Model**: pre-LN causal Transformer, 4 layers × 256-d × 4 heads, GELU, learned positions, **tied embeddings** (~3.2 M params), FlashAttention-style `scaled_dot_product_attention`.
* **Regularisation**: dropout 0.2, AdamW weight decay 0.1, label smoothing 0.05, **word-dropout** (10 % of input words → `<unk>`, also trains the OOV path), grad clipping, warmup + cosine LR, **EMA weights** for evaluation, fp16 AMP on GPU.
* **Early stopping** on held-out-stem perplexity; the best EMA checkpoint is saved.
* **Honest metrics**: perplexity and top-1/3/5 overall *and* split into `stem` (novel content), `choice` (first ending word), `ending`, `eos`, each next to the oracle floor.

## Reference numbers computed in the build environment (no GPU/torch there)
| Method | Test perplexity (unseen stems) |
|---|---|
| Oracle entropy floor (knows the whole dataset; unbeatable) | 1.86 overall · choice 9.73 · ending 1.01 |
| 3-gram / 4-gram absolute-discount baseline | 4.39 / 4.38 |
| **Transformer / LSTM** | produced by your run → `runs/*/metrics.json` |

The floor for `stem` tokens (2.15) assumes the oracle has already seen those stems, so a real model
on unseen stems should land above it there; closing the gap on `choice`/`ending` tokens and beating
the n-gram baseline are the meaningful targets.

## Caveats
* **The neural code was not executed in the build environment** (no PyTorch, no network). Tested there: the text layer, split/leakage logic, entropy floor, n-gram baseline, audit, and byte-compilation of all modules. `tests/test_model.py` checks shapes, weight tying and causality — run it first.
* The corpus is templated, so strong results here do **not** imply generalization to natural English. For a natural corpus, run `--set data=my_text.txt` (one sentence per line; `.csv` with a `sentence` column also works): the same grouped split, training and evaluation apply (set `group_words` to control the dedupe granularity).
* Predictions are limited to the 598-word training vocabulary; other words become `<unk>`.

## Layout
`nwp/text.py` tokenizer · vocab · grouped split · entropy floor  |  `nwp/data.py` datasets/augmentation  |  `nwp/model.py` Transformer & LSTM  |  `nwp/train.py` training  |  `nwp/evaluate.py` metrics  |  `nwp/predict.py` inference  |  `nwp/baseline_ngram.py` baseline  |  `nwp/audit.py` data audit

## Deploy
See `deploy/`: `app.py` (Gradio demo / Hugging Face Space), `api.py` (FastAPI REST), `Dockerfile`. Train first so `runs/transformer/model.pt` exists.
