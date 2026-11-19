#!/usr/bin/env bash
# End-to-end: install -> tests -> audit -> baseline -> train (Transformer + LSTM) -> evaluate -> demo
set -euo pipefail
cd "$(dirname "$0")/.."
pip install -r requirements.txt            # on an NVIDIA GPU install the CUDA build of torch first (see README)
python -m pytest -q tests
python -m nwp.audit
python -m nwp.baseline_ngram --order 3
python -m nwp.train --config config.yaml
python -m nwp.train --config config_lstm.yaml
python -m nwp.evaluate --ckpt runs/transformer/model.pt --split test
python -m nwp.predict --ckpt runs/transformer/model.pt --text "The student starts the day with" --complete
