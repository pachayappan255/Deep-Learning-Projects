"""Gradio demo. Local: python deploy/app.py   |  Hugging Face Space: copy to repo root as app.py"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
from nwp.predict import Predictor

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = APP_DIR if os.path.isdir(os.path.join(APP_DIR, "nwp")) else os.path.dirname(APP_DIR)
CKPT = os.environ.get("CKPT", os.path.join(PROJECT_ROOT, "runs", "transformer", "model.pt"))
if not os.path.isfile(CKPT):
    raise FileNotFoundError(
        f"Checkpoint not found: {CKPT}. Train the model with "
        "`python -m nwp.train --config config.yaml` from the project root, "
        "or set the CKPT environment variable to an existing checkpoint."
    )

pr = Predictor(CKPT, device="cpu")


def suggest(text, k, temperature):
    if not text.strip():
        return {}, ""
    probs = dict(pr.next_words(text, int(k), 1.0))
    return probs, pr.complete(text, temperature=float(temperature))


demo = gr.Interface(
    fn=suggest,
    inputs=[gr.Textbox(label="Start of a sentence", value="The student starts the day with"),
            gr.Slider(1, 10, value=5, step=1, label="Top-k suggestions"),
            gr.Slider(0.3, 1.5, value=0.8, label="Completion temperature")],
    outputs=[gr.Label(label="Next-word probabilities"), gr.Textbox(label="Full completion")],
    title="Next-Word Prediction (Transformer)",
    description="Trained on a 598-word vocabulary; unknown words are treated as <unk>.",
    flagging_mode="never",
)
if __name__ == "__main__":
    demo.launch(share=True)
