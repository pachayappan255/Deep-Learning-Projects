"""REST API.  uvicorn deploy.api:app --host 0.0.0.0 --port 8000   (docs at /docs)"""
import os
from pydantic import BaseModel, Field
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from nwp.predict import Predictor

app = FastAPI(title="Next-Word Prediction")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
pr = Predictor(os.environ.get("CKPT", "runs/transformer/model.pt"), device="cpu")


class Req(BaseModel):
    text: str = Field(..., max_length=500)
    k: int = Field(5, ge=1, le=20)
    complete: bool = False


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(r: Req):
    out = {"suggestions": [{"word": w, "prob": p} for w, p in pr.next_words(r.text, r.k)],
           "unknown_words": pr.unknown_words(r.text)}
    if r.complete:
        out["completion"] = pr.complete(r.text)
    return out
