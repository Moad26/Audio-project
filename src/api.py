"""
api.py — FastAPI REST endpoint for Speech Emotion Recognition.

Endpoints:
    GET  /            → health check
    POST /predict     → upload WAV file, get emotion + probabilities
    GET  /emotions    → list supported emotion labels

Launch:
    uv run uvicorn src.api:app --reload --port 8000
    # or from src/ directory:
    uv run uvicorn api:app --reload --port 8000
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# ── Path bootstrap so imports resolve from src/ ──────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

from data import EMOTION_MAP
from datautils import SpectrogramUtils
from model import EmotionModel

EMOTION_LABELS = [EMOTION_MAP[i + 1] for i in range(len(EMOTION_MAP))]
NUM_CLASSES    = len(EMOTION_LABELS)
TARGET_SR      = 22050
TARGET_DURATION = 3.0
N_MELS         = 40
N_FFT          = 2048
HOP_LENGTH     = 512
N_MFCC         = 13

# ── Model registry (loaded lazily on first request) ───────────────────────
_model: Optional[torch.nn.Module] = None
_ckpt_path: Optional[str] = None   # set via startup event / env var


def get_model() -> torch.nn.Module:
    global _model
    if _model is not None:
        return _model

    model = EmotionModel(num_classes=NUM_CLASSES)
    if _ckpt_path and Path(_ckpt_path).exists():
        try:
            from train import EmotionLitModel
            lit = EmotionLitModel.load_from_checkpoint(_ckpt_path, map_location="cpu")
            model = lit.model
        except Exception as exc:
            print(f"[api] Warning: could not load checkpoint — {exc}")
    model.eval()
    _model = model
    return _model


# ── FastAPI app ───────────────────────────────────────────────────────────
app = FastAPI(
    title="Speech Emotion Recognition API",
    description=(
        "Predict the emotional content of a speech audio file.\n\n"
        "Upload a WAV (or MP3) file to `/predict` and receive the top predicted "
        "emotion and full class probability distribution."
    ),
    version="1.0.0",
)


@app.get("/", tags=["Health"])
def health():
    """Simple health-check endpoint."""
    return {"status": "ok", "model": "CNN-LSTM Emotion Recogniser", "classes": NUM_CLASSES}


@app.get("/emotions", tags=["Info"])
def list_emotions():
    """Return the list of supported emotion labels."""
    return {"emotions": EMOTION_LABELS}


@app.post("/predict", tags=["Inference"])
async def predict(
    file: UploadFile = File(..., description="Audio file (WAV or MP3)"),
    top_k: int = Query(default=3, ge=1, le=NUM_CLASSES, description="Number of top emotions to return"),
):
    """
    Predict the emotion in an uploaded audio file.

    Returns:
        - **emotion**: top-1 predicted emotion label
        - **confidence**: probability of the top-1 prediction
        - **top_k**: list of (emotion, probability) pairs
        - **probabilities**: full probability dict for all classes
    """
    import torch.nn.functional as F

    # ── Save upload to temp file ─────────────────────────────────────────
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        import librosa
        aud, sr = librosa.load(str(tmp_path), sr=None, mono=False)
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=f"Could not decode audio: {exc}")
    finally:
        tmp_path.unlink(missing_ok=True)

    # ── Pre-processing ────────────────────────────────────────────────────
    aud_sr = SpectrogramUtils.rechannel((aud, sr), new_ch=1)
    aud_sr = SpectrogramUtils.resample(aud_sr, new_sr=TARGET_SR)
    aud_sr = SpectrogramUtils.pad_or_truncate(aud_sr, target_duration=TARGET_DURATION)
    aud, sr = aud_sr

    audio_dict = {
        "audio": aud, "sr": sr,
        "path": Path("api_upload"),
        "source_type": "live",
    }
    result = SpectrogramUtils.load_spectrogram(
        audio_dict=audio_dict, new_sr=TARGET_SR,
        n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS,
        n_mfcc=N_MFCC, target_duration=TARGET_DURATION,
    )

    def _resize(arr, h):
        t = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        t = F.interpolate(t, size=(h, arr.shape[1]), mode="bilinear", align_corners=False)
        return t.squeeze().numpy()

    feat = np.stack([
        result["mel_spectrogram"],
        _resize(result["mfcc"],       N_MELS),
        _resize(result["mfcc_delta"], N_MELS),
        _resize(result["chromagram"], N_MELS),
    ], axis=0)
    x = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        log_probs = get_model()(x)
        probs = torch.exp(log_probs).squeeze().tolist()

    # ── Build response ────────────────────────────────────────────────────
    ranked = sorted(
        [(EMOTION_LABELS[i], probs[i]) for i in range(NUM_CLASSES)],
        key=lambda t: t[1], reverse=True,
    )
    return JSONResponse({
        "emotion":       ranked[0][0],
        "confidence":    round(ranked[0][1], 4),
        "top_k":         [{"emotion": e, "probability": round(p, 4)} for e, p in ranked[:top_k]],
        "probabilities": {EMOTION_LABELS[i]: round(probs[i], 4) for i in range(NUM_CLASSES)},
    })


# ── Dev launcher ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--reload", action="store_true")
    a = parser.parse_args()

    _ckpt_path = a.ckpt_path
    uvicorn.run("api:app", host=a.host, port=a.port, reload=a.reload)
