"""
app.py — Interactive Gradio demo for Speech Emotion Recognition.

Supports:
  • Microphone recording or file upload.
  • Live emotion prediction with probability bar chart.
  • Cumulative confusion matrix visualisation.
  • Optional Wav2Vec2 backend (falls back to CNN-LSTM).

Launch:
    uv run python src/app.py
    uv run python src/app.py --ckpt_path logs/checkpoints/last.ckpt
    uv run python src/app.py --wav2vec2  # use Wav2Vec2 backend
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np
import torch

# ── Path bootstrap so imports resolve from src/ ──────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from data import EMOTION_MAP
from datautils import SpectrogramUtils
from model import EmotionModel

EMOTION_LABELS = [EMOTION_MAP[i + 1] for i in range(len(EMOTION_MAP))]  # 0-indexed list
NUM_CLASSES = len(EMOTION_LABELS)
TARGET_SR = 22050
TARGET_DURATION = 3.0
N_MELS = 40
N_FFT = 2048
HOP_LENGTH = 512
N_MFCC = 13


def load_cnn_lstm_model(ckpt_path: str | None) -> torch.nn.Module:
    """Load CNN-LSTM model from a checkpoint or return a random-weight model."""
    model = EmotionModel(num_classes=NUM_CLASSES)
    if ckpt_path and Path(ckpt_path).exists():
        from train import EmotionLitModel
        lit = EmotionLitModel.load_from_checkpoint(ckpt_path, map_location="cpu", strict=False)
        model = lit.model
        print(f"Loaded CNN-LSTM checkpoint: {ckpt_path}")
    else:
        print("No checkpoint found — using randomly initialised CNN-LSTM weights.")
    model.eval()
    return model


def load_wav2vec2_model(ckpt_path: str | None) -> torch.nn.Module:
    """Load Wav2Vec2 model from a checkpoint or return a fresh model."""
    from wav2vec2_model import Wav2Vec2EmotionModel
    model = Wav2Vec2EmotionModel(num_classes=NUM_CLASSES)
    if ckpt_path and Path(ckpt_path).exists():
        from wav2vec2_train import Wav2Vec2LitModel
        lit = Wav2Vec2LitModel.load_from_checkpoint(ckpt_path, map_location="cpu", strict=False)
        model = lit.model
        print(f"Loaded Wav2Vec2 checkpoint: {ckpt_path}")
    else:
        print("No checkpoint found — using pre-trained Wav2Vec2 base weights only.")
    model.eval()
    return model


def predict_cnn_lstm(model: torch.nn.Module, audio_array: np.ndarray, sr: int) -> np.ndarray:
    """Run CNN-LSTM inference on a waveform; return class probabilities."""
    import torch.nn.functional as F

    aud = audio_array.astype(np.float32)
    aud_sr = SpectrogramUtils.rechannel((aud, sr), new_ch=1)
    aud_sr = SpectrogramUtils.resample(aud_sr, new_sr=TARGET_SR)
    aud_sr = SpectrogramUtils.pad_or_truncate(aud_sr, target_duration=TARGET_DURATION)
    aud, sr = aud_sr
    aud_mono = aud[0] if aud.ndim > 1 else aud

    audio_dict = {"audio": aud, "sr": sr, "path": Path("live"), "source_type": "live"}
    result = SpectrogramUtils.load_spectrogram(
        audio_dict=audio_dict, new_sr=TARGET_SR, n_fft=N_FFT,
        hop_length=HOP_LENGTH, n_mels=N_MELS, n_mfcc=N_MFCC,
        target_duration=TARGET_DURATION,
    )

    def _resize(arr, h):
        t = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        t = F.interpolate(t, size=(h, arr.shape[1]), mode="bilinear", align_corners=False)
        return t.squeeze().numpy()

    mel   = result["mel_spectrogram"]
    mfcc  = _resize(result["mfcc"], N_MELS)
    mfcc_d = _resize(result["mfcc_delta"], N_MELS)
    chro  = _resize(result["chromagram"], N_MELS)

    feat = np.stack([mel, mfcc, mfcc_d, chro], axis=0)  # (4, n_mels, T)
    x = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)  # (1, 4, n_mels, T)

    with torch.no_grad():
        log_probs = model(x)
        probs = torch.exp(log_probs).squeeze().numpy()
    return probs


def predict_wav2vec2(model: torch.nn.Module, audio_array: np.ndarray, sr: int) -> np.ndarray:
    """Run Wav2Vec2 inference on a waveform; return class probabilities."""
    import librosa
    target_sr = 16000
    aud = audio_array.astype(np.float32)
    if aud.ndim > 1:
        aud = aud.mean(axis=0)
    if sr != target_sr:
        aud = librosa.resample(aud, orig_sr=sr, target_sr=target_sr)
    target_samples = int(target_sr * TARGET_DURATION)
    if len(aud) < target_samples:
        aud = np.pad(aud, (0, target_samples - len(aud)))
    else:
        aud = aud[:target_samples]
    x = torch.tensor(aud, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        log_probs = model(x)
        probs = torch.exp(log_probs).squeeze().numpy()
    return probs


# ── Gradio app factory ────────────────────────────────────────────────────
def build_demo(model, predict_fn, use_confusion: bool = True):
    import gradio as gr
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    ground_truth_store: list[int | None] = [None]

    def predict(audio):
        """Called by Gradio with (sample_rate, numpy_array) audio tuple."""
        if audio is None:
            return {}, None

        sr, audio_array = audio
        audio_array = audio_array.astype(np.float32)
        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=-1)

        probs = predict_fn(model, audio_array, sr)
        pred_idx = int(np.argmax(probs))
        pred_label = EMOTION_LABELS[pred_idx]

        # Probability bar chart
        fig, ax = plt.subplots(figsize=(7, 3))
        colors = ["#5C6BC0" if i != pred_idx else "#EF5350" for i in range(NUM_CLASSES)]
        ax.barh(EMOTION_LABELS, probs, color=colors, edgecolor="none")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability")
        ax.set_title(f"Predicted: {pred_label.upper()}  ({probs[pred_idx]*100:.1f}%)")
        ax.invert_yaxis()
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120)
        plt.close(fig)
        buf.seek(0)
        
        from PIL import Image
        img = Image.open(buf)

        label_dict = {EMOTION_LABELS[i]: float(probs[i]) for i in range(NUM_CLASSES)}
        return label_dict, img

    def update_confusion(true_label_str: str):
        """Record one confusion matrix cell when the user provides the true label."""
        if true_label_str not in EMOTION_LABELS:
            return plot_confusion()
        true_idx = EMOTION_LABELS.index(true_label_str)
        pred = ground_truth_store[0]
        if pred is not None:
            confusion_matrix[true_idx, pred] += 1
        return plot_confusion()

    def plot_confusion():
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(
            confusion_matrix, annot=True, fmt="d", ax=ax,
            xticklabels=EMOTION_LABELS, yticklabels=EMOTION_LABELS,
            cmap="Blues",
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix (cumulative)")
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120)
        plt.close(fig)
        buf.seek(0)
        from PIL import Image
        return Image.open(buf)

    def reset_confusion():
        confusion_matrix[:] = 0
        ground_truth_store[0] = None
        return plot_confusion()

    with gr.Blocks(title="Speech Emotion Recognition", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# 🎙️ Speech Emotion Recognition\n"
            "Record or upload audio (WAV/MP3) to detect the expressed emotion."
        )

        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(sources=["microphone", "upload"],
                                       type="numpy", label="Input Audio")
                predict_btn = gr.Button("Predict Emotion", variant="primary")

            with gr.Column(scale=2):
                label_out = gr.Label(num_top_classes=4, label="Emotion Probabilities")
                chart_out = gr.Image(label="Probability Chart", type="pil")

        if use_confusion:
            gr.Markdown("### Confusion Matrix (provide ground truth to record)")
            with gr.Row():
                true_label = gr.Dropdown(
                    EMOTION_LABELS, label="True Emotion (optional)", scale=1
                )
                record_btn = gr.Button("Record result", scale=1)
                reset_btn  = gr.Button("Reset matrix",  scale=1, variant="stop")
            confusion_out = gr.Image(label="Cumulative Confusion Matrix", type="pil")
            record_btn.click(update_confusion, inputs=[true_label], outputs=[confusion_out])
            reset_btn.click(reset_confusion, outputs=[confusion_out])

        predict_btn.click(predict, inputs=[audio_input], outputs=[label_out, chart_out])

    return demo


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_path", type=str, default=None)
    p.add_argument("--wav2vec2", action="store_true",
                   help="Use Wav2Vec2 backend instead of CNN-LSTM")
    p.add_argument("--share", action="store_true",
                   help="Create a public Gradio share link")
    p.add_argument("--server_port", type=int, default=7860)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.wav2vec2:
        model = load_wav2vec2_model(args.ckpt_path)
        predict_fn = predict_wav2vec2
    else:
        model = load_cnn_lstm_model(args.ckpt_path)
        predict_fn = predict_cnn_lstm

    demo = build_demo(model, predict_fn)
    demo.launch(server_port=args.server_port, share=args.share)
