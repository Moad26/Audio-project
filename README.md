# Speech Emotion Recognition

A comprehensive system for detecting emotions in speech audio using a **CNN-LSTM hybrid model** and optional **Wav2Vec2 fine-tuning**, trained on RAVDESS and TESS datasets.

## Overview

| Component | Description |
|-----------|-------------|
| **Feature extraction** | Mel-spectrogram + 13 MFCCs + delta-MFCCs + Chromagram (4-channel tensor) |
| **CNN-LSTM model** | Residual CNN feature extractor → Bidirectional LSTM → softmax |
| **Wav2Vec2 model** | `facebook/wav2vec2-base` fine-tuned with a linear emotion head |
| **Training** | PyTorch Lightning · class-weighted NLLLoss · macro F1 · MLflow + TensorBoard |
| **Deployment** | Gradio interactive demo · FastAPI REST endpoint |

## Datasets

| Dataset | Speakers | Emotions | Files |
|---------|----------|----------|-------|
| [RAVDESS](https://zenodo.org/record/1188976) | 24 actors | 8 | ~1,440 |
| [TESS](https://tspace.library.utoronto.ca/handle/1807/24487) | 2 actors (F) | 7 | ~2,800 |

**8 emotion classes:** neutral · calm · happy · sad · angry · fearful · disgust · surprised

## Emotions & Labels

```
0 neutral   1 calm   2 happy   3 sad
4 angry     5 fearful 6 disgust 7 surprised
```

## Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/audio-project.git
cd audio-project

# Install dependencies (using uv — recommended)
pip install uv
uv sync

# Or with standard pip
pip install -r requirements.txt
```

Download datasets and place them as:
```
data/
├── ravdess/
│   ├── Actor_01/
│   │   └── 03-01-01-01-01-01-01.wav
│   └── ...
└── toronto/
    ├── OAF_Fear/
    │   └── OAF_back_fear.wav
    └── ...
```

## Quick Start

### Train CNN-LSTM

```bash
uv run python src/train.py \
    --ravdess_dir data/ravdess \
    --tess_dir data/toronto \
    --max_epochs 30 \
    --batch_size 32 \
    --mlflow_tracking_uri mlruns
```

**Key CLI arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--ravdess_dir` | `data/ravdess` | RAVDESS root directory |
| `--tess_dir` | `data/toronto` | TESS root directory |
| `--max_epochs` | 30 | Training epochs |
| `--batch_size` | 32 | Mini-batch size |
| `--lr` | 1e-3 | Learning rate |
| `--n_mels` | 40 | Mel filter banks |
| `--accelerator` | `auto` | `cpu` / `gpu` / `auto` |
| `--mlflow_tracking_uri` | `mlruns` | MLflow tracking server URI |

### Fine-tune Wav2Vec2

```bash
uv run python src/wav2vec2_train.py \
    --ravdess_dir data/ravdess \
    --tess_dir data/toronto \
    --max_epochs 20 \
    --batch_size 8 \
    --freeze_feature_extractor
```

## Deployment

### Gradio Demo (interactive)

```bash
# CNN-LSTM (default)
uv run python src/app.py --ckpt_path logs/checkpoints/last.ckpt

# Wav2Vec2 backend
uv run python src/app.py --wav2vec2 --ckpt_path logs/wav2vec2_checkpoints/last.ckpt

# With public share link
uv run python src/app.py --share
```

The Gradio UI opens at **http://localhost:7860** and provides:
- Microphone recording or file upload
- Real-time probability bar chart per emotion
- Cumulative confusion matrix with ground-truth labelling

### FastAPI REST Endpoint

```bash
uv run uvicorn src.api:app --reload --port 8000
```

Interactive docs at **http://localhost:8000/docs**

**Example request:**

```bash
curl -X POST "http://localhost:8000/predict" \
     -F "file=@path/to/speech.wav" | python -m json.tool
```

**Response:**

```json
{
  "emotion": "happy",
  "confidence": 0.8732,
  "top_k": [
    {"emotion": "happy",   "probability": 0.8732},
    {"emotion": "neutral", "probability": 0.0741},
    {"emotion": "calm",    "probability": 0.0312}
  ],
  "probabilities": { "neutral": 0.0741, "calm": 0.0312, ... }
}
```

## Project Structure

```
audio-project/
├── src/
│   ├── data.py            # RAVDESS & TESS Dataset classes
│   ├── datautils.py       # SpectrogramUtils (mel + MFCC + chroma)
│   ├── model.py           # CNN-LSTM EmotionModel
│   ├── train.py           # CNN-LSTM Lightning training script
│   ├── wav2vec2_model.py  # Wav2Vec2 emotion head
│   ├── wav2vec2_train.py  # Wav2Vec2 fine-tuning script
│   ├── app.py             # Gradio demo
│   └── api.py             # FastAPI REST endpoint
├── notebooks/
│   ├── 01_eda.ipynb        # Exploratory data analysis
│   ├── 02_features.ipynb   # Feature extraction visualisation
│   └── 03_evaluation.ipynb # Model evaluation & confusion matrix
├── data/
│   ├── ravdess/
│   └── toronto/
├── logs/                  # Checkpoints & TensorBoard logs
├── mlruns/                # MLflow experiment tracking
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Feature Engineering

Each audio clip is transformed into a **4-channel tensor (C=4, H=n_mels, W=T)**:

| Channel | Feature | Shape |
|---------|---------|-------|
| 0 | Mel-spectrogram (dB) | `(n_mels, T)` |
| 1 | 13 MFCCs (resized) | `(n_mels, T)` |
| 2 | Delta-MFCCs (resized) | `(n_mels, T)` |
| 3 | Chromagram (resized) | `(n_mels, T)` |

MFCC and chromagram maps are resized to `n_mels` rows via bilinear interpolation.

## Model Architecture

```
Input (B, 4, 40, T)
   │
   ▼
Residual CNN  (3 stages: 32→64→128 channels)
   │  (B, T', 128*F')
   ▼
Bidirectional LSTM  (2 layers, hidden=256)
   │  (B, 512)
   ▼
Linear(512→128) → ReLU → Dropout → Linear(128→8)
   │
   ▼
log_softmax  →  loss: NLLLoss (class-weighted)
```

## Training Metrics

- **Loss:** weighted NLLLoss (inverse class frequency weights)
- **Accuracy:** multi-class accuracy per split
- **F1-score:** macro-averaged F1 per emotion (target > 0.80)
- **Logging:** TensorBoard (`logs/`) + MLflow (`mlruns/`)

View TensorBoard:
```bash
tensorboard --logdir logs/
```

View MLflow:
```bash
mlflow ui --port 5000
```

## Notebooks

| Notebook | Content |
|----------|---------|
| [`01_eda.ipynb`](notebooks/01_eda.ipynb) | Class distributions, audio durations, waveforms, spectrograms |
| [`02_features.ipynb`](notebooks/02_features.ipynb) | MFCC, delta, chroma extraction and 4-channel visualisation |
| [`03_evaluation.ipynb`](notebooks/03_evaluation.ipynb) | Confusion matrix, per-class F1, classification report |

## Uploading to Hugging Face Hub

```python
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path="logs/checkpoints",
    repo_id="<your-username>/speech-emotion-recognition",
    repo_type="model",
)
```

## Requirements

See [`pyproject.toml`](pyproject.toml) or [`requirements.txt`](requirements.txt) for the full dependency list.

Core dependencies: `torch`, `torchaudio`, `pytorch-lightning`, `librosa`, `transformers`, `gradio`, `fastapi`, `mlflow`.

## License

MIT
