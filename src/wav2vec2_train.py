"""
wav2vec2_train.py — Fine-tune Wav2Vec2 for speech emotion recognition.

Requires:
    uv add transformers accelerate

Usage:
    uv run python src/wav2vec2_train.py
    uv run python src/wav2vec2_train.py --max_epochs 20 --freeze_feature_extractor
    uv run python src/wav2vec2_train.py --accelerator gpu --batch_size 8
"""

from __future__ import annotations

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import argparse
import random
from collections import Counter
from pathlib import Path
from typing import List, Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split

try:
    from pytorch_lightning.loggers import MLFlowLogger
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False

from data import EMOTION_MAP

try:
    from transformers import Wav2Vec2Processor
except ImportError as exc:
    raise ImportError("Install transformers: uv add transformers accelerate") from exc

from wav2vec2_model import Wav2Vec2EmotionModel


# ── Raw Waveform Dataset ───────────────────────────────────────────────────
class RawWaveformDataset(Dataset):
    """
    Yields (waveform, label) pairs suitable for Wav2Vec2.

    Loads raw audio at 16 kHz (Wav2Vec2's expected rate) and
    pads/truncates to a fixed duration.
    """

    def __init__(
        self,
        data_dir: Path,
        target_sr: int = 16000,
        target_duration: float = 3.0,
        apply_augmentation: bool = True,
        augmentation_prob: float = 0.5,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.target_sr = target_sr
        self.target_samples = int(target_sr * target_duration)
        self.apply_augmentation = apply_augmentation
        self.augmentation_prob = augmentation_prob

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.file_paths: List[Path] = []
        self.labels: List[int] = []
        self._load()

    def _load(self) -> None:
        """Discover .wav files in Actor_* subdirs (RAVDESS) or flat dirs (TESS)."""
        for wav_file in sorted(self.data_dir.rglob("*.wav")):
            label = self._parse_label(wav_file)
            if label is not None:
                self.file_paths.append(wav_file)
                self.labels.append(label)

    @staticmethod
    def _parse_label(path: Path) -> Optional[int]:
        parts = path.stem.split("-")
        if len(parts) == 7:                           # RAVDESS format
            try:
                return int(parts[2]) - 1              # 0-indexed
            except ValueError:
                return None
        # TESS format: OAF_word_emotion or YAF_word_emotion
        emotion_map = {
            "neutral": 0, "happy": 2, "sad": 3,
            "angry": 4, "fear": 5, "disgust": 6, "ps": 7,
        }
        tail = path.stem.split("_")[-1].lower()
        return emotion_map.get(tail, None)

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int):
        import librosa
        path = self.file_paths[idx]
        label = self.labels[idx]

        waveform, _ = librosa.load(str(path), sr=self.target_sr, mono=True)

        # Augmentation: time shift
        if self.apply_augmentation and random.random() < self.augmentation_prob:
            shift = int(random.uniform(-0.2, 0.2) * len(waveform))
            waveform = np.roll(waveform, shift)

        # Pad or truncate
        if len(waveform) < self.target_samples:
            waveform = np.pad(waveform, (0, self.target_samples - len(waveform)))
        else:
            waveform = waveform[: self.target_samples]

        return torch.tensor(waveform, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


def collate_fn(batch):
    waveforms, labels = zip(*batch)
    return torch.stack(waveforms), torch.stack(labels)


# ── DataModule ─────────────────────────────────────────────────────────────
class Wav2Vec2DataModule(pl.LightningDataModule):
    def __init__(
        self,
        ravdess_dir: str = "data/ravdess",
        tess_dir: str = "data/toronto",
        batch_size: int = 16,
        num_workers: int = 4,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        target_sr: int = 16000,
        target_duration: float = 3.0,
        seed: int = 42,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.ravdess_dir = Path(ravdess_dir)
        self.tess_dir = Path(tess_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.target_sr = target_sr
        self.target_duration = target_duration
        self.seed = seed

    def _make_combined(self, augmentation: bool) -> ConcatDataset:
        shared = dict(
            target_sr=self.target_sr,
            target_duration=self.target_duration,
            apply_augmentation=augmentation,
            seed=self.seed,
        )
        ravdess = RawWaveformDataset(self.ravdess_dir, **shared)
        tess = RawWaveformDataset(self.tess_dir, **shared)
        return ConcatDataset([ravdess, tess])

    def setup(self, stage=None):
        aug = self._make_combined(augmentation=True)
        clean = self._make_combined(augmentation=False)
        total = len(aug)
        train_n = int(self.train_ratio * total)
        val_n = int(self.val_ratio * total)
        test_n = total - train_n - val_n
        g = torch.Generator().manual_seed(self.seed)
        self.train_ds, _, _ = random_split(aug, [train_n, val_n, test_n], generator=g)
        g = torch.Generator().manual_seed(self.seed)
        _, self.val_ds, self.test_ds = random_split(clean, [train_n, val_n, test_n], generator=g)

    def _loader(self, ds, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds, batch_size=self.batch_size, shuffle=shuffle,
            num_workers=self.num_workers, pin_memory=True,
            collate_fn=collate_fn,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self): return self._loader(self.train_ds, True)
    def val_dataloader(self):   return self._loader(self.val_ds, False)
    def test_dataloader(self):  return self._loader(self.test_ds, False)

    def compute_class_weights(self) -> torch.Tensor:
        label_counts: Counter = Counter()
        for idx in self.train_ds.indices:
            cumulative = 0
            for ds in self.train_ds.dataset.datasets:
                if idx < cumulative + len(ds):
                    label_counts[ds.labels[idx - cumulative]] += 1
                    break
                cumulative += len(ds)
        num_classes = len(EMOTION_MAP)
        total = sum(label_counts.values())
        weights = torch.zeros(num_classes)
        for cls in range(num_classes):
            weights[cls] = total / (num_classes * label_counts.get(cls, 1))
        return weights


# ── Lightning Module ───────────────────────────────────────────────────────
class Wav2Vec2LitModel(pl.LightningModule):
    def __init__(
        self,
        num_classes: int = 8,
        dropout: float = 0.3,
        freeze_feature_extractor: bool = True,
        pretrained_model_name: str = "facebook/wav2vec2-base",
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        scheduler_patience: int = 3,
        scheduler_factor: float = 0.5,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])

        self.model = Wav2Vec2EmotionModel(
            num_classes=num_classes,
            dropout=dropout,
            freeze_feature_extractor=freeze_feature_extractor,
            pretrained_model_name=pretrained_model_name,
        )

        self.criterion = nn.NLLLoss(weight=class_weights)

        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc   = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc  = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_f1    = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.test_f1   = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average="macro")

    def forward(self, waveform):
        return self.model(waveform)

    def _shared_step(self, batch):
        waveforms, labels = batch
        log_probs = self(waveforms)
        loss = self.criterion(log_probs, labels)
        preds = log_probs.argmax(dim=-1)
        return loss, preds, labels

    def training_step(self, batch, _):
        loss, preds, labels = self._shared_step(batch)
        self.train_acc(preds, labels)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc",  self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        loss, preds, labels = self._shared_step(batch)
        self.val_acc(preds, labels)
        self.val_f1(preds, labels)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc",  self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/f1",   self.val_f1,  on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, _):
        loss, preds, labels = self._shared_step(batch)
        self.test_acc(preds, labels)
        self.test_f1(preds, labels)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc",  self.test_acc, on_step=False, on_epoch=True)
        self.log("test/f1",   self.test_f1,  on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min",
            patience=self.hparams.scheduler_patience,
            factor=self.hparams.scheduler_factor,
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val/loss"}}


# ── CLI ────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune Wav2Vec2 for speech emotion recognition")
    p.add_argument("--ravdess_dir", type=str, default="data/ravdess")
    p.add_argument("--tess_dir", type=str, default="data/toronto")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--target_sr", type=int, default=16000)
    p.add_argument("--target_duration", type=float, default=3.0)
    p.add_argument("--pretrained_model_name", type=str, default="facebook/wav2vec2-base")
    p.add_argument("--freeze_feature_extractor", action="store_true", default=True)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--max_epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--scheduler_patience", type=int, default=3)
    p.add_argument("--scheduler_factor", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--accelerator", type=str, default="auto")
    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--log_dir", type=str, default="logs")
    p.add_argument("--mlflow_tracking_uri", type=str, default="mlruns")
    p.add_argument("--ckpt_path", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")

    datamodule = Wav2Vec2DataModule(
        ravdess_dir=args.ravdess_dir,
        tess_dir=args.tess_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        target_sr=args.target_sr,
        target_duration=args.target_duration,
        seed=args.seed,
    )
    datamodule.setup()
    class_weights = datamodule.compute_class_weights()

    model = Wav2Vec2LitModel(
        num_classes=len(EMOTION_MAP),
        dropout=args.dropout,
        freeze_feature_extractor=args.freeze_feature_extractor,
        pretrained_model_name=args.pretrained_model_name,
        lr=args.lr,
        weight_decay=args.weight_decay,
        scheduler_patience=args.scheduler_patience,
        scheduler_factor=args.scheduler_factor,
        class_weights=class_weights,
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=f"{args.log_dir}/wav2vec2_checkpoints",
            filename="wav2vec2-{epoch:02d}-{val/acc:.3f}",
            monitor="val/acc", mode="max", save_top_k=3, save_last=True,
        ),
        EarlyStopping(monitor="val/loss", patience=args.patience, mode="min", verbose=True),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    loggers = [TensorBoardLogger(save_dir=args.log_dir, name="wav2vec2_emotion")]
    if _MLFLOW_AVAILABLE:
        loggers.append(MLFlowLogger(
            experiment_name="wav2vec2_emotion",
            tracking_uri=args.mlflow_tracking_uri,
            log_model=True,
        ))

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        callbacks=callbacks,
        logger=loggers,
        deterministic=True,
        log_every_n_steps=1,
    )

    trainer.fit(model, datamodule=datamodule, ckpt_path=args.ckpt_path)
    trainer.test(model, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()
