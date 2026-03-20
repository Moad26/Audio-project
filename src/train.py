"""
train.py — PyTorch Lightning training script for RAVDESS + TESS emotion recognition.

Usage:
    uv run python src/train.py                          # train with defaults
    uv run python src/train.py --max_epochs 50          # override epochs
    uv run python src/train.py --accelerator gpu        # use GPU
    uv run python src/train.py --batch_size 64 --lr 3e-4
"""

import argparse
from collections import Counter
from pathlib import Path

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

try:
    from pytorch_lightning.loggers import MLFlowLogger

    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False
from torch.utils.data import ConcatDataset, DataLoader, random_split

from data import EMOTION_MAP, RavdessDataset, TorontoDataset
from model import EmotionModel


# ── Lightning DataModule ────────────────────────────────────────────────
class CombinedDataModule(pl.LightningDataModule):
    """Handles train / val / test splits and DataLoaders for RAVDESS + TESS."""

    def __init__(
        self,
        ravdess_dir: str = "data/ravdess",
        tess_dir: str = "data/toronto",
        batch_size: int = 32,
        num_workers: int = 4,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        n_mels: int = 40,
        n_fft: int = 2048,
        hop_length: int = 512,
        target_duration: float = 3.0,
        new_sr: int = 22050,
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
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.target_duration = target_duration
        self.new_sr = new_sr
        self.seed = seed

    def _make_datasets(self, apply_augmentation: bool) -> ConcatDataset:
        """Instantiate RAVDESS + TESS and return a ConcatDataset."""
        shared = dict(
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            target_duration=self.target_duration,
            new_sr=self.new_sr,
            apply_augmentation=apply_augmentation,
            seed=self.seed,
        )
        ravdess_ds = RavdessDataset(data_dir=self.ravdess_dir, **shared)
        tess_ds = TorontoDataset(data_dir=self.tess_dir, **shared)
        return ConcatDataset([ravdess_ds, tess_ds])

    def setup(self, stage=None):
        # Two independent ConcatDatasets: augmented for train, clean for val/test
        aug_combined = self._make_datasets(apply_augmentation=True)
        noaug_combined = self._make_datasets(apply_augmentation=False)

        total = len(aug_combined)
        train_size = int(self.train_ratio * total)
        val_size = int(self.val_ratio * total)
        test_size = total - train_size - val_size

        # Same seed → identical index partitions across both splits
        self.train_dataset, _, _ = random_split(
            aug_combined,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(self.seed),
        )
        _, self.val_dataset, self.test_dataset = random_split(
            noaug_combined,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(self.seed),
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def compute_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights from the training subset."""
        label_counts: Counter = Counter()
        for idx in self.train_dataset.indices:
            # Peek into the underlying ConcatDataset to get the label quickly
            cumulative = 0
            for ds in self.train_dataset.dataset.datasets:
                if idx < cumulative + len(ds):
                    label_counts[ds.labels[idx - cumulative]] += 1
                    break
                cumulative += len(ds)
        num_classes = len(EMOTION_MAP)
        total = sum(label_counts.values())
        weights = torch.zeros(num_classes)
        for cls in range(num_classes):
            count = label_counts.get(cls, 1)  # avoid division by zero
            weights[cls] = total / (num_classes * count)
        return weights


# ── Lightning Module ────────────────────────────────────────────────────
class EmotionLitModel(pl.LightningModule):
    """Wraps EmotionModel with training / validation / test logic."""

    def __init__(
        self,
        num_classes: int = 8,
        hidden_size: int = 256,
        num_lstm_layers: int = 2,
        dropout: float = 0.3,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        scheduler_patience: int = 5,
        scheduler_factor: float = 0.5,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])

        self.model = EmotionModel(
            num_classes=num_classes,
            hidden_size=hidden_size,
            num_lstm_layers=num_lstm_layers,
            dropout=dropout,
        )

        # NLLLoss expects log-probabilities (which our model outputs)
        self.criterion = nn.NLLLoss(weight=class_weights)

        # Accuracy metrics
        self.train_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )

        # Macro F1-score metrics (val + test only)
        self.val_f1 = torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.test_f1 = torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )

    # ── Forward ──
    def forward(self, x):
        return self.model(x)

    # ── Shared step ──
    def _shared_step(self, batch, stage: str):
        specs = batch.spectrogram  # (B, 1, n_mels, T)
        labels = batch.label  # (B,)

        log_probs = self(specs)
        loss = self.criterion(log_probs, labels)
        preds = log_probs.argmax(dim=-1)

        return loss, preds, labels

    # ── Training ──
    def training_step(self, batch, batch_idx):
        loss, preds, labels = self._shared_step(batch, "train")
        self.train_acc(preds, labels)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True
        )
        return loss

    # ── Validation ──
    def validation_step(self, batch, batch_idx):
        loss, preds, labels = self._shared_step(batch, "val")
        self.val_acc(preds, labels)
        self.val_f1(preds, labels)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)

    # ── Test ──
    def test_step(self, batch, batch_idx):
        loss, preds, labels = self._shared_step(batch, "test")
        self.test_acc(preds, labels)
        self.test_f1(preds, labels)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True)
        self.log("test/f1", self.test_f1, on_step=False, on_epoch=True)

    # ── Optimizer & Scheduler ──
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=self.hparams.scheduler_patience,
            factor=self.hparams.scheduler_factor,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
            },
        }


# ── CLI ─────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Train RAVDESS + TESS Emotion Recognition")

    # Data
    p.add_argument("--ravdess_dir", type=str, default="data/ravdess")
    p.add_argument("--tess_dir", type=str, default="data/toronto")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--val_ratio", type=float, default=0.1)

    # Audio
    p.add_argument("--n_mels", type=int, default=40)
    p.add_argument("--n_fft", type=int, default=2048)
    p.add_argument("--hop_length", type=int, default=512)
    p.add_argument("--target_duration", type=float, default=3.0)
    p.add_argument("--new_sr", type=int, default=22050)

    # Model
    p.add_argument("--hidden_size", type=int, default=256)
    p.add_argument("--num_lstm_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.3)

    # Training
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--max_epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=10, help="Early-stop patience")
    p.add_argument("--scheduler_patience", type=int, default=5)
    p.add_argument("--scheduler_factor", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)

    # Trainer
    p.add_argument("--accelerator", type=str, default="auto")
    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--log_dir", type=str, default="logs")
    p.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Path to a checkpoint to resume training from (e.g. logs/checkpoints/last.ckpt)",
    )
    p.add_argument(
        "--mlflow_tracking_uri",
        type=str,
        default="mlruns",
        help="MLflow tracking URI (local path or remote server URL)",
    )

    return p.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")  # leverage Tensor Cores on RTX GPUs

    # ── DataModule ──
    datamodule = CombinedDataModule(
        ravdess_dir=args.ravdess_dir,
        tess_dir=args.tess_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        target_duration=args.target_duration,
        new_sr=args.new_sr,
        seed=args.seed,
    )

    # ── DataModule setup (needed for class weights) ──
    datamodule.setup()
    class_weights = datamodule.compute_class_weights()

    # ── Model ──
    model = EmotionLitModel(
        num_classes=len(EMOTION_MAP),
        hidden_size=args.hidden_size,
        num_lstm_layers=args.num_lstm_layers,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        scheduler_patience=args.scheduler_patience,
        scheduler_factor=args.scheduler_factor,
        class_weights=class_weights,
    )

    # ── Callbacks ──
    callbacks = [
        ModelCheckpoint(
            dirpath=f"{args.log_dir}/checkpoints",
            filename="emotion-{epoch:02d}-{val/acc:.3f}",
            monitor="val/acc",
            mode="max",
            save_top_k=3,
            save_last=True,
        ),
        EarlyStopping(
            monitor="val/loss",
            patience=args.patience,
            mode="min",
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    # ── Logger ──
    tb_logger = TensorBoardLogger(save_dir=args.log_dir, name="emotion_recognition")
    loggers = [tb_logger]
    if _MLFLOW_AVAILABLE:
        mlflow_logger = MLFlowLogger(
            experiment_name="emotion_recognition",
            tracking_uri=args.mlflow_tracking_uri,
            log_model=True,
        )
        loggers.append(mlflow_logger)
        print(f"MLflow logging to: {args.mlflow_tracking_uri}")
    else:
        print("mlflow not installed — using TensorBoard only.")

    # ── Trainer ──
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        callbacks=callbacks,
        logger=loggers,
        deterministic=True,
        log_every_n_steps=1,
    )

    # ── Train & Test ──
    trainer.fit(model, datamodule=datamodule, ckpt_path=args.ckpt_path)
    trainer.test(model, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()
