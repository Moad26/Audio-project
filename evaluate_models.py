#!/usr/bin/env python3
"""
evaluate_models.py
==================
Evaluate the CNN-LSTM and Wav2Vec2 models on the held-out test set and
print every number needed to fill in the LaTeX tables in report.tex.

Usage
-----
    # Evaluate both models (default):
    uv run python evaluate_models.py

    # Evaluate only CNN-LSTM:
    uv run python evaluate_models.py --model cnn_lstm \
        --cnn_ckpt logs/checkpoints/last.ckpt

    # Evaluate only Wav2Vec2:
    uv run python evaluate_models.py --model wav2vec2 \
        --w2v_ckpt logs/wav2vec2/checkpoints/last.ckpt

    # Save confusion-matrix PNGs:
    uv run python evaluate_models.py --save_plots --plots_dir results/

Arguments
---------
  --cnn_ckpt     Path to the CNN-LSTM PyTorch-Lightning checkpoint (.ckpt)
  --w2v_ckpt     Path to the Wav2Vec2 PyTorch-Lightning checkpoint (.ckpt)
  --ravdess_dir  Path to RAVDESS data   (default: data/ravdess)
  --tess_dir     Path to TESS data      (default: data/toronto)
  --batch_size   DataLoader batch size  (default: 32)
  --num_workers  DataLoader workers     (default: 4)
  --n_mels       Mel bands used during training (default: 40)
  --seed         Random seed            (default: 42)
  --model        Which model(s) to run: cnn_lstm | wav2vec2 | both (default: both)
  --save_plots   Save confusion-matrix PNGs instead of only displaying them
  --plots_dir    Directory for saved plots (default: results/)
  --latency_runs Number of forward passes for latency benchmark (default: 100)
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, ConcatDataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EMOTION_LABELS = {
    0: "neutral",
    1: "calm",
    2: "happy",
    3: "sad",
    4: "angry",
    5: "fearful",
    6: "disgust",
    7: "surprised",
}

# Map emotion name → LaTeX-friendly display name (capitalised)
LATEX_LABELS = {v: v.capitalize() for v in EMOTION_LABELS.values()}


def _sep(char="=", width=72):
    print(char * width)


def _header(title: str):
    _sep()
    print(f"  {title}")
    _sep()


def _latex_float(v: float) -> str:
    """Format a float as a 4-decimal LaTeX-ready string."""
    return f"{v:.4f}"


# ---------------------------------------------------------------------------
# Results printer
# ---------------------------------------------------------------------------

def print_latex_tables(
    model_name: str,
    y_true: list,
    y_pred: list,
    label_names: list[str],
):
    """Print all metrics in the exact format needed to fill in report.tex."""

    _header(f"RESULTS — {model_name}")

    # ── Overall metrics ──
    acc   = accuracy_score(y_true, y_pred)
    prec  = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec   = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1    = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print("\n📋  TABLE: tab:overall — Overall Performance")
    print(f"    Accuracy       : {_latex_float(acc)}")
    print(f"    Macro Precision: {_latex_float(prec)}")
    print(f"    Macro Recall   : {_latex_float(rec)}")
    print(f"    Macro F1       : {_latex_float(f1)}")

    # ── Per-class metrics ──
    print(f"\n📋  TABLE: tab:peremotion — Per-Emotion Recall & F1")
    print(f"    {'Emotion':<12}  {'Recall':>8}  {'F1':>8}")
    print("    " + "-" * 32)

    per_class_f1  = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_class_rec = recall_score(y_true, y_pred, average=None, zero_division=0)

    present_labels = sorted(set(y_true))
    for idx, lbl_idx in enumerate(present_labels):
        name = label_names[lbl_idx] if lbl_idx < len(label_names) else str(lbl_idx)
        r = per_class_rec[idx] if idx < len(per_class_rec) else float("nan")
        f = per_class_f1[idx]  if idx < len(per_class_f1)  else float("nan")
        print(f"    {name.capitalize():<12}  {r:>8.4f}  {f:>8.4f}")

    print("    " + "-" * 32)
    print(f"    {'Macro Avg':<12}  {rec:>8.4f}  {f1:>8.4f}")

    # ── Full sklearn classification report (reference) ──
    print(f"\n📄  Full Classification Report ({model_name}):")
    report = classification_report(
        y_true, y_pred,
        target_names=[label_names[i] for i in sorted(set(y_true))],
        zero_division=0,
    )
    print(report)

    return {
        "accuracy": acc,
        "macro_precision": prec,
        "macro_recall": rec,
        "macro_f1": f1,
        "per_class_recall": per_class_rec,
        "per_class_f1": per_class_f1,
    }


def print_ablation_row(tag: str, y_true: list, y_pred: list):
    """Print one row for the ablation study table."""
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    print(f"    {tag:<35}  {_latex_float(f1)}")
    return f1


def print_confusion_matrix(model_name: str, y_true, y_pred, label_names, save: bool, out_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg" if save else "TkAgg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        present = sorted(set(y_true))
        names   = [label_names[i].capitalize() for i in present]
        cm = confusion_matrix(y_true, y_pred, labels=present)

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=names, yticklabels=names, ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix — {model_name}")
        plt.tight_layout()

        if save:
            out_dir.mkdir(parents=True, exist_ok=True)
            fname = out_dir / f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
            fig.savefig(fname, dpi=150)
            print(f"  ✅  Confusion matrix saved → {fname}")
        else:
            plt.show()

        plt.close(fig)

    except ImportError:
        print("  ⚠️  matplotlib/seaborn not installed — skipping confusion matrix plot.")


# ---------------------------------------------------------------------------
# Latency benchmark
# ---------------------------------------------------------------------------

def benchmark_latency(model, dummy_input, n_runs: int, device: torch.device, label: str):
    """Run `n_runs` forward passes and report mean ± std latency in ms."""
    model.eval()
    dummy_input = dummy_input.to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy_input)

    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

    mean_ms = float(np.mean(times))
    std_ms  = float(np.std(times))
    print(f"\n⏱️  Latency — {label} ({device.type.upper()}): "
          f"{mean_ms:.1f} ± {std_ms:.1f} ms  (n={n_runs})")
    print(f"   → LaTeX value: \\(\\mathbf{{{mean_ms:.1f} \\pm {std_ms:.1f}}}\\)\\,ms")
    return mean_ms, std_ms


# ---------------------------------------------------------------------------
# Data helpers (mirrors train.py setup)
# ---------------------------------------------------------------------------

def build_test_loader(args) -> tuple[DataLoader, list[str]]:
    """Reconstruct the exact same 80/10/10 split used in train.py."""
    sys.path.insert(0, str(Path(__file__).parent / "src"))
    from data import RavdessDataset, TorontoDataset, EMOTION_MAP

    shared = dict(
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        target_duration=args.target_duration,
        new_sr=args.new_sr,
        apply_augmentation=False,   # never augment the test set
        seed=args.seed,
    )

    ravdess_ds = RavdessDataset(data_dir=Path(args.ravdess_dir), **shared)
    tess_ds    = TorontoDataset(data_dir=Path(args.tess_dir),    **shared)
    combined   = ConcatDataset([ravdess_ds, tess_ds])

    total      = len(combined)
    train_size = int(args.train_ratio * total)
    val_size   = int(args.val_ratio   * total)
    test_size  = total - train_size - val_size

    _, _, test_ds = random_split(
        combined,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # 0-indexed label → emotion name
    label_names = [EMOTION_MAP[k] for k in sorted(EMOTION_MAP)]
    return loader, label_names


# ---------------------------------------------------------------------------
# CNN-LSTM evaluation
# ---------------------------------------------------------------------------

def evaluate_cnn_lstm(args, test_loader, label_names, device):
    sys.path.insert(0, str(Path(__file__).parent / "src"))
    from train import EmotionLitModel
    from data  import EMOTION_MAP

    ckpt = Path(args.cnn_ckpt)
    if not ckpt.exists():
        print(f"  ⚠️  CNN-LSTM checkpoint not found: {ckpt}  — skipping.")
        return None

    print(f"\n🔄  Loading CNN-LSTM from: {ckpt}")
    lit_model = EmotionLitModel.load_from_checkpoint(
        str(ckpt),
        num_classes=len(EMOTION_MAP),
        map_location=device,
        strict=False,   # ignore criterion.weight stored in ckpt (not needed for eval)
    )
    lit_model.eval()
    lit_model.to(device)
    model = lit_model.model  # bare nn.Module

    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in test_loader:
            specs  = batch.spectrogram.to(device)  # (B, 4, n_mels, T)
            labels = batch.label                    # (B,)
            log_probs = model(specs)
            preds = log_probs.argmax(dim=-1).cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(labels.tolist())

    metrics = print_latex_tables("CNN-LSTM", y_true, y_pred, label_names)

    print_confusion_matrix(
        "CNN-LSTM", y_true, y_pred, label_names,
        save=args.save_plots, out_dir=Path(args.plots_dir),
    )

    # Latency benchmark (single sample)
    dummy = torch.zeros(1, 4, args.n_mels, 130, device=device)
    benchmark_latency(model, dummy, args.latency_runs, device, "CNN-LSTM")

    return y_true, y_pred, metrics


# ---------------------------------------------------------------------------
# Wav2Vec2 evaluation
# ---------------------------------------------------------------------------

def evaluate_wav2vec2(args, device):
    sys.path.insert(0, str(Path(__file__).parent / "src"))
    from wav2vec2_train import Wav2Vec2LitModel
    from data import EMOTION_MAP

    ckpt = Path(args.w2v_ckpt)
    if not ckpt.exists():
        print(f"  ⚠️  Wav2Vec2 checkpoint not found: {ckpt}  — skipping.")
        return None

    print(f"\n🔄  Loading Wav2Vec2 from: {ckpt}")
    lit_model = Wav2Vec2LitModel.load_from_checkpoint(
        str(ckpt),
        num_classes=len(EMOTION_MAP),
        map_location=device,
        strict=False,
    )
    lit_model.eval()
    lit_model.to(device)
    model = lit_model.model

    # Build test loader for raw waveforms (wav2vec2 takes raw audio, not spectrograms)
    from wav2vec2_train import Wav2Vec2DataModule
    dm = Wav2Vec2DataModule(
        ravdess_dir=args.ravdess_dir,
        tess_dir=args.tess_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_sr=16000,  # wav2vec2 expects 16 kHz
        target_duration=args.target_duration,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    dm.setup()

    label_names = [EMOTION_MAP[k] for k in sorted(EMOTION_MAP)]
    y_true, y_pred = [], []

    with torch.no_grad():
        for batch in dm.test_dataloader():
            waveform, labels = batch
            waveform = waveform.to(device)
            # Wav2Vec2DataModule currently doesn't produce attention_masks (all same padded length)
            log_probs = model(waveform, attention_mask=None)
            preds = log_probs.argmax(dim=-1).cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(labels.tolist())

    metrics = print_latex_tables("Wav2Vec2", y_true, y_pred, label_names)

    print_confusion_matrix(
        "Wav2Vec2", y_true, y_pred, label_names,
        save=args.save_plots, out_dir=Path(args.plots_dir),
    )

    # Latency benchmark — 3-second clip at 16 kHz
    dummy = torch.zeros(1, 16000 * 3, device=device)
    benchmark_latency(model, dummy, args.latency_runs, device, "Wav2Vec2")

    return y_true, y_pred, metrics


# ---------------------------------------------------------------------------
# Ablation study
# ---------------------------------------------------------------------------

def run_ablation(args, device):
    """
    Print rows for tab:ablation.

    Requires four separate CNN-LSTM checkpoints trained under different configs:
      --ablation_ckpt_ravdess_noaug
      --ablation_ckpt_ravdess_aug
      --ablation_ckpt_both_noaug
      --ablation_ckpt_both_aug   (same as --cnn_ckpt if trained with both + aug)
    """
    sys.path.insert(0, str(Path(__file__).parent / "src"))
    from train  import EmotionLitModel
    from data   import EMOTION_MAP, RavdessDataset, TorontoDataset

    ablation_configs = [
        ("RAVDESS only  | No aug",  args.ablation_ckpt_ravdess_noaug, False, False),
        ("RAVDESS only  | Aug",     args.ablation_ckpt_ravdess_aug,   False, True),
        ("RAVDESS + TESS| No aug",  args.ablation_ckpt_both_noaug,    True,  False),
        ("RAVDESS + TESS| Aug",     args.ablation_ckpt_both_aug,      True,  True),
    ]

    _header("ABLATION STUDY — tab:ablation")
    print(f"\n    {'Config':<35}  {'Macro F1':>10}  {'Δ vs Full':>10}")
    print("    " + "-" * 60)

    results = {}
    shared_ds = dict(
        n_mels=args.n_mels, n_fft=args.n_fft, hop_length=args.hop_length,
        target_duration=args.target_duration, new_sr=args.new_sr,
        apply_augmentation=False, seed=args.seed,   # no aug on test set
    )

    label_names = [EMOTION_MAP[k] for k in sorted(EMOTION_MAP)]

    for tag, ckpt_path, use_tess, _ in ablation_configs:
        ckpt = Path(ckpt_path) if ckpt_path else None
        if ckpt is None or not ckpt.exists():
            print(f"    {tag:<35}  {'N/A (ckpt missing)':>20}")
            results[tag] = None
            continue

        lit = EmotionLitModel.load_from_checkpoint(
            str(ckpt), num_classes=len(EMOTION_MAP), map_location=device,
            strict=False,  # ignore criterion.weight
        )
        lit.eval(); lit.to(device)
        nn_model = lit.model

        # Build matching test set
        ravdess_ds = RavdessDataset(data_dir=Path(args.ravdess_dir), **shared_ds)
        datasets   = [ravdess_ds]
        if use_tess:
            tess_ds = TorontoDataset(data_dir=Path(args.tess_dir), **shared_ds)
            datasets.append(tess_ds)
        combined = ConcatDataset(datasets)
        total    = len(combined)
        train_sz = int(args.train_ratio * total)
        val_sz   = int(args.val_ratio   * total)
        test_sz  = total - train_sz - val_sz
        _, _, test_ds = random_split(
            combined, [train_sz, val_sz, test_sz],
            generator=torch.Generator().manual_seed(args.seed),
        )
        loader = DataLoader(test_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)

        y_true_abl, y_pred_abl = [], []
        with torch.no_grad():
            for batch in loader:
                specs  = batch.spectrogram.to(device)
                labels = batch.label
                preds  = nn_model(specs).argmax(dim=-1).cpu().tolist()
                y_pred_abl.extend(preds)
                y_true_abl.extend(labels.tolist())

        f1 = print_ablation_row(tag, y_true_abl, y_pred_abl)
        results[tag] = f1

    # Print Delta column
    full_tag  = "RAVDESS + TESS| Aug"
    full_f1   = results.get(full_tag)
    if full_f1 is not None:
        print(f"\n    Delta column (vs. {full_tag}):")
        for tag, f1 in results.items():
            if f1 is not None and tag != full_tag:
                delta = f1 - full_f1
                sign  = "+" if delta >= 0 else ""
                print(f"      {tag:<35}  Δ = {sign}{delta:.4f}")

    return results


# ---------------------------------------------------------------------------
# LaTeX snippet generator
# ---------------------------------------------------------------------------

def print_latex_snippet(cnn_metrics, w2v_metrics, label_names):
    """Print copy-pasteable LaTeX for the three report tables."""
    _header("COPY-PASTE LaTeX SNIPPETS")

    # ── tab:overall ──
    print("\n%% --- Table: tab:overall ---")
    print("\\begin{tabular}{lcccc}")
    print("\\toprule")
    print("\\textbf{Model} & \\textbf{Accuracy} & \\textbf{Macro Precision} & "
          "\\textbf{Macro Recall} & \\textbf{Macro F1} \\\\")
    print("\\midrule")
    if cnn_metrics:
        m = cnn_metrics
        print(f"CNN-LSTM Hybrid & {m['accuracy']:.4f} & {m['macro_precision']:.4f} "
              f"& {m['macro_recall']:.4f} & {m['macro_f1']:.4f} \\\\")
    if w2v_metrics:
        m = w2v_metrics
        print(f"Wav2Vec2 (fine-tuned) & {m['accuracy']:.4f} & {m['macro_precision']:.4f} "
              f"& {m['macro_recall']:.4f} & \\textbf{{{m['macro_f1']:.4f}}} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")

    # ── tab:peremotion ──
    print("\n%% --- Table: tab:peremotion ---")
    print("\\begin{tabular}{lcccc}")
    print("\\toprule")
    print("\\multirow{2}{*}{\\textbf{Emotion}} & \\multicolumn{2}{c}{\\textbf{CNN-LSTM}} "
          "& \\multicolumn{2}{c}{\\textbf{Wav2Vec2}} \\\\")
    print("\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}")
    print("& Recall & F1 & Recall & F1 \\\\")
    print("\\midrule")

    present = list(range(len(label_names)))
    for idx in present:
        name = label_names[idx].capitalize() if idx < len(label_names) else str(idx)
        c_rec = cnn_metrics["per_class_recall"][idx] if cnn_metrics and idx < len(cnn_metrics["per_class_recall"]) else float("nan")
        c_f1  = cnn_metrics["per_class_f1"][idx]     if cnn_metrics and idx < len(cnn_metrics["per_class_f1"])     else float("nan")
        w_rec = w2v_metrics["per_class_recall"][idx]  if w2v_metrics and idx < len(w2v_metrics["per_class_recall"]) else float("nan")
        w_f1  = w2v_metrics["per_class_f1"][idx]      if w2v_metrics and idx < len(w2v_metrics["per_class_f1"])     else float("nan")
        print(f"{name:<12} & {c_rec:.4f} & {c_f1:.4f} & {w_rec:.4f} & {w_f1:.4f} \\\\")

    c_r = cnn_metrics["macro_recall"]  if cnn_metrics else float("nan")
    c_f = cnn_metrics["macro_f1"]      if cnn_metrics else float("nan")
    w_r = w2v_metrics["macro_recall"]  if w2v_metrics else float("nan")
    w_f = w2v_metrics["macro_f1"]      if w2v_metrics else float("nan")
    print("\\midrule")
    print(f"\\textbf{{Macro Avg}} & {c_r:.4f} & {c_f:.4f} & {w_r:.4f} & \\textbf{{{w_f:.4f}}} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate SER models and produce LaTeX table values.")

    # Checkpoints
    p.add_argument("--cnn_ckpt", default="logs/checkpoints/last.ckpt",
                   help="Path to CNN-LSTM Lightning checkpoint (.ckpt)")
    p.add_argument("--w2v_ckpt", default="logs/wav2vec2/checkpoints/last.ckpt",
                   help="Path to Wav2Vec2 Lightning checkpoint (.ckpt)")

    # Ablation checkpoints (optional)
    p.add_argument("--ablation_ckpt_ravdess_noaug", default=None)
    p.add_argument("--ablation_ckpt_ravdess_aug",   default=None)
    p.add_argument("--ablation_ckpt_both_noaug",    default=None)
    p.add_argument("--ablation_ckpt_both_aug",      default=None)

    # Data
    p.add_argument("--ravdess_dir",      default="data/ravdess")
    p.add_argument("--tess_dir",         default="data/toronto")
    p.add_argument("--batch_size",       type=int,   default=32)
    p.add_argument("--num_workers",      type=int,   default=4)
    p.add_argument("--train_ratio",      type=float, default=0.8)
    p.add_argument("--val_ratio",        type=float, default=0.1)
    p.add_argument("--n_mels",           type=int,   default=40)
    p.add_argument("--n_fft",            type=int,   default=2048)
    p.add_argument("--hop_length",       type=int,   default=512)
    p.add_argument("--target_duration",  type=float, default=3.0)
    p.add_argument("--new_sr",           type=int,   default=22050)
    p.add_argument("--seed",             type=int,   default=42)

    # Which model(s) to evaluate
    p.add_argument("--model", choices=["cnn_lstm", "wav2vec2", "both"], default="both")

    # Ablation
    p.add_argument("--ablation", action="store_true",
                   help="Run ablation study (requires four separate checkpoints)")

    # Plots
    p.add_argument("--save_plots", action="store_true",
                   help="Save confusion-matrix PNGs to --plots_dir")
    p.add_argument("--plots_dir", default="results/")

    # Latency
    p.add_argument("--latency_runs", type=int, default=100,
                   help="Number of forward passes for the latency benchmark")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️   Device: {device}")

    cnn_metrics = None
    w2v_metrics = None
    label_names = None

    # ── CNN-LSTM ──
    if args.model in ("cnn_lstm", "both"):
        test_loader, label_names = build_test_loader(args)
        result = evaluate_cnn_lstm(args, test_loader, label_names, device)
        if result is not None:
            _, _, cnn_metrics = result

    # ── Wav2Vec2 ──
    if args.model in ("wav2vec2", "both"):
        result = evaluate_wav2vec2(args, device)
        if result is not None:
            _, _, w2v_metrics = result

    # ── Ablation study ──
    if args.ablation:
        run_ablation(args, device)

    # ── Copy-paste LaTeX snippets ──
    if cnn_metrics or w2v_metrics:
        if label_names is None:
            import sys
            sys.path.insert(0, str(Path(__file__).parent / "src"))
            from data import EMOTION_MAP
            label_names = [EMOTION_MAP[k] for k in sorted(EMOTION_MAP)]
        print_latex_snippet(cnn_metrics, w2v_metrics, label_names)

    _sep()
    print("  ✅  Evaluation complete.  Copy the values above into report.tex.")
    _sep()


if __name__ == "__main__":
    main()
