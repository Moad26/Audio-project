"""
wav2vec2_model.py — Wav2Vec2-based speech emotion classifier.

Fine-tunes facebook/wav2vec2-base with a custom emotion head for sequence
classification on the 8-class RAVDESS/TESS emotion set.

Architecture:
    Wav2Vec2Model (frozen / partially frozen) → mean-pool last hidden state
    → Dropout → Linear(hidden_dim, num_classes) → log_softmax
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Wav2Vec2EmotionModel(nn.Module):
    """
    Wav2Vec2 fine-tuned for speech emotion recognition.

    Args:
        num_classes:    Number of emotion categories (default 8).
        dropout:        Dropout probability before the classifier head.
        freeze_feature_extractor: Whether to freeze the CNN feature encoder
                        (leaves the transformer layers trainable).
        pretrained_model_name: HuggingFace model hub name.
    """

    def __init__(
        self,
        num_classes: int = 8,
        dropout: float = 0.3,
        freeze_feature_extractor: bool = True,
        pretrained_model_name: str = "facebook/wav2vec2-base",
    ) -> None:
        super().__init__()

        try:
            from transformers import Wav2Vec2Model
        except ImportError as exc:
            raise ImportError(
                "Install 'transformers' to use Wav2Vec2EmotionModel: "
                "uv add transformers accelerate"
            ) from exc

        self.wav2vec2 = Wav2Vec2Model.from_pretrained(pretrained_model_name)

        # Optionally freeze the CNN feature extractor (recommended for small datasets)
        if freeze_feature_extractor:
            self.wav2vec2.feature_extractor._freeze_parameters()

        hidden_size = self.wav2vec2.config.hidden_size  # 768 for wav2vec2-base

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, waveform: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            waveform:       (batch, seq_len) raw 16 kHz float waveform.
            attention_mask: (batch, seq_len) optional padding mask.

        Returns:
            log_probs: (batch, num_classes) log-softmax probabilities.
        """
        outputs = self.wav2vec2(
            input_values=waveform,
            attention_mask=attention_mask,
        )
        # Mean-pool the last hidden state over time
        hidden_states = outputs.last_hidden_state   # (batch, T, hidden_size)
        if attention_mask is not None:
            # Build mask aligned to hidden state time dimension
            mask = self._get_feature_mask(attention_mask, hidden_states.size(1))
            mask = mask.unsqueeze(-1).float()       # (batch, T, 1)
            pooled = (hidden_states * mask).sum(1) / mask.sum(1).clamp(min=1)
        else:
            pooled = hidden_states.mean(dim=1)      # (batch, hidden_size)

        logits = self.classifier(pooled)            # (batch, num_classes)
        return torch.log_softmax(logits, dim=-1)

    def _get_feature_mask(self, attention_mask: torch.Tensor, target_len: int) -> torch.Tensor:
        """Down-sample attention mask from waveform length to hidden-state length."""
        # Use the model's built-in helper when available
        if hasattr(self.wav2vec2, "_get_feat_extract_output_lengths"):
            lengths = self.wav2vec2._get_feat_extract_output_lengths(
                attention_mask.sum(-1)
            )
            mask = torch.zeros(
                attention_mask.size(0), target_len, device=attention_mask.device
            )
            for i, length in enumerate(lengths):
                mask[i, : length.item()] = 1
            return mask
        # Fallback: expand / shrink naively
        return torch.ones(
            attention_mask.size(0), target_len, device=attention_mask.device
        )
