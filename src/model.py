import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def conv3x3(in_channels, out_channels, stride=1):
    """3×3 convolution with padding (no bias when followed by BatchNorm)."""
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
    )


# ── Residual Block ──────────────────────────────────────────────────────
class ResidualBlock(nn.Module):
    """Basic residual block with optional downsampling."""

    def __init__(
        self, in_channels, out_channels, stride=1, downsample=None, dropout=0.1
    ):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return self.dropout(out)


# ── Bidirectional LSTM ──────────────────────────────────────────────────
class BidirectionalLSTM(nn.Module):
    """Bidirectional LSTM that processes the temporal sequence from the CNN."""

    def __init__(self, input_size, hidden_size, num_layers, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

    def forward(self, x):
        # x: (batch, time_steps, features)
        output, (h_n, _) = self.lstm(x)
        # h_n: (num_layers * 2, batch, hidden_size) – concatenate last forward & backward
        forward_last = h_n[-2]  # last layer, forward direction
        backward_last = h_n[-1]  # last layer, backward direction
        return torch.cat(
            [forward_last, backward_last], dim=-1
        )  # (batch, hidden_size*2)


# ── CNN Feature Extractor ───────────────────────────────────────────────
class EmotionCNNFeatureExtractor(nn.Module):
    """
    Residual CNN that turns a mel-spectrogram into a sequence of feature
    vectors suitable for the LSTM.

    Input shape:  (batch, 1, n_mels, time_steps)   e.g. (B, 1, 40, 130)
    Output shape: (batch, T', C * F')              sequence ready for LSTM
    """

    def __init__(self, input_channels=4, dropout=0.1):
        super(EmotionCNNFeatureExtractor, self).__init__()

        # Stage 1: stride (2,1) → halve freq, keep time
        self.stage1 = self._make_stage(
            input_channels, 32, num_blocks=2, first_stride=(2, 1), dropout=dropout
        )
        # Stage 2: stride (2,2) → halve both
        self.stage2 = self._make_stage(
            32, 64, num_blocks=2, first_stride=(2, 2), dropout=dropout
        )
        # Stage 3: stride (2,1) → halve freq, keep time
        self.stage3 = self._make_stage(
            64, 128, num_blocks=2, first_stride=(2, 1), dropout=dropout
        )

    def _make_stage(
        self, in_channels, out_channels, num_blocks, first_stride=(1, 1), dropout=0.1
    ):
        layers = []
        downsample = None

        if first_stride != (1, 1) or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=first_stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

        layers.append(
            ResidualBlock(
                in_channels,
                out_channels,
                stride=first_stride,
                downsample=downsample,
                dropout=dropout,
            )
        )
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, dropout=dropout))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, 1, 40, 130)
        x = self.stage1(x)  # → (B, 32, 20, 130)
        x = self.stage2(x)  # → (B, 64, 10, 65)
        x = self.stage3(x)  # → (B, 128, 5, 65)

        batch_size, channels, freq, time = x.size()
        # Reshape to (batch, time_steps, channels * freq) so the LSTM
        # receives one feature vector per time step.
        x = x.permute(0, 3, 1, 2)  # (B, T, C, F)
        x = x.contiguous().view(batch_size, time, channels * freq)  # (B, T, C*F)
        return x


# ── Full Emotion Model ─────────────────────────────────────────────────
class EmotionModel(nn.Module):
    """
    CNN → LSTM → softmax for speech emotion recognition.

    Pipeline:
        1. CNN extracts spectral feature maps from the mel spectrogram.
        2. Feature maps are reshaped into a temporal sequence.
        3. Bidirectional LSTM captures temporal dynamics.
        4. Final hidden states are projected to class logits.
        5. Softmax (log) produces class probabilities.
    """

    def __init__(self, num_classes=8, hidden_size=256, num_lstm_layers=2, dropout=0.1):
        super(EmotionModel, self).__init__()

        self.cnn = EmotionCNNFeatureExtractor(dropout=dropout)

        # Build a tiny dummy input to auto-detect the CNN output feature size
        # so the model is robust to changes in n_mels / duration / hop_length.
        with torch.no_grad():
            dummy = torch.zeros(1, 4, 40, 130)
            cnn_out = self.cnn(dummy)
            self._cnn_feature_size = cnn_out.size(-1)  # channels * freq'

        self.lstm = BidirectionalLSTM(
            input_size=self._cnn_feature_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            dropout=dropout,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: (batch, 1, n_mels, time_steps) — mel spectrogram
        x = self.cnn(x)  # → (batch, T', features)
        x = self.lstm(x)  # → (batch, hidden_size * 2)
        x = self.classifier(x)  # → (batch, num_classes)
        x = torch.log_softmax(x, dim=-1)
        return x
