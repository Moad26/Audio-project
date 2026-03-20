import random
from collections import namedtuple
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from datautils import SpectrogramUtils


def _resize_to_height(arr: np.ndarray, target_h: int) -> np.ndarray:
    """Resize a 2-D feature map (freq, time) to (target_h, time) using interpolation."""
    if arr.shape[0] == target_h:
        return arr
    t = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1,1,H,T)
    t = F.interpolate(t, size=(target_h, arr.shape[1]), mode="bilinear", align_corners=False)
    return t.squeeze(0).squeeze(0).numpy()

# RAVDESS emotion codes → labels
EMOTION_MAP = {
    1: "neutral",
    2: "calm",
    3: "happy",
    4: "sad",
    5: "angry",
    6: "fearful",
    7: "disgust",
    8: "surprised",
}

BatchItem = namedtuple("BatchItem", ["spectrogram", "label", "emotion", "file_path"])


class RavdessDataset(Dataset):
    """
    PyTorch Dataset for the RAVDESS speech emotion dataset.

    Expected folder layout:
        data_dir/
            Actor_01/
                03-01-01-01-01-01-01.wav
                ...
            Actor_02/
                ...

    Filename identifiers (dash-separated):
        1  Modality        (03 = audio-only)
        2  Vocal channel   (01 = speech, 02 = song)
        3  Emotion         (01–08, see EMOTION_MAP)
        4  Intensity       (01 = normal, 02 = strong)
        5  Statement       (01 or 02)
        6  Repetition      (01 or 02)
        7  Actor           (01–24, odd = male, even = female)
    """

    def __init__(
        self,
        data_dir: Path,
        new_ch: int = 1,
        new_sr: int = 22050,
        target_duration: float = 3.0,
        n_mels: int = 40,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mfcc: int = 13,
        cache_dir: Optional[Path] = None,
        save_cache: bool = False,
        apply_augmentation: bool = True,
        augmentation_prob: float = 0.8,
        shift_limit: float = 0.3,
        seed: int = 42,
    ) -> None:
        super().__init__()

        self.data_dir = Path(data_dir)
        self.spec_utils = SpectrogramUtils()
        self.new_ch = new_ch
        self.new_sr = new_sr
        self.target_duration = target_duration
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mfcc = n_mfcc
        self.cache_dir = cache_dir
        self.save_cache = save_cache
        self.apply_augmentation = apply_augmentation
        self.augmentation_prob = augmentation_prob
        self.shift_limit = shift_limit

        self.set_seed(seed)

        # Discover all .wav files and extract labels from filenames
        self.file_paths: List[Path] = []
        self.labels: List[int] = []  # 0-indexed emotion labels
        self._load_file_list()

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------
    @staticmethod
    def set_seed(seed: int):
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

    def _load_file_list(self) -> None:
        """Walk every Actor_* subfolder and collect .wav paths + emotion labels."""
        for actor_dir in sorted(self.data_dir.glob("Actor_*")):
            if not actor_dir.is_dir():
                continue
            for wav_file in sorted(actor_dir.glob("*.wav")):
                emotion_code = self._parse_emotion(wav_file.stem)
                if emotion_code is not None:
                    self.file_paths.append(wav_file)
                    self.labels.append(emotion_code - 1)  # 0-indexed

    @staticmethod
    def _parse_emotion(filename_stem: str) -> Optional[int]:
        """Extract the emotion code (3rd identifier) from a RAVDESS filename."""
        parts = filename_stem.split("-")
        if len(parts) != 7:
            return None
        try:
            return int(parts[2])
        except ValueError:
            return None

    # ------------------------------------------------------------------
    # Augmentation
    # ------------------------------------------------------------------
    @staticmethod
    def time_shift(
        aud_sr: tuple[np.ndarray, float],
        shift_limit: float = 0.3,
    ) -> tuple[np.ndarray, float]:
        """Randomly shift audio left or right."""
        aud, sr = aud_sr
        n_samples = aud.shape[-1] if aud.ndim > 1 else len(aud)
        shift_amount = int(random.uniform(-shift_limit, shift_limit) * n_samples)
        return np.roll(aud, shift_amount, axis=-1), sr

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, index: int) -> BatchItem:
        audio_path = self.file_paths[index]
        label = self.labels[index]
        emotion_name = EMOTION_MAP[label + 1]  # map back to 1-indexed key

        # Load raw audio via librosa
        aud, sr = self._load_audio(audio_path)

        # Pre-processing pipeline: rechannel → resample → pad/truncate
        aud_sr = SpectrogramUtils.rechannel((aud, sr), new_ch=self.new_ch)
        aud_sr = SpectrogramUtils.resample(aud_sr, new_sr=self.new_sr)
        aud_sr = SpectrogramUtils.pad_or_truncate(
            aud_sr, target_duration=self.target_duration
        )

        # Optional time-shift augmentation
        if self.apply_augmentation and random.random() < self.augmentation_prob:
            aud_sr = self.time_shift(aud_sr, self.shift_limit)

        aud, sr = aud_sr

        # Build the dict that SpectrogramUtils.load_spectrogram expects
        audio_dict = {
            "audio": aud,
            "sr": sr,
            "path": audio_path,
            "source_type": "path",
        }

        result = SpectrogramUtils.load_spectrogram(
            audio_dict=audio_dict,
            new_sr=self.new_sr,
            new_ch=self.new_ch,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            n_mfcc=self.n_mfcc,
            target_duration=self.target_duration,
            cache_dir=self.cache_dir,
            save_cache=self.save_cache,
        )

        # Resize all feature maps to (n_mels, T) and stack into (4, n_mels, T)
        mel  = result["mel_spectrogram"]                          # (n_mels, T)
        mfcc = _resize_to_height(result["mfcc"],  self.n_mels)   # (n_mfcc→n_mels, T)
        mfcc_d = _resize_to_height(result["mfcc_delta"], self.n_mels)
        chro = _resize_to_height(result["chromagram"], self.n_mels)

        feature = np.stack([mel, mfcc, mfcc_d, chro], axis=0)    # (4, n_mels, T)
        feature_tensor = torch.tensor(feature, dtype=torch.float32)

        return BatchItem(
            spectrogram=feature_tensor,
            label=torch.tensor(label, dtype=torch.long),
            emotion=emotion_name,
            file_path=str(audio_path),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _load_audio(path: Path) -> tuple[np.ndarray, float]:
        """Load an audio file and return (waveform, sample_rate)."""
        import librosa

        aud, sr = librosa.load(str(path), sr=None, mono=False)
        return aud, sr

    @property
    def num_classes(self) -> int:
        return len(EMOTION_MAP)

    @property
    def emotion_labels(self) -> dict[int, str]:
        """Return the 0-indexed label → emotion name mapping."""
        return {k - 1: v for k, v in EMOTION_MAP.items()}


class TorontoDataset(Dataset):
    """
    PyTorch Dataset for the TESS (Toronto emotional speech set) dataset.

    Expected folder layout:
        data_dir/
            OAF_Fear/
                OAF_back_fear.wav
                ...
            YAF_angry/
                ...
    """

    def __init__(
        self,
        data_dir: Path,
        new_ch: int = 1,
        new_sr: int = 22050,
        target_duration: float = 3.0,
        n_mels: int = 40,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mfcc: int = 13,
        cache_dir: Optional[Path] = None,
        save_cache: bool = False,
        apply_augmentation: bool = True,
        augmentation_prob: float = 0.8,
        shift_limit: float = 0.3,
        seed: int = 42,
    ) -> None:
        super().__init__()

        self.data_dir = Path(data_dir)
        self.spec_utils = SpectrogramUtils()
        self.new_ch = new_ch
        self.new_sr = new_sr
        self.target_duration = target_duration
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mfcc = n_mfcc
        self.cache_dir = cache_dir
        self.save_cache = save_cache
        self.apply_augmentation = apply_augmentation
        self.augmentation_prob = augmentation_prob
        self.shift_limit = shift_limit

        self.set_seed(seed)

        # Discover all .wav files and extract labels from filenames
        self.file_paths: List[Path] = []
        self.labels: List[int] = []  # 0-indexed emotion labels
        self._load_file_list()

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------
    @staticmethod
    def set_seed(seed: int):
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

    def _load_file_list(self) -> None:
        """Walk every directory and collect .wav paths + emotion labels."""
        for wav_file in sorted(self.data_dir.rglob("*.wav")):
            emotion_code = self._parse_emotion(wav_file.stem)
            if emotion_code is not None:
                self.file_paths.append(wav_file)
                self.labels.append(emotion_code - 1)  # 0-indexed

    @staticmethod
    def _parse_emotion(filename_stem: str) -> Optional[int]:
        """Extract the emotion code from a TESS filename."""
        parts = filename_stem.split("_")
        if len(parts) < 3:
            return None

        emotion_str = "_".join(parts[2:]).lower()

        mapping = {
            "neutral": 1,
            "happy": 3,
            "sad": 4,
            "angry": 5,
            "fear": 6,
            "disgust": 7,
            "ps": 8,
            "pleasant_surprise": 8,
            "pleasant_surprised": 8,
            "surprised": 8,
        }
        return mapping.get(emotion_str, None)

    # ------------------------------------------------------------------
    # Augmentation
    # ------------------------------------------------------------------
    @staticmethod
    def time_shift(
        aud_sr: tuple[np.ndarray, float],
        shift_limit: float = 0.3,
    ) -> tuple[np.ndarray, float]:
        """Randomly shift audio left or right."""
        aud, sr = aud_sr
        n_samples = aud.shape[-1] if aud.ndim > 1 else len(aud)
        shift_amount = int(random.uniform(-shift_limit, shift_limit) * n_samples)
        return np.roll(aud, shift_amount, axis=-1), sr

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, index: int) -> BatchItem:
        audio_path = self.file_paths[index]
        label = self.labels[index]
        emotion_name = EMOTION_MAP[label + 1]  # map back to 1-indexed key

        # Load raw audio via librosa
        aud, sr = self._load_audio(audio_path)

        # Pre-processing pipeline: rechannel → resample → pad/truncate
        aud_sr = SpectrogramUtils.rechannel((aud, sr), new_ch=self.new_ch)
        aud_sr = SpectrogramUtils.resample(aud_sr, new_sr=self.new_sr)
        aud_sr = SpectrogramUtils.pad_or_truncate(
            aud_sr, target_duration=self.target_duration
        )

        # Optional time-shift augmentation
        if self.apply_augmentation and random.random() < self.augmentation_prob:
            aud_sr = self.time_shift(aud_sr, self.shift_limit)

        aud, sr = aud_sr

        # Build the dict that SpectrogramUtils.load_spectrogram expects
        audio_dict = {
            "audio": aud,
            "sr": sr,
            "path": audio_path,
            "source_type": "path",
        }

        result = SpectrogramUtils.load_spectrogram(
            audio_dict=audio_dict,
            new_sr=self.new_sr,
            new_ch=self.new_ch,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            n_mfcc=self.n_mfcc,
            target_duration=self.target_duration,
            cache_dir=self.cache_dir,
            save_cache=self.save_cache,
        )

        # Resize all feature maps to (n_mels, T) and stack into (4, n_mels, T)
        mel    = result["mel_spectrogram"]
        mfcc   = _resize_to_height(result["mfcc"],       self.n_mels)
        mfcc_d = _resize_to_height(result["mfcc_delta"], self.n_mels)
        chro   = _resize_to_height(result["chromagram"],  self.n_mels)

        feature = np.stack([mel, mfcc, mfcc_d, chro], axis=0)    # (4, n_mels, T)
        feature_tensor = torch.tensor(feature, dtype=torch.float32)

        return BatchItem(
            spectrogram=feature_tensor,
            label=torch.tensor(label, dtype=torch.long),
            emotion=emotion_name,
            file_path=str(audio_path),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _load_audio(path: Path) -> tuple[np.ndarray, float]:
        """Load an audio file and return (waveform, sample_rate)."""
        import librosa

        aud, sr = librosa.load(str(path), sr=None, mono=False)
        return aud, sr

    @property
    def num_classes(self) -> int:
        return len(EMOTION_MAP)

    @property
    def emotion_labels(self) -> dict[int, str]:
        """Return the 0-indexed label → emotion name mapping."""
        return {k - 1: v for k, v in EMOTION_MAP.items()}
