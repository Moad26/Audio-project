from pathlib import Path
from typing import Optional, Tuple

import librosa
import numpy as np
from tqdm import tqdm


class SpectrogramUtils:
    @staticmethod
    def resample(
        aud_sr: Tuple[np.ndarray, float],
        new_sr: int = 22050,
    ) -> Tuple[np.ndarray, float]:
        aud, sr = aud_sr
        if sr == new_sr:
            return aud, sr
        if aud.ndim == 1:
            res_aud = librosa.resample(aud, orig_sr=sr, target_sr=new_sr)
        else:
            res_aud = np.stack(
                [
                    librosa.resample(channel, orig_sr=sr, target_sr=new_sr)
                    for channel in aud
                ]
            )
        return res_aud, new_sr

    @staticmethod
    def rechannel(
        aud_sr: Tuple[np.ndarray, float],
        new_ch: int = 1,
    ) -> Tuple[np.ndarray, float]:
        aud, sr = aud_sr
        n_ch = 1 if aud.ndim == 1 else aud.shape[0]
        if n_ch == new_ch:
            return aud_sr
        if new_ch == 1:
            res_aud = np.mean(aud, axis=0, keepdims=True)
            return res_aud, sr
        if new_ch == 2:
            res_aud = np.stack([aud, aud])
            return res_aud, sr
        raise ValueError(f"Unsupported number of channels: {new_ch}")

    @staticmethod
    def pad_or_truncate(
        aud_sr: Tuple[np.ndarray, float],
        target_duration: float = 30.0,
    ) -> Tuple[np.ndarray, float]:
        aud, sr = aud_sr
        target_samples = int(sr * target_duration)
        current_samples = aud.shape[-1] if aud.ndim > 1 else len(aud)

        if current_samples > target_samples:
            return (
                aud[:, :target_samples] if aud.ndim > 1 else aud[:target_samples]
            ), sr

        if current_samples < target_samples:
            pad_samples = target_samples - current_samples
            pad_width = ((0, 0), (0, pad_samples)) if aud.ndim > 1 else (0, pad_samples)
            return np.pad(aud, pad_width, mode="constant", constant_values=0), sr

        return aud, sr

    @staticmethod
    def get_cache_path(
        audio_path: Path,
        cache_dir: Path,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 40,
        n_mfcc: int = 13,
        new_sr: int = 22050,
        new_ch: int = 1,
        target_duration: float = 30.0,
    ) -> Path:
        """Derive a deterministic cache path based on file name and extraction params."""
        params_hash = (
            f"{n_fft}_{hop_length}_{n_mels}_{n_mfcc}_{new_sr}_{new_ch}_{target_duration}"
        )
        return cache_dir / f"{audio_path.stem}_{params_hash}.npz"

    @staticmethod
    def load_spectrogram(
        audio_dict: dict,
        new_sr: int = 22050,
        new_ch: int = 1,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 40,
        n_mfcc: int = 13,
        target_duration: float = 30.0,
        cache_dir: Optional[Path] = None,
        save_cache: bool = False,
    ) -> dict:
        """
        Compute mel spectrogram, STFT, MFCCs, delta-MFCCs, and chromagram
        from a single audio dict.

        Args:
            audio_dict:      Dict with keys: audio, sr, path, source_type.
            new_sr:          Target sample rate.
            new_ch:          Target number of channels.
            n_fft:           FFT window size.
            hop_length:      Hop length for STFT / mel spectrogram.
            n_mels:          Number of mel filter banks.
            n_mfcc:          Number of MFCC coefficients (default 13).
            target_duration: Clip duration in seconds (pad or truncate).
            cache_dir:       Directory for .npz cache files; required when save_cache=True.
            save_cache:      Whether to read from / write to disk cache.

        Returns:
            Dict with keys:
              mel_spectrogram, stft_spectrogram, mfcc, mfcc_delta,
              chromagram, audio, sr, path.
        """
        audio = audio_dict["audio"]
        sr = audio_dict["sr"]
        audio_path = audio_dict["path"]
        source_type = audio_dict["source_type"]

        # --- Cache read (real file paths only) ---
        if save_cache and cache_dir and source_type == "path":
            cache_path = SpectrogramUtils.get_cache_path(
                audio_path=audio_path,
                cache_dir=cache_dir,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
                n_mfcc=n_mfcc,
                new_sr=new_sr,
                new_ch=new_ch,
                target_duration=target_duration,
            )
            if cache_path.exists():
                try:
                    cached = np.load(cache_path)
                    return {
                        "mel_spectrogram": cached["mel_spectrogram"],
                        "stft_spectrogram": cached["stft_spectrogram"],
                        "mfcc": cached["mfcc"],
                        "mfcc_delta": cached["mfcc_delta"],
                        "chromagram": cached["chromagram"],
                        "audio": cached["audio"],
                        "sr": float(cached["sr"]),
                        "path": audio_path,
                    }
                except Exception:
                    pass  # Corrupted cache — fall through and recompute

        try:
            aud_sr = (audio, sr)
            aud_sr = SpectrogramUtils.pad_or_truncate(
                aud_sr, target_duration=target_duration
            )
            aud_sr = SpectrogramUtils.rechannel(aud_sr, new_ch=new_ch)
            aud_sr = SpectrogramUtils.resample(aud_sr, new_sr=new_sr)
            aud, sr = aud_sr

            # Flatten to 1-D for librosa feature functions
            aud_mono = aud[0] if aud.ndim > 1 else aud

            stft_mag = np.abs(librosa.stft(aud_mono, n_fft=n_fft, hop_length=hop_length))

            mel_spect_db = librosa.power_to_db(
                librosa.feature.melspectrogram(
                    y=aud_mono, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
                ),
                ref=np.max,
            )

            # 13 MFCCs + delta MFCCs
            mfcc = librosa.feature.mfcc(
                y=aud_mono, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length
            )  # (n_mfcc, T)
            mfcc_delta = librosa.feature.delta(mfcc)   # (n_mfcc, T)

            # Chromagram (12 chroma bins)
            chromagram = librosa.feature.chroma_stft(
                y=aud_mono, sr=sr, n_fft=n_fft, hop_length=hop_length
            )  # (12, T)

        except Exception as e:
            raise ValueError(f"Error processing audio from {audio_path}: {e}") from e

        # --- Cache write (real file paths only) ---
        if save_cache and cache_dir and source_type == "path":
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = SpectrogramUtils.get_cache_path(
                audio_path=audio_path,
                cache_dir=cache_dir,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
                n_mfcc=n_mfcc,
                new_sr=new_sr,
                new_ch=new_ch,
                target_duration=target_duration,
            )
            np.savez_compressed(
                cache_path,
                mel_spectrogram=mel_spect_db,
                stft_spectrogram=stft_mag,
                mfcc=mfcc,
                mfcc_delta=mfcc_delta,
                chromagram=chromagram,
                audio=aud,
                sr=sr,
            )

        return {
            "mel_spectrogram": mel_spect_db,
            "stft_spectrogram": stft_mag,
            "mfcc": mfcc,
            "mfcc_delta": mfcc_delta,
            "chromagram": chromagram,
            "audio": aud,
            "sr": sr,
            "path": audio_path,
        }

    @staticmethod
    def batch_load_spectrograms(
        audio_dicts: list,
        new_sr: int = 22050,
        new_ch: int = 1,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 40,
        n_mfcc: int = 13,
        target_duration: float = 30.0,
        cache_dir: Optional[Path] = None,
        save_cache: bool = False,
    ) -> np.ndarray:
        """
        Process a list of audio dicts and return an object array of result dicts.
        Mirrors the original transform() behaviour, but as a plain static method.
        """
        results = []
        for audio_dict in tqdm(audio_dicts, desc="Computing spectrograms"):
            result = SpectrogramUtils.load_spectrogram(
                audio_dict=audio_dict,
                new_sr=new_sr,
                new_ch=new_ch,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
                n_mfcc=n_mfcc,
                target_duration=target_duration,
                cache_dir=cache_dir,
                save_cache=save_cache,
            )
            results.append(result)
        return np.array(results, dtype=object)
