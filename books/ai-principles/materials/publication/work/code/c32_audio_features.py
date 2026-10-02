"""Explicit 16-kHz waveform -> 25-ms / 10-ms-hop log-mel features."""

from __future__ import annotations

import functools

import numpy as np
from scipy.signal import stft

SAMPLE_RATE = 16000
WINDOW_SAMPLES = 400
HOP_SAMPLES = 160
FFT_SIZE = 512
MEL_BANDS = 80


def hertz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hertz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (np.power(10.0, mel / 2595.0) - 1.0)


@functools.lru_cache(maxsize=1)
def mel_filters() -> np.ndarray:
    """80 triangular bands, each normalized to sum one over FFT bins."""
    fft_hz = np.fft.rfftfreq(FFT_SIZE, 1 / SAMPLE_RATE)
    centers_mel = np.linspace(hertz_to_mel(np.array([0.0]))[0],
                              hertz_to_mel(np.array([SAMPLE_RATE / 2]))[0],
                              MEL_BANDS + 2)
    edges = mel_to_hertz(centers_mel)
    filters = np.zeros((MEL_BANDS, len(fft_hz)), dtype=np.float32)
    for i in range(MEL_BANDS):
        left, center, right = edges[i:i + 3]
        rise = (fft_hz - left) / (center - left)
        fall = (right - fft_hz) / (right - center)
        band = np.maximum(0.0, np.minimum(rise, fall))
        if not np.any(band > 0):
            raise RuntimeError(f"empty mel band {i}")
        filters[i] = band / band.sum()
    return filters


def complex_stft(wave: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if wave.ndim != 1:
        raise ValueError("mono waveform required")
    freqs, times, spectrum = stft(
        wave.astype(np.float32, copy=False), fs=SAMPLE_RATE,
        window="hann", nperseg=WINDOW_SAMPLES,
        noverlap=WINDOW_SAMPLES - HOP_SAMPLES, nfft=FFT_SIZE,
        boundary="zeros", padded=True,
    )
    return freqs, times, spectrum


def log_mel(wave: np.ndarray) -> np.ndarray:
    _, _, spectrum = complex_stft(wave)
    power = np.abs(spectrum).astype(np.float32) ** 2
    bands = mel_filters() @ power
    return np.log(np.maximum(bands, 1e-10)).T.astype(np.float32, copy=False)


def normalize_utterance(features: np.ndarray) -> np.ndarray:
    """Per-band, per-utterance centering/scaling; no other clips are seen."""
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    return ((features - mean) / np.maximum(std, 1e-4)).astype(np.float32)
