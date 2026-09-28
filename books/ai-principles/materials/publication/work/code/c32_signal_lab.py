"""Use one real speech waveform to inspect sampling, phase and short-time spectra."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import istft, resample_poly
import soundfile as sf

from c32_audio_features import (FFT_SIZE, HOP_SAMPLES, SAMPLE_RATE,
                                WINDOW_SAMPLES, complex_stft, log_mel)


ROOT = Path(__file__).resolve().parents[2]
SPLIT = ROOT / "work/data/librispeech_dev_clean/chapter_split.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    if out.exists():
        raise FileExistsError(f"preserve existing signal experiment: {out}")
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    selected = next(row for row in split["rows"] if
                    row["split"] == "train" and 4 <= row["duration_seconds"] <= 6)
    wave, rate = sf.read(ROOT / selected["flac"], dtype="float32")
    if rate != SAMPLE_RATE or wave.ndim != 1:
        raise ValueError("expected 16-kHz mono speech")
    freqs, times, spectrum = complex_stft(wave)
    _, original_phase_inverse = istft(
        spectrum, fs=SAMPLE_RATE, window="hann",
        nperseg=WINDOW_SAMPLES, noverlap=WINDOW_SAMPLES - HOP_SAMPLES,
        nfft=FFT_SIZE, input_onesided=True, boundary=True)
    reconstructed = original_phase_inverse[:len(wave)]
    if len(reconstructed) != len(wave):
        raise ValueError("incomplete inverse short-time transform")
    _, zero_phase_inverse = istft(
        np.abs(spectrum).astype(np.complex64), fs=SAMPLE_RATE,
        window="hann", nperseg=WINDOW_SAMPLES,
        noverlap=WINDOW_SAMPLES - HOP_SAMPLES,
        nfft=FFT_SIZE, input_onesided=True, boundary=True)
    zero_phase = zero_phase_inverse[:len(wave)]
    phase_gain = float(np.sqrt(np.mean(wave ** 2)) /
                       max(np.sqrt(np.mean(zero_phase ** 2)), 1e-12))
    zero_phase_audible = np.clip(zero_phase * phase_gain, -1, 1)

    # A known 5.4-kHz tone is added to real speech only for the alias check.
    # Subtract the clean-speech downsampled result to isolate that added tone.
    tone = (0.1 * np.sin(2 * np.pi * 5400 *
                         np.arange(len(wave)) / SAMPLE_RATE)).astype(np.float32)
    naive_added = (wave + tone)[::4] - wave[::4]
    filtered_added = resample_poly(wave + tone, 1, 4) - resample_poly(wave, 1, 4)
    original_low_rate = resample_poly(wave, 1, 4)
    naive_low_rate = wave[::4]
    sample_count = min(4000, len(naive_added))
    f4k = np.fft.rfftfreq(sample_count, 1 / 4000)
    win = np.hanning(sample_count)
    naive_fft = np.abs(np.fft.rfft(naive_added[:sample_count] * win))
    filtered_fft = np.abs(np.fft.rfft(filtered_added[:sample_count] * win))
    alias_bin = int(np.argmin(np.abs(f4k - 1400)))

    out.mkdir(parents=True)
    sf.write(out / "original_speech.wav", wave, SAMPLE_RATE)
    sf.write(out / "complex_spectrum_reconstruction.wav", reconstructed,
             SAMPLE_RATE)
    sf.write(out / "magnitude_with_zero_phase.wav", zero_phase_audible,
             SAMPLE_RATE)
    sf.write(out / "proper_4khz_speech.wav", original_low_rate, 4000)
    sf.write(out / "naive_4khz_speech.wav", naive_low_rate, 4000)

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), layout="constrained")
    segment_samples = min(int(0.14 * SAMPLE_RATE), len(wave))
    candidates = range(0, len(wave) - segment_samples + 1, HOP_SAMPLES)
    begin = max(candidates, key=lambda i: float(np.mean(wave[i:i + segment_samples] ** 2)))
    end = begin + segment_samples
    axes[0].plot(np.arange(begin, end) / SAMPLE_RATE, wave[begin:end], color="#255797",
                 linewidth=0.85)
    axes[0].set(xlabel="时间（秒）", ylabel="空气压力的离散记录",
                title="真实朗读录音中有声的一小段：相邻采样相隔 1/16000 秒")
    db = 20 * np.log10(np.maximum(np.abs(spectrum), 1e-6))
    axes[1].pcolormesh(times, freqs, db, shading="auto", cmap="magma",
                       vmin=-80, vmax=-15)
    axes[1].set(ylim=(0, 4000), xlabel="时间（秒）", ylabel="频率（Hz）",
                title="25 毫秒窗、每 10 毫秒移动一次的短时幅度谱")
    axes[2].plot(f4k, naive_fft, color="#b9533d", label="不经低通直接每四点取一项")
    axes[2].plot(f4k, filtered_fft, color="#277c67", label="先低通再降到 4 kHz")
    axes[2].axvline(1400, linestyle="--", color="#444444", linewidth=0.9)
    axes[2].set(xlim=(900, 1800), xlabel="降采样后看到的频率（Hz）",
                ylabel="已知加进的 5.4 kHz 分量幅度",
                title="同一真实语音另加已知高音：直接抽样会在 1.4 kHz 出现假低音")
    axes[2].legend(loc="upper right")
    fig.savefig(out / "waveform_spectrum_alias.png", dpi=160)
    plt.close(fig)
    report = {
        "clip_id": selected["id"],
        "source_flac": selected["flac"],
        "split": selected["split"],
        "source_archive_sha256": split["archive_sha256"],
        "chapter_split_sha256": sha256(SPLIT),
        "sample_rate": SAMPLE_RATE,
        "duration_seconds": len(wave) / SAMPLE_RATE,
        "samples": len(wave),
        "displayed_waveform_start_seconds": begin / SAMPLE_RATE,
        "window_samples": WINDOW_SAMPLES,
        "hop_samples": HOP_SAMPLES,
        "fft_size": FFT_SIZE,
        "complex_spectrum_shape": list(spectrum.shape),
        "log_mel_shape": list(log_mel(wave).shape),
        "complex_reconstruction_max_abs_error": float(np.max(np.abs(wave - reconstructed))),
        "complex_reconstruction_rmse": float(np.sqrt(np.mean((wave - reconstructed) ** 2))),
        "zero_phase_gain_for_playback": phase_gain,
        "zero_phase_playback_rmse_to_original": float(np.sqrt(np.mean((wave - zero_phase_audible) ** 2))),
        "injected_tone_hz": 5400,
        "downsampled_rate": 4000,
        "expected_alias_hz": 1400,
        "isolated_alias_fft_amplitude_no_filter": float(naive_fft[alias_bin]),
        "isolated_alias_fft_amplitude_with_filter": float(filtered_fft[alias_bin]),
        "code_sha256": sha256(Path(__file__)),
    }
    (out / "receipt.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                      encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
