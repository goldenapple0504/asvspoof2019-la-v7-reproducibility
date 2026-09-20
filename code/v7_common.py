"""Shared deterministic geometry and I/O helpers for rebuild_v7."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SEED = 20260819
EPS = 1e-12
N_FFT = 512
FRAME_MS = 15
FRAME_TIMES_MS = np.arange(-20, 21, 5, dtype=np.int16)
FLUX_TIMES_MS = FRAME_TIMES_MS[1:]
BASE_HALF_SUPPORT_MS = 27.5
CPPS_LOCAL_HALF_SUPPORT_MS = 20.0
CPPS_TRAJECTORY_HALF_SUPPORT_MS = 40.0
REPRODUCTION_TOLERANCE = 1e-6
SYSTEMS = ("A01", "A02", "A03", "A04", "A07", "A08", "A09", "A10", "A11", "A12")


def atomic_tsv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, sep="\t", index=False)
    os.replace(tmp, path)


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def frame_length_samples(sample_rate: int) -> int:
    n = round(int(sample_rate) * FRAME_MS / 1000)
    if abs(n / int(sample_rate) - FRAME_MS / 1000) > 1e-12:
        raise ValueError("15ms_not_integer_samples")
    return int(n)


def v6_frame_indices(center_sec: float, relative_ms: int, sample_rate: int) -> tuple[int, int, float]:
    """Reproduce the official V6 Python-round frame geometry exactly."""
    n = frame_length_samples(sample_rate)
    requested_center_sample = float(center_sec) * sample_rate + int(relative_ms) * sample_rate / 1000
    start = round(requested_center_sample - n / 2)
    end = start + n
    realized_center_sample = (start + end) / 2
    return int(start), int(end), float(realized_center_sample - requested_center_sample)


def trajectory_support(
    phone_start_sec: float,
    phone_end_sec: float,
    midpoint_sec: float,
    sample_rate: int,
    audio_frames: int,
) -> dict[str, float | int | bool]:
    """Return exact short-frame support and strict phone-interior eligibility.

    Frame slices are the same half-open sample intervals used by V6.  The user
    specification forbids a frame from touching a phone boundary, so both
    continuous sample-edge margins must be strictly positive.
    """
    first_start, _, first_error = v6_frame_indices(midpoint_sec, int(FRAME_TIMES_MS[0]), sample_rate)
    _, last_end, last_error = v6_frame_indices(midpoint_sec, int(FRAME_TIMES_MS[-1]), sample_rate)
    phone_start_sample = float(phone_start_sec) * sample_rate
    phone_end_sample = float(phone_end_sec) * sample_rate
    left_margin = float(first_start - phone_start_sample)
    right_margin = float(phone_end_sample - last_end)
    audio_ok = first_start >= 0 and last_end <= int(audio_frames)
    strict_phone_ok = left_margin > 0.0 and right_margin > 0.0
    return {
        "midpoint_sample_unrounded": float(midpoint_sec * sample_rate),
        "midpoint_sample_round": int(round(midpoint_sec * sample_rate)),
        "first_required_sample": int(first_start),
        "last_required_sample_exclusive": int(last_end),
        "left_phone_margin_samples": left_margin,
        "right_phone_margin_samples": right_margin,
        "first_center_error_samples": first_error,
        "last_center_error_samples": last_error,
        "audio_support_ok": bool(audio_ok),
        "strict_phone_support_ok": bool(strict_phone_ok),
        "eligible": bool(audio_ok and strict_phone_ok),
    }


def mono_float64(audio: np.ndarray) -> np.ndarray:
    """Official V6 stereo-to-mono conversion on float64 samples."""
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if x.ndim != 1:
        raise ValueError(f"unexpected_audio_shape:{x.shape}")
    return x


def spectral_frame_features(frame: np.ndarray, sample_rate: int) -> tuple[np.ndarray, dict[str, float]]:
    x = np.asarray(frame, dtype=np.float64)
    magnitude = np.abs(np.fft.rfft(x * np.hanning(len(x)), n=N_FFT))
    power = magnitude**2
    frequency = np.fft.rfftfreq(N_FFT, d=1 / sample_rate)
    use = (frequency >= 300) & (frequency <= 4000)
    values = {
        "energy": float(20 * np.log10(np.sqrt(np.mean(x**2)) + EPS)),
        "centroid": float(np.dot(frequency, magnitude) / (magnitude.sum() + EPS)),
        "tilt": float(np.polyfit(frequency[use], 10 * np.log10(power[use] + EPS), 1)[0] * 1000),
        "flatness": float(np.exp(np.mean(np.log(power + EPS))) / np.mean(power + EPS)),
    }
    return magnitude, values


def spectral_flux(previous: np.ndarray, current: np.ndarray) -> float:
    a = previous / (np.linalg.norm(previous) + EPS)
    b = current / (np.linalg.norm(current) + EPS)
    return float(np.sqrt(np.sum((b - a) ** 2)))


def required_directories(root: Path) -> Iterable[Path]:
    for name in ("audit", "extracted_features", "matching", "models", "meta_analysis", "diagnostics", "figures", "scripts", "tests", "logs"):
        yield root / name

