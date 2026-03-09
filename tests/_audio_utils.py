from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def tests_root() -> Path:
    return Path(__file__).resolve().parent


def audio_outputs_root() -> Path:
    return tests_root() / "audio_outputs"


def require_soundfile():
    return pytest.importorskip("soundfile")


def trim_or_pad(y: np.ndarray, target_len: int) -> np.ndarray:
    out = np.asarray(y, dtype=np.float32).reshape(-1)
    target_len = int(target_len)
    if out.size < target_len:
        out = np.pad(out, (0, target_len - out.size))
    return np.ascontiguousarray(out[:target_len], dtype=np.float32)


def dominant_frequency_hz(y: np.ndarray, fs: int) -> float:
    samples = np.asarray(y, dtype=np.float32).reshape(-1)
    if samples.size < 4:
        return 0.0
    window = np.hanning(samples.size).astype(np.float32)
    spec = np.abs(np.fft.rfft(samples * window))
    if spec.size <= 1:
        return 0.0
    freqs = np.fft.rfftfreq(samples.size, d=1.0 / float(fs))
    peak_idx = int(np.argmax(spec[1:]) + 1)
    return float(freqs[peak_idx])


def write_audio_output(category: str, name: str, y: np.ndarray, fs: int, *, subtype: str = "FLOAT") -> Path:
    sf = require_soundfile()
    out_dir = audio_outputs_root() / Path(category)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / name
    sf.write(str(out_path), np.asarray(y, dtype=np.float32), int(fs), subtype=subtype)
    return out_path
