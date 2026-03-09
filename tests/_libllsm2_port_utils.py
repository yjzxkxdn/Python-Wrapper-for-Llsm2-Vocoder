from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import pyllsm2 as m

from tests._audio_utils import write_audio_output


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[1]

def project_root() -> Path:
    return workspace_root().parent

def libllsm2_test_dir() -> Path:
    return project_root() / "libllsm2" / "test"


def require_soundfile():
    return pytest.importorskip("soundfile")


def require_parselmouth():
    return pytest.importorskip("parselmouth")


def read_test_wav(name: str, max_seconds: float = 2.0) -> tuple[np.ndarray, int]:
    sf = require_soundfile()
    wav_path = libllsm2_test_dir() / name
    x, fs = sf.read(str(wav_path), dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = np.mean(x, axis=1, dtype=np.float32)
    if max_seconds > 0:
        x = x[: int(max_seconds * fs)]
    return np.ascontiguousarray(x, dtype=np.float32), int(fs)


def write_test_output(name: str, y: np.ndarray, fs: int) -> Path:
    require_soundfile()
    return write_audio_output("integration/pyllsm2_ports", name, y, fs, subtype="PCM_16")


def extract_f0_parselmouth(
    x: np.ndarray,
    sampling_rate: int,
    hop_length: int,
    target_len: int,
    f0_min: float = 50.0,
) -> np.ndarray:
    parselmouth = require_parselmouth()
    l_pad = int(np.ceil(1.5 / f0_min * sampling_rate))
    r_pad = hop_length * ((len(x) - 1) // hop_length + 1) - len(x) + l_pad + 1
    s = parselmouth.Sound(np.pad(x, (l_pad, r_pad)), sampling_rate).to_pitch_ac(
        time_step=hop_length / sampling_rate,
        voicing_threshold=0.6,
        pitch_floor=f0_min,
        pitch_ceiling=1100,
    )
    f0 = np.asarray(s.selected_array["frequency"], dtype=np.float32)
    if f0.size < target_len:
        f0 = np.pad(f0, (0, target_len - f0.size))
    f0 = f0[:target_len]
    return np.ascontiguousarray(f0, dtype=np.float32)


def empirical_kld(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    nx = x.size
    ny = y.size
    if nx < 3 or ny < 3:
        return 0.0
    xs = np.sort(x)
    ys = np.sort(y)
    ys[0] = xs[0]
    ys[-1] = xs[-1]
    yi = 1
    dsum = 0.0
    for i in range(1, nx):
        while yi < ny - 1 and ys[yi] < xs[i]:
            yi += 1
        d_x = max(xs[i] - xs[i - 1], 1e-10)
        d_y = max(ys[yi] - ys[max(0, yi - 1)], 1e-10)
        dsum += np.log((ny * d_y) / (nx * d_x))
    return float(dsum / max(1, nx - 1) - 1.0)


def _stft_mag(x: np.ndarray, nfft: int = 2048, hop: int = 512) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size < nfft:
        x = np.pad(x, (0, nfft - x.size))
    nfrm = 1 + (x.size - nfft) // hop
    win = np.hanning(nfft).astype(np.float32)
    out = np.empty((nfrm, nfft // 2 + 1), dtype=np.float32)
    for i in range(nfrm):
        fr = x[i * hop : i * hop + nfft] * win
        out[i] = np.abs(np.fft.rfft(fr)).astype(np.float32)
    return out


def verify_data_distribution(x: np.ndarray, y: np.ndarray) -> None:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    n = min(x.size, y.size)
    x = x[:n]
    y = y[:n]
    assert np.isfinite(x).all() and np.isfinite(y).all()
    std_x = float(np.std(x)) + 1e-12
    std_y = float(np.std(y)) + 1e-12
    ratio = std_y / std_x
    assert 0.01 < ratio < 100.0
    cc = float(np.corrcoef(x, y)[0, 1])
    assert cc > -0.3


def verify_spectral_distribution(x: np.ndarray, y: np.ndarray) -> None:
    sx = _stft_mag(x)
    sy = _stft_mag(y)
    nfrm = min(sx.shape[0], sy.shape[0])
    sx = sx[:nfrm]
    sy = sy[:nfrm]
    c = np.corrcoef(sx.reshape(-1), sy.reshape(-1))[0, 1]
    assert float(c) > 0.55
    kld = empirical_kld(
        sx.reshape(-1) + np.random.randn(sx.size) * 1e-6,
        sy.reshape(-1) + np.random.randn(sy.size) * 1e-6,
    )
    assert kld < 0.9


def create_analysis_options(fs: int, nhop: int = 128) -> m.AnalysisOptions:
    opt_a = m.AnalysisOptions()
    opt_a.thop = nhop / float(fs)
    opt_a.npsd = 128
    opt_a.maxnhar = 120
    opt_a.maxnhar_e = 5
    return opt_a


def create_synthesis_options(fs: int) -> m.SynthesisOptions:
    return m.SynthesisOptions(fs)


def output_dict(out: m.Output) -> dict[str, np.ndarray]:
    return {"y": out.y, "y_sin": out.y_sin, "y_noise": out.y_noise}


def raw_output_dict(raw, out_ptr) -> dict[str, np.ndarray]:
    ffi = raw.ffi
    ny = int(out_ptr.ny)
    return {
        "y": np.frombuffer(ffi.buffer(out_ptr.y, ny * ffi.sizeof("FP_TYPE")), dtype=np.float32).copy(),
        "y_sin": np.frombuffer(ffi.buffer(out_ptr.y_sin, ny * ffi.sizeof("FP_TYPE")), dtype=np.float32).copy(),
        "y_noise": np.frombuffer(ffi.buffer(out_ptr.y_noise, ny * ffi.sizeof("FP_TYPE")), dtype=np.float32).copy(),
    }


def raw_create_analysis_options(raw, fs: int, nhop: int = 128, hm_method: int | None = None):
    opt = raw.lib.llsm_create_aoptions()
    if opt == raw.ffi.NULL:
        raise MemoryError("llsm_create_aoptions returned NULL")
    opt.thop = nhop / float(fs)
    opt.npsd = 128
    opt.maxnhar = 120
    opt.maxnhar_e = 5
    if hm_method is not None:
        opt.hm_method = int(hm_method)
    return opt


def raw_create_synthesis_options(raw, fs: int, use_l1: bool = False):
    opt = raw.lib.llsm_create_soptions(float(fs))
    if opt == raw.ffi.NULL:
        raise MemoryError("llsm_create_soptions returned NULL")
    opt.use_l1 = 1 if use_l1 else 0
    return opt


def raw_analyze_chunk(raw, opt_a, x: np.ndarray, fs: int, f0: np.ndarray):
    x_arr = raw.as_f32_array(x, "x")
    f0_arr = raw.as_f32_array(f0, "f0")
    chunk = raw.lib.llsm_analyze(
        opt_a,
        raw.ffi.from_buffer("FP_TYPE[]", x_arr),
        x_arr.size,
        float(fs),
        raw.ffi.from_buffer("FP_TYPE[]", f0_arr),
        f0_arr.size,
        raw.ffi.NULL,
    )
    if chunk == raw.ffi.NULL:
        raise RuntimeError("llsm_analyze returned NULL")
    return chunk


def raw_synthesize_output(raw, opt_s, chunk_ptr) -> dict[str, np.ndarray]:
    out = raw.lib.llsm_synthesize(opt_s, chunk_ptr)
    if out == raw.ffi.NULL:
        raise RuntimeError("llsm_synthesize returned NULL")
    try:
        return raw_output_dict(raw, out)
    finally:
        raw.lib.llsm_delete_output(out)
