from __future__ import annotations

import numpy as np

import pyllsm2 as m

from tests._libllsm2_port_utils import (
    create_analysis_options,
    create_synthesis_options,
    extract_f0_parselmouth,
    raw_analyze_chunk,
    raw_create_analysis_options,
    raw_create_synthesis_options,
    raw_synthesize_output,
    read_test_wav,
    verify_data_distribution,
    verify_spectral_distribution,
    write_test_output,
)

ffi = m.raw.ffi
lib = m.raw.lib
LLSM_FRAME_GROWLSTRENGTH = 15


def approx(a: float, b: float, eps: float = 1e-3) -> bool:
    return abs(a - b) <= eps * max(1.0, abs(a), abs(b))


def speech_case(name: str = "arctic_a0001.wav", max_seconds: float = 1.2, nhop: int = 128):
    x, fs = read_test_wav(name, max_seconds=max_seconds)
    nfrm = max(1, x.size // nhop)
    f0 = extract_f0_parselmouth(x, fs, nhop, nfrm)
    return x, fs, nhop, nfrm, f0


def python_analysis_pair(
    *,
    wav_name: str = "arctic_a0001.wav",
    max_seconds: float = 1.2,
    nhop: int = 128,
    hm_method: int | None = None,
):
    x, fs, nhop, nfrm, f0 = speech_case(wav_name, max_seconds=max_seconds, nhop=nhop)
    opt_a = create_analysis_options(fs, nhop=nhop)
    if hm_method is not None:
        opt_a.hm_method = int(hm_method)
    opt_s = create_synthesis_options(fs)
    return x, fs, nhop, nfrm, f0, opt_a, opt_s


def raw_analysis_pair(
    *,
    wav_name: str = "arctic_a0001.wav",
    max_seconds: float = 1.2,
    nhop: int = 128,
    hm_method: int | None = None,
    use_l1: bool = False,
):
    x, fs, nhop, nfrm, f0 = speech_case(wav_name, max_seconds=max_seconds, nhop=nhop)
    opt_a = m.AnalysisOptions(ptr=raw_create_analysis_options(m.raw, fs, nhop=nhop, hm_method=hm_method))
    opt_s = m.SynthesisOptions(ptr=raw_create_synthesis_options(m.raw, fs, use_l1=use_l1))
    return x, fs, nhop, nfrm, f0, opt_a, opt_s


def raw_chunk_to_layer1(chunk: m.Chunk, nfft: int) -> m.Layer1Features:
    ptr = lib.llsm_copy_chunk(chunk.ptr)
    if ptr == ffi.NULL:
        raise MemoryError("llsm_copy_chunk returned NULL")
    lib.llsm_chunk_tolayer1(ptr, int(nfft))
    return m.Layer1Features(ptr)


def raw_chunk_to_layer0(chunk: m.Chunk) -> m.Layer0Features:
    ptr = lib.llsm_copy_chunk(chunk.ptr)
    if ptr == ffi.NULL:
        raise MemoryError("llsm_copy_chunk returned NULL")
    lib.llsm_chunk_tolayer0(ptr)
    return m.Layer0Features(ptr)


def raw_output(opt_s: m.SynthesisOptions, chunk: m.Chunk | m.Layer0Features | m.Layer1Features) -> dict[str, np.ndarray]:
    ptr = chunk.ptr if hasattr(chunk, "ptr") else chunk
    return raw_synthesize_output(m.raw, opt_s.ptr, ptr)


def make_spectrum(size: int, use_db: bool, no_modulation: bool) -> np.ndarray:
    idx = np.arange(size, dtype=np.float32)
    spec = np.exp(-(idx / size) * 10.0)
    if not no_modulation:
        spec *= np.exp(np.cos(idx * 0.1) * 3.0 - 3.0)
    if use_db:
        spec = 20.0 * np.log10(np.maximum(spec, 1e-12))
    return spec.astype(np.float32)


def synthetic_harmonic_case():
    nx = 24000
    fs = 20000.0
    f0_start = 100.0
    f0_end = 200.0
    thop = 0.005
    nfrm = int(np.floor(nx / fs / thop))
    idx = np.arange(nx, dtype=np.float32)
    rate = idx / float(nx)
    f0_inst = f0_start + (f0_end - f0_start) * rate
    ampl0 = rate
    phase = np.cumsum(f0_inst / fs * 2.0 * np.pi, dtype=np.float32)
    x = (ampl0 * np.sin(phase) + 0.5 * np.sin(2.0 * phase) + 0.25 * np.sin(3.0 * phase)).astype(np.float32)
    centers = np.rint(np.arange(nfrm, dtype=np.float32) * thop * fs).astype(np.int32)
    rate_f = centers / float(nx)
    f0 = (f0_start + (f0_end - f0_start) * rate_f).astype(np.float32)
    ampl0_truth = rate_f.astype(np.float32)
    return x, fs, f0, thop, nfrm, ampl0_truth


def verify_harmonic_analysis_result(h: dict[str, np.ndarray], f0: np.ndarray, thop: float, ampl0_truth: np.ndarray) -> None:
    nfrm = int(f0.size)
    lo, hi = 5, max(6, nfrm - 5)
    a0 = h["ampl"][:, 0]
    a1 = h["ampl"][:, 1]
    a2 = h["ampl"][:, 2]
    assert float(np.abs(np.mean(a0[lo:hi] - ampl0_truth[lo:hi]))) < 0.03
    assert float(np.std(a0[lo:hi] - ampl0_truth[lo:hi])) < 0.03
    assert float(np.abs(np.mean(a1[lo:hi] - 0.5))) < 0.03
    assert float(np.std(a1[lo:hi] - 0.5)) < 0.03
    assert float(np.abs(np.mean(a2[lo:hi] - 0.25))) < 0.03
    assert float(np.std(a2[lo:hi] - 0.25)) < 0.03
    phase_err = []
    for idx in range(lo + 1, hi):
        inc = f0[idx] * 2.0 * np.pi * thop
        prev_p = float(h["phse"][idx - 1, 0]) + inc
        curr_p = float(h["phse"][idx, 0])
        delta = (prev_p - curr_p + np.pi) % (2.0 * np.pi) - np.pi
        phase_err.append(delta)
    phase_err = np.asarray(phase_err, dtype=np.float32)
    assert float(np.abs(np.mean(phase_err))) < 0.2
    assert float(np.std(phase_err)) < 0.2


def raw_harmonic_analysis_matrix(x: np.ndarray, fs: float, f0: np.ndarray, thop: float, method: int) -> dict[str, np.ndarray]:
    max_nhar = 3
    ampl = np.full((f0.size, max_nhar), 0.0, dtype=np.float32)
    phse = np.full((f0.size, max_nhar), 0.0, dtype=np.float32)
    nhar = np.zeros(f0.size, dtype=np.int32)
    x_arr = m.raw.as_f32_array(x, "x")
    f0_arr = m.raw.as_f32_array(f0, "f0")
    rc = int(
        lib.llsm_py_harmonic_analysis_matrix(
            ffi.from_buffer("FP_TYPE[]", x_arr),
            x_arr.size,
            float(fs),
            ffi.from_buffer("FP_TYPE[]", f0_arr),
            f0_arr.size,
            float(thop),
            4.0,
            max_nhar,
            int(method),
            0.0,
            ffi.from_buffer("FP_TYPE[]", ampl.reshape(-1)),
            ffi.from_buffer("FP_TYPE[]", phse.reshape(-1)),
            ffi.from_buffer("int[]", nhar),
            max_nhar,
        )
    )
    if rc != 0:
        raise RuntimeError(f"llsm_py_harmonic_analysis_matrix failed with code {rc}")
    return {"ampl": ampl, "phse": phse, "nhar": nhar}


__all__ = [
    "LLSM_FRAME_GROWLSTRENGTH",
    "approx",
    "ffi",
    "lib",
    "make_spectrum",
    "python_analysis_pair",
    "raw_analysis_pair",
    "raw_chunk_to_layer0",
    "raw_chunk_to_layer1",
    "raw_harmonic_analysis_matrix",
    "raw_output",
    "speech_case",
    "synthetic_harmonic_case",
    "verify_data_distribution",
    "verify_harmonic_analysis_result",
    "verify_spectral_distribution",
    "write_test_output",
]