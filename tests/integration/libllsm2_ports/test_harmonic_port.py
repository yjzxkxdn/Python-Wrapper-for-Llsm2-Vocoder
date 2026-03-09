from __future__ import annotations

import time

import numpy as np

import pyllsm2 as m


def _runtime_ratio_python(ampl: np.ndarray, phse: np.ndarray, nhar: int, nx: int, niter: int = 10) -> float:
    ampl = np.asarray(ampl[:nhar], dtype=np.float32)
    phse = np.asarray(phse[:nhar], dtype=np.float32)
    t0 = time.perf_counter()
    for _ in range(niter):
        m.synthesize_harmonic_frame(ampl, phse, 0.001, nx, use_iczt=False)
    t_gensins = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(niter):
        m.synthesize_harmonic_frame(ampl, phse, 0.001, nx, use_iczt=True)
    t_iczt = time.perf_counter() - t0
    return float(t_iczt / max(t_gensins, 1e-9))


def _runtime_ratio_raw(ampl: np.ndarray, phse: np.ndarray, nhar: int, nx: int, niter: int = 10) -> float:
    ffi = m.raw.ffi
    lib = m.raw.lib
    ampl = m.raw.as_f32_array(ampl[:nhar], "ampl")
    phse = m.raw.as_f32_array(phse[:nhar], "phse")
    t0 = time.perf_counter()
    for _ in range(niter):
        ptr = lib.llsm_synthesize_harmonic_frame(ffi.from_buffer("FP_TYPE[]", ampl), ffi.from_buffer("FP_TYPE[]", phse), nhar, 0.001, nx)
        m.raw.copy_fp_ptr(ptr, nx, free_after=True)
    t_gensins = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(niter):
        ptr = lib.llsm_synthesize_harmonic_frame_iczt(ffi.from_buffer("FP_TYPE[]", ampl), ffi.from_buffer("FP_TYPE[]", phse), nhar, 0.001, nx)
        m.raw.copy_fp_ptr(ptr, nx, free_after=True)
    t_iczt = time.perf_counter() - t0
    return float(t_iczt / max(t_gensins, 1e-9))


def test_harmonic_python_api() -> None:
    rng = np.random.default_rng(42)
    ampl = rng.normal(0, 1.0, 256).astype(np.float32)
    phse = rng.normal(0, 100.0, 256).astype(np.float32)
    y1 = m.synthesize_harmonic_frame(ampl, phse, 0.01, 1024, use_iczt=True)
    y2 = m.synthesize_harmonic_frame(ampl, phse, 0.01, 1024, use_iczt=False)
    err = y1 - y2
    snr = 20.0 * np.log10((np.std(err) + 1e-12) / (np.std(y2) + 1e-12))
    assert float(snr) < -10.0
    ratio = _runtime_ratio_python(ampl, phse, nhar=80, nx=512)
    assert np.isfinite(ratio)
    assert ratio > 0.0


def test_harmonic_raw_api() -> None:
    rng = np.random.default_rng(7)
    ampl = rng.normal(0, 1.0, 256).astype(np.float32)
    phse = rng.normal(0, 100.0, 256).astype(np.float32)
    ampl_arr = m.raw.as_f32_array(ampl, "ampl")
    phse_arr = m.raw.as_f32_array(phse, "phse")
    y1 = m.raw.copy_fp_ptr(
        m.raw.lib.llsm_synthesize_harmonic_frame_iczt(
            m.raw.ffi.from_buffer("FP_TYPE[]", ampl_arr),
            m.raw.ffi.from_buffer("FP_TYPE[]", phse_arr),
            ampl_arr.size,
            0.01,
            1024,
        ),
        1024,
        free_after=True,
    )
    y2 = m.raw.copy_fp_ptr(
        m.raw.lib.llsm_synthesize_harmonic_frame(
            m.raw.ffi.from_buffer("FP_TYPE[]", ampl_arr),
            m.raw.ffi.from_buffer("FP_TYPE[]", phse_arr),
            ampl_arr.size,
            0.01,
            1024,
        ),
        1024,
        free_after=True,
    )
    err = y1 - y2
    snr = 20.0 * np.log10((np.std(err) + 1e-12) / (np.std(y2) + 1e-12))
    assert float(snr) < -10.0
    ratio = _runtime_ratio_raw(ampl, phse, nhar=80, nx=512)
    assert np.isfinite(ratio)
    assert ratio > 0.0