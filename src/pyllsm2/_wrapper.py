from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np

from ._pyllsm2_cffi import ffi, lib

FP_DTYPE = np.float32


def as_f32_array(data: Iterable[float], name: str = "data") -> np.ndarray:
    arr = np.asarray(data, dtype=FP_DTYPE).reshape(-1)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D")
    return np.ascontiguousarray(arr)


def as_i32_array(data: Iterable[int], name: str = "data") -> np.ndarray:
    arr = np.asarray(data, dtype=np.int32).reshape(-1)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D")
    return np.ascontiguousarray(arr)


def as_char_array(value: str | bytes):
    if isinstance(value, bytes):
        raw = value
    else:
        raw = value.encode("ascii")
    return ffi.new("char[]", raw)


def free(ptr) -> None:
    if ptr != ffi.NULL:
        lib.llsm_py_free(ptr)


def copy_fp_ptr(ptr, n: int, free_after: bool = True) -> np.ndarray:
    if ptr == ffi.NULL:
        raise MemoryError("native call returned NULL pointer")
    out = np.frombuffer(ffi.buffer(ptr, int(n) * ffi.sizeof("FP_TYPE")), dtype=FP_DTYPE).copy()
    if free_after:
        lib.llsm_py_free(ptr)
    return out


def copy_int_ptr(ptr, n: int, free_after: bool = True) -> np.ndarray:
    if ptr == ffi.NULL:
        raise MemoryError("native call returned NULL pointer")
    out = np.frombuffer(ffi.buffer(ptr, int(n) * ffi.sizeof("int")), dtype=np.int32).copy()
    if free_after:
        lib.llsm_py_free(ptr)
    return out


def container_attach(dst, index: int, ptr, dtor=ffi.NULL, copyctor=ffi.NULL) -> None:
    lib.llsm_py_container_attach(dst, int(index), ptr, dtor, copyctor)


def output_to_numpy(out_ptr, free_after: bool = False):
    if out_ptr == ffi.NULL:
        raise MemoryError("llsm_output is NULL")
    ny = int(out_ptr.ny)
    ret = {
        "y": np.array([out_ptr.y[i] for i in range(ny)], dtype=FP_DTYPE),
        "y_sin": np.array([out_ptr.y_sin[i] for i in range(ny)], dtype=FP_DTYPE),
        "y_noise": np.array([out_ptr.y_noise[i] for i in range(ny)], dtype=FP_DTYPE),
        "ny": ny,
        "fs": float(out_ptr.fs),
    }
    if free_after:
        lib.llsm_delete_output(out_ptr)
    return ret


def analyze(options, x: Iterable[float], fs: float, f0: Iterable[float], return_xap: bool = False):
    x_arr = as_f32_array(x, "x")
    f0_arr = as_f32_array(f0, "f0")
    x_ptr = ffi.from_buffer("FP_TYPE[]", x_arr)
    f0_ptr = ffi.from_buffer("FP_TYPE[]", f0_arr)
    xap_ptr = ffi.new("FP_TYPE **") if return_xap else ffi.NULL
    chunk = lib.llsm_analyze(options, x_ptr, x_arr.size, float(fs), f0_ptr, f0_arr.size, xap_ptr)
    if chunk == ffi.NULL:
        raise RuntimeError("llsm_analyze returned NULL")
    if not return_xap:
        return chunk
    xap = copy_fp_ptr(xap_ptr[0], x_arr.size, free_after=True)
    return chunk, xap


def synthesize(options, chunk, return_numpy: bool = False):
    out = lib.llsm_synthesize(options, chunk)
    if out == ffi.NULL:
        raise RuntimeError("llsm_synthesize returned NULL")
    if return_numpy:
        return output_to_numpy(out, free_after=True)
    return out


def qifft(magn: Iterable[float], k: int) -> Tuple[float, float]:
    magn_arr = as_f32_array(magn, "magn")
    f = ffi.new("FP_TYPE *")
    v = lib.cig_qifft(ffi.from_buffer("FP_TYPE[]", magn_arr), int(k), f)
    return float(v), float(f[0])


def spec2env(S: Iterable[float], nfft: int, f0_ratio: float, nhar: int, Cout: Iterable[float] | None = None):
    s_arr = as_f32_array(S, "S")
    cout_ptr = ffi.NULL
    if Cout is not None:
        c_arr = as_f32_array(Cout, "Cout")
        cout_ptr = ffi.from_buffer("FP_TYPE[]", c_arr)
    out = lib.cig_spec2env(
        ffi.from_buffer("FP_TYPE[]", s_arr), int(nfft), float(f0_ratio), int(nhar), cout_ptr
    )
    return copy_fp_ptr(out, int(nfft) // 2 + 1, free_after=True)


def lfmodel_from_rd(rd: float, T0: float, Ee: float):
    return lib.cig_lfmodel_from_rd(float(rd), float(T0), float(Ee))


def lfmodel_spectrum(model, freq: Iterable[float], return_phase: bool = False):
    freq_arr = as_f32_array(freq, "freq")
    phase_ptr = ffi.new("FP_TYPE[]", freq_arr.size) if return_phase else ffi.NULL
    out = lib.cig_lfmodel_spectrum(model, ffi.from_buffer("FP_TYPE[]", freq_arr), freq_arr.size, phase_ptr)
    ampl = copy_fp_ptr(out, freq_arr.size, free_after=True)
    if not return_phase:
        return ampl
    phase = np.frombuffer(ffi.buffer(phase_ptr, freq_arr.size * ffi.sizeof("FP_TYPE")), dtype=FP_DTYPE).copy()
    return ampl, phase


def lfmodel_period(model, fs: int, n: int):
    out = lib.cig_lfmodel_period(model, int(fs), int(n))
    return copy_fp_ptr(out, int(n), free_after=True)


def ifdetector_estimate(ifd, x: Iterable[float]) -> float:
    x_arr = as_f32_array(x, "x")
    return float(lib.cig_ifdetector_estimate(ifd, ffi.from_buffer("FP_TYPE[]", x_arr), x_arr.size))

