"""Python-first wrappers around the native :mod:`libllsm2` API.

This module owns the high-level objects used by :mod:`pyllsm2`. It keeps the
native pointers and memory management details hidden behind NumPy-friendly
classes and functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple

import numpy as np
from numpy.typing import NDArray

from ._pyllsm2_cffi import ffi, lib

FP_DTYPE = np.float32
FloatArray1D = NDArray[np.float32]
FloatArray2D = NDArray[np.float32]
IntArray1D = NDArray[np.int32]
_LLSM_FRAME_F0 = 0
_LLSM_FRAME_HM = 1
_LLSM_FRAME_NM = 2
_LLSM_FRAME_PBPSYN = 9
_LLSM_FRAME_RD = 10
_LLSM_FRAME_VTMAGN = 11
_LLSM_FRAME_VSPHSE = 12
_LLSM_CONF_NFRM = 0
_LLSM_CONF_MAXNHAR = 2
_LLSM_CONF_MAXNHAR_E = 3
_LLSM_CONF_NPSD = 4
_LLSM_CONF_NCHANNEL = 7
_LLSM_CONF_NSPEC = 10


@dataclass(frozen=True)
class HarmonicsView:
    """Dense layer-0 harmonic matrices.

    Attributes:
        ampl: ``float32`` array with shape ``(nfrm, max_nhar)``.
            Axis 0 is frame index over time.
            Axis 1 is 0-based harmonic index within each frame.
        phse: ``float32`` array with shape ``(nfrm, max_nhar)``.
            Axis meanings match ``ampl``.
        nhar: ``int32`` array with shape ``(nfrm,)``.
            Axis 0 is frame index; each value is the valid harmonic count for
            the corresponding frame.
    """

    ampl: FloatArray2D
    phse: FloatArray2D
    nhar: IntArray1D


def as_f32_array(data: Iterable[float], name: str = "data") -> FloatArray1D:
    """Convert ``data`` to a contiguous ``float32`` vector.

    Returns an array with shape ``(n,)`` where axis 0 is the only sample/frame
    axis for the requested input.
    """

    arr = np.asarray(data, dtype=FP_DTYPE)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D")
    return np.ascontiguousarray(arr)


def as_i32_array(data: Iterable[int], name: str = "data") -> IntArray1D:
    """Convert ``data`` to a contiguous ``int32`` vector.

    Returns an array with shape ``(n,)`` where axis 0 is the only sample/frame
    axis for the requested input.
    """

    arr = np.asarray(data, dtype=np.int32)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D")
    return np.ascontiguousarray(arr)


def as_char_array(value: str | bytes):
    """Allocate a native ``char[]`` buffer from ASCII text or raw bytes."""

    if isinstance(value, bytes):
        raw = value
    else:
        raw = value.encode("ascii")
    return ffi.new("char[]", raw)


def free(ptr) -> None:
    """Free memory that was allocated by the Python compatibility layer."""

    if ptr != ffi.NULL:
        lib.llsm_py_free(ptr)


def copy_fp_ptr(ptr, n: int, free_after: bool = True) -> np.ndarray:
    """Copy a native ``FP_TYPE*`` buffer into a standalone NumPy array."""

    if ptr == ffi.NULL:
        raise MemoryError("native call returned NULL pointer")
    out = np.frombuffer(ffi.buffer(ptr, int(n) * ffi.sizeof("FP_TYPE")), dtype=FP_DTYPE).copy()
    if free_after:
        lib.llsm_py_free(ptr)
    return out


def copy_int_ptr(ptr, n: int, free_after: bool = True) -> np.ndarray:
    """Copy a native ``int*`` buffer into a standalone NumPy array."""

    if ptr == ffi.NULL:
        raise MemoryError("native call returned NULL pointer")
    out = np.frombuffer(ffi.buffer(ptr, int(n) * ffi.sizeof("int")), dtype=np.int32).copy()
    if free_after:
        lib.llsm_py_free(ptr)
    return out


def container_attach(dst, index: int, ptr, dtor=ffi.NULL, copyctor=ffi.NULL) -> None:
    """Attach a raw pointer to an ``llsm_container`` member slot."""

    lib.llsm_py_container_attach(dst, int(index), ptr, dtor, copyctor)


def output_to_numpy(out_ptr, free_after: bool = False):
    """Convert a native ``llsm_output*`` into the default waveform array."""

    if out_ptr == ffi.NULL:
        raise MemoryError("llsm_output is NULL")
    ny = int(out_ptr.ny)
    ret = np.frombuffer(ffi.buffer(out_ptr.y, ny * ffi.sizeof("FP_TYPE")), dtype=FP_DTYPE).copy()
    if free_after:
        lib.llsm_delete_output(out_ptr)
    return ret


def _f32_view_1d(ptr, n: int) -> FloatArray1D:
    if ptr == ffi.NULL or n <= 0:
        return np.empty((max(int(n), 0),), dtype=FP_DTYPE)
    return np.frombuffer(ffi.buffer(ptr, int(n) * ffi.sizeof("FP_TYPE")), dtype=FP_DTYPE)


def _i32_view_1d(ptr, n: int) -> IntArray1D:
    if ptr == ffi.NULL or n <= 0:
        return np.empty((max(int(n), 0),), dtype=np.int32)
    return np.frombuffer(ffi.buffer(ptr, int(n) * ffi.sizeof("int")), dtype=np.int32)


def _f32_view_2d(ptr, rows: int, cols: int) -> FloatArray2D:
    if ptr == ffi.NULL or rows <= 0 or cols <= 0:
        return np.empty((max(int(rows), 0), max(int(cols), 0)), dtype=FP_DTYPE)
    return np.frombuffer(
        ffi.buffer(ptr, int(rows) * int(cols) * ffi.sizeof("FP_TYPE")),
        dtype=FP_DTYPE,
    ).reshape(int(rows), int(cols))


def _f32_strided_block_view(block_ptr, rows: int, cols: int) -> FloatArray2D:
    if block_ptr == ffi.NULL or rows <= 0 or cols <= 0:
        return np.empty((max(int(rows), 0), max(int(cols), 0)), dtype=FP_DTYPE)
    row_bytes = ffi.sizeof("int") + int(cols) * ffi.sizeof("FP_TYPE")
    total_bytes = int(rows) * row_bytes
    buf = ffi.buffer(ffi.cast("char *", block_ptr), total_bytes)
    return np.ndarray(
        shape=(int(rows), int(cols)),
        dtype=FP_DTYPE,
        buffer=buf,
        offset=ffi.sizeof("int"),
        strides=(row_bytes, ffi.sizeof("FP_TYPE")),
    )


def _i32_strided_block_lengths_view(block_ptr, rows: int, cols: int) -> IntArray1D:
    if block_ptr == ffi.NULL or rows <= 0:
        return np.empty((max(int(rows), 0),), dtype=np.int32)
    row_bytes = ffi.sizeof("int") + int(max(cols, 0)) * ffi.sizeof("FP_TYPE")
    total_bytes = int(rows) * row_bytes
    buf = ffi.buffer(ffi.cast("char *", block_ptr), total_bytes)
    return np.ndarray(
        shape=(int(rows),),
        dtype=np.int32,
        buffer=buf,
        offset=0,
        strides=(row_bytes,),
    )


def _ensure_chunk_f0_view(ptr) -> None:
    if ptr.py_frame_f0 == ffi.NULL:
        rc = int(lib.llsm_py_chunk_refresh_f0_view(ptr))
        if rc != 0:
            raise RuntimeError(f"llsm_py_chunk_refresh_f0_view failed with code {rc}")


def _ensure_chunk_rd_view(ptr) -> None:
    if ptr.py_frame_rd == ffi.NULL:
        rc = int(lib.llsm_py_chunk_refresh_rd_view(ptr))
        if rc != 0:
            raise RuntimeError(f"llsm_py_chunk_refresh_rd_view failed with code {rc}")


def _ensure_chunk_harmonics_view(ptr, max_nhar: int | None = None) -> None:
    if ptr.py_hm_ampl == ffi.NULL or (max_nhar is not None and int(ptr.py_hm_max_nhar) != int(max_nhar)):
        rc = int(lib.llsm_py_chunk_refresh_harmonics_view(ptr, 0 if max_nhar is None else int(max_nhar)))
        if rc != 0:
            raise RuntimeError(f"llsm_py_chunk_refresh_harmonics_view failed with code {rc}")


def _ensure_chunk_vector_view(ptr, member_idx: int, max_len: int | None = None) -> None:
    block_ptr = ptr.py_vtmagn_block if member_idx == _LLSM_FRAME_VTMAGN else ptr.py_vsphse_block
    block_len = ptr.py_vtmagn_max_len if member_idx == _LLSM_FRAME_VTMAGN else ptr.py_vsphse_max_len
    if block_ptr == ffi.NULL or (max_len is not None and int(block_len) != int(max_len)):
        rc = int(lib.llsm_py_chunk_refresh_vector_view(ptr, int(member_idx), 0 if max_len is None else int(max_len)))
        if rc != 0:
            raise RuntimeError(f"llsm_py_chunk_refresh_vector_view failed with code {rc}")


def _chunk_nfrm(ptr) -> int:
    nfrm = int(lib.llsm_py_chunk_nfrm(ptr))
    if nfrm < 0:
        raise RuntimeError("failed to get chunk frame count")
    return nfrm


def _chunk_harmonics_views(ptr) -> tuple[FloatArray2D, FloatArray2D, IntArray1D]:
    nfrm = _chunk_nfrm(ptr)
    _ensure_chunk_harmonics_view(ptr)
    width = int(ptr.py_hm_max_nhar)
    ampl = _f32_view_2d(ptr.py_hm_ampl, nfrm, width)
    phse = _f32_view_2d(ptr.py_hm_phse, nfrm, width)
    nhar = _i32_view_1d(ptr.py_frame_nhar, nfrm)
    return ampl, phse, nhar


def _chunk_vector_lengths_view(ptr, member_idx: int) -> IntArray1D:
    nfrm = _chunk_nfrm(ptr)
    _ensure_chunk_vector_view(ptr, member_idx)
    if member_idx == _LLSM_FRAME_VTMAGN:
        return _i32_strided_block_lengths_view(ptr.py_vtmagn_block, nfrm, int(ptr.py_vtmagn_max_len))
    return _i32_strided_block_lengths_view(ptr.py_vsphse_block, nfrm, int(ptr.py_vsphse_max_len))


def _chunk_vector_view(ptr, member_idx: int) -> tuple[FloatArray2D, IntArray1D]:
    nfrm = _chunk_nfrm(ptr)
    _ensure_chunk_vector_view(ptr, member_idx)
    if member_idx == _LLSM_FRAME_VTMAGN:
        values = _f32_strided_block_view(ptr.py_vtmagn_block, nfrm, int(ptr.py_vtmagn_max_len))
    else:
        values = _f32_strided_block_view(ptr.py_vsphse_block, nfrm, int(ptr.py_vsphse_max_len))
    lengths = _chunk_vector_lengths_view(ptr, member_idx)
    return values, lengths


def _unwrap_analysis_options(options):
    if isinstance(options, AnalysisOptions):
        return options.ptr
    return options


def _unwrap_synthesis_options(options):
    if isinstance(options, SynthesisOptions):
        return options.ptr
    return options


def _unwrap_chunk(chunk):
    if isinstance(chunk, _FeatureContainer):
        return chunk.ptr
    if isinstance(chunk, Chunk):
        return chunk.ptr
    return chunk

def _normalize_frame_mask(mask: Iterable[int] | Iterable[bool] | None, nfrm: int, name: str = "mask") -> np.ndarray:
    if mask is None:
        return np.ones(int(nfrm), dtype=bool)
    arr = np.asarray(mask)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D")
    if arr.dtype == np.bool_:
        if arr.size != int(nfrm):
            raise ValueError(f"{name} size mismatch: expected {nfrm}, got {arr.size}")
        return np.ascontiguousarray(arr, dtype=bool)
    idx = np.asarray(mask, dtype=np.int64)
    if idx.ndim != 1:
        raise ValueError(f"{name} must be 1-D")
    idx = idx.copy()
    idx[idx < 0] += int(nfrm)
    if idx.size > 0 and (np.min(idx) < 0 or np.max(idx) >= int(nfrm)):
        raise IndexError(f"{name} contains out-of-range frame index")
    out = np.zeros(int(nfrm), dtype=bool)
    out[idx] = True
    return out


def _chunk_matches_layer(ptr, layer: int) -> bool:
    if ptr == ffi.NULL:
        return False
    nfrm = int(lib.llsm_py_chunk_nfrm(ptr))
    if nfrm < 0:
        return False
    checker = lib.llsm_frame_checklayer0 if layer == 0 else lib.llsm_frame_checklayer1
    for i in range(nfrm):
        frame = ptr.frames[i]
        if frame == ffi.NULL or int(checker(frame)) == 0:
            return False
    return True


def _copy_chunk_ptr(ptr):
    copied = lib.llsm_copy_chunk(ptr)
    if copied == ffi.NULL:
        raise MemoryError("llsm_copy_chunk returned NULL")
    return copied

def _chunk_set_vector_matrix(ptr, member_idx: int, values: FloatArray2D, lengths: Iterable[int] | None = None) -> None:
    """Replace dense variable-length frame data.

    Args:
        values: shape ``(nfrm, max_len)``; axis 1 is per-frame element index.
        lengths: optional shape ``(nfrm,)`` valid prefix lengths per frame.
    """
    values_arr = np.asarray(values, dtype=FP_DTYPE)
    if values_arr.ndim != 2:
        raise ValueError("values must be 2-D")
    nfrm = int(lib.llsm_py_chunk_nfrm(ptr))
    if values_arr.shape[0] != nfrm:
        raise ValueError(f"frame count mismatch: expected {nfrm}, got {values_arr.shape[0]}")
    if lengths is None:
        len_arr = np.full(nfrm, values_arr.shape[1], dtype=np.int32)
    else:
        len_arr = as_i32_array(lengths, "lengths")
        if len_arr.size != nfrm:
            raise ValueError(f"lengths size mismatch: expected {nfrm}, got {len_arr.size}")
    len_arr = np.maximum(len_arr, 0).astype(np.int32, copy=False)
    rc = int(
        lib.llsm_py_chunk_set_vector_matrix(
            ptr,
            int(member_idx),
            ffi.from_buffer("FP_TYPE[]", np.ascontiguousarray(values_arr).reshape(-1)),
            ffi.from_buffer("int[]", len_arr),
            nfrm,
            values_arr.shape[1],
        )
    )
    if rc != 0:
        raise RuntimeError(f"llsm_py_chunk_set_vector_matrix failed with code {rc}")


def _reshape_matrix_like(current: FloatArray2D, target: FloatArray2D) -> FloatArray2D:
    """Resize one dense matrix to match another while preserving overlap."""

    if current.shape == target.shape:
        return current
    out = np.zeros_like(target, dtype=FP_DTYPE)
    rows = min(current.shape[0], target.shape[0])
    cols = min(current.shape[1], target.shape[1])
    if rows > 0 and cols > 0:
        out[:rows, :cols] = current[:rows, :cols]
    return out


def _resolve_harmonic_lengths(current: IntArray1D, width: int) -> IntArray1D:
    """Resolve per-frame harmonic counts for a dense matrix assignment.

    If the current chunk has no active harmonics yet, assigning a dense matrix
    implies that every column up to ``width`` is valid.
    """

    current_arr = np.asarray(current, dtype=np.int32)
    if current_arr.size == 0:
        return current_arr
    if width <= 0:
        return np.zeros_like(current_arr, dtype=np.int32)
    if int(np.max(current_arr)) <= 0:
        return np.full(current_arr.shape, int(width), dtype=np.int32)
    return np.minimum(current_arr, int(width)).astype(np.int32, copy=False)


def _chunk_conf_int(ptr, index: int, default: int = 0) -> int:
    raw = ffi.cast("int*", lib.llsm_container_get(ptr.conf, int(index)))
    return int(raw[0]) if raw != ffi.NULL else int(default)


def _create_chunk_for_layer(
    nfrm: int,
    *,
    fs: float,
    options: AnalysisOptions | Any | None,
    layer: int,
    max_nhar: int,
    nspec: int | None = None,
) -> "Chunk":
    """Allocate a chunk with concrete per-frame members for one target layer."""

    conf = _make_chunk_conf(options, int(nfrm), float(fs), maxnhar=int(max_nhar), nspec=nspec)
    try:
        chunk_ptr = lib.llsm_create_chunk(conf, 0)
    finally:
        lib.llsm_delete_container(conf)
    if chunk_ptr == ffi.NULL:
        raise MemoryError("llsm_create_chunk returned NULL")
    chunk = Chunk(chunk_ptr)
    try:
        nchannel = _chunk_conf_int(chunk.ptr, _LLSM_CONF_NCHANNEL, default=1)
        npsd = _chunk_conf_int(chunk.ptr, _LLSM_CONF_NPSD, default=0)
        nhar_e = _chunk_conf_int(chunk.ptr, _LLSM_CONF_MAXNHAR_E, default=0)
        for i in range(chunk.nfrm):
            frame = lib.llsm_create_frame(int(max_nhar) if int(layer) == 0 else 0, nchannel, nhar_e, npsd)
            if frame == ffi.NULL:
                raise MemoryError("llsm_create_frame returned NULL")
            if int(layer) == 1:
                lib.llsm_container_remove(frame, _LLSM_FRAME_HM)
            chunk.ptr.frames[i] = frame
        return chunk
    except Exception:
        chunk.close()
        raise


def _copy_synthesis_options(options: SynthesisOptions | Any, use_l1: bool | None = None) -> "SynthesisOptions":
    src = options if isinstance(options, SynthesisOptions) else SynthesisOptions(ptr=_unwrap_synthesis_options(options), owns_memory=False)
    src._ensure_open()
    dst = SynthesisOptions(fs=float(src.fs))
    dst.use_iczt = int(src.use_iczt)
    dst.use_l1 = int(src.use_l1)
    dst.iczt_param_a = float(src.iczt_param_a)
    dst.iczt_param_b = float(src.iczt_param_b)
    if use_l1 is not None:
        dst.use_l1 = 1 if use_l1 else 0
    return dst


def _copy_analysis_options(options: AnalysisOptions | Any | None = None) -> "AnalysisOptions":
    if options is None:
        return AnalysisOptions()
    src = options if isinstance(options, AnalysisOptions) else AnalysisOptions(ptr=_unwrap_analysis_options(options), owns_memory=False)
    src._ensure_open()
    dst = AnalysisOptions()
    dst.thop = float(src.thop)
    dst.maxnhar = int(src.maxnhar)
    dst.maxnhar_e = int(src.maxnhar_e)
    dst.npsd = int(src.npsd)
    dst.nchannel = int(src.nchannel)
    dst.lip_radius = float(src.lip_radius)
    dst.f0_refine = int(src.f0_refine)
    dst.hm_method = int(src.hm_method)
    dst.rel_winsize = float(src.rel_winsize)
    return dst


def _attach_conf_int(conf, index: int, value: int) -> None:
    ptr = lib.llsm_create_int(int(value))
    if ptr == ffi.NULL:
        raise MemoryError("llsm_create_int returned NULL")
    container_attach(
        conf,
        int(index),
        ffi.cast("void*", ptr),
        ffi.cast("llsm_fdestructor", lib.llsm_delete_int),
        ffi.cast("llsm_fcopy", lib.llsm_copy_int),
    )


def _make_chunk_conf(options: AnalysisOptions | Any | None, nfrm: int, fs: float, *, maxnhar: int | None = None, nspec: int | None = None):
    opt = _copy_analysis_options(options)
    try:
        if maxnhar is not None:
            opt.maxnhar = int(maxnhar)
        conf = lib.llsm_aoptions_toconf(opt.ptr, float(fs) / 2.0)
    finally:
        opt.close()
    if conf == ffi.NULL:
        raise MemoryError("llsm_aoptions_toconf returned NULL")
    _attach_conf_int(conf, _LLSM_CONF_NFRM, int(nfrm))
    if maxnhar is not None:
        _attach_conf_int(conf, _LLSM_CONF_MAXNHAR, int(maxnhar))
    if nspec is not None:
        _attach_conf_int(conf, _LLSM_CONF_NSPEC, int(nspec))
    return conf


class _FeatureContainer:
    """Base class for user-facing feature containers derived from ``Chunk``."""

    _EXPECTED_LAYER: int | None = None

    def __init__(self, chunk: Chunk | Any, *, validate: bool = True, owns_memory: bool = True):
        self._chunk = chunk if isinstance(chunk, Chunk) else Chunk(chunk, owns_memory=owns_memory)
        if validate and self._EXPECTED_LAYER is not None and not _chunk_matches_layer(self.ptr, self._EXPECTED_LAYER):
            raise ValueError(f"chunk does not contain a valid layer {self._EXPECTED_LAYER} representation")

    def __enter__(self):
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    @property
    def ptr(self):
        return self._chunk.ptr

    @property
    def nfrm(self) -> int:
        """Number of frames shared by all per-frame arrays in this container."""
        return self._chunk.nfrm

    @property
    def closed(self) -> bool:
        return self._chunk.closed

    @property
    def f0(self) -> np.ndarray:
        """Fundamental-frequency vector with shape ``(nfrm,)``.

        Axis 0 is frame index over time.
        """
        return self._chunk.f0

    @f0.setter
    def f0(self, value: Iterable[float]) -> None:
        self._chunk.f0 = value

    def close(self) -> None:
        self._chunk.close()

    def copy_chunk(self) -> Chunk:
        return Chunk(_copy_chunk_ptr(self.ptr))

    def phasepropagate(self, sign: int = 1) -> None:
        self._chunk.phasepropagate(sign=sign)

    def phasesync_rps(self, layer1_based: bool = False) -> None:
        self._chunk.phasesync_rps(layer1_based=layer1_based)


class Layer0Features(_FeatureContainer):
    """User-facing container for layer-0 speech features.

    Layer 0 is the harmonic-plus-noise representation produced directly by
    analysis. It is the clearest entry point for the analysis workflow.

    Array axis meanings used by this class:

    - axis 0: frame index over time, shape ``(nfrm, ...)``
    - axis 1 for harmonic matrices: harmonic index, shape ``(..., max_nhar)``
    """

    _EXPECTED_LAYER = 0

    @classmethod
    def from_arrays(
        cls,
        f0: Iterable[float],
        ampl: FloatArray2D,
        phse: FloatArray2D,
        nhar: Iterable[int] | None = None,
        *,
        fs: float,
        options: AnalysisOptions | Any | None = None,
    ) -> "Layer0Features":
        """Create layer-0 features from dense NumPy arrays.

        Args:
            f0: shape ``(nfrm,)``; axis 0 is frame index over time.
            ampl: shape ``(nfrm, max_nhar)``; axis 1 is harmonic index.
            phse: shape ``(nfrm, max_nhar)``; axis 1 is harmonic index.
            nhar: optional shape ``(nfrm,)`` valid harmonic counts per frame.
            fs: sampling rate in Hz.
        """
        return cls(Chunk.from_layer0_arrays(f0, ampl, phse, nhar=nhar, fs=fs, options=options), validate=False)

    @property
    def nhar(self) -> IntArray1D:
        """Valid harmonic counts with shape ``(nfrm,)``.

        Axis 0 is frame index over time.
        """
        return self._chunk.nhar

    @nhar.setter
    def nhar(self, value: Iterable[int]) -> None:
        self._chunk.nhar = value

    @property
    def ampl(self) -> FloatArray2D:
        """Harmonic amplitude matrix with shape ``(nfrm, max_nhar)``.

        Axis 0 is frame index over time.
        Axis 1 is harmonic index within each frame.
        """
        return self._chunk.ampl

    @ampl.setter
    def ampl(self, values: FloatArray2D) -> None:
        self._chunk.ampl = values

    @property
    def phse(self) -> FloatArray2D:
        """Harmonic phase matrix with shape ``(nfrm, max_nhar)``.

        Axis 0 is frame index over time.
        Axis 1 is harmonic index within each frame.
        """
        return self._chunk.phse

    @phse.setter
    def phse(self, values: FloatArray2D) -> None:
        self._chunk.phse = values

    def copy(self) -> "Layer0Features":
        return Layer0Features(_copy_chunk_ptr(self.ptr))

    def frame_scalar(self, member_idx: int, default: float = np.nan) -> FloatArray1D:
        """Read one scalar member as shape ``(nfrm,)``.

        Axis 0 is frame index over time.
        """
        return self._chunk.frame_scalar(member_idx, default=default)

    def set_frame_scalar(self, member_idx: int, values: Iterable[float]) -> None:
        self._chunk.set_frame_scalar(member_idx, values)

    def clear_member(self, member_idx: int, mask: Iterable[int] | Iterable[bool] | None = None) -> None:
        self._chunk.clear_member(member_idx, mask=mask)

    def enable_pulse_by_pulse(self, mask: Iterable[int] | Iterable[bool], clear_harmonics: bool = True) -> None:
        self._chunk.enable_pulse_by_pulse(mask, clear_harmonics=clear_harmonics)

    def resample_linear_f0(self, nfrm_new: int) -> "Layer0Features":
        return Layer0Features(self._chunk.resample_linear_f0(nfrm_new))

    def to_layer1(self, nfft: int) -> "Layer1Features":
        chunk_ptr = _copy_chunk_ptr(self.ptr)
        lib.llsm_chunk_tolayer1(chunk_ptr, int(nfft))
        return Layer1Features(chunk_ptr)


class Layer1Features(_FeatureContainer):
    """User-facing container for layer-1 speech features.

    Layer 1 reinterprets the signal in a source-filter form with glottal and
    vocal-tract parameters.

    Array axis meanings used by this class:

    - axis 0: frame index over time, shape ``(nfrm, ...)``
    - axis 1 for spectral/source-phase matrices: per-frame bin/index axis
    """

    _EXPECTED_LAYER = 1

    @classmethod
    def from_arrays(
        cls,
        f0: Iterable[float],
        rd: Iterable[float],
        vtmagn: FloatArray2D,
        vsphse: FloatArray2D,
        *,
        fs: float,
        options: AnalysisOptions | Any | None = None,
        vsphse_lengths: Iterable[int] | None = None,
    ) -> "Layer1Features":
        """Create layer-1 features from dense NumPy arrays.

        Args:
            f0: shape ``(nfrm,)``; axis 0 is frame index over time.
            rd: shape ``(nfrm,)``; axis 0 is frame index over time.
            vtmagn: shape ``(nfrm, nspec)``; axis 1 is spectral-bin index.
            vsphse: shape ``(nfrm, max_nhar)``; axis 1 is source-harmonic index.
            vsphse_lengths: optional shape ``(nfrm,)`` valid source-phase counts.
            fs: sampling rate in Hz.
        """
        return cls(
            Chunk.from_layer1_arrays(
                f0,
                rd,
                vtmagn,
                vsphse,
                fs=fs,
                options=options,
                vsphse_lengths=vsphse_lengths,
            ),
            validate=False,
        )

    @property
    def rd(self) -> FloatArray1D:
        """Rd vector with shape ``(nfrm,)`` where axis 0 is frame index."""
        return self._chunk.rd

    @rd.setter
    def rd(self, values: Iterable[float]) -> None:
        self._chunk.rd = values

    @property
    def vtmagn(self) -> FloatArray2D:
        """Vocal-tract magnitude matrix with shape ``(nfrm, nspec)``.

        Axis 0 is frame index over time.
        Axis 1 is spectral-bin index.
        """
        return self._chunk.vtmagn

    @vtmagn.setter
    def vtmagn(self, values: FloatArray2D) -> None:
        self._chunk.vtmagn = values

    @property
    def vsphse(self) -> FloatArray2D:
        """Vocal-source phase matrix with shape ``(nfrm, max_nhar)``.

        Axis 0 is frame index over time.
        Axis 1 is source-harmonic index.
        """
        return self._chunk.vsphse

    @vsphse.setter
    def vsphse(self, values: np.ndarray) -> None:
        self._chunk.vsphse = values

    @property
    def vsphse_lengths(self) -> IntArray1D:
        """Valid source-phase counts with shape ``(nfrm,)``.

        Axis 0 is frame index over time.
        """
        return self._chunk.vsphse_lengths

    @property
    def nspec(self) -> int:
        return self._chunk.nspec

    def copy(self) -> "Layer1Features":
        return Layer1Features(_copy_chunk_ptr(self.ptr))

    def enable_pulse_by_pulse(self, mask: Iterable[int] | Iterable[bool], clear_harmonics: bool = True) -> None:
        self._chunk.enable_pulse_by_pulse(mask, clear_harmonics=clear_harmonics)

    def resample_linear_f0(self, nfrm_new: int) -> "Layer1Features":
        return Layer1Features(self._chunk.resample_linear_f0(nfrm_new), validate=False)

    def pitch_shift(self, ratio: float, compensate_vtmagn_db: bool = True, clear_harmonics: bool = True) -> None:
        self._chunk.pitch_shift_layer1(
            ratio, compensate_vtmagn_db=compensate_vtmagn_db, clear_harmonics=clear_harmonics
        )

    def to_layer0(self) -> Layer0Features:
        chunk_ptr = _copy_chunk_ptr(self.ptr)
        lib.llsm_chunk_tolayer0(chunk_ptr)
        return Layer0Features(chunk_ptr)


class AnalysisOptions:
    """Analysis configuration for converting audio into an LLSM chunk.

    Instances expose the native ``llsm_aoptions`` fields as regular Python
    attributes such as ``thop`` and ``maxnhar``.
    """

    _FIELDS = {
        "thop",
        "maxnhar",
        "maxnhar_e",
        "npsd",
        "nchannel",
        "chanfreq",
        "lip_radius",
        "f0_refine",
        "hm_method",
        "rel_winsize",
    }

    def __init__(self, ptr=None, owns_memory: bool = True):
        if ptr is None:
            ptr = lib.llsm_create_aoptions()
        if ptr == ffi.NULL:
            raise MemoryError("llsm_create_aoptions returned NULL")
        self.ptr = ptr
        self._owns_memory = owns_memory
        self._closed = False

    def __enter__(self) -> "AnalysisOptions":
        self._ensure_open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    @property
    def closed(self) -> bool:
        return self._closed

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("AnalysisOptions is closed")

    def close(self) -> None:
        """Release the owned native options object."""

        if not self._closed and self._owns_memory:
            lib.llsm_delete_aoptions(self.ptr)
        self._closed = True

    def __getattr__(self, name: str):
        if name in self._FIELDS:
            self._ensure_open()
            return getattr(self.ptr, name)
        raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"ptr", "_owns_memory", "_closed"}:
            object.__setattr__(self, name, value)
            return
        if name in self._FIELDS:
            self._ensure_open()
            setattr(self.ptr, name, value)
            return
        object.__setattr__(self, name, value)


class SynthesisOptions:
    """Synthesis configuration for rendering audio from an LLSM chunk."""

    _FIELDS = {"fs", "use_iczt", "use_l1", "iczt_param_a", "iczt_param_b"}

    def __init__(self, fs: float | None = None, ptr=None, owns_memory: bool = True):
        if ptr is None:
            if fs is None:
                raise ValueError("fs must be provided when ptr is None")
            ptr = lib.llsm_create_soptions(float(fs))
        if ptr == ffi.NULL:
            raise MemoryError("llsm_create_soptions returned NULL")
        self.ptr = ptr
        self._owns_memory = owns_memory
        self._closed = False

    def __enter__(self) -> "SynthesisOptions":
        self._ensure_open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    @property
    def closed(self) -> bool:
        return self._closed

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("SynthesisOptions is closed")

    def close(self) -> None:
        """Release the owned native options object."""

        if not self._closed and self._owns_memory:
            lib.llsm_delete_soptions(self.ptr)
        self._closed = True

    def __getattr__(self, name: str):
        if name in self._FIELDS:
            self._ensure_open()
            return getattr(self.ptr, name)
        raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"ptr", "_owns_memory", "_closed"}:
            object.__setattr__(self, name, value)
            return
        if name in self._FIELDS:
            self._ensure_open()
            setattr(self.ptr, name, value)
            return
        object.__setattr__(self, name, value)


class Chunk:
    """High-level wrapper around an ``llsm_chunk`` native pointer.

    A chunk stores one analyzed or synthesized sequence of LLSM frames. The
    native C structure can hold both layer-0 and layer-1 members at once, but
    this wrapper exposes layer-specific ndarray properties so Python code can
    stay array-oriented instead of container-oriented.
    """

    @classmethod
    def allocate_layer0(
        cls,
        nfrm: int,
        *,
        fs: float,
        max_nhar: int,
        options: AnalysisOptions | Any | None = None,
    ) -> "Chunk":
        """Allocate a layer-0-oriented chunk.

        The allocated chunk exposes writable ndarray properties:

        - ``f0`` with shape ``(nfrm,)``
        - ``ampl`` with shape ``(nfrm, max_nhar)``
        - ``phse`` with shape ``(nfrm, max_nhar)``
        - ``nhar`` with shape ``(nfrm,)``
        """

        nfrm = int(nfrm)
        max_nhar = int(max_nhar)
        if nfrm <= 0:
            raise ValueError("nfrm must be positive")
        if max_nhar < 0:
            raise ValueError("max_nhar must be non-negative")
        return _create_chunk_for_layer(nfrm, fs=float(fs), options=options, layer=0, max_nhar=max_nhar)

    @classmethod
    def allocate_layer1(
        cls,
        nfrm: int,
        *,
        fs: float,
        nspec: int,
        max_nhar: int,
        options: AnalysisOptions | Any | None = None,
    ) -> "Chunk":
        """Allocate a layer-1-oriented chunk.

        The allocated chunk exposes writable ndarray properties:

        - ``f0`` with shape ``(nfrm,)``
        - ``rd`` with shape ``(nfrm,)``
        - ``vtmagn`` with shape ``(nfrm, nspec)``
        - ``vsphse`` with shape ``(nfrm, max_nhar)``
        - ``vsphse_lengths`` with shape ``(nfrm,)``
        """

        nfrm = int(nfrm)
        nspec = int(nspec)
        max_nhar = int(max_nhar)
        if nfrm <= 0:
            raise ValueError("nfrm must be positive")
        if nspec < 0:
            raise ValueError("nspec must be non-negative")
        if max_nhar < 0:
            raise ValueError("max_nhar must be non-negative")
        return _create_chunk_for_layer(
            nfrm,
            fs=float(fs),
            options=options,
            layer=1,
            max_nhar=max_nhar,
            nspec=nspec,
        )

    @classmethod
    def from_layer0_arrays(
        cls,
        f0: Iterable[float],
        ampl: FloatArray2D,
        phse: FloatArray2D,
        nhar: Iterable[int] | None = None,
        *,
        fs: float,
        options: AnalysisOptions | Any | None = None,
    ) -> "Chunk":
        """Create a layer-0-oriented chunk from dense arrays.

        Args:
            f0: shape ``(nfrm,)``; axis 0 is frame index over time.
            ampl: shape ``(nfrm, max_nhar)``; axis 1 is harmonic index.
            phse: shape ``(nfrm, max_nhar)``; axis 1 is harmonic index.
            nhar: optional shape ``(nfrm,)`` valid harmonic counts per frame.
        """
        f0_arr = as_f32_array(f0, "f0")
        ampl_arr = np.asarray(ampl, dtype=FP_DTYPE)
        phse_arr = np.asarray(phse, dtype=FP_DTYPE)
        if ampl_arr.shape != phse_arr.shape:
            raise ValueError("ampl/phse shape mismatch")
        if ampl_arr.ndim != 2:
            raise ValueError("ampl/phse must be 2-D")
        if ampl_arr.shape[0] != f0_arr.size:
            raise ValueError(f"frame count mismatch: expected {f0_arr.size}, got {ampl_arr.shape[0]}")
        nhar_arr = np.full(f0_arr.size, ampl_arr.shape[1], dtype=np.int32) if nhar is None else as_i32_array(nhar, "nhar")
        if nhar_arr.size != f0_arr.size:
            raise ValueError(f"nhar size mismatch: expected {f0_arr.size}, got {nhar_arr.size}")
        nhar_arr = np.maximum(nhar_arr, 0).astype(np.int32, copy=False)
        max_nhar = int(ampl_arr.shape[1])
        if nhar_arr.size > 0 and int(np.max(nhar_arr)) > max_nhar:
            raise ValueError("nhar contains values larger than harmonic matrix width")
        chunk = cls.allocate_layer0(f0_arr.size, fs=float(fs), max_nhar=max_nhar, options=options)
        try:
            chunk.f0 = f0_arr
            chunk.ampl = ampl_arr
            chunk.phse = phse_arr
            chunk.nhar = nhar_arr
            return chunk
        except Exception:
            chunk.close()
            raise

    @classmethod
    def from_layer1_arrays(
        cls,
        f0: Iterable[float],
        rd: Iterable[float],
        vtmagn: FloatArray2D,
        vsphse: FloatArray2D,
        *,
        fs: float,
        options: AnalysisOptions | Any | None = None,
        vsphse_lengths: Iterable[int] | None = None,
    ) -> "Chunk":
        """Create a layer-1-oriented chunk from dense arrays.

        Args:
            f0: shape ``(nfrm,)``; axis 0 is frame index over time.
            rd: shape ``(nfrm,)``; axis 0 is frame index over time.
            vtmagn: shape ``(nfrm, nspec)``; axis 1 is spectral-bin index.
            vsphse: shape ``(nfrm, max_nhar)``; axis 1 is source-harmonic index.
            vsphse_lengths: optional shape ``(nfrm,)`` valid source-phase counts.
        """
        f0_arr = as_f32_array(f0, "f0")
        rd_arr = as_f32_array(rd, "rd")
        vtmagn_arr = np.asarray(vtmagn, dtype=FP_DTYPE)
        vsphse_arr = np.asarray(vsphse, dtype=FP_DTYPE)
        if rd_arr.size != f0_arr.size:
            raise ValueError(f"rd size mismatch: expected {f0_arr.size}, got {rd_arr.size}")
        if vtmagn_arr.ndim != 2 or vtmagn_arr.shape[0] != f0_arr.size:
            raise ValueError("vtmagn must be 2-D with frame dimension matching f0")
        if vsphse_arr.ndim != 2 or vsphse_arr.shape[0] != f0_arr.size:
            raise ValueError("vsphse must be 2-D with frame dimension matching f0")
        lengths_arr = np.full(f0_arr.size, vsphse_arr.shape[1], dtype=np.int32) if vsphse_lengths is None else as_i32_array(vsphse_lengths, "vsphse_lengths")
        if lengths_arr.size != f0_arr.size:
            raise ValueError(f"vsphse_lengths size mismatch: expected {f0_arr.size}, got {lengths_arr.size}")
        lengths_arr = np.maximum(lengths_arr, 0).astype(np.int32, copy=False)
        if lengths_arr.size > 0 and int(np.max(lengths_arr)) > vsphse_arr.shape[1]:
            raise ValueError("vsphse_lengths contains values larger than vsphse width")
        max_nhar = int(vsphse_arr.shape[1])
        chunk = cls.allocate_layer1(
            f0_arr.size,
            fs=float(fs),
            nspec=int(vtmagn_arr.shape[1]),
            max_nhar=max_nhar,
            options=options,
        )
        try:
            chunk.f0 = f0_arr
            chunk.rd = rd_arr
            chunk.vtmagn = vtmagn_arr
            chunk.vsphse = vsphse_arr
            chunk.vsphse_lengths = lengths_arr
            return chunk
        except Exception:
            chunk.close()
            raise

    def __init__(self, ptr, owns_memory: bool = True):
        if ptr == ffi.NULL:
            raise MemoryError("llsm_chunk is NULL")
        self.ptr = ptr
        self._owns_memory = owns_memory
        self._closed = False

    def __enter__(self) -> "Chunk":
        self._ensure_open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    @property
    def closed(self) -> bool:
        return self._closed

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Chunk is closed")

    def close(self) -> None:
        """Release the owned native chunk."""

        if not self._closed and self._owns_memory:
            lib.llsm_delete_chunk(self.ptr)
        self._closed = True

    def copy(self) -> "Chunk":
        """Return a deep copy of the chunk."""

        self._ensure_open()
        return Chunk(lib.llsm_copy_chunk(self.ptr))

    @property
    def nfrm(self) -> int:
        """Number of frames stored in the chunk."""

        self._ensure_open()
        nfrm = int(lib.llsm_py_chunk_nfrm(self.ptr))
        if nfrm < 0:
            raise RuntimeError("failed to get chunk frame count")
        return nfrm

    @property
    def f0(self) -> FloatArray1D:
        """Frame-wise fundamental frequency with shape ``(nfrm,)``.

        Axis 0 is frame index over time.
        """

        return self.frame_f0()

    @f0.setter
    def f0(self, value: Iterable[float]) -> None:
        self.set_frame_f0(value)

    @property
    def nhar(self) -> IntArray1D:
        """Frame-wise harmonic counts with shape ``(nfrm,)``.

        Axis 0 is frame index over time.
        """

        self._ensure_open()
        _ensure_chunk_harmonics_view(self.ptr)
        return _i32_view_1d(self.ptr.py_frame_nhar, self.nfrm)

    @nhar.setter
    def nhar(self, value: Iterable[int]) -> None:
        nhar_arr = as_i32_array(value, "nhar")
        if nhar_arr.size != self.nfrm:
            raise ValueError(f"nhar size mismatch: expected {self.nfrm}, got {nhar_arr.size}")
        self.set_harmonics(self.ampl, self.phse, nhar_arr)

    @property
    def ampl(self) -> FloatArray2D:
        """Harmonic amplitude matrix with shape ``(nfrm, max_nhar)``.

        Axis 0 is frame index over time.
        Axis 1 is harmonic index within each frame.
        """

        return self.harmonics(fill_value=0.0).ampl

    @ampl.setter
    def ampl(self, values: FloatArray2D) -> None:
        values_arr = np.asarray(values, dtype=FP_DTYPE)
        if values_arr.ndim != 2:
            raise ValueError("ampl must be 2-D")
        if values_arr.shape[0] != self.nfrm:
            raise ValueError(f"frame count mismatch: expected {self.nfrm}, got {values_arr.shape[0]}")
        phse = _reshape_matrix_like(self.phse, values_arr)
        nhar = _resolve_harmonic_lengths(self.nhar, values_arr.shape[1])
        self.set_harmonics(values_arr, phse, nhar)

    @property
    def phse(self) -> FloatArray2D:
        """Harmonic phase matrix with shape ``(nfrm, max_nhar)``.

        Axis 0 is frame index over time.
        Axis 1 is harmonic index within each frame.
        """

        return self.harmonics(fill_value=0.0).phse

    @phse.setter
    def phse(self, values: FloatArray2D) -> None:
        values_arr = np.asarray(values, dtype=FP_DTYPE)
        if values_arr.ndim != 2:
            raise ValueError("phse must be 2-D")
        if values_arr.shape[0] != self.nfrm:
            raise ValueError(f"frame count mismatch: expected {self.nfrm}, got {values_arr.shape[0]}")
        ampl = _reshape_matrix_like(self.ampl, values_arr)
        nhar = _resolve_harmonic_lengths(self.nhar, values_arr.shape[1])
        self.set_harmonics(ampl, values_arr, nhar)

    @property
    def rd(self) -> FloatArray1D:
        """Rd vector with shape ``(nfrm,)`` where axis 0 is frame index."""

        self._ensure_open()
        _ensure_chunk_rd_view(self.ptr)
        return _f32_view_1d(self.ptr.py_frame_rd, self.nfrm)

    @rd.setter
    def rd(self, values: Iterable[float]) -> None:
        self.set_frame_scalar(_LLSM_FRAME_RD, values)

    @property
    def vtmagn(self) -> FloatArray2D:
        """Vocal-tract magnitude matrix with shape ``(nfrm, nspec)``.

        Axis 0 is frame index over time.
        Axis 1 is spectral-bin index.
        """

        values, _ = _chunk_vector_view(self.ptr, _LLSM_FRAME_VTMAGN)
        return values

    @vtmagn.setter
    def vtmagn(self, values: FloatArray2D) -> None:
        _chunk_set_vector_matrix(self.ptr, _LLSM_FRAME_VTMAGN, values)

    @property
    def vsphse(self) -> FloatArray2D:
        """Vocal-source phase matrix with shape ``(nfrm, max_nhar)``.

        Axis 0 is frame index over time.
        Axis 1 is source-harmonic index.
        """

        values, _ = _chunk_vector_view(self.ptr, _LLSM_FRAME_VSPHSE)
        return values

    @vsphse.setter
    def vsphse(self, values: FloatArray2D) -> None:
        values_arr = np.asarray(values, dtype=FP_DTYPE)
        if values_arr.ndim != 2:
            raise ValueError("vsphse must be 2-D")
        lengths = _resolve_harmonic_lengths(self.vsphse_lengths, values_arr.shape[1])
        _chunk_set_vector_matrix(self.ptr, _LLSM_FRAME_VSPHSE, values_arr, lengths=lengths)

    @property
    def vsphse_lengths(self) -> IntArray1D:
        """Valid source-phase counts with shape ``(nfrm,)``.

        Axis 0 is frame index over time.
        """

        return _chunk_vector_lengths_view(self.ptr, _LLSM_FRAME_VSPHSE)

    @vsphse_lengths.setter
    def vsphse_lengths(self, lengths: Iterable[int]) -> None:
        lengths_arr = as_i32_array(lengths, "vsphse_lengths")
        if lengths_arr.size != self.nfrm:
            raise ValueError(f"vsphse_lengths size mismatch: expected {self.nfrm}, got {lengths_arr.size}")
        _chunk_set_vector_matrix(self.ptr, _LLSM_FRAME_VSPHSE, self.vsphse, lengths=lengths_arr)

    @property
    def nspec(self) -> int:
        """Spectral width for layer-1 vocal-tract magnitude matrices."""

        self._ensure_open()
        return _chunk_conf_int(self.ptr, _LLSM_CONF_NSPEC, default=0)

    def frame_f0(self) -> FloatArray1D:
        """Return the frame-wise F0 track with shape ``(nfrm,)``."""

        self._ensure_open()
        _ensure_chunk_f0_view(self.ptr)
        return _f32_view_1d(self.ptr.py_frame_f0, self.nfrm)

    def set_frame_f0(self, f0: Iterable[float]) -> None:
        """Replace the frame-wise F0 track."""

        self._ensure_open()
        arr = as_f32_array(f0, "f0")
        if arr.size != self.nfrm:
            raise ValueError(f"f0 size mismatch: expected {self.nfrm}, got {arr.size}")
        rc = int(lib.llsm_py_chunk_set_f0(self.ptr, ffi.from_buffer("FP_TYPE[]", arr), arr.size))
        if rc != 0:
            raise RuntimeError(f"llsm_py_chunk_set_f0 failed with code {rc}")

    def frame_nhar(self) -> IntArray1D:
        """Return valid harmonic counts with shape ``(nfrm,)``."""

        self._ensure_open()
        nfrm = self.nfrm
        out = np.empty(nfrm, dtype=np.int32)
        rc = int(lib.llsm_py_chunk_fill_nhar(self.ptr, ffi.from_buffer("int[]", out), nfrm))
        if rc != 0:
            raise RuntimeError(f"llsm_py_chunk_fill_nhar failed with code {rc}")
        return out

    def get_frame_harmonics(self, frame_idx: int) -> tuple[FloatArray1D, FloatArray1D]:
        """Return one frame's harmonic vectors.

        Returns:
            Tuple ``(ampl, phse)`` where both arrays have shape ``(nhar_i,)`` and
            axis 0 is harmonic index within the selected frame.
        """

        self._ensure_open()
        frame = self.ptr.frames[int(frame_idx)]
        hm_ptr = ffi.cast("llsm_hmframe*", lib.llsm_container_get(frame, _LLSM_FRAME_HM))
        if hm_ptr == ffi.NULL or int(hm_ptr.nhar) <= 0:
            return np.zeros(0, dtype=FP_DTYPE), np.zeros(0, dtype=FP_DTYPE)
        nhar = int(hm_ptr.nhar)
        ampl = np.frombuffer(ffi.buffer(hm_ptr.ampl, nhar * ffi.sizeof("FP_TYPE")), dtype=FP_DTYPE).copy()
        phse = np.frombuffer(ffi.buffer(hm_ptr.phse, nhar * ffi.sizeof("FP_TYPE")), dtype=FP_DTYPE).copy()
        return ampl, phse

    def set_frame_harmonics(self, frame_idx: int, ampl: Iterable[float], phse: Iterable[float]) -> None:
        """Replace the harmonic amplitudes and phases for one frame."""

        self._ensure_open()
        ampl_arr = as_f32_array(ampl, "ampl")
        phse_arr = as_f32_array(phse, "phse")
        if ampl_arr.size != phse_arr.size:
            raise ValueError("ampl/phse size mismatch")
        rc = int(
            lib.llsm_py_chunk_frame_set_hm(
                self.ptr,
                int(frame_idx),
                ffi.from_buffer("FP_TYPE[]", ampl_arr) if ampl_arr.size > 0 else ffi.NULL,
                ffi.from_buffer("FP_TYPE[]", phse_arr) if phse_arr.size > 0 else ffi.NULL,
                int(ampl_arr.size),
            )
        )
        if rc != 0:
            raise RuntimeError(f"llsm_py_chunk_frame_set_hm failed with code {rc}")

    def to_layer1(self, nfft: int) -> None:
        """Augment the chunk with layer-1 parameters in place."""

        self._ensure_open()
        lib.llsm_chunk_tolayer1(self.ptr, int(nfft))
        if self.ptr.py_frame_f0 != ffi.NULL:
            lib.llsm_py_chunk_refresh_f0_view(self.ptr)
        if self.ptr.py_frame_rd != ffi.NULL:
            lib.llsm_py_chunk_refresh_rd_view(self.ptr)
        if self.ptr.py_vtmagn_block != ffi.NULL:
            lib.llsm_py_chunk_refresh_vector_view(self.ptr, _LLSM_FRAME_VTMAGN, int(self.ptr.py_vtmagn_max_len))
        if self.ptr.py_vsphse_block != ffi.NULL:
            lib.llsm_py_chunk_refresh_vector_view(self.ptr, _LLSM_FRAME_VSPHSE, int(self.ptr.py_vsphse_max_len))

    def to_layer0(self) -> None:
        """Rebuild layer-0 harmonics from existing layer-1 parameters."""

        self._ensure_open()
        lib.llsm_chunk_tolayer0(self.ptr)
        if self.ptr.py_hm_ampl != ffi.NULL:
            lib.llsm_py_chunk_refresh_harmonics_view(self.ptr, int(self.ptr.py_hm_max_nhar))

    def phasesync_rps(self, layer1_based: bool = False) -> None:
        """Synchronize harmonic phases using relative phase shift."""

        self._ensure_open()
        lib.llsm_chunk_phasesync_rps(self.ptr, 1 if layer1_based else 0)

    def phasepropagate(self, sign: int = 1) -> None:
        """Add or subtract integrated F0 from harmonic phases in place."""

        self._ensure_open()
        lib.llsm_chunk_phasepropagate(self.ptr, int(sign))

    def harmonics(self, fill_value: float = 0.0) -> HarmonicsView:
        """Return all frame harmonics as dense NumPy matrices.

        Returns:
            `HarmonicsView` with ``ampl/phse`` shape ``(nfrm, max_nhar)`` and
            ``nhar`` shape ``(nfrm,)``.
        """

        self._ensure_open()
        ampl, phse, nhar = _chunk_harmonics_views(self.ptr)
        if fill_value != 0.0 and ampl.size > 0:
            mask = np.arange(ampl.shape[1], dtype=np.int32)[None, :] >= nhar[:, None]
            ampl = ampl.copy()
            phse = phse.copy()
            ampl[mask] = np.float32(fill_value)
            phse[mask] = np.float32(fill_value)
        return HarmonicsView(ampl=ampl, phse=phse, nhar=nhar)

    def set_harmonics(self, ampl: FloatArray2D, phse: FloatArray2D, nhar: Iterable[int] | None = None) -> None:
        """Replace all frame harmonics from dense NumPy matrices.

        Args:
            ampl: shape ``(nfrm, max_nhar)``; axis 1 is harmonic index.
            phse: shape ``(nfrm, max_nhar)``; axis 1 is harmonic index.
            nhar: optional shape ``(nfrm,)`` valid harmonic counts per frame.
        """

        self._ensure_open()
        ampl_arr = np.asarray(ampl, dtype=FP_DTYPE)
        phse_arr = np.asarray(phse, dtype=FP_DTYPE)
        if ampl_arr.shape != phse_arr.shape:
            raise ValueError("ampl/phse shape mismatch")
        if ampl_arr.ndim != 2:
            raise ValueError("ampl/phse must be 2-D")
        if ampl_arr.shape[0] != self.nfrm:
            raise ValueError(f"frame count mismatch: expected {self.nfrm}, got {ampl_arr.shape[0]}")
        if nhar is None:
            nhar_arr = np.full(self.nfrm, ampl_arr.shape[1], dtype=np.int32)
        else:
            nhar_arr = as_i32_array(nhar, "nhar")
            if nhar_arr.size != self.nfrm:
                raise ValueError(f"nhar size mismatch: expected {self.nfrm}, got {nhar_arr.size}")
        nhar_arr = np.maximum(nhar_arr, 0).astype(np.int32, copy=False)
        rc = int(
            lib.llsm_py_chunk_set_harmonics_matrix(
                self.ptr,
                ffi.from_buffer("FP_TYPE[]", np.ascontiguousarray(ampl_arr).reshape(-1)),
                ffi.from_buffer("FP_TYPE[]", np.ascontiguousarray(phse_arr).reshape(-1)),
                ffi.from_buffer("int[]", nhar_arr),
                self.nfrm,
                ampl_arr.shape[1],
            )
        )
        if rc != 0:
            raise RuntimeError(f"llsm_py_chunk_set_harmonics_matrix failed with code {rc}")

    def clear_member(self, member_idx: int, mask: Iterable[int] | Iterable[bool] | None = None) -> None:
        """Clear one frame member across selected frames."""

        self._ensure_open()
        mask_arr = _normalize_frame_mask(mask, self.nfrm)
        mask_u8 = np.ascontiguousarray(mask_arr.astype(np.uint8))
        rc = int(
            lib.llsm_py_chunk_clear_member_mask(
                self.ptr, int(member_idx), ffi.from_buffer("unsigned char[]", mask_u8), mask_u8.size
            )
        )
        if rc != 0:
            raise RuntimeError(f"llsm_py_chunk_clear_member_mask failed with code {rc}")
        if member_idx == _LLSM_FRAME_HM and self.ptr.py_hm_ampl != ffi.NULL:
            lib.llsm_py_chunk_refresh_harmonics_view(self.ptr, int(self.ptr.py_hm_max_nhar))
        if member_idx == _LLSM_FRAME_RD and self.ptr.py_frame_rd != ffi.NULL:
            lib.llsm_py_chunk_refresh_rd_view(self.ptr)
        if member_idx == _LLSM_FRAME_VTMAGN and self.ptr.py_vtmagn_block != ffi.NULL:
            lib.llsm_py_chunk_refresh_vector_view(self.ptr, _LLSM_FRAME_VTMAGN, int(self.ptr.py_vtmagn_max_len))
        if member_idx == _LLSM_FRAME_VSPHSE and self.ptr.py_vsphse_block != ffi.NULL:
            lib.llsm_py_chunk_refresh_vector_view(self.ptr, _LLSM_FRAME_VSPHSE, int(self.ptr.py_vsphse_max_len))

    def enable_pulse_by_pulse(
        self, mask: Iterable[int] | Iterable[bool], clear_harmonics: bool = True
    ) -> None:
        """Enable pulse-by-pulse synthesis on selected frames."""

        self._ensure_open()
        mask_arr = _normalize_frame_mask(mask, self.nfrm)
        mask_u8 = np.ascontiguousarray(mask_arr.astype(np.uint8))
        rc = int(
            lib.llsm_py_chunk_enable_pbp_mask(
                self.ptr, ffi.from_buffer("unsigned char[]", mask_u8), mask_u8.size, 1 if clear_harmonics else 0
            )
        )
        if rc != 0:
            raise RuntimeError(f"llsm_py_chunk_enable_pbp_mask failed with code {rc}")
        if clear_harmonics and self.ptr.py_hm_ampl != ffi.NULL:
            lib.llsm_py_chunk_refresh_harmonics_view(self.ptr, int(self.ptr.py_hm_max_nhar))

    def frame_scalar(self, member_idx: int, default: float = np.nan) -> FloatArray1D:
        """Read a scalar frame member across the whole chunk.

        Returns a ``float32`` vector with shape ``(nfrm,)`` where axis 0 is
        frame index over time.
        """

        self._ensure_open()
        out = np.empty(self.nfrm, dtype=FP_DTYPE)
        rc = int(
            lib.llsm_py_chunk_get_scalar(
                self.ptr, int(member_idx), np.float32(default), ffi.from_buffer("FP_TYPE[]", out), out.size
            )
        )
        if rc != 0:
            raise RuntimeError(f"llsm_py_chunk_get_scalar failed with code {rc}")
        return out

    def set_frame_scalar(self, member_idx: int, values: Iterable[float]) -> None:
        """Set a scalar frame member across the whole chunk."""

        self._ensure_open()
        arr = as_f32_array(values, "values")
        if arr.size != self.nfrm:
            raise ValueError(f"values size mismatch: expected {self.nfrm}, got {arr.size}")
        rc = int(
            lib.llsm_py_chunk_set_scalar(self.ptr, int(member_idx), ffi.from_buffer("FP_TYPE[]", arr), arr.size)
        )
        if rc != 0:
            raise RuntimeError(f"llsm_py_chunk_set_scalar failed with code {rc}")

    def resample_linear_f0(self, nfrm_new: int) -> "Chunk":
        """Create a new chunk with linearly resampled frame count and F0."""

        self._ensure_open()
        nfrm_new = int(nfrm_new)
        if nfrm_new <= 0:
            raise ValueError("nfrm_new must be positive")
        old_nfrm = self.nfrm
        conf_new = lib.llsm_copy_container(self.ptr.conf)
        if conf_new == ffi.NULL:
            raise MemoryError("llsm_copy_container returned NULL")
        try:
            nfrm_ptr = lib.llsm_create_int(nfrm_new)
            container_attach(
                conf_new,
                _LLSM_CONF_NFRM,
                ffi.cast("void*", nfrm_ptr),
                ffi.cast("llsm_fdestructor", lib.llsm_delete_int),
                ffi.cast("llsm_fcopy", lib.llsm_copy_int),
            )
            chunk_new = lib.llsm_create_chunk(conf_new, 0)
        finally:
            lib.llsm_delete_container(conf_new)
        if chunk_new == ffi.NULL:
            raise MemoryError("llsm_create_chunk returned NULL")
        for i in range(nfrm_new):
            mapped = i * max(0, old_nfrm - 1) / max(1, nfrm_new - 1)
            base = int(mapped)
            nxt = min(base + 1, max(0, old_nfrm - 1))
            ratio = float(mapped - base)
            fr = lib.llsm_copy_container(self.ptr.frames[base])
            f0_base = ffi.cast("FP_TYPE*", lib.llsm_container_get(self.ptr.frames[base], _LLSM_FRAME_F0))
            f0_nxt = ffi.cast("FP_TYPE*", lib.llsm_container_get(self.ptr.frames[nxt], _LLSM_FRAME_F0))
            if f0_base != ffi.NULL and f0_nxt != ffi.NULL:
                f0_new = lib.llsm_create_fp((1.0 - ratio) * float(f0_base[0]) + ratio * float(f0_nxt[0]))
                container_attach(
                    fr,
                    _LLSM_FRAME_F0,
                    ffi.cast("void*", f0_new),
                    ffi.cast("llsm_fdestructor", lib.llsm_delete_fp),
                    ffi.cast("llsm_fcopy", lib.llsm_copy_fp),
                )
            chunk_new.frames[i] = fr
        return Chunk(chunk_new)

    def pitch_shift_layer1(
        self, ratio: float, compensate_vtmagn_db: bool = True, clear_harmonics: bool = True
    ) -> None:
        """Apply an in-place pitch shift to existing layer-1 parameters."""

        self._ensure_open()
        ratio = float(ratio)
        if ratio <= 0.0:
            raise ValueError("ratio must be positive")
        rc = int(
            lib.llsm_py_chunk_pitch_shift_layer1(
                self.ptr, ratio, 1 if compensate_vtmagn_db else 0, 1 if clear_harmonics else 0
            )
        )
        if rc != 0:
            raise RuntimeError(f"llsm_py_chunk_pitch_shift_layer1 failed with code {rc}")


class Coder:
    """Lossy encoder/decoder for fixed-dimensional LLSM feature vectors."""

    def __init__(self, conf_or_chunk, order_spec: int = 64, order_bap: int = 5, ptr=None, owns_memory: bool = True):
        if ptr is None:
            if isinstance(conf_or_chunk, _FeatureContainer):
                conf = conf_or_chunk.ptr.conf
            elif isinstance(conf_or_chunk, Chunk):
                conf = conf_or_chunk.ptr.conf
            else:
                conf = conf_or_chunk
            ptr = lib.llsm_create_coder(conf, int(order_spec), int(order_bap))
        if ptr == ffi.NULL:
            raise MemoryError("llsm_create_coder returned NULL")
        self.ptr = ptr
        self._owns_memory = owns_memory
        self._closed = False

    def __enter__(self) -> "Coder":
        self._ensure_open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    @property
    def closed(self) -> bool:
        return self._closed

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Coder is closed")

    def close(self) -> None:
        """Release the owned native coder."""

        if not self._closed and self._owns_memory:
            lib.llsm_delete_coder(self.ptr)
        self._closed = True

    def encode_frame(self, frame) -> np.ndarray:
        """Encode one native LLSM frame into a feature vector."""

        self._ensure_open()
        enc = lib.llsm_coder_encode(self.ptr, frame)
        if enc == ffi.NULL:
            raise RuntimeError("llsm_coder_encode returned NULL")
        try:
            n = int(lib.llsm_fparray_length(enc))
            return np.frombuffer(ffi.buffer(enc, n * ffi.sizeof("FP_TYPE")), dtype=FP_DTYPE).copy()
        finally:
            free(enc)

    def reconstruct_chunk(
        self, chunk: Chunk | Any, decode_layer0: bool = True, decode_layer1: bool = True
    ) -> Any:
        """Reconstruct one or two chunks from coded frame features."""

        self._ensure_open()
        if not decode_layer0 and not decode_layer1:
            raise ValueError("at least one of decode_layer0/decode_layer1 must be True")
        chunk_obj = chunk if isinstance(chunk, Chunk) else Chunk(_unwrap_chunk(chunk), owns_memory=False)
        nfrm = chunk_obj.nfrm
        out0 = Chunk(lib.llsm_create_chunk(chunk_obj.ptr.conf, 0)) if decode_layer0 else None
        out1 = Chunk(lib.llsm_create_chunk(chunk_obj.ptr.conf, 0)) if decode_layer1 else None
        if decode_layer0 and out0.ptr == ffi.NULL:
            raise MemoryError("llsm_create_chunk returned NULL for layer0 decode")
        if decode_layer1 and out1.ptr == ffi.NULL:
            raise MemoryError("llsm_create_chunk returned NULL for layer1 decode")
        rc = int(
            lib.llsm_py_coder_reconstruct_chunk(
                self.ptr,
                chunk_obj.ptr,
                out0.ptr if decode_layer0 else ffi.NULL,
                out1.ptr if decode_layer1 else ffi.NULL,
            )
        )
        if rc != 0:
            raise RuntimeError(f"llsm_py_coder_reconstruct_chunk failed with code {rc}")
        if decode_layer0 and decode_layer1:
            return Layer0Features(out0), Layer1Features(out1)
        if decode_layer0:
            return Layer0Features(out0)
        return Layer1Features(out1)


class RTSynthBuffer:
    """Realtime synthesis helper backed by ``llsm_rtsynth_buffer``."""

    def __init__(self, options, conf_or_chunk, capacity_samples: int = 4096, ptr=None, owns_memory: bool = True):
        self._owned_options = None
        if ptr is None:
            if isinstance(conf_or_chunk, Layer1Features):
                opt_obj = _copy_synthesis_options(options, use_l1=True)
                opt_ptr = opt_obj.ptr
                self._owned_options = opt_obj
            else:
                opt_ptr = _unwrap_synthesis_options(options)
            if isinstance(conf_or_chunk, _FeatureContainer):
                conf_ptr = conf_or_chunk.ptr.conf
            elif isinstance(conf_or_chunk, Chunk):
                conf_ptr = conf_or_chunk.ptr.conf
            else:
                conf_ptr = conf_or_chunk
            ptr = lib.llsm_create_rtsynth_buffer(opt_ptr, conf_ptr, int(capacity_samples))
        if ptr == ffi.NULL:
            raise MemoryError("llsm_create_rtsynth_buffer returned NULL")
        self.ptr = ptr
        self._owns_memory = owns_memory
        self._closed = False

    def __enter__(self) -> "RTSynthBuffer":
        self._ensure_open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    @property
    def closed(self) -> bool:
        return self._closed

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("RTSynthBuffer is closed")

    def close(self) -> None:
        """Release the owned realtime synthesis buffer."""

        if not self._closed and self._owns_memory:
            lib.llsm_delete_rtsynth_buffer(self.ptr)
        if self._owned_options is not None:
            self._owned_options.close()
            self._owned_options = None
        self._closed = True

    @property
    def latency(self) -> int:
        """Current output latency, in samples."""

        self._ensure_open()
        return int(lib.llsm_rtsynth_buffer_getlatency(self.ptr))

    def clear(self) -> None:
        """Reset the realtime synthesis buffer state."""

        self._ensure_open()
        lib.llsm_rtsynth_buffer_clear(self.ptr)

    def render_chunk_decomposed(self, chunk: Chunk | Any, nx: int, trim_latency: bool = True) -> dict[str, np.ndarray]:
        """Render periodic and aperiodic components for a chunk.

        Returns a dictionary with ``p``, ``ap`` and ``y`` arrays.
        """

        self._ensure_open()
        chunk_obj = chunk if isinstance(chunk, Chunk) else Chunk(_unwrap_chunk(chunk), owns_memory=False)
        nx = int(nx)
        if nx <= 0:
            raise ValueError("nx must be positive")
        y_p = np.zeros(nx, dtype=FP_DTYPE)
        y_ap = np.zeros(nx, dtype=FP_DTYPE)
        ny_ptr = ffi.new("int *")
        rc = int(
            lib.llsm_py_rtsynth_render_chunk_decomposed(
                self.ptr,
                chunk_obj.ptr,
                ffi.from_buffer("FP_TYPE[]", y_p),
                ffi.from_buffer("FP_TYPE[]", y_ap),
                nx,
                1 if trim_latency else 0,
                ny_ptr,
            )
        )
        if rc != 0:
            raise RuntimeError(f"llsm_py_rtsynth_render_chunk_decomposed failed with code {rc}")
        y = y_p + y_ap
        if trim_latency:
            end = int(ny_ptr[0])
            y_p = y_p[:end]
            y_ap = y_ap[:end]
            y = y[:end]
        return {"p": y_p, "ap": y_ap, "y": y}


class Output:
    """Owned wrapper around a native ``llsm_output`` structure."""

    def __init__(self, ptr, owns_memory: bool = True):
        if ptr == ffi.NULL:
            raise MemoryError("llsm_output is NULL")
        self.ptr = ptr
        self._owns_memory = owns_memory
        self._closed = False

    def __enter__(self) -> "Output":
        self._ensure_open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    @property
    def closed(self) -> bool:
        return self._closed

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Output is closed")

    def close(self) -> None:
        """Release the owned native output."""

        if not self._closed and self._owns_memory:
            lib.llsm_delete_output(self.ptr)
        self._closed = True

    def _copy_signal(self, member_name: str) -> np.ndarray:
        self._ensure_open()
        ny = int(self.ptr.ny)
        return np.frombuffer(
            ffi.buffer(getattr(self.ptr, member_name), ny * ffi.sizeof("FP_TYPE")),
            dtype=FP_DTYPE,
        ).copy()

    @property
    def y(self) -> np.ndarray:
        """Return the full synthesized waveform as a standalone NumPy array."""

        return self._copy_signal("y")

    @property
    def y_sin(self) -> np.ndarray:
        """Return the periodic component as a standalone NumPy array."""

        return self._copy_signal("y_sin")

    @property
    def y_noise(self) -> np.ndarray:
        """Return the noise component as a standalone NumPy array."""

        return self._copy_signal("y_noise")

    @property
    def ny(self) -> int:
        """Return the synthesized sample count."""

        self._ensure_open()
        return int(self.ptr.ny)

    @property
    def fs(self) -> float:
        """Return the synthesis sample rate."""

        self._ensure_open()
        return float(self.ptr.fs)

    def __array__(self, dtype=None) -> np.ndarray:
        """Convert the output to its default waveform array."""

        arr = self.y
        if dtype is not None:
            return arr.astype(dtype, copy=False)
        return arr


def create_analysis_options() -> AnalysisOptions:
    """Create a default :class:`AnalysisOptions` instance."""

    return AnalysisOptions()


def create_synthesis_options(fs: float) -> SynthesisOptions:
    """Create a default :class:`SynthesisOptions` instance."""

    return SynthesisOptions(fs=float(fs))


def _analyze_native(options, x: Iterable[float], fs: float, f0: Iterable[float], return_xap: bool = False):
    """Low-level analysis entry point returning native pointers.

    Most users should call :func:`analyze_chunk` instead.
    """

    options_ptr = _unwrap_analysis_options(options)
    x_arr = as_f32_array(x, "x")
    f0_arr = as_f32_array(f0, "f0")
    x_ptr = ffi.from_buffer("FP_TYPE[]", x_arr)
    f0_ptr = ffi.from_buffer("FP_TYPE[]", f0_arr)
    xap_ptr = ffi.new("FP_TYPE **") if return_xap else ffi.NULL
    chunk = lib.llsm_analyze(options_ptr, x_ptr, x_arr.size, float(fs), f0_ptr, f0_arr.size, xap_ptr)
    if chunk == ffi.NULL:
        raise RuntimeError("llsm_analyze returned NULL")
    if not return_xap:
        return chunk
    xap = copy_fp_ptr(xap_ptr[0], x_arr.size, free_after=True)
    return chunk, xap


def analyze_chunk(
    options: AnalysisOptions | Any, x: Iterable[float], fs: float, f0: Iterable[float], return_xap: bool = False
) -> Chunk | tuple[Chunk, np.ndarray]:
    """Analyze audio into a :class:`Chunk`.

    Parameters are expected as one-dimensional NumPy-compatible arrays. When
    ``return_xap`` is true, the function also returns the aperiodic waveform
    component produced by the native analyzer.
    """

    ret = _analyze_native(options, x, fs, f0, return_xap=return_xap)
    if not return_xap:
        return Chunk(ret)
    chunk_ptr, xap = ret
    return Chunk(chunk_ptr), xap


def _synthesize_native(options, chunk, return_numpy: bool = False):
    """Low-level synthesis entry point returning native pointers or arrays.

    Most users should call :func:`synthesize_output` instead.
    """

    opt_ptr = _unwrap_synthesis_options(options)
    chunk_ptr = _unwrap_chunk(chunk)
    out = lib.llsm_synthesize(opt_ptr, chunk_ptr)
    if out == ffi.NULL:
        raise RuntimeError("llsm_synthesize returned NULL")
    if return_numpy:
        return output_to_numpy(out, free_after=True)
    return out


def synthesize(options: SynthesisOptions | Any, chunk: Chunk | Any) -> Output:
    """Synthesize audio from a chunk into an :class:`Output` object."""

    use_l1 = isinstance(chunk, Layer1Features)
    if isinstance(options, SynthesisOptions):
        old_use_l1 = int(options.use_l1)
        if use_l1:
            options.use_l1 = 1
        try:
            out_ptr = _synthesize_native(options, chunk, return_numpy=False)
        finally:
            if use_l1:
                options.use_l1 = old_use_l1
    else:
        opt = _copy_synthesis_options(options, use_l1=use_l1)
        try:
            out_ptr = _synthesize_native(opt, chunk, return_numpy=False)
        finally:
            opt.close()
    return Output(out_ptr)


synthesize_output = synthesize


def analyze(
    options: AnalysisOptions | Any, x: Iterable[float], fs: float, f0: Iterable[float], return_xap: bool = False
) -> Layer0Features | tuple[Layer0Features, np.ndarray]:
    """Analyze audio into a :class:`Layer0Features` container."""

    ret = analyze_chunk(options, x, fs, f0, return_xap=return_xap)
    if not return_xap:
        return Layer0Features(ret)
    chunk_obj, xap = ret
    return Layer0Features(chunk_obj), xap


def to_layer1(features: Layer0Features | Chunk | Any, nfft: int) -> Layer1Features:
    """Derive a layer-1 feature container from layer-0 features."""

    ptr = _copy_chunk_ptr(_unwrap_chunk(features))
    lib.llsm_chunk_tolayer1(ptr, int(nfft))
    return Layer1Features(ptr)


def to_layer0(features: Layer1Features | Chunk | Any) -> Layer0Features:
    """Rebuild a layer-0 feature container from layer-1 features."""

    ptr = _copy_chunk_ptr(_unwrap_chunk(features))
    lib.llsm_chunk_tolayer0(ptr)
    return Layer0Features(ptr)


def create_coder(conf_or_chunk, order_spec: int = 64, order_bap: int = 5) -> Coder:
    """Create a :class:`Coder` for a configuration or chunk."""

    return Coder(conf_or_chunk, order_spec=order_spec, order_bap=order_bap)


def create_rtsynth_buffer(options, conf_or_chunk, capacity_samples: int = 4096) -> RTSynthBuffer:
    """Create a :class:`RTSynthBuffer` for realtime synthesis."""

    return RTSynthBuffer(options, conf_or_chunk, capacity_samples=capacity_samples)


def warp_frequency(fmin: float, fmax: float, n: int, warp_const: float) -> np.ndarray:
    """Compute a warped frequency axis."""

    out_ptr = lib.llsm_warp_frequency(float(fmin), float(fmax), int(n), float(warp_const))
    return copy_fp_ptr(out_ptr, int(n), free_after=True)


def spectral_mean(spectrum: Iterable[float], fnyq: float, freq_axis: Iterable[float]) -> np.ndarray:
    """Compute spectral means over a target frequency axis."""

    spec_arr = as_f32_array(spectrum, "spectrum")
    freq_arr = as_f32_array(freq_axis, "freq_axis")
    out_ptr = lib.llsm_spectral_mean(
        ffi.from_buffer("FP_TYPE[]", spec_arr),
        spec_arr.size,
        float(fnyq),
        ffi.from_buffer("FP_TYPE[]", freq_arr),
        freq_arr.size,
    )
    return copy_fp_ptr(out_ptr, freq_arr.size, free_after=True)


def spectrum_from_envelope(freq_axis: Iterable[float], env: Iterable[float], nspec: int, fnyq: float) -> np.ndarray:
    """Resample an envelope defined on ``freq_axis`` to an FFT spectrum grid."""

    freq_arr = as_f32_array(freq_axis, "freq_axis")
    env_arr = as_f32_array(env, "env")
    if freq_arr.size != env_arr.size:
        raise ValueError("freq_axis/env size mismatch")
    out_ptr = lib.llsm_spectrum_from_envelope(
        ffi.from_buffer("FP_TYPE[]", freq_arr),
        ffi.from_buffer("FP_TYPE[]", env_arr),
        freq_arr.size,
        int(nspec),
        float(fnyq),
    )
    return copy_fp_ptr(out_ptr, int(nspec), free_after=True)


def harmonic_analysis(
    x: Iterable[float],
    fs: float,
    f0: Iterable[float],
    thop: float,
    rel_winsize: float = 4.0,
    maxnhar: int = 120,
    method: int = 0,
    pad_value: float = 0.0,
) -> dict[str, np.ndarray]:
    """Run batch harmonic analysis and return dense result matrices."""

    x_arr = as_f32_array(x, "x")
    f0_arr = as_f32_array(f0, "f0")
    nfrm = int(f0_arr.size)
    max_nhar = int(max(0, maxnhar))
    nhar = np.zeros(nfrm, dtype=np.int32)
    ampl = np.full((nfrm, max_nhar), np.float32(pad_value), dtype=FP_DTYPE)
    phse = np.full((nfrm, max_nhar), np.float32(pad_value), dtype=FP_DTYPE)
    rc = int(
        lib.llsm_py_harmonic_analysis_matrix(
            ffi.from_buffer("FP_TYPE[]", x_arr),
            x_arr.size,
            float(fs),
            ffi.from_buffer("FP_TYPE[]", f0_arr),
            nfrm,
            float(thop),
            float(rel_winsize),
            int(maxnhar),
            int(method),
            float(pad_value),
            ffi.from_buffer("FP_TYPE[]", ampl.reshape(-1)),
            ffi.from_buffer("FP_TYPE[]", phse.reshape(-1)),
            ffi.from_buffer("int[]", nhar),
            max_nhar,
        )
    )
    if rc != 0:
        raise RuntimeError(f"llsm_py_harmonic_analysis_matrix failed with code {rc}")
    nhar = np.minimum(np.maximum(nhar, 0), max_nhar).astype(np.int32, copy=False)
    return {"ampl": ampl, "phse": phse, "nhar": nhar}


def synthesize_harmonic_frame(
    ampl: Iterable[float], phse: Iterable[float], f0: float, nx: int, use_iczt: bool = False
) -> np.ndarray:
    """Synthesize one harmonic frame directly into a waveform segment."""

    ampl_arr = as_f32_array(ampl, "ampl")
    phse_arr = as_f32_array(phse, "phse")
    if ampl_arr.size != phse_arr.size:
        raise ValueError("ampl/phse size mismatch")
    func = lib.llsm_synthesize_harmonic_frame_iczt if bool(use_iczt) else lib.llsm_synthesize_harmonic_frame
    out_ptr = func(
        ffi.from_buffer("FP_TYPE[]", ampl_arr),
        ffi.from_buffer("FP_TYPE[]", phse_arr),
        int(ampl_arr.size),
        float(f0),
        int(nx),
    )
    return copy_fp_ptr(out_ptr, int(nx), free_after=True)


def qifft(magn: Iterable[float], k: int) -> Tuple[float, float]:
    """Quadratically interpolate one spectral peak around bin ``k``."""

    magn_arr = as_f32_array(magn, "magn")
    f = ffi.new("FP_TYPE *")
    v = lib.cig_qifft(ffi.from_buffer("FP_TYPE[]", magn_arr), int(k), f)
    return float(v), float(f[0])


def spec2env(S: Iterable[float], nfft: int, f0_ratio: float, nhar: int, Cout: Iterable[float] | None = None):
    """Estimate a smooth spectral envelope from a magnitude spectrum."""

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
    """Create an LF glottal model from Rd and period parameters."""

    return lib.cig_lfmodel_from_rd(float(rd), float(T0), float(Ee))


def lfmodel_spectrum(model, freq: Iterable[float], return_phase: bool = False):
    """Evaluate the LF glottal model spectrum on a frequency axis."""

    freq_arr = as_f32_array(freq, "freq")
    phase_ptr = ffi.new("FP_TYPE[]", freq_arr.size) if return_phase else ffi.NULL
    out = lib.cig_lfmodel_spectrum(model, ffi.from_buffer("FP_TYPE[]", freq_arr), freq_arr.size, phase_ptr)
    ampl = copy_fp_ptr(out, freq_arr.size, free_after=True)
    if not return_phase:
        return ampl
    phase = np.frombuffer(ffi.buffer(phase_ptr, freq_arr.size * ffi.sizeof("FP_TYPE")), dtype=FP_DTYPE).copy()
    return ampl, phase


def lfmodel_period(model, fs: int, n: int):
    """Render one LF glottal period to a waveform array."""

    out = lib.cig_lfmodel_period(model, int(fs), int(n))
    return copy_fp_ptr(out, int(n), free_after=True)


def ifdetector_estimate(ifd, x: Iterable[float]) -> float:
    """Estimate instantaneous frequency related statistics for ``x``."""

    x_arr = as_f32_array(x, "x")
    return float(lib.cig_ifdetector_estimate(ifd, ffi.from_buffer("FP_TYPE[]", x_arr), x_arr.size))
