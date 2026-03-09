from __future__ import annotations

import numpy as np

import pyllsm2 as m

from tests._sample_data import sample_layer0_arrays


def _sample_layer0() -> m.Layer0Features:
    f0, ampl, phse, nhar = sample_layer0_arrays()
    return m.Layer0Features.from_arrays(f0, ampl, phse, nhar=nhar, fs=16000)


def test_python_objects_work_with_raw_chunk_functions() -> None:
    layer0 = _sample_layer0()

    chunk_ptr = m.raw.llsm_copy_chunk(layer0.ptr)
    assert chunk_ptr != m.raw.ffi.NULL
    assert int(m.raw.llsm_frame_checklayer0(chunk_ptr.frames[0])) == 1

    m.raw.llsm_chunk_tolayer1(chunk_ptr, 256)
    layer1 = m.Layer1Features(chunk_ptr)
    assert isinstance(layer1, m.Layer1Features)
    assert int(m.raw.llsm_frame_checklayer1(layer1.ptr.frames[0])) == 1
    assert layer1.rd.shape == (layer1.nfrm,)
    assert layer1.vtmagn.shape[0] == layer1.nfrm


def test_raw_container_access_matches_python_views() -> None:
    layer0 = _sample_layer0()
    ffi = m.raw.ffi

    nfrm_ptr = ffi.cast("int*", m.raw.llsm_container_get(layer0.ptr.conf, m.raw.LLSM_CONF_NFRM))
    assert int(nfrm_ptr[0]) == layer0.nfrm

    hm_ptr = ffi.cast("llsm_hmframe*", m.raw.llsm_container_get(layer0.ptr.frames[0], m.raw.LLSM_FRAME_HM))
    assert hm_ptr != ffi.NULL
    raw_ampl = np.frombuffer(ffi.buffer(hm_ptr.ampl, int(hm_ptr.nhar) * ffi.sizeof("FP_TYPE")), dtype=np.float32).copy()
    np.testing.assert_allclose(raw_ampl, layer0.ampl[0, : int(hm_ptr.nhar)])


def test_raw_roundtrip_preserves_ndarray_created_features() -> None:
    layer0 = _sample_layer0()

    ptr = m.raw.llsm_copy_chunk(layer0.ptr)
    assert ptr != m.raw.ffi.NULL

    m.raw.llsm_chunk_tolayer1(ptr, 256)
    m.raw.llsm_chunk_tolayer0(ptr)
    roundtrip = m.Layer0Features(ptr)

    np.testing.assert_allclose(roundtrip.f0, layer0.f0)
    np.testing.assert_array_equal(roundtrip.nhar, layer0.nhar)
    mask = np.arange(layer0.ampl.shape[1], dtype=np.int32)[None, :] < layer0.nhar[:, None]
    assert np.isfinite(roundtrip.ampl[mask]).all()
    assert np.isfinite(roundtrip.phse[mask]).all()
    assert float(np.mean(roundtrip.ampl[mask])) > 0.05
    assert float(np.max(roundtrip.ampl[mask])) > 0.1


def test_raw_synthesize_output_matches_high_level_waveform() -> None:
    layer0 = _sample_layer0()
    options = m.SynthesisOptions(16000)
    raw_out = m.raw.llsm_synthesize(options.ptr, layer0.ptr)
    assert raw_out != m.raw.ffi.NULL
    try:
        y_raw = m.raw.output_to_numpy(raw_out, free_after=False)
        y_high = np.asarray(m.synthesize(options, layer0).y, dtype=np.float32)
    finally:
        m.raw.llsm_delete_output(raw_out)

    assert y_raw.ndim == 1
    assert y_raw.size == y_high.size
    np.testing.assert_allclose(y_raw, y_high, atol=1e-6, rtol=1e-6)
