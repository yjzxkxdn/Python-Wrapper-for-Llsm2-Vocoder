from __future__ import annotations

import math

import numpy as np

import pyllsm2 as m

from ._shared import approx, ffi, lib


def test_structs_python_api() -> None:
    f0 = np.array([120.0, 121.0, 122.0], dtype=np.float32)
    ampl = np.array([[1.0, 0.5, 0.2], [1.1, 0.4, 0.1], [0.9, 0.6, 0.3]], dtype=np.float32)
    phse = np.array([[1.0, -0.5, 2.5], [1.2, -0.4, 2.4], [0.9, -0.6, 2.6]], dtype=np.float32)
    layer0 = m.Layer0Features.from_arrays(f0=f0, ampl=ampl, phse=phse, fs=22050.0)
    layer0_copy = layer0.copy()
    assert np.allclose(layer0_copy.f0, f0)
    assert np.allclose(layer0_copy.ampl, ampl)
    assert np.allclose(layer0_copy.phse, phse)
    chunk = m.Chunk.allocate_layer1(4, fs=22050.0, nspec=20, max_nhar=6)
    chunk.f0 = np.array([100.0, 110.0, 120.0, 130.0], dtype=np.float32)
    chunk.rd = np.array([1.0, 0.9, 0.8, 0.7], dtype=np.float32)
    chunk.vtmagn = np.tile(np.linspace(-5.0, 5.0, 20, dtype=np.float32), (4, 1))
    chunk.vsphse = np.tile(np.linspace(-1.0, 1.0, 6, dtype=np.float32), (4, 1))
    chunk_copy = chunk.copy()
    assert np.allclose(chunk_copy.f0, chunk.f0)
    assert np.allclose(chunk_copy.rd, chunk.rd)
    assert np.allclose(chunk_copy.vtmagn, chunk.vtmagn)
    assert np.allclose(chunk_copy.vsphse, chunk.vsphse)


def test_structs_raw_api() -> None:
    h1 = lib.llsm_create_hmframe(3)
    assert h1 != ffi.NULL
    h1.ampl[0], h1.phse[0] = 1.0, 1.0
    h1.ampl[1], h1.phse[1] = 0.5, -0.5
    h1.ampl[2], h1.phse[2] = 0.2, 2.5
    h2 = lib.llsm_copy_hmframe(h1)
    assert h2 != ffi.NULL
    assert h2.nhar == 3
    lib.llsm_hmframe_phaseshift(h2, 3.14)
    lib.llsm_hmframe_phaseshift(h2, 3.14)
    lib.llsm_hmframe_phaseshift(h2, -6.28)
    assert approx(float(h2.phse[0]), 1.0, eps=5e-3)
    assert approx(float(h2.phse[1]), -0.5, eps=5e-3)
    assert approx(float(h2.phse[2]), 2.5, eps=5e-3)
    lib.llsm_delete_hmframe(h1)
    lib.llsm_delete_hmframe(h2)

    n1 = lib.llsm_create_nmframe(3, 2, 20)
    assert n1 != ffi.NULL
    for i in range(20):
        n1.psd[i] = i - 10.0
    for i in range(3):
        n1.edc[i] = i * 0.1
        n1.eenv[i].ampl[0] = 1.0
        n1.eenv[i].ampl[1] = 0.5
    n2 = lib.llsm_copy_nmframe(n1)
    assert n2 != ffi.NULL
    assert n2.npsd == 20
    assert n2.nchannel == 3
    for i in range(20):
        assert approx(float(n2.psd[i]), i - 10.0)
    lib.llsm_delete_nmframe(n1)
    lib.llsm_delete_nmframe(n2)

    c1 = lib.llsm_create_container(10)
    assert c1 != ffi.NULL
    p0 = lib.llsm_create_fp(5.0)
    p1 = lib.llsm_create_fp(10.0)
    m.raw.container_attach(c1, 0, ffi.cast("void*", p0), ffi.cast("llsm_fdestructor", lib.llsm_delete_fp), ffi.cast("llsm_fcopy", lib.llsm_copy_fp))
    m.raw.container_attach(c1, 1, ffi.cast("void*", p1), ffi.NULL, ffi.cast("llsm_fcopy", lib.llsm_copy_fp))
    c2 = lib.llsm_copy_container(c1)
    assert c2 != ffi.NULL
    v0 = ffi.cast("FP_TYPE*", lib.llsm_container_get(c2, 0))
    v1 = ffi.cast("FP_TYPE*", lib.llsm_container_get(c2, 1))
    assert approx(float(v0[0]), 5.0)
    assert approx(float(v1[0]), 10.0)
    lib.llsm_container_remove(c1, 0)
    assert c1.members[0] == ffi.NULL
    lib.llsm_delete_fp(ffi.cast("FP_TYPE*", c1.members[1]))
    lib.llsm_delete_fp(ffi.cast("FP_TYPE*", c2.members[1]))
    lib.llsm_delete_container(c1)
    lib.llsm_delete_container(c2)

    opt = lib.llsm_create_aoptions()
    conf = lib.llsm_aoptions_toconf(opt, 22050.0)
    assert conf != ffi.NULL
    nfrm_ptr = ffi.cast("int*", lib.llsm_container_get(conf, m.raw.LLSM_CONF_NFRM))
    nfrm_ptr[0] = 40
    npsd = int(ffi.cast("int*", lib.llsm_container_get(conf, m.raw.LLSM_CONF_NPSD))[0])
    chunk1 = lib.llsm_create_chunk(conf, 1)
    assert chunk1 != ffi.NULL
    for frame_idx in range(40):
        frame = chunk1.frames[frame_idx]
        nm = ffi.cast("llsm_nmframe*", lib.llsm_container_get(frame, m.raw.LLSM_FRAME_NM))
        for j in range(npsd):
            nm.psd[j] = math.sin(j * 0.1)
    chunk2 = lib.llsm_copy_chunk(chunk1)
    assert chunk2 != ffi.NULL
    for frame_idx in range(40):
        frame = chunk2.frames[frame_idx]
        nm = ffi.cast("llsm_nmframe*", lib.llsm_container_get(frame, m.raw.LLSM_FRAME_NM))
        for j in range(npsd):
            assert approx(float(nm.psd[j]), math.sin(j * 0.1), eps=1e-5)
    lib.llsm_delete_chunk(chunk1)
    lib.llsm_delete_chunk(chunk2)
    lib.llsm_delete_container(conf)
    lib.llsm_delete_aoptions(opt)