import math

import pyllsm2 as m


ffi = m.ffi
lib = m.lib


def _approx(a: float, b: float, eps: float = 1e-3) -> bool:
    return abs(a - b) <= eps * max(1.0, abs(a), abs(b))


def test_hmframe_copy_and_phaseshift():
    h1 = lib.llsm_create_hmframe(3)
    assert h1 != ffi.NULL
    h1.ampl[0], h1.phse[0] = 1.0, 1.0
    h1.ampl[1], h1.phse[1] = 0.5, -0.5
    h1.ampl[2], h1.phse[2] = 0.2, 2.5

    h2 = lib.llsm_copy_hmframe(h1)
    assert h2 != ffi.NULL
    assert h2.nhar == 3
    assert _approx(float(h2.ampl[2]), 0.2)
    assert _approx(float(h2.phse[2]), 2.5)

    lib.llsm_hmframe_phaseshift(h2, 3.14)
    lib.llsm_hmframe_phaseshift(h2, 3.14)
    lib.llsm_hmframe_phaseshift(h2, -6.28)
    assert _approx(float(h2.phse[0]), 1.0, eps=5e-3)
    assert _approx(float(h2.phse[1]), -0.5, eps=5e-3)
    assert _approx(float(h2.phse[2]), 2.5, eps=5e-3)

    lib.llsm_delete_hmframe(h1)
    lib.llsm_delete_hmframe(h2)


def test_nmframe_copy():
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
        assert _approx(float(n2.psd[i]), i - 10.0)
    for i in range(3):
        assert _approx(float(n2.edc[i]), i * 0.1)
        assert _approx(float(n2.eenv[i].ampl[0]), 1.0)
        assert _approx(float(n2.eenv[i].ampl[1]), 0.5)

    lib.llsm_delete_nmframe(n1)
    lib.llsm_delete_nmframe(n2)


def test_container_attach_copy_remove():
    c1 = lib.llsm_create_container(10)
    assert c1 != ffi.NULL

    p0 = lib.llsm_create_fp(5.0)
    dtor_fp = ffi.cast("llsm_fdestructor", lib.llsm_delete_fp)
    copy_fp = ffi.cast("llsm_fcopy", lib.llsm_copy_fp)
    m.container_attach(c1, 0, ffi.cast("void*", p0), dtor_fp, copy_fp)

    p1 = lib.llsm_create_fp(10.0)
    m.container_attach(c1, 1, ffi.cast("void*", p1), ffi.NULL, copy_fp)

    c2 = lib.llsm_copy_container(c1)
    assert c2 != ffi.NULL

    v0 = ffi.cast("FP_TYPE*", lib.llsm_container_get(c2, 0))
    v1 = ffi.cast("FP_TYPE*", lib.llsm_container_get(c2, 1))
    assert _approx(float(v0[0]), 5.0)
    assert _approx(float(v1[0]), 10.0)

    lib.llsm_container_remove(c1, 0)
    assert c1.members[0] == ffi.NULL

    # member 1 has no destructor in containers, free manually.
    lib.llsm_delete_fp(ffi.cast("FP_TYPE*", c1.members[1]))
    lib.llsm_delete_fp(ffi.cast("FP_TYPE*", c2.members[1]))

    lib.llsm_delete_container(c1)
    lib.llsm_delete_container(c2)
