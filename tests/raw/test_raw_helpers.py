from __future__ import annotations

import numpy as np
import pytest

import pyllsm2 as m


def test_raw_array_helpers_cast_dtype_and_make_contiguous() -> None:
    f32 = m.raw.as_f32_array(np.array([1.0, 2.0, 3.0], dtype=np.float64), "f32")
    i32 = m.raw.as_i32_array(np.array([1.0, 2.0, 3.0], dtype=np.float64), "i32")

    assert f32.dtype == np.float32
    assert i32.dtype == np.int32
    assert f32.flags.c_contiguous
    assert i32.flags.c_contiguous
    np.testing.assert_allclose(f32, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(i32, [1, 2, 3])


def test_raw_array_helpers_reject_non_1d_inputs() -> None:
    with pytest.raises(ValueError, match="must be 1-D"):
        m.raw.as_f32_array(np.zeros((2, 2), dtype=np.float32), "bad")

    with pytest.raises(ValueError, match="must be 1-D"):
        m.raw.as_i32_array(np.zeros((2, 2), dtype=np.int32), "bad")


def test_raw_char_buffer_accepts_ascii_text_and_bytes() -> None:
    assert bytes(m.raw.ffi.string(m.raw.as_char_array("abc"))) == b"abc"
    assert bytes(m.raw.ffi.string(m.raw.as_char_array(b"xyz"))) == b"xyz"


def test_raw_copy_helpers_copy_and_detach_from_native_memory() -> None:
    fp_ptr = m.raw.ffi.new("FP_TYPE[]", [0.25, 0.5, 0.75])
    int_ptr = m.raw.ffi.new("int[]", [2, 4, 6])

    fp_copy = m.raw.copy_fp_ptr(fp_ptr, 3, free_after=False)
    int_copy = m.raw.copy_int_ptr(int_ptr, 3, free_after=False)

    fp_ptr[0] = np.float32(9.0)
    int_ptr[0] = 99
    np.testing.assert_allclose(fp_copy, [0.25, 0.5, 0.75])
    np.testing.assert_array_equal(int_copy, [2, 4, 6])


def test_raw_copy_helpers_raise_on_null_pointer() -> None:
    with pytest.raises(MemoryError, match="NULL pointer"):
        m.raw.copy_fp_ptr(m.raw.ffi.NULL, 4)

    with pytest.raises(MemoryError, match="NULL pointer"):
        m.raw.copy_int_ptr(m.raw.ffi.NULL, 4)


def test_raw_container_attach_roundtrip_with_integer_payload() -> None:
    container = m.raw.llsm_create_container(10)
    assert container != m.raw.ffi.NULL
    try:
        value_ptr = m.raw.llsm_create_fp(np.float32(1.25))
        m.raw.container_attach(
            container,
            0,
            m.raw.ffi.cast("void*", value_ptr),
            m.raw.ffi.cast("llsm_fdestructor", m.raw.llsm_delete_fp),
            m.raw.ffi.cast("llsm_fcopy", m.raw.llsm_copy_fp),
        )
        got_ptr = m.raw.ffi.cast("FP_TYPE*", m.raw.llsm_container_get(container, 0))
        assert float(got_ptr[0]) == pytest.approx(1.25, abs=1e-6)
    finally:
        m.raw.llsm_delete_container(container)
