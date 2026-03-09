from __future__ import annotations

import numpy as np
import pytest

import pyllsm2 as m

from tests._sample_data import mask_from_lengths, sample_layer0_arrays, sample_layer1_arrays


def test_direct_write_updates_layer0_arrays() -> None:
    f0, ampl, phse, nhar = sample_layer0_arrays()
    layer0 = m.Layer0Features.from_arrays(f0, ampl, phse, nhar=nhar, fs=16000)

    layer0.f0[:] = layer0.f0 + np.float32(5.0)
    np.testing.assert_allclose(layer0.f0, f0 + np.float32(5.0))

    layer0.ampl[:, :2] = np.float32(0.25)
    np.testing.assert_allclose(layer0.ampl[:, :2], np.full((layer0.nfrm, 2), 0.25, dtype=np.float32))

    new_phse = np.full_like(layer0.phse, 0.5, dtype=np.float32)
    layer0.phse = new_phse
    mask0 = mask_from_lengths(layer0.nhar, layer0.phse.shape[1])
    np.testing.assert_allclose(layer0.phse[mask0], new_phse[mask0])


def test_direct_write_updates_layer1_arrays() -> None:
    f0, rd, vtmagn, vsphse, lengths = sample_layer1_arrays()
    layer1 = m.Layer1Features.from_arrays(f0, rd, vtmagn, vsphse, fs=16000, vsphse_lengths=lengths)

    layer1.rd[:] = np.linspace(1.1, 1.4, layer1.nfrm, dtype=np.float32)
    np.testing.assert_allclose(layer1.rd, np.linspace(1.1, 1.4, layer1.nfrm, dtype=np.float32))

    layer1.vtmagn = np.zeros_like(layer1.vtmagn, dtype=np.float32)
    np.testing.assert_allclose(layer1.vtmagn, 0.0)

    layer1.vsphse[:, 0] = np.float32(0.75)
    np.testing.assert_allclose(layer1.vsphse[:, 0], np.full(layer1.nfrm, 0.75, dtype=np.float32))


def test_direct_write_validates_frame_aligned_assignments() -> None:
    f0, ampl, phse, nhar = sample_layer0_arrays()
    layer0 = m.Layer0Features.from_arrays(f0, ampl, phse, nhar=nhar, fs=16000)
    with pytest.raises(ValueError, match="frame count mismatch"):
        layer0.ampl = np.zeros((layer0.nfrm - 1, ampl.shape[1]), dtype=np.float32)

    f0_1, rd, vtmagn, vsphse, lengths = sample_layer1_arrays()
    layer1 = m.Layer1Features.from_arrays(f0_1, rd, vtmagn, vsphse, fs=16000, vsphse_lengths=lengths)
    with pytest.raises(ValueError, match="frame count mismatch"):
        layer1.vsphse = np.zeros((layer1.nfrm - 1, vsphse.shape[1]), dtype=np.float32)
