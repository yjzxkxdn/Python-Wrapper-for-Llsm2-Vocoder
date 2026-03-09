from __future__ import annotations

import numpy as np
import pytest

import pyllsm2 as m

from tests._sample_data import mask_from_lengths, sample_layer0_arrays, sample_layer1_arrays


def test_create_layer0_and_layer1_from_feature_objects() -> None:
    f0, ampl, phse, nhar = sample_layer0_arrays()

    layer0 = m.Layer0Features.from_arrays(f0, ampl, phse, nhar=nhar, fs=16000)
    assert isinstance(layer0, m.Layer0Features)
    np.testing.assert_allclose(layer0.f0, f0)
    np.testing.assert_array_equal(layer0.nhar, nhar)
    mask0 = mask_from_lengths(nhar, ampl.shape[1])
    np.testing.assert_allclose(layer0.ampl[mask0], ampl[mask0])

    layer1 = layer0.to_layer1(256)
    assert isinstance(layer1, m.Layer1Features)
    assert layer1.rd.shape == (f0.size,)
    assert layer1.vtmagn.shape[0] == f0.size
    assert layer1.vsphse.shape[0] == f0.size
    assert np.isfinite(layer1.rd).all()


def test_create_layer0_and_layer1_directly_from_ndarrays() -> None:
    f0_0, ampl, phse, nhar = sample_layer0_arrays()
    layer0 = m.Layer0Features.from_arrays(f0_0, ampl, phse, nhar=nhar, fs=16000)
    mask0 = mask_from_lengths(nhar, ampl.shape[1])
    np.testing.assert_allclose(layer0.ampl[mask0], ampl[mask0])
    np.testing.assert_allclose(layer0.phse[mask0], phse[mask0])

    f0_1, rd, vtmagn, vsphse, lengths = sample_layer1_arrays()
    layer1 = m.Layer1Features.from_arrays(f0_1, rd, vtmagn, vsphse, fs=16000, vsphse_lengths=lengths)
    mask1 = mask_from_lengths(lengths, vsphse.shape[1])
    np.testing.assert_allclose(layer1.f0, f0_1)
    np.testing.assert_allclose(layer1.rd, rd)
    np.testing.assert_allclose(layer1.vtmagn, vtmagn)
    np.testing.assert_allclose(layer1.vsphse[mask1], vsphse[mask1])
    np.testing.assert_array_equal(layer1.vsphse_lengths, lengths)


def test_chunk_allocate_layer0_and_layer1_support_direct_ndarray_properties() -> None:
    f0_0, ampl, phse, nhar = sample_layer0_arrays()
    chunk0 = m.Chunk.allocate_layer0(f0_0.size, fs=16000, max_nhar=ampl.shape[1])
    chunk0.f0 = f0_0
    chunk0.ampl = ampl
    chunk0.phse = phse
    chunk0.nhar = nhar
    mask0 = mask_from_lengths(nhar, ampl.shape[1])
    np.testing.assert_allclose(np.log(chunk0.ampl[mask0] + np.float32(1e-6)), np.log(ampl[mask0] + np.float32(1e-6)))
    np.testing.assert_allclose(chunk0.phse[mask0], phse[mask0])

    f0_1, rd, vtmagn, vsphse, lengths = sample_layer1_arrays()
    chunk1 = m.Chunk.allocate_layer1(f0_1.size, fs=16000, nspec=vtmagn.shape[1], max_nhar=vsphse.shape[1])
    chunk1.f0 = f0_1
    chunk1.rd = rd
    chunk1.vtmagn = vtmagn
    chunk1.vsphse = vsphse
    chunk1.vsphse_lengths = lengths
    mask1 = mask_from_lengths(lengths, vsphse.shape[1])
    np.testing.assert_allclose(np.mean(chunk1.vtmagn, axis=0), np.mean(vtmagn, axis=0))
    np.testing.assert_allclose(chunk1.vsphse[mask1], vsphse[mask1])
    np.testing.assert_array_equal(chunk1.vsphse_lengths, lengths)


def test_layer_copies_are_independent() -> None:
    f0, ampl, phse, nhar = sample_layer0_arrays()
    layer0 = m.Layer0Features.from_arrays(f0, ampl, phse, nhar=nhar, fs=16000)
    clone0 = layer0.copy()
    clone0.f0[:] += np.float32(10.0)
    assert not np.allclose(layer0.f0, clone0.f0)

    f0_1, rd, vtmagn, vsphse, lengths = sample_layer1_arrays()
    layer1 = m.Layer1Features.from_arrays(f0_1, rd, vtmagn, vsphse, fs=16000, vsphse_lengths=lengths)
    clone1 = layer1.copy()
    clone1.rd[:] *= np.float32(0.8)
    assert not np.allclose(layer1.rd, clone1.rd)


def test_layer_creation_rejects_mismatched_shapes() -> None:
    f0, ampl, phse, nhar = sample_layer0_arrays()
    with pytest.raises(ValueError, match="ampl/phse shape mismatch"):
        m.Layer0Features.from_arrays(f0, ampl[:, :-1], phse, nhar=nhar, fs=16000)

    with pytest.raises(ValueError, match="nhar contains values larger than harmonic matrix width"):
        m.Layer0Features.from_arrays(f0, ampl, phse, nhar=nhar + 10, fs=16000)

    f0_1, rd, vtmagn, vsphse, lengths = sample_layer1_arrays()
    with pytest.raises(ValueError, match="rd size mismatch"):
        m.Layer1Features.from_arrays(f0_1, rd[:-1], vtmagn, vsphse, fs=16000, vsphse_lengths=lengths)

    with pytest.raises(ValueError, match="vsphse_lengths contains values larger than vsphse width"):
        m.Layer1Features.from_arrays(f0_1, rd, vtmagn, vsphse, fs=16000, vsphse_lengths=lengths + 10)


def test_chunk_allocate_rejects_invalid_dimensions() -> None:
    with pytest.raises(ValueError, match="nfrm must be positive"):
        m.Chunk.allocate_layer0(0, fs=16000, max_nhar=4)

    with pytest.raises(ValueError, match="max_nhar must be non-negative"):
        m.Chunk.allocate_layer0(3, fs=16000, max_nhar=-1)

    with pytest.raises(ValueError, match="nspec must be non-negative"):
        m.Chunk.allocate_layer1(3, fs=16000, nspec=-1, max_nhar=4)
