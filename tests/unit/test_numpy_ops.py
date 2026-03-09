from __future__ import annotations

import numpy as np

import pyllsm2 as m

from tests._sample_data import sample_layer0_arrays, sample_layer1_arrays


def test_numpy_functions_work_on_layer0_arrays() -> None:
    f0, ampl, phse, nhar = sample_layer0_arrays()
    layer0 = m.Layer0Features.from_arrays(f0, ampl, phse, nhar=nhar, fs=16000)

    logged = np.log(layer0.ampl + np.float32(1e-6))
    summed = np.sum(layer0.phse, axis=1)
    stacked = np.stack([layer0.f0, layer0.nhar.astype(np.float32)], axis=1)

    assert logged.shape == layer0.ampl.shape
    assert summed.shape == (layer0.nfrm,)
    assert stacked.shape == (layer0.nfrm, 2)
    assert np.isfinite(logged).all()


def test_numpy_functions_work_on_layer1_arrays() -> None:
    f0, rd, vtmagn, vsphse, lengths = sample_layer1_arrays()
    layer1 = m.Layer1Features.from_arrays(f0, rd, vtmagn, vsphse, fs=16000, vsphse_lengths=lengths)

    vt_mean = np.mean(layer1.vtmagn, axis=0)
    phase_sin = np.sin(layer1.vsphse)
    rd_scaled = np.clip(layer1.rd * np.float32(1.1), 0.0, 3.0)

    assert vt_mean.shape == (layer1.vtmagn.shape[1],)
    assert phase_sin.shape == layer1.vsphse.shape
    assert rd_scaled.shape == layer1.rd.shape


def test_numpy_stacking_and_masking_match_valid_lengths() -> None:
    f0, rd, vtmagn, vsphse, lengths = sample_layer1_arrays()
    layer1 = m.Layer1Features.from_arrays(f0, rd, vtmagn, vsphse, fs=16000, vsphse_lengths=lengths)

    valid_mask = np.arange(layer1.vsphse.shape[1], dtype=np.int32)[None, :] < layer1.vsphse_lengths[:, None]
    stacked = np.stack([layer1.f0, layer1.rd], axis=1)
    phase_energy = np.sum(np.square(layer1.vsphse[valid_mask]), dtype=np.float32)

    assert stacked.shape == (layer1.nfrm, 2)
    assert np.isfinite(stacked).all()
    assert float(phase_energy) >= 0.0
