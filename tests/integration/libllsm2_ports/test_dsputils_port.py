from __future__ import annotations

import numpy as np

import pyllsm2 as m

from ._shared import ffi, lib, make_spectrum, raw_harmonic_analysis_matrix, synthetic_harmonic_case, verify_harmonic_analysis_result


def test_dsputils_python_api() -> None:
    fnyq = 22050.0
    spec1 = make_spectrum(1024, use_db=False, no_modulation=False)
    mod1 = make_spectrum(1024, use_db=False, no_modulation=True)
    warp_axis = m.warp_frequency(0.0, fnyq, 100, 15000.0)
    env = m.spectral_mean(spec1, fnyq, warp_axis)
    spec2 = m.spectrum_from_envelope(warp_axis, env, spec1.size, fnyq)
    err = np.abs(spec1 - spec2)
    assert np.all(err < (mod1 + 1e-4))
    x, fs, f0, thop, _, ampl0_truth = synthetic_harmonic_case()
    for method in (m.raw.LLSM_AOPTION_HMPP, m.raw.LLSM_AOPTION_HMCZT):
        h = m.harmonic_analysis(x, fs, f0, thop, rel_winsize=4.0, maxnhar=3, method=int(method))
        verify_harmonic_analysis_result(h, f0, thop, ampl0_truth)


def test_dsputils_raw_api() -> None:
    fnyq = 22050.0
    spec1 = make_spectrum(1024, use_db=False, no_modulation=False)
    mod1 = make_spectrum(1024, use_db=False, no_modulation=True)
    warp_axis = m.raw.copy_fp_ptr(lib.llsm_warp_frequency(0.0, fnyq, 100, 15000.0), 100, free_after=True)
    env = m.raw.copy_fp_ptr(
        lib.llsm_spectral_mean(
            ffi.from_buffer("FP_TYPE[]", spec1),
            spec1.size,
            fnyq,
            ffi.from_buffer("FP_TYPE[]", warp_axis),
            warp_axis.size,
        ),
        warp_axis.size,
        free_after=True,
    )
    spec2 = m.raw.copy_fp_ptr(
        lib.llsm_spectrum_from_envelope(
            ffi.from_buffer("FP_TYPE[]", warp_axis),
            ffi.from_buffer("FP_TYPE[]", env),
            warp_axis.size,
            spec1.size,
            fnyq,
        ),
        spec1.size,
        free_after=True,
    )
    err = np.abs(spec1 - spec2)
    assert np.all(err < (mod1 + 1e-4))
    x, fs, f0, thop, _, ampl0_truth = synthetic_harmonic_case()
    for method in (m.raw.LLSM_AOPTION_HMPP, m.raw.LLSM_AOPTION_HMCZT):
        h = raw_harmonic_analysis_matrix(x, fs, f0, thop, method)
        verify_harmonic_analysis_result(h, f0, thop, ampl0_truth)
    nhar = 20
    nparam = 64
    param_list = np.linspace(0.02, 3.0, nparam, dtype=np.float32)
    cgm = lib.llsm_create_cached_glottal_model(ffi.from_buffer("FP_TYPE[]", param_list), nparam, nhar)
    assert cgm != ffi.NULL
    try:
        f0_ref = 200.0
        freq = (f0_ref * (np.arange(nhar, dtype=np.float32) + 1.0)).astype(np.float32)
        for step in range(0, 120, 12):
            test_param = 0.3 + (2.5 - 0.3) / 120.0 * step
            lf = lib.cig_lfmodel_from_rd(test_param, 1.0 / f0_ref, 1.0)
            ampl_ptr = lib.cig_lfmodel_spectrum(lf, ffi.from_buffer("FP_TYPE[]", freq), freq.size, ffi.NULL)
            ampl = m.raw.copy_fp_ptr(ampl_ptr, freq.size, free_after=True)
            ampl = (ampl / (np.arange(nhar, dtype=np.float32) + 1.0) * (0.5 + 0.02 * step)).astype(np.float32)
            est = float(lib.llsm_spectral_glottal_fitting(ffi.from_buffer("FP_TYPE[]", ampl), nhar, cgm))
            assert abs(est - test_param) < 0.05
    finally:
        lib.llsm_delete_cached_glottal_model(cgm)