import numpy as np

import pyllsm2 as m


ffi = m.ffi
lib = m.lib


def _make_spectrum(size: int, use_db: bool, no_modulation: bool) -> np.ndarray:
    x = np.zeros(size, dtype=np.float32)
    for i in range(size):
        v = np.exp(-(i / size) * 10.0)
        if not no_modulation:
            v *= np.exp(np.cos(i * 0.1) * 3.0 - 3.0)
        if use_db:
            v = 20.0 * np.log10(v)
        x[i] = v
    return x


def test_spectral_envelope_reconstruction_bound():
    fnyq = 22050.0
    spec1 = _make_spectrum(1024, use_db=False, no_modulation=False)
    mod1 = _make_spectrum(1024, use_db=False, no_modulation=True)

    warp_ptr = lib.llsm_warp_frequency(0.0, fnyq, 100, 15000.0)
    warp_axis = m.copy_fp_ptr(warp_ptr, 100, free_after=True)

    env_ptr = lib.llsm_spectral_mean(
        ffi.from_buffer("FP_TYPE[]", spec1),
        spec1.size,
        fnyq,
        ffi.from_buffer("FP_TYPE[]", warp_axis),
        warp_axis.size,
    )
    env = m.copy_fp_ptr(env_ptr, warp_axis.size, free_after=True)

    spec2_ptr = lib.llsm_spectrum_from_envelope(
        ffi.from_buffer("FP_TYPE[]", warp_axis),
        ffi.from_buffer("FP_TYPE[]", env),
        warp_axis.size,
        spec1.size,
        fnyq,
    )
    spec2 = m.copy_fp_ptr(spec2_ptr, spec1.size, free_after=True)

    err = np.abs(spec1 - spec2)
    assert np.all(err < (mod1 + 1e-5))


def test_harmonic_iczt_is_close_to_recurrent():
    rng = np.random.default_rng(1)
    ampl = rng.normal(0, 1.0, 100).astype(np.float32)
    phse = rng.normal(0, 100.0, 100).astype(np.float32)

    y1_ptr = lib.llsm_synthesize_harmonic_frame_iczt(
        ffi.from_buffer("FP_TYPE[]", ampl),
        ffi.from_buffer("FP_TYPE[]", phse),
        ampl.size,
        0.01,
        1024,
    )
    y2_ptr = lib.llsm_synthesize_harmonic_frame(
        ffi.from_buffer("FP_TYPE[]", ampl),
        ffi.from_buffer("FP_TYPE[]", phse),
        ampl.size,
        0.01,
        1024,
    )
    y1 = m.copy_fp_ptr(y1_ptr, 1024, free_after=True)
    y2 = m.copy_fp_ptr(y2_ptr, 1024, free_after=True)

    diff_std = float(np.std(y1 - y2))
    ref_std = float(np.std(y2)) + 1e-12
    snr_db = 20.0 * np.log10((diff_std + 1e-12) / ref_std)
    # Corresponds to test-harmonic.c behavior: error should be much smaller.
    assert snr_db < -10.0

