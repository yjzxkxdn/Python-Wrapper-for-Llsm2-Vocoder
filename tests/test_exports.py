import pyllsm2 as m


def test_core_exports_exist():
    required = [
        "ffi",
        "lib",
        "llsm_create_aoptions",
        "llsm_create_soptions",
        "llsm_analyze",
        "llsm_synthesize",
        "llsm_create_chunk",
        "llsm_create_coder",
        "llsm_create_rtsynth_buffer",
        "llsm_harmonic_analysis",
        "llsm_warp_frequency",
        "llsm_spectral_mean",
        "llsm_spectrum_from_envelope",
        "qifft",
        "spec2env",
        "lfmodel_from_rd",
        "lfmodel_spectrum",
    ]
    for name in required:
        assert hasattr(m, name), f"missing export: {name}"

