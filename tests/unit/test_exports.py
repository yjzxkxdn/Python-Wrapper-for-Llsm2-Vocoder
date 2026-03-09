import pyllsm2 as m


def test_top_level_exports_are_python_first() -> None:
    required = [
        "AnalysisOptions",
        "SynthesisOptions",
        "Layer0Features",
        "Layer1Features",
        "analyze",
        "to_layer1",
        "to_layer0",
        "synthesize",
        "Chunk",
        "Coder",
        "RTSynthBuffer",
        "Output",
        "analyze_chunk",
        "synthesize_output",
        "warp_frequency",
        "spectral_mean",
        "spectrum_from_envelope",
        "harmonic_analysis",
        "synthesize_harmonic_frame",
        "qifft",
        "spec2env",
        "lfmodel_from_rd",
        "lfmodel_spectrum",
        "lfmodel_period",
        "ifdetector_estimate",
        "raw",
    ]
    for name in required:
        assert hasattr(m, name), f"missing export: {name}"


def test_top_level_api_omits_selected_raw_symbol_spam() -> None:
    unexpected = [
        "llsm_analyze",
        "llsm_synthesize",
        "llsm_create_chunk",
        "LLSM_AOPTION_HMPP",
        "LLSM_FRAME_HM",
    ]
    for name in unexpected:
        assert not hasattr(m, name), f"unexpected raw symbol leaked into top level: {name}"
