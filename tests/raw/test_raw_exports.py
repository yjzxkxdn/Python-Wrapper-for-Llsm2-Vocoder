import pyllsm2 as m


def test_raw_exports_exist_under_raw_module() -> None:
    required = [
        "ffi",
        "lib",
        "container_attach",
        "llsm_create_aoptions",
        "llsm_create_soptions",
        "llsm_analyze",
        "llsm_synthesize",
        "llsm_harmonic_analysis",
        "LLSM_AOPTION_HMPP",
        "LLSM_AOPTION_HMCZT",
    ]
    for name in required:
        assert hasattr(m.raw, name), f"missing raw export: {name}"


def test_raw_aliases_and_version_constants_are_consistent() -> None:
    assert m.raw.LLSM_VERSION_STRING == "2.1.0"
    assert m.raw.LLSM_VERSION_MAJOR == 2
    assert m.raw.LLSM_VERSION_MINOR == 1
    assert m.raw.LLSM_VERSION_REVISION == 0
    assert m.raw.create_ifdetector is m.raw.lib.cig_create_ifdetector
    assert m.raw.delete_ifdetector is m.raw.lib.cig_delete_ifdetector
    assert m.raw.create_filterbank is m.raw.lib.cig_create_empty_filterbank
