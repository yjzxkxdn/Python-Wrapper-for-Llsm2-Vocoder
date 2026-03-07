import os

import numpy as np
import pytest

import pyllsm2 as m


ffi = m.ffi
lib = m.lib


def _make_sine_test_signal(fs: int, sec: float = 1.0, f0: float = 180.0) -> np.ndarray:
    t = np.arange(int(fs * sec), dtype=np.float32) / fs
    x = (
        0.8 * np.sin(2 * np.pi * f0 * t)
        + 0.3 * np.sin(2 * np.pi * f0 * 2 * t)
        + 0.1 * np.sin(2 * np.pi * f0 * 3 * t)
    )
    return x.astype(np.float32)


def test_layer0_analyze_synthesize_with_python_f0():
    fs = 16000
    nhop = 128
    x = _make_sine_test_signal(fs=fs, sec=1.0, f0=180.0)
    nfrm = int(np.floor((x.size / fs) / (nhop / fs)))
    f0 = np.full(nfrm, 180.0, dtype=np.float32)

    opt_a = lib.llsm_create_aoptions()
    opt_s = lib.llsm_create_soptions(float(fs))
    chunk = ffi.NULL
    try:
        opt_a.thop = nhop / float(fs)
        opt_a.npsd = 128
        opt_a.maxnhar = 60
        opt_a.maxnhar_e = 5
        chunk = m.analyze(opt_a, x, float(fs), f0, return_xap=False)
        out = m.synthesize(opt_s, chunk, return_numpy=True)
        assert out["ny"] > 1000
        assert np.isfinite(out["y"]).all()
        assert np.std(out["y"]) > 1e-5
    finally:
        if chunk != ffi.NULL:
            lib.llsm_delete_chunk(chunk)
        lib.llsm_delete_aoptions(opt_a)
        lib.llsm_delete_soptions(opt_s)


@pytest.mark.skipif(
    os.environ.get("PYLLSM2_TEST_LIBROSA", "0") != "1",
    reason="set PYLLSM2_TEST_LIBROSA=1 to run the optional librosa.pyin test",
)
def test_layer0_librosa_pyin_smoke():
    librosa = pytest.importorskip("librosa")
    x = _make_sine_test_signal(fs=16000, sec=0.6, f0=200.0)
    f0, _, _ = librosa.pyin(
        x,
        fmin=50.0,
        fmax=500.0,
        sr=16000,
        hop_length=128,
        frame_length=1024,
    )
    assert np.sum(np.isfinite(f0)) > 0
