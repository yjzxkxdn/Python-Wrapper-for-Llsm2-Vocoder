from __future__ import annotations

import numpy as np

import pyllsm2 as m

from ._shared import raw_analyze_chunk, python_analysis_pair, raw_analysis_pair, raw_output, write_test_output


def test_layer0_edgecase_python_api() -> None:
    x, fs, _, nfrm, _, opt_a, opt_s = python_analysis_pair(max_seconds=1.4, nhop=100)
    f0 = np.zeros(nfrm, dtype=np.float32)
    with opt_a, opt_s:
        opt_a.thop = 100.5 / fs
        layer0 = m.analyze(opt_a, x, float(fs), f0)
        out = m.synthesize(opt_s, layer0)
        data = {"y": out.y, "y_sin": out.y_sin, "y_noise": out.y_noise}
        write_test_output("layer0_edgecase_python.wav", data["y"], fs)
        write_test_output("layer0_edgecase_python_sin.wav", data["y_sin"], fs)
        write_test_output("layer0_edgecase_python_noise.wav", data["y_noise"], fs)
        assert data["y"].size > 0
        assert np.max(np.abs(data["y_sin"])) < 1e-4
        assert np.std(data["y_noise"]) > 1e-6


def test_layer0_edgecase_raw_api() -> None:
    x, fs, _, nfrm, _, opt_a, opt_s = raw_analysis_pair(max_seconds=1.4, nhop=100)
    f0 = np.zeros(nfrm, dtype=np.float32)
    with opt_a, opt_s:
        opt_a.thop = 100.5 / fs
        with m.Chunk(raw_analyze_chunk(m.raw, opt_a.ptr, x, fs, f0)) as chunk:
            data = raw_output(opt_s, chunk)
            write_test_output("layer0_edgecase_raw.wav", data["y"], fs)
            write_test_output("layer0_edgecase_raw_sin.wav", data["y_sin"], fs)
            write_test_output("layer0_edgecase_raw_noise.wav", data["y_noise"], fs)
            assert data["y"].size > 0
            assert np.max(np.abs(data["y_sin"])) < 1e-4
            assert np.std(data["y_noise"]) > 1e-6