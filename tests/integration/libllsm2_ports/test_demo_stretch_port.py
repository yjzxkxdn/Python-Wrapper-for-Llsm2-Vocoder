from __future__ import annotations

import numpy as np

import pyllsm2 as m

from ._shared import raw_analyze_chunk, lib, python_analysis_pair, raw_analysis_pair, raw_chunk_to_layer1, raw_output, write_test_output


def test_demo_stretch_python_api() -> None:
    x, fs, _, _, f0, opt_a, opt_s = python_analysis_pair(max_seconds=1.2, hm_method=m.raw.LLSM_AOPTION_HMCZT)
    with opt_a, opt_s:
        layer0 = m.analyze(opt_a, x, float(fs), f0)
        y0 = m.synthesize(opt_s, layer0).y
        write_test_output("demo_stretch_python_orig.wav", y0, fs)
        layer1 = m.to_layer1(layer0, 2048)
        layer1.phasepropagate(sign=-1)
        layer1_new = layer1.resample_linear_f0(max(2, layer1.nfrm * 2))
        layer0_new = m.to_layer0(layer1_new)
        layer0_new.phasepropagate(sign=1)
        y = m.synthesize(opt_s, layer0_new).y
        write_test_output("demo_stretch_python.wav", y, fs)
        assert y.size > y0.size * 1.5
        assert np.isfinite(y).all()


def test_demo_stretch_raw_api() -> None:
    x, fs, _, _, f0, opt_a, opt_s = raw_analysis_pair(max_seconds=1.2, hm_method=m.raw.LLSM_AOPTION_HMCZT)
    with opt_a, opt_s:
        with m.Chunk(raw_analyze_chunk(m.raw, opt_a.ptr, x, fs, f0)) as chunk:
            y0 = raw_output(opt_s, chunk)["y"]
            write_test_output("demo_stretch_raw_orig.wav", y0, fs)
            with raw_chunk_to_layer1(chunk, 2048) as layer1:
                lib.llsm_chunk_phasepropagate(layer1.ptr, -1)
                stretched = layer1.resample_linear_f0(max(2, layer1.nfrm * 2))
                layer0_new = m.to_layer0(stretched)
                layer0_new.phasepropagate(sign=1)
                y = raw_output(opt_s, layer0_new)["y"]
                write_test_output("demo_stretch_raw.wav", y, fs)
                assert y.size > y0.size * 1.5
                assert np.isfinite(y).all()
