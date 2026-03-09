from __future__ import annotations

import numpy as np

import pyllsm2 as m

from ._shared import raw_analyze_chunk, ffi, lib, python_analysis_pair, raw_analysis_pair, raw_chunk_to_layer1, raw_output, verify_data_distribution, write_test_output


def test_coder_python_api() -> None:
    x, fs, _, _, f0, opt_a, opt_s = python_analysis_pair(max_seconds=1.6)
    with opt_a, opt_s:
        layer0 = m.analyze(opt_a, x, float(fs), f0)
        layer1 = m.to_layer1(layer0, 2048)
        with m.Coder(layer1, 64, 5) as coder:
            reconstructed_0, reconstructed_1 = coder.reconstruct_chunk(layer1)
        reconstructed_0.phasepropagate(sign=1)
        y = m.synthesize(opt_s, reconstructed_0).y
        write_test_output("coder_python.wav", y, fs)
        verify_data_distribution(x, y)
        assert reconstructed_1.vtmagn.shape[0] == layer1.nfrm
        assert reconstructed_1.vsphse.shape[0] == layer1.nfrm


def test_coder_raw_api() -> None:
    x, fs, _, _, f0, opt_a, opt_s = raw_analysis_pair(max_seconds=1.6)
    with opt_a, opt_s:
        with m.Chunk(raw_analyze_chunk(m.raw, opt_a.ptr, x, fs, f0)) as chunk:
            with raw_chunk_to_layer1(chunk, 2048) as layer1:
                with m.Coder(layer1.ptr.conf, 64, 5) as coder:
                    enc = lib.llsm_coder_encode(coder.ptr, layer1.ptr.frames[0])
                    try:
                        assert enc != ffi.NULL
                    finally:
                        m.raw.free(enc)
                    with m.Chunk(lib.llsm_create_chunk(layer1.ptr.conf, 0)) as out0, m.Chunk(lib.llsm_create_chunk(layer1.ptr.conf, 0)) as out1:
                        rc = int(lib.llsm_py_coder_reconstruct_chunk(coder.ptr, layer1.ptr, out0.ptr, out1.ptr))
                        assert rc == 0
                        lib.llsm_chunk_phasepropagate(out0.ptr, 1)
                        y = raw_output(opt_s, out0)["y"]
                        write_test_output("coder_raw.wav", y, fs)
                        verify_data_distribution(x, y)
                        assert out1.vtmagn.shape[0] == layer1.nfrm
                        assert out1.vsphse.shape[0] == layer1.nfrm