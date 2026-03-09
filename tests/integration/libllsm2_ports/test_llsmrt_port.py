from __future__ import annotations

import numpy as np

import pyllsm2 as m

from ._shared import raw_analyze_chunk, ffi, lib, python_analysis_pair, raw_analysis_pair, raw_chunk_to_layer1, raw_output, verify_data_distribution, verify_spectral_distribution, write_test_output


def test_llsmrt_python_api() -> None:
    x, fs, _, _, f0, opt_a, opt_s = python_analysis_pair(max_seconds=1.6)
    with opt_a, opt_s:
        layer0 = m.analyze(opt_a, x, float(fs), f0)
        layer1 = m.to_layer1(layer0, 2048)
        layer1.phasesync_rps(layer1_based=True)
        pbp_mask = (np.arange(layer1.nfrm, dtype=np.int32) % 100) > 50
        layer1.enable_pulse_by_pulse(pbp_mask, clear_harmonics=True)
        layer1.phasepropagate(sign=1)
        with m.RTSynthBuffer(opt_s, layer1, 4096) as rt:
            dec = rt.render_chunk_decomposed(layer1, nx=x.size, trim_latency=True)
        write_test_output("llsmrt_python_p.wav", dec["p"], fs)
        write_test_output("llsmrt_python_ap.wav", dec["ap"], fs)
        y_ref = m.synthesize(opt_s, layer1).y[: dec["y"].size]
        verify_data_distribution(y_ref, dec["y"])
        verify_spectral_distribution(y_ref, dec["y"])


def test_llsmrt_raw_api() -> None:
    x, fs, _, _, f0, opt_a, opt_s = raw_analysis_pair(max_seconds=1.6, use_l1=True)
    with opt_a, opt_s:
        with m.Chunk(raw_analyze_chunk(m.raw, opt_a.ptr, x, fs, f0)) as chunk:
            with raw_chunk_to_layer1(chunk, 2048) as layer1:
                lib.llsm_chunk_phasesync_rps(layer1.ptr, 1)
                pbp_mask = ((np.arange(layer1.nfrm, dtype=np.int32) % 100) > 50).astype(np.uint8)
                rc = int(lib.llsm_py_chunk_enable_pbp_mask(layer1.ptr, ffi.from_buffer("unsigned char[]", pbp_mask), pbp_mask.size, 1))
                assert rc == 0
                lib.llsm_chunk_phasepropagate(layer1.ptr, 1)
                rt = lib.llsm_create_rtsynth_buffer(opt_s.ptr, layer1.ptr.conf, 4096)
                assert rt != ffi.NULL
                try:
                    y_p = np.zeros(x.size, dtype=np.float32)
                    y_ap = np.zeros(x.size, dtype=np.float32)
                    ny_ptr = ffi.new("int *")
                    rc = int(
                        lib.llsm_py_rtsynth_render_chunk_decomposed(
                            rt,
                            layer1.ptr,
                            ffi.from_buffer("FP_TYPE[]", y_p),
                            ffi.from_buffer("FP_TYPE[]", y_ap),
                            x.size,
                            1,
                            ny_ptr,
                        )
                    )
                    assert rc == 0
                    end = int(ny_ptr[0])
                    y_rt = (y_p + y_ap)[:end]
                    write_test_output("llsmrt_raw_p.wav", y_p[:end], fs)
                    write_test_output("llsmrt_raw_ap.wav", y_ap[:end], fs)
                    y_ref = raw_output(opt_s, layer1)["y"][:end]
                    verify_data_distribution(y_ref, y_rt)
                    verify_spectral_distribution(y_ref, y_rt)
                finally:
                    lib.llsm_delete_rtsynth_buffer(rt)