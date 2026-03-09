from __future__ import annotations

import pyllsm2 as m

from ._shared import (
    lib,
    python_analysis_pair,
    raw_analysis_pair,
    raw_output,
    verify_data_distribution,
    verify_spectral_distribution,
    write_test_output,
)


def test_layer0_anasynth_python_api() -> None:
    for method, suffix in (
        (m.raw.LLSM_AOPTION_HMPP, "hmpp"),
        (m.raw.LLSM_AOPTION_HMCZT, "hmczt"),
    ):
        x, fs, _, _, f0, opt_a, opt_s = python_analysis_pair(max_seconds=1.6, hm_method=method)
        with opt_a, opt_s:
            layer0 = m.analyze(opt_a, x, float(fs), f0)
            out0 = m.synthesize(opt_s, layer0)
            d0 = {"y": out0.y, "y_sin": out0.y_sin, "y_noise": out0.y_noise}
            write_test_output(f"layer0_python_{suffix}.wav", d0["y"], fs)
            verify_data_distribution(x, d0["y"])
            verify_spectral_distribution(x, d0["y"])
            layer0.phasesync_rps(layer1_based=False)
            layer0.phasepropagate(sign=1)
            out1 = m.synthesize(opt_s, layer0)
            d1 = {"y": out1.y, "y_sin": out1.y_sin, "y_noise": out1.y_noise}
            write_test_output(f"layer0_python_{suffix}_phasesync.wav", d1["y"], fs)
            verify_data_distribution(x, d1["y"])
            verify_spectral_distribution(x, d1["y"])


def test_layer0_anasynth_raw_api() -> None:
    for method, suffix in (
        (m.raw.LLSM_AOPTION_HMPP, "hmpp"),
        (m.raw.LLSM_AOPTION_HMCZT, "hmczt"),
    ):
        x, fs, _, _, f0, opt_a, opt_s = raw_analysis_pair(max_seconds=1.6, hm_method=method)
        with opt_a, opt_s:
            with m.Chunk(m.raw.lib.llsm_analyze(
                opt_a.ptr,
                m.raw.ffi.from_buffer("FP_TYPE[]", m.raw.as_f32_array(x, "x")),
                x.size,
                float(fs),
                m.raw.ffi.from_buffer("FP_TYPE[]", m.raw.as_f32_array(f0, "f0")),
                f0.size,
                m.raw.ffi.NULL,
            )) as chunk:
                d0 = raw_output(opt_s, chunk)
                write_test_output(f"layer0_raw_{suffix}.wav", d0["y"], fs)
                verify_data_distribution(x, d0["y"])
                verify_spectral_distribution(x, d0["y"])
                lib.llsm_chunk_phasesync_rps(chunk.ptr, 0)
                lib.llsm_chunk_phasepropagate(chunk.ptr, 1)
                d1 = raw_output(opt_s, chunk)
                write_test_output(f"layer0_raw_{suffix}_phasesync.wav", d1["y"], fs)
                verify_data_distribution(x, d1["y"])
                verify_spectral_distribution(x, d1["y"])