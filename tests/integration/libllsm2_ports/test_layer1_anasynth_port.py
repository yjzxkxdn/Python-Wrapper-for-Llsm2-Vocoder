from __future__ import annotations

import numpy as np

import pyllsm2 as m

from ._shared import raw_analyze_chunk, ffi, lib, python_analysis_pair, raw_analysis_pair, raw_chunk_to_layer1, raw_output, verify_spectral_distribution, write_test_output


def test_layer1_anasynth_python_api() -> None:
    x, fs, _, _, f0, opt_a, opt_s = python_analysis_pair(max_seconds=1.6)
    with opt_a, opt_s:
        layer0 = m.analyze(opt_a, x, float(fs), f0)
        y0 = m.synthesize(opt_s, layer0).y
        layer1 = m.to_layer1(layer0, 2048)
        layer1.phasesync_rps(layer1_based=True)
        pbp_mask = (np.arange(layer1.nfrm, dtype=np.int32) % 100) > 50
        layer1.enable_pulse_by_pulse(pbp_mask, clear_harmonics=True)
        layer1.phasepropagate(sign=1)
        y1 = m.synthesize(opt_s, layer1).y
        write_test_output("layer1_python.wav", y1, fs)
        verify_spectral_distribution(x, y1)
        verify_spectral_distribution(y0, y1)
        layer1.phasepropagate(sign=-1)
        layer1.pitch_shift(1.5, compensate_vtmagn_db=True, clear_harmonics=True)
        layer1.phasepropagate(sign=1)
        y_shift = m.synthesize(opt_s, layer1).y
        write_test_output("layer1_python_pitchshift.wav", y_shift, fs)
        assert np.std(y_shift) > 1e-6


def test_layer1_anasynth_raw_api() -> None:
    x, fs, _, _, f0, opt_a, opt_s = raw_analysis_pair(max_seconds=1.6, use_l1=True)
    with opt_a, opt_s:
        with m.Chunk(raw_analyze_chunk(m.raw, opt_a.ptr, x, fs, f0)) as chunk:
            with raw_chunk_to_layer1(chunk, 2048) as layer1:
                lib.llsm_chunk_phasesync_rps(layer1.ptr, 1)
                pbp_mask = ((np.arange(layer1.nfrm, dtype=np.int32) % 100) > 50).astype(np.uint8)
                rc = int(
                    lib.llsm_py_chunk_enable_pbp_mask(
                        layer1.ptr,
                        ffi.from_buffer("unsigned char[]", pbp_mask),
                        pbp_mask.size,
                        1,
                    )
                )
                assert rc == 0
                lib.llsm_chunk_phasepropagate(layer1.ptr, 1)
                d1 = raw_output(opt_s, layer1)
                write_test_output("layer1_raw.wav", d1["y"], fs)
                verify_spectral_distribution(x, d1["y"])
                lib.llsm_chunk_phasepropagate(layer1.ptr, -1)
                rc = int(lib.llsm_py_chunk_pitch_shift_layer1(layer1.ptr, 1.5, 1, 1))
                assert rc == 0
                lib.llsm_chunk_phasepropagate(layer1.ptr, 1)
                d_shift = raw_output(opt_s, layer1)
                write_test_output("layer1_raw_pitchshift.wav", d_shift["y"], fs)
                assert np.std(d_shift["y"]) > 1e-6