from __future__ import annotations

import math

import numpy as np

import pyllsm2 as m

from ._shared import raw_analyze_chunk, LLSM_FRAME_GROWLSTRENGTH, ffi, lib, python_analysis_pair, raw_analysis_pair, raw_chunk_to_layer1, raw_output, write_test_output


def test_pbpeffects_python_api() -> None:
    x, fs, _, _, f0, opt_a, opt_s = python_analysis_pair(wav_name="are-you-ready.wav", max_seconds=2.0)
    with opt_a, opt_s:
        opt_a.rel_winsize = 4.0
        layer0 = m.analyze(opt_a, x, float(fs), f0)
        layer1 = m.to_layer1(layer0, 2048)
        layer1.phasepropagate(sign=-1)
        mask = np.zeros(layer1.nfrm, dtype=bool)
        begin = int(0.5 / opt_a.thop)
        end = min(layer1.nfrm, int(1.8 / opt_a.thop))
        mask[begin:end] = True
        layer1.enable_pulse_by_pulse(mask, clear_harmonics=True)
        if np.any(mask):
            strength = np.linspace(0.0, 1.0, mask.sum(), dtype=np.float32)
            layer1.rd[mask] = np.clip(layer1.rd[mask] * (1.0 - 0.25 * strength), 0.2, 3.0)
            layer1.vtmagn[mask] = layer1.vtmagn[mask] + (np.sin(np.linspace(0.0, np.pi, layer1.nspec, dtype=np.float32)) * 3.0)
            layer1.vsphse[mask] = layer1.vsphse[mask] + 0.03 * np.arange(layer1.vsphse.shape[1], dtype=np.float32)
        layer1.phasepropagate(sign=1)
        y = m.synthesize(opt_s, layer1).y
        write_test_output("pbpeffects_python.wav", y, fs)
        assert y.size > 0
        assert np.isfinite(y).all()
        assert np.std(y) > 1e-6


def test_pbpeffects_raw_api() -> None:
    x, fs, _, _, f0, opt_a, opt_s = raw_analysis_pair(wav_name="are-you-ready.wav", max_seconds=2.0, use_l1=True)
    with opt_a, opt_s:
        opt_a.rel_winsize = 4.0
        with m.Chunk(raw_analyze_chunk(m.raw, opt_a.ptr, x, fs, f0)) as chunk:
            with raw_chunk_to_layer1(chunk, 2048) as layer1:
                lib.llsm_chunk_phasepropagate(layer1.ptr, -1)
                state = {"period_count": 0, "osc": 0.0}
                handle = ffi.new_handle(state)

                @ffi.callback("void(llsm_gfm*, FP_TYPE*, void*, llsm_container*)")
                def _growl(gfm, delta_t, info, src_frame):
                    st = ffi.from_handle(info)
                    ptr = ffi.cast("FP_TYPE*", lib.llsm_container_get(src_frame, LLSM_FRAME_GROWLSTRENGTH))
                    strength = float(ptr[0]) if ptr != ffi.NULL else 1.0
                    st["period_count"] += 1
                    lfo = math.sin(st["period_count"] * 2.0 * math.pi / 50.0)
                    st["osc"] += 2.0 * math.pi / (6.0 + lfo)
                    osc = math.sin(st["osc"])
                    delta_t[0] = gfm.T0 * 0.01 * strength
                    gfm.Fa *= 1.0 - osc * 0.5 * strength
                    gfm.Rk *= 1.0 + osc * 0.3 * strength
                    gfm.Ee *= 1.0 - osc * 0.5 * strength

                n_effect_begin = int(0.5 / opt_a.thop)
                n_effect_end = min(layer1.nfrm, int(1.8 / opt_a.thop))
                n_fade = 20
                for frame_idx in range(layer1.nfrm):
                    frame = layer1.ptr.frames[frame_idx]
                    m.raw.container_attach(frame, m.raw.LLSM_FRAME_HM, ffi.NULL, ffi.NULL, ffi.NULL)
                    if n_effect_begin < frame_idx < n_effect_end:
                        strength = lib.llsm_create_fp(1.0)
                        if frame_idx < n_effect_begin + n_fade:
                            strength[0] = float(frame_idx - n_effect_begin) / n_fade
                        if frame_idx > n_effect_end - n_fade:
                            strength[0] = float(n_effect_end - frame_idx) / n_fade
                        pbpsyn = lib.llsm_create_int(1)
                        pbeff = lib.llsm_create_pbpeffect(_growl, handle)
                        m.raw.container_attach(frame, m.raw.LLSM_FRAME_PBPSYN, ffi.cast("void*", pbpsyn), ffi.cast("llsm_fdestructor", lib.llsm_delete_int), ffi.cast("llsm_fcopy", lib.llsm_copy_int))
                        m.raw.container_attach(frame, m.raw.LLSM_FRAME_PBPEFF, ffi.cast("void*", pbeff), ffi.cast("llsm_fdestructor", lib.llsm_delete_pbpeffect), ffi.cast("llsm_fcopy", lib.llsm_copy_pbpeffect))
                        m.raw.container_attach(frame, LLSM_FRAME_GROWLSTRENGTH, ffi.cast("void*", strength), ffi.cast("llsm_fdestructor", lib.llsm_delete_fp), ffi.cast("llsm_fcopy", lib.llsm_copy_fp))

                lib.llsm_chunk_phasepropagate(layer1.ptr, 1)
                y = raw_output(opt_s, layer1)["y"]
                write_test_output("pbpeffects_raw.wav", y, fs)
                assert y.size > 0
                assert np.isfinite(y).all()
                assert np.std(y) > 1e-6