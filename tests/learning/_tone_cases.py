from __future__ import annotations

import numpy as np

import pyllsm2 as m

from tests._audio_utils import trim_or_pad


def make_layer0_tone(*, fs: int = 44100, duration_s: float = 0.5, f0_hz: float = 440.0, nhop: int = 128):
    ny = int(round(duration_s * fs))
    nfrm = int(np.ceil(ny / nhop))
    nhar = int(np.floor((fs / 2.0) / f0_hz))

    base_db = -6.0
    step_db = -3.0
    harmonic_db = base_db + step_db * np.arange(nhar, dtype=np.float32)
    harmonic_ampl = np.power(np.float32(10.0), harmonic_db / np.float32(20.0), dtype=np.float32)

    f0 = np.full(nfrm, np.float32(f0_hz), dtype=np.float32)
    ampl = np.tile(harmonic_ampl[None, :], (nfrm, 1)).astype(np.float32, copy=False)
    phse = np.zeros((nfrm, nhar), dtype=np.float32)
    nhar_arr = np.full(nfrm, nhar, dtype=np.int32)

    opt_a = m.AnalysisOptions()
    opt_a.thop = nhop / float(fs)
    opt_a.maxnhar = nhar
    opt_a.maxnhar_e = 0
    opt_a.npsd = 64

    layer0 = m.Layer0Features.from_arrays(f0, ampl, phse, nhar=nhar_arr, fs=float(fs), options=opt_a)
    meta = {"fs": fs, "duration_s": duration_s, "f0_hz": f0_hz, "nhop": nhop, "ny": ny, "nfrm": nfrm, "nhar": nhar}
    return layer0, meta


def make_seeded_layer1_tone(*, fs: int = 44100, duration_s: float = 0.5, f0_hz: float = 440.0, nhop: int = 128, nfft: int = 1024):
    layer0, meta = make_layer0_tone(fs=fs, duration_s=duration_s, f0_hz=f0_hz, nhop=nhop)
    layer1_seed = m.to_layer1(layer0, nfft)

    layer1 = m.Layer1Features.from_arrays(
        np.asarray(layer0.f0, dtype=np.float32),
        np.asarray(layer1_seed.rd, dtype=np.float32),
        np.asarray(layer1_seed.vtmagn, dtype=np.float32),
        np.asarray(layer1_seed.vsphse, dtype=np.float32),
        fs=float(fs),
        vsphse_lengths=np.asarray(layer1_seed.vsphse_lengths, dtype=np.int32),
    )
    meta.update({"nfft": nfft, "nspec": layer1.nspec})
    return layer0, layer1, meta


def make_manual_layer1_tone(*, fs: int = 44100, duration_s: float = 0.5, f0_hz: float = 440.0, nhop: int = 128, nfft: int = 1024):
    fnyq = fs / 2.0
    ny = int(round(duration_s * fs))
    nfrm = int(np.ceil(ny / nhop))
    nhar = int(np.floor(fnyq / f0_hz))
    nspec = nfft // 2 + 1

    base_db = -6.0
    step_db = -3.0
    harm_index = np.arange(nhar, dtype=np.float32)
    target_db = base_db + step_db * harm_index
    target_ampl = np.power(np.float32(10.0), target_db / np.float32(20.0), dtype=np.float32)

    rd_value = np.float32(1.0)
    freq = (f0_hz * (np.arange(nhar, dtype=np.float32) + np.float32(1.0))).astype(np.float32)
    model = m.lfmodel_from_rd(float(rd_value), 1.0 / float(f0_hz), 1.0)
    vs_ampl = np.asarray(m.lfmodel_spectrum(model, freq), dtype=np.float32)
    if nhar > 1:
        vs_ampl[1:] /= ((1.0 + np.arange(1, nhar, dtype=np.float32)) * vs_ampl[0])
    vs_ampl[0] = np.float32(1.0)
    vs_ampl = np.maximum(vs_ampl, np.float32(1e-8))

    opt_a = m.AnalysisOptions()
    opt_a.thop = nhop / float(fs)
    opt_a.maxnhar = nhar
    opt_a.maxnhar_e = 0
    opt_a.npsd = 64
    lip_radius = float(opt_a.lip_radius)

    pre_lip_ampl = np.asarray(target_ampl, dtype=np.float32).copy()
    pre_lip_phse = np.zeros(nhar, dtype=np.float32)
    m.raw.lib.llsm_lipfilter(
        np.float32(lip_radius),
        np.float32(f0_hz),
        int(nhar),
        m.raw.ffi.from_buffer("FP_TYPE[]", pre_lip_ampl),
        m.raw.ffi.from_buffer("FP_TYPE[]", pre_lip_phse),
        1,
    )

    vt_harm_ampl = np.maximum(pre_lip_ampl / vs_ampl, np.float32(1e-8))
    vt_harm_db = (20.0 * np.log10(vt_harm_ampl)).astype(np.float32)
    vt_harm_phse = m.raw.copy_fp_ptr(
        m.raw.lib.llsm_harmonic_minphase(
            m.raw.ffi.from_buffer("FP_TYPE[]", np.ascontiguousarray(vt_harm_ampl)),
            int(nhar),
        ),
        int(nhar),
        free_after=True,
    )
    vsphse_harm = (pre_lip_phse - vt_harm_phse).astype(np.float32)

    spec_freq = np.linspace(0.0, fnyq, nspec, dtype=np.float32)
    interp_x = np.concatenate(([0.0], freq, [fnyq])).astype(np.float32)
    interp_y = np.concatenate(([vt_harm_db[0]], vt_harm_db, [vt_harm_db[-1]])).astype(np.float32)
    vtmagn_frame = np.interp(spec_freq, interp_x, interp_y).astype(np.float32)

    f0 = np.full(nfrm, np.float32(f0_hz), dtype=np.float32)
    rd = np.full(nfrm, rd_value, dtype=np.float32)
    vtmagn = np.tile(vtmagn_frame[None, :], (nfrm, 1)).astype(np.float32, copy=False)
    vsphse = np.tile(vsphse_harm[None, :], (nfrm, 1)).astype(np.float32, copy=False)
    vsphse_lengths = np.full(nfrm, nhar, dtype=np.int32)

    layer1 = m.Layer1Features.from_arrays(
        f0,
        rd,
        vtmagn,
        vsphse,
        fs=float(fs),
        options=opt_a,
        vsphse_lengths=vsphse_lengths,
    )
    meta = {
        "fs": fs,
        "duration_s": duration_s,
        "f0_hz": f0_hz,
        "nhop": nhop,
        "nfft": nfft,
        "ny": ny,
        "nfrm": nfrm,
        "nhar": nhar,
        "nspec": nspec,
    }
    return layer1, meta


def render_chunk(chunk: m.Chunk | m.Layer0Features | m.Layer1Features, *, fs: int, ny: int) -> np.ndarray:
    out = m.synthesize(m.SynthesisOptions(fs), chunk)
    return trim_or_pad(np.asarray(out.y, dtype=np.float32), ny)
