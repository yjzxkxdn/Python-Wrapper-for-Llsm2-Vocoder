"""Low-level CFFI bindings for :mod:`pyllsm2`.

This module provides direct access to the native ``libllsm2`` symbols and
constants. It is intended for advanced users who need one-to-one access to the
underlying C API.

Most users should import from :mod:`pyllsm2` instead, which exposes the
high-level Python-first API.
"""

from ._pyllsm2_cffi import ffi, lib
from ._wrapper import (
    FP_DTYPE,
    as_char_array,
    as_f32_array,
    as_i32_array,
    container_attach,
    copy_fp_ptr,
    copy_int_ptr,
    free,
    output_to_numpy,
)

LLSM_VERSION_STRING = "2.1.0"
LLSM_VERSION_MAJOR = 2
LLSM_VERSION_MINOR = 1
LLSM_VERSION_REVISION = 0

LLSM_FRAME_F0 = 0
LLSM_FRAME_HM = 1
LLSM_FRAME_NM = 2
LLSM_FRAME_PSDRES = 3
LLSM_FRAME_PBPEFF = 8
LLSM_FRAME_PBPSYN = 9
LLSM_FRAME_RD = 10
LLSM_FRAME_VTMAGN = 11
LLSM_FRAME_VSPHSE = 12

LLSM_CONF_NFRM = 0
LLSM_CONF_THOP = 1
LLSM_CONF_MAXNHAR = 2
LLSM_CONF_MAXNHAR_E = 3
LLSM_CONF_NPSD = 4
LLSM_CONF_NOSWARP = 5
LLSM_CONF_FNYQ = 6
LLSM_CONF_NCHANNEL = 7
LLSM_CONF_CHANFREQ = 8
LLSM_CONF_NSPEC = 10
LLSM_CONF_LIPRADIUS = 11

LLSM_AOPTION_HMPP = 0
LLSM_AOPTION_HMCZT = 1


RAW_SYMBOLS = [
    "llsm_create_fp",
    "llsm_create_int",
    "llsm_create_fparray",
    "llsm_copy_fp",
    "llsm_copy_int",
    "llsm_copy_fparray",
    "llsm_delete_fp",
    "llsm_delete_int",
    "llsm_delete_fparray",
    "llsm_fparray_length",
    "llsm_create_container",
    "llsm_copy_container",
    "llsm_copy_container_inplace",
    "llsm_delete_container",
    "llsm_container_get",
    "llsm_container_attach_",
    "llsm_container_remove",
    "llsm_create_hmframe",
    "llsm_copy_hmframe",
    "llsm_copy_hmframe_inplace",
    "llsm_delete_hmframe",
    "llsm_hmframe_phaseshift",
    "llsm_hmframe_harpsd",
    "llsm_create_nmframe",
    "llsm_copy_nmframe",
    "llsm_copy_nmframe_inplace",
    "llsm_delete_nmframe",
    "llsm_create_pbpeffect",
    "llsm_copy_pbpeffect",
    "llsm_delete_pbpeffect",
    "llsm_create_frame",
    "llsm_frame_tolayer0",
    "llsm_frame_phaseshift",
    "llsm_frame_phasesync_rps",
    "llsm_frame_compute_snr",
    "llsm_frame_checklayer0",
    "llsm_frame_checklayer1",
    "llsm_delete_output",
    "llsm_create_aoptions",
    "llsm_delete_aoptions",
    "llsm_aoptions_toconf",
    "llsm_create_soptions",
    "llsm_delete_soptions",
    "llsm_create_chunk",
    "llsm_copy_chunk",
    "llsm_delete_chunk",
    "llsm_chunk_tolayer1",
    "llsm_chunk_tolayer0",
    "llsm_chunk_phasesync_rps",
    "llsm_chunk_phasepropagate",
    "llsm_chunk_getf0",
    "llsm_analyze",
    "llsm_synthesize",
    "llsm_create_coder",
    "llsm_delete_coder",
    "llsm_coder_encode",
    "llsm_coder_decode_layer1",
    "llsm_coder_decode_layer0",
    "llsm_refine_f0",
    "llsm_compute_spectrogram",
    "llsm_compute_dc",
    "llsm_harmonic_peakpicking",
    "llsm_harmonic_czt",
    "llsm_harmonic_analysis",
    "llsm_subband_energy",
    "llsm_fft_to_psd",
    "llsm_estimate_psd",
    "llsm_warp_frequency",
    "llsm_spectral_mean",
    "llsm_spectrum_from_envelope",
    "llsm_get_fftsize",
    "llsm_synthesize_harmonic_frame",
    "llsm_synthesize_harmonic_frame_iczt",
    "llsm_generate_white_noise",
    "llsm_generate_bandlimited_noise",
    "llsm_lipfilter",
    "llsm_lipfilter_reim",
    "llsm_harmonic_spectrum",
    "llsm_harmonic_envelope",
    "llsm_harmonic_minphase",
    "llsm_create_cached_glottal_model",
    "llsm_delete_cached_glottal_model",
    "llsm_spectral_glottal_fitting",
    "llsm_smoothing_filter",
    "llsm_create_rtsynth_buffer",
    "llsm_delete_rtsynth_buffer",
    "llsm_rtsynth_buffer_getlatency",
    "llsm_rtsynth_buffer_numoutput",
    "llsm_rtsynth_buffer_feed",
    "llsm_rtsynth_buffer_fetch",
    "llsm_rtsynth_buffer_fetch_decomposed",
    "llsm_rtsynth_buffer_clear",
    "llsm_lfmodel_to_gfm",
    "llsm_gfm_to_lfmodel",
    "llsm_synthesize_harmonic_frame_auto",
    "llsm_make_filtered_pulse",
    "cig_qifft",
    "cig_spec2env",
    "cig_lfmodel_from_rd",
    "cig_lfmodel_spectrum",
    "cig_lfmodel_period",
    "cig_create_ifdetector",
    "cig_delete_ifdetector",
    "cig_ifdetector_estimate",
    "cig_create_empty_filterbank",
    "cig_create_plp_filterbank",
    "cig_create_melfreq_filterbank",
    "cig_delete_filterbank",
    "cig_filterbank_spectrogram",
    "cig_filterbank_spectrum",
    "llsm_py_container_attach",
    "llsm_py_free",
]

for _name in RAW_SYMBOLS:
    globals()[_name] = getattr(lib, _name)


create_ifdetector = lib.cig_create_ifdetector
delete_ifdetector = lib.cig_delete_ifdetector
create_filterbank = lib.cig_create_empty_filterbank
create_plpfilterbank = lib.cig_create_plp_filterbank
create_melfilterbank = lib.cig_create_melfreq_filterbank
delete_filterbank = lib.cig_delete_filterbank
filterbank_spgm = lib.cig_filterbank_spectrogram
filterbank_spec = lib.cig_filterbank_spectrum


__all__ = [
    "ffi",
    "lib",
    "FP_DTYPE",
    "as_f32_array",
    "as_i32_array",
    "as_char_array",
    "free",
    "copy_fp_ptr",
    "copy_int_ptr",
    "container_attach",
    "output_to_numpy",
    "create_ifdetector",
    "delete_ifdetector",
    "create_filterbank",
    "create_plpfilterbank",
    "create_melfilterbank",
    "delete_filterbank",
    "filterbank_spgm",
    "filterbank_spec",
    "LLSM_VERSION_STRING",
    "LLSM_VERSION_MAJOR",
    "LLSM_VERSION_MINOR",
    "LLSM_VERSION_REVISION",
    "LLSM_FRAME_F0",
    "LLSM_FRAME_HM",
    "LLSM_FRAME_NM",
    "LLSM_FRAME_PSDRES",
    "LLSM_FRAME_PBPEFF",
    "LLSM_FRAME_PBPSYN",
    "LLSM_FRAME_RD",
    "LLSM_FRAME_VTMAGN",
    "LLSM_FRAME_VSPHSE",
    "LLSM_CONF_NFRM",
    "LLSM_CONF_THOP",
    "LLSM_CONF_MAXNHAR",
    "LLSM_CONF_MAXNHAR_E",
    "LLSM_CONF_NPSD",
    "LLSM_CONF_NOSWARP",
    "LLSM_CONF_FNYQ",
    "LLSM_CONF_NCHANNEL",
    "LLSM_CONF_CHANFREQ",
    "LLSM_CONF_NSPEC",
    "LLSM_CONF_LIPRADIUS",
    "LLSM_AOPTION_HMPP",
    "LLSM_AOPTION_HMCZT",
]
__all__.extend(RAW_SYMBOLS)
