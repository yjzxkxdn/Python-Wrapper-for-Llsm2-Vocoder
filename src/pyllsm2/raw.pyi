from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from numpy.typing import NDArray

FloatArray1D = NDArray[np.float32]
IntArray1D = NDArray[np.int32]

ffi: Any
lib: Any
FP_DTYPE: Any

def as_f32_array(data: Iterable[float], name: str = ...) -> FloatArray1D: ...
def as_i32_array(data: Iterable[int], name: str = ...) -> IntArray1D: ...
def as_char_array(value: str | bytes) -> Any: ...
def free(ptr: Any) -> None: ...
def copy_fp_ptr(ptr: Any, n: int, free_after: bool = ...) -> FloatArray1D: ...
def copy_int_ptr(ptr: Any, n: int, free_after: bool = ...) -> IntArray1D: ...
def container_attach(dst: Any, index: int, ptr: Any, dtor: Any = ..., copyctor: Any = ...) -> None: ...
def output_to_numpy(out_ptr: Any, free_after: bool = ...) -> FloatArray1D: ...

LLSM_VERSION_STRING: str
LLSM_VERSION_MAJOR: int
LLSM_VERSION_MINOR: int
LLSM_VERSION_REVISION: int

LLSM_FRAME_F0: int
LLSM_FRAME_HM: int
LLSM_FRAME_NM: int
LLSM_FRAME_PSDRES: int
LLSM_FRAME_PBPEFF: int
LLSM_FRAME_PBPSYN: int
LLSM_FRAME_RD: int
LLSM_FRAME_VTMAGN: int
LLSM_FRAME_VSPHSE: int

LLSM_CONF_NFRM: int
LLSM_CONF_THOP: int
LLSM_CONF_MAXNHAR: int
LLSM_CONF_MAXNHAR_E: int
LLSM_CONF_NPSD: int
LLSM_CONF_NOSWARP: int
LLSM_CONF_FNYQ: int
LLSM_CONF_NCHANNEL: int
LLSM_CONF_CHANFREQ: int
LLSM_CONF_NSPEC: int
LLSM_CONF_LIPRADIUS: int

LLSM_AOPTION_HMPP: int
LLSM_AOPTION_HMCZT: int

RAW_SYMBOLS: list[str]

llsm_create_fp: Any
llsm_create_int: Any
llsm_create_fparray: Any
llsm_copy_fp: Any
llsm_copy_int: Any
llsm_copy_fparray: Any
llsm_delete_fp: Any
llsm_delete_int: Any
llsm_delete_fparray: Any
llsm_fparray_length: Any
llsm_create_container: Any
llsm_copy_container: Any
llsm_copy_container_inplace: Any
llsm_delete_container: Any
llsm_container_get: Any
llsm_container_attach_: Any
llsm_container_remove: Any
llsm_create_hmframe: Any
llsm_copy_hmframe: Any
llsm_copy_hmframe_inplace: Any
llsm_delete_hmframe: Any
llsm_hmframe_phaseshift: Any
llsm_hmframe_harpsd: Any
llsm_create_nmframe: Any
llsm_copy_nmframe: Any
llsm_copy_nmframe_inplace: Any
llsm_delete_nmframe: Any
llsm_create_pbpeffect: Any
llsm_copy_pbpeffect: Any
llsm_delete_pbpeffect: Any
llsm_create_frame: Any
llsm_frame_tolayer0: Any
llsm_frame_phaseshift: Any
llsm_frame_phasesync_rps: Any
llsm_frame_compute_snr: Any
llsm_frame_checklayer0: Any
llsm_frame_checklayer1: Any
llsm_delete_output: Any
llsm_create_aoptions: Any
llsm_delete_aoptions: Any
llsm_aoptions_toconf: Any
llsm_create_soptions: Any
llsm_delete_soptions: Any
llsm_create_chunk: Any
llsm_copy_chunk: Any
llsm_delete_chunk: Any
llsm_chunk_tolayer1: Any
llsm_chunk_tolayer0: Any
llsm_chunk_phasesync_rps: Any
llsm_chunk_phasepropagate: Any
llsm_chunk_getf0: Any
llsm_analyze: Any
llsm_synthesize: Any
llsm_create_coder: Any
llsm_delete_coder: Any
llsm_coder_encode: Any
llsm_coder_decode_layer1: Any
llsm_coder_decode_layer0: Any
llsm_refine_f0: Any
llsm_compute_spectrogram: Any
llsm_compute_dc: Any
llsm_harmonic_peakpicking: Any
llsm_harmonic_czt: Any
llsm_harmonic_analysis: Any
llsm_subband_energy: Any
llsm_fft_to_psd: Any
llsm_estimate_psd: Any
llsm_warp_frequency: Any
llsm_spectral_mean: Any
llsm_spectrum_from_envelope: Any
llsm_get_fftsize: Any
llsm_synthesize_harmonic_frame: Any
llsm_synthesize_harmonic_frame_iczt: Any
llsm_generate_white_noise: Any
llsm_generate_bandlimited_noise: Any
llsm_lipfilter: Any
llsm_lipfilter_reim: Any
llsm_harmonic_spectrum: Any
llsm_harmonic_envelope: Any
llsm_harmonic_minphase: Any
llsm_create_cached_glottal_model: Any
llsm_delete_cached_glottal_model: Any
llsm_spectral_glottal_fitting: Any
llsm_smoothing_filter: Any
llsm_create_rtsynth_buffer: Any
llsm_delete_rtsynth_buffer: Any
llsm_rtsynth_buffer_getlatency: Any
llsm_rtsynth_buffer_numoutput: Any
llsm_rtsynth_buffer_feed: Any
llsm_rtsynth_buffer_fetch: Any
llsm_rtsynth_buffer_fetch_decomposed: Any
llsm_rtsynth_buffer_clear: Any
llsm_lfmodel_to_gfm: Any
llsm_gfm_to_lfmodel: Any
llsm_synthesize_harmonic_frame_auto: Any
llsm_make_filtered_pulse: Any
cig_qifft: Any
cig_spec2env: Any
cig_lfmodel_from_rd: Any
cig_lfmodel_spectrum: Any
cig_lfmodel_period: Any
cig_create_ifdetector: Any
cig_delete_ifdetector: Any
cig_ifdetector_estimate: Any
cig_create_empty_filterbank: Any
cig_create_plp_filterbank: Any
cig_create_melfreq_filterbank: Any
cig_delete_filterbank: Any
cig_filterbank_spectrogram: Any
cig_filterbank_spectrum: Any
llsm_py_container_attach: Any
llsm_py_free: Any

create_ifdetector: Any
delete_ifdetector: Any
create_filterbank: Any
create_plpfilterbank: Any
create_melfilterbank: Any
delete_filterbank: Any
filterbank_spgm: Any
filterbank_spec: Any

