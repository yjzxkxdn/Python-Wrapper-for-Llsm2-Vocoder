from pathlib import Path
import platform

from cffi import FFI


ffibuilder = FFI()

CDEF = r"""
typedef float FP_TYPE;

typedef struct {
  FP_TYPE T0;
  FP_TYPE te;
  FP_TYPE tp;
  FP_TYPE ta;
  FP_TYPE Ee;
} lfmodel;

typedef void (*llsm_fdestructor)(void*);
typedef void* (*llsm_fcopy)(void*);

typedef struct {
  void** members;
  llsm_fdestructor* destructors;
  llsm_fcopy* copyctors;
  int nmember;
} llsm_container;

typedef struct {
  FP_TYPE* ampl;
  FP_TYPE* phse;
  int nhar;
} llsm_hmframe;

typedef struct {
  llsm_hmframe** eenv;
  FP_TYPE* edc;
  FP_TYPE* psd;
  int npsd;
  int nchannel;
} llsm_nmframe;

typedef struct {
  FP_TYPE Fa;
  FP_TYPE Rk;
  FP_TYPE Rg;
  FP_TYPE T0;
  FP_TYPE Ee;
} llsm_gfm;

typedef void (*llsm_fgfm)(llsm_gfm* dst, FP_TYPE* delta_t, void* info,
  llsm_container* src_frame);

typedef struct {
  llsm_fgfm modifier;
  void* info;
} llsm_pbpeffect;

typedef struct {
  int ny;
  FP_TYPE fs;
  FP_TYPE* y;
  FP_TYPE* y_sin;
  FP_TYPE* y_noise;
} llsm_output;

typedef struct {
  FP_TYPE thop;
  int maxnhar;
  int maxnhar_e;
  int npsd;
  int nchannel;
  FP_TYPE* chanfreq;
  FP_TYPE lip_radius;
  int f0_refine;
  int hm_method;
  FP_TYPE rel_winsize;
} llsm_aoptions;

typedef struct {
  FP_TYPE fs;
  int use_iczt;
  int use_l1;
  FP_TYPE iczt_param_a;
  FP_TYPE iczt_param_b;
} llsm_soptions;

typedef struct {
  llsm_container* conf;
  llsm_container** frames;
} llsm_chunk;

typedef void llsm_coder;
typedef void llsm_cached_glottal_model;
typedef void llsm_rtsynth_buffer;

typedef struct {
  FP_TYPE fc;
  int nh;
  FP_TYPE* hr;
  FP_TYPE* hi;
  FP_TYPE* hdr;
  FP_TYPE* hdi;
} ifdetector;

typedef struct {
  int nchannel;
  int nf;
  FP_TYPE fnyq;
  FP_TYPE** fresp;
  int* lower_idx;
  int* upper_idx;
} filterbank;

FP_TYPE* llsm_create_fp(FP_TYPE x);
int* llsm_create_int(int x);
FP_TYPE* llsm_create_fparray(int size);
FP_TYPE* llsm_copy_fp(FP_TYPE* src);
int* llsm_copy_int(int* src);
FP_TYPE* llsm_copy_fparray(FP_TYPE* src);
void llsm_delete_fp(FP_TYPE* dst);
void llsm_delete_int(int* dst);
void llsm_delete_fparray(FP_TYPE* dst);
int llsm_fparray_length(FP_TYPE* src);

llsm_container* llsm_create_container(int nmember);
llsm_container* llsm_copy_container(llsm_container* src);
void llsm_copy_container_inplace(llsm_container* dst, llsm_container* src);
void llsm_delete_container(llsm_container* dst);
void* llsm_container_get(llsm_container* src, int index);
void llsm_container_attach_(llsm_container* dst, int index, void* ptr,
  llsm_fdestructor dtor, llsm_fcopy copyctor);
void llsm_container_remove(llsm_container* dst, int index);

llsm_hmframe* llsm_create_hmframe(int nhar);
llsm_hmframe* llsm_copy_hmframe(llsm_hmframe* src);
void llsm_copy_hmframe_inplace(llsm_hmframe* dst, llsm_hmframe* src);
void llsm_delete_hmframe(llsm_hmframe* dst);
void llsm_hmframe_phaseshift(llsm_hmframe* dst, FP_TYPE theta);
FP_TYPE* llsm_hmframe_harpsd(llsm_hmframe* src, int db_scale);

llsm_nmframe* llsm_create_nmframe(int nchannel, int nhar_e, int npsd);
llsm_nmframe* llsm_copy_nmframe(llsm_nmframe* src);
void llsm_copy_nmframe_inplace(llsm_nmframe* dst, llsm_nmframe* src);
void llsm_delete_nmframe(llsm_nmframe* dst);

llsm_pbpeffect* llsm_create_pbpeffect(llsm_fgfm modifier, void* info);
llsm_pbpeffect* llsm_copy_pbpeffect(llsm_pbpeffect* src);
void llsm_delete_pbpeffect(llsm_pbpeffect* dst);

llsm_container* llsm_create_frame(int nhar, int nchannel, int nhar_e, int npsd);
void llsm_frame_tolayer0(llsm_container* dst, llsm_container* conf);
void llsm_frame_phaseshift(llsm_container* dst, FP_TYPE theta);
void llsm_frame_phasesync_rps(llsm_container* dst, int layer1_based);
FP_TYPE* llsm_frame_compute_snr(llsm_container* src, llsm_container* conf,
  int as_aperiodicity);
int llsm_frame_checklayer0(llsm_container* src);
int llsm_frame_checklayer1(llsm_container* src);
void llsm_delete_output(llsm_output* dst);

llsm_aoptions* llsm_create_aoptions();
void llsm_delete_aoptions(llsm_aoptions* dst);
llsm_container* llsm_aoptions_toconf(llsm_aoptions* src, FP_TYPE fnyq);
llsm_soptions* llsm_create_soptions(FP_TYPE fs);
void llsm_delete_soptions(llsm_soptions* dst);

llsm_chunk* llsm_create_chunk(llsm_container* conf, int init_frames);
llsm_chunk* llsm_copy_chunk(llsm_chunk* src);
void llsm_delete_chunk(llsm_chunk* dst);
void llsm_chunk_tolayer1(llsm_chunk* dst, int nfft);
void llsm_chunk_tolayer0(llsm_chunk* dst);
void llsm_chunk_phasesync_rps(llsm_chunk* dst, int layer1_based);
void llsm_chunk_phasepropagate(llsm_chunk* dst, int sign);
FP_TYPE* llsm_chunk_getf0(llsm_chunk* src, int* dst_nfrm);

llsm_chunk* llsm_analyze(llsm_aoptions* options, FP_TYPE* x, int nx,
  FP_TYPE fs, FP_TYPE* f0, int nfrm, FP_TYPE** x_ap);
llsm_output* llsm_synthesize(llsm_soptions* options, llsm_chunk* src);

llsm_coder* llsm_create_coder(llsm_container* conf, int order_spec, int order_bap);
void llsm_delete_coder(llsm_coder* dst);
FP_TYPE* llsm_coder_encode(llsm_coder* c, llsm_container* src);
llsm_container* llsm_coder_decode_layer1(llsm_coder* c, FP_TYPE* src);
llsm_container* llsm_coder_decode_layer0(llsm_coder* c, FP_TYPE* src);

void llsm_refine_f0(FP_TYPE* x, int nx, FP_TYPE fs, FP_TYPE* f0, int nfrm,
  FP_TYPE thop);
void llsm_compute_spectrogram(FP_TYPE* x, int nx, int* center, int* winsize,
  int nfrm, int nfft, char* wintype, FP_TYPE** dst_spec, FP_TYPE** dst_phse);
void llsm_compute_dc(FP_TYPE* x, int nx, int* center, int* winsize, int nfrm,
  FP_TYPE* dst_dc);
void llsm_harmonic_peakpicking(FP_TYPE* spectrum, FP_TYPE* phase,
  int nfft, FP_TYPE fs, int nhar, FP_TYPE f0, FP_TYPE* dst_ampl, FP_TYPE* dst_phse);
void llsm_harmonic_czt(FP_TYPE* x, int nx, FP_TYPE f0, FP_TYPE fs, int nhar,
  FP_TYPE* dst_ampl, FP_TYPE* dst_phse);
void llsm_harmonic_analysis(FP_TYPE* x, int nx, FP_TYPE fs, FP_TYPE* f0,
  int nfrm, FP_TYPE thop, FP_TYPE rel_winsize, int maxnhar, int method,
  int* dst_nhar, FP_TYPE** dst_ampl, FP_TYPE** dst_phse);
FP_TYPE* llsm_subband_energy(FP_TYPE* x, int nx, FP_TYPE fmin, FP_TYPE fmax);
void llsm_fft_to_psd(FP_TYPE* X_re, FP_TYPE* X_im, int nfft, FP_TYPE wsqr,
  FP_TYPE* dst_psd);
void llsm_estimate_psd(FP_TYPE* x, int nx, int nfft, FP_TYPE* dst_psd);
FP_TYPE* llsm_warp_frequency(FP_TYPE fmin, FP_TYPE fmax, int n, FP_TYPE warp_const);
FP_TYPE* llsm_spectral_mean(FP_TYPE* spectrum, int nspec, FP_TYPE fnyq,
  FP_TYPE* freq, int nfreq);
FP_TYPE* llsm_spectrum_from_envelope(FP_TYPE* freq, FP_TYPE* ampl, int nfreq,
  int nspec, FP_TYPE fnyq);
int llsm_get_fftsize(FP_TYPE* f0, int nfrm, FP_TYPE fs, FP_TYPE rel_winsize);
FP_TYPE* llsm_synthesize_harmonic_frame(FP_TYPE* ampl, FP_TYPE* phse, int nhar,
  FP_TYPE f0, int nx);
FP_TYPE* llsm_synthesize_harmonic_frame_iczt(FP_TYPE* ampl, FP_TYPE* phse,
  int nhar, FP_TYPE f0, int nx);
FP_TYPE* llsm_generate_white_noise(int nx);
FP_TYPE* llsm_generate_bandlimited_noise(int nx, FP_TYPE fmin, FP_TYPE fmax);
void llsm_lipfilter(FP_TYPE radius, FP_TYPE f0, int nhar,
  FP_TYPE* dst_ampl, FP_TYPE* dst_phse, int inverse);
void llsm_lipfilter_reim(FP_TYPE radius, FP_TYPE f0, int nhar,
  FP_TYPE* dst_re, FP_TYPE* dst_im, int inverse);
FP_TYPE* llsm_harmonic_spectrum(FP_TYPE* ampl, int nhar, FP_TYPE f0, int nfft);
FP_TYPE* llsm_harmonic_envelope(FP_TYPE* ampl, int nhar, FP_TYPE f0, int nfft);
FP_TYPE* llsm_harmonic_minphase(FP_TYPE* ampl, int nhar);
llsm_cached_glottal_model* llsm_create_cached_glottal_model(FP_TYPE* param,
  int nparam, int nhar);
void llsm_delete_cached_glottal_model(llsm_cached_glottal_model* dst);
FP_TYPE llsm_spectral_glottal_fitting(FP_TYPE* ampl, int nhar,
  llsm_cached_glottal_model* model);
FP_TYPE* llsm_smoothing_filter(FP_TYPE* x, int nx, int order);

llsm_rtsynth_buffer* llsm_create_rtsynth_buffer(llsm_soptions* options,
  llsm_container* conf, int capacity_samples);
void llsm_delete_rtsynth_buffer(llsm_rtsynth_buffer* dst);
int llsm_rtsynth_buffer_getlatency(llsm_rtsynth_buffer* src);
int llsm_rtsynth_buffer_numoutput(llsm_rtsynth_buffer* src);
void llsm_rtsynth_buffer_feed(llsm_rtsynth_buffer* dst, llsm_container* frame);
int llsm_rtsynth_buffer_fetch(llsm_rtsynth_buffer* src, FP_TYPE* dst);
int llsm_rtsynth_buffer_fetch_decomposed(
  llsm_rtsynth_buffer* src, FP_TYPE* dst_p, FP_TYPE* dst_ap);
void llsm_rtsynth_buffer_clear(llsm_rtsynth_buffer* dst);

llsm_gfm llsm_lfmodel_to_gfm(lfmodel src);
lfmodel llsm_gfm_to_lfmodel(llsm_gfm src);
FP_TYPE* llsm_synthesize_harmonic_frame_auto(llsm_soptions* options,
  FP_TYPE* ampl, FP_TYPE* phse, int nhar, FP_TYPE f0, int nx);
FP_TYPE* llsm_make_filtered_pulse(llsm_container* src, lfmodel* sources,
  FP_TYPE* offsets, int num_pulses, int pre_rotate, int size, FP_TYPE fnyq,
  FP_TYPE lip_radius, FP_TYPE fs);

FP_TYPE cig_qifft(FP_TYPE* magn, int k, FP_TYPE* dst_freq);
FP_TYPE* cig_spec2env(FP_TYPE* S, int nfft, FP_TYPE f0, int nhar, FP_TYPE* Cout);
lfmodel cig_lfmodel_from_rd(FP_TYPE rd, FP_TYPE T0, FP_TYPE Ee);
FP_TYPE* cig_lfmodel_spectrum(lfmodel model, FP_TYPE* freq, int nf, FP_TYPE* dst_phase);
FP_TYPE* cig_lfmodel_period(lfmodel model, int fs, int n);
ifdetector* cig_create_ifdetector(FP_TYPE fc, FP_TYPE fres);
void cig_delete_ifdetector(ifdetector* dst);
FP_TYPE cig_ifdetector_estimate(ifdetector* ifd, FP_TYPE* x, int nx);
filterbank* cig_create_empty_filterbank(int nf, FP_TYPE fnyq, int nchannel);
filterbank* cig_create_plp_filterbank(int nf, FP_TYPE fnyq, int nchannel);
filterbank* cig_create_melfreq_filterbank(int nf, FP_TYPE fnyq, int nchannel,
  FP_TYPE min_freq, FP_TYPE max_freq, FP_TYPE scale, FP_TYPE min_width);
void cig_delete_filterbank(filterbank* dst);
FP_TYPE** cig_filterbank_spectrogram(filterbank* fbank, FP_TYPE** S, int nfrm,
  int nfft, int fs, int crtenergy);
FP_TYPE* cig_filterbank_spectrum(filterbank* fbank, FP_TYPE* S, int nfft, int fs,
  int crtenergy);

void llsm_py_container_attach(llsm_container* dst, int index, void* ptr,
  llsm_fdestructor dtor, llsm_fcopy copyctor);
void llsm_py_free(void* p);
"""

ffibuilder.cdef(CDEF)

_libllsm2 = Path("..") / "libllsm2"

SOURCES = [
    _libllsm2 / "container.c",
    _libllsm2 / "frame.c",
    _libllsm2 / "dsputils.c",
    _libllsm2 / "llsmutils.c",
    _libllsm2 / "layer0.c",
    _libllsm2 / "layer1.c",
    _libllsm2 / "coder.c",
    _libllsm2 / "llsmrt.c",
    _libllsm2 / "external" / "ciglet" / "ciglet.c",
    _libllsm2 / "external" / "ciglet" / "external" / "fftsg_h.c",
    _libllsm2 / "external" / "ciglet" / "external" / "fast_median.c",
    _libllsm2 / "external" / "ciglet" / "external" / "wavfile.c",
]

INCLUDE_DIRS = [
    _libllsm2,
    _libllsm2 / "external",
    _libllsm2 / "external" / "ciglet",
    _libllsm2 / "external" / "ciglet" / "external",
]

libraries = []
if platform.system() != "Windows":
    libraries.append("m")

extra_compile_args = []
if platform.system() != "Windows":
    extra_compile_args.append("-std=c99")

ffibuilder.set_source(
    "pyllsm2._pyllsm2_cffi",
    r"""
    #include <stdlib.h>
    #define FP_TYPE float
    #include "llsm.h"
    #include "dsputils.h"
    #include "llsmrt.h"
    #include "llsmutils.h"
    #include <ciglet/ciglet.h>

    void llsm_py_container_attach(llsm_container* dst, int index, void* ptr,
      llsm_fdestructor dtor, llsm_fcopy copyctor) {
      llsm_container_attach_(dst, index, ptr, dtor, copyctor);
    }

    void llsm_py_free(void* p) {
      free(p);
    }
    """,
    sources=[str(p).replace("\\", "/") for p in SOURCES],
    include_dirs=[str(p).replace("\\", "/") for p in INCLUDE_DIRS],
    define_macros=[("FP_TYPE", "float")],
    libraries=libraries,
    extra_compile_args=extra_compile_args,
)

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
