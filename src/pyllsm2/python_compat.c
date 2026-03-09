#include <stdlib.h>
#include <string.h>
#include <math.h>

#define FP_TYPE float
#include "llsm.h"
#include "dsputils.h"
#include "llsmrt.h"

/* pyllsm2 modification: this compatibility layer now builds persistent
 * C-backed array views so Layer0Features/Layer1Features can expose NumPy
 * ndarray-compatible memory directly instead of Python-side auto-commit
 * wrappers.
 *
 * pyllsm2 modification: after the ndarray-backed chunk refactor, the
 * remaining llsm_py_* helpers are limited to wrappers that still serve the
 * current Python API. Legacy copy/export helpers tied to the pre-ndarray
 * compatibility layer are retired here instead of being kept half-working. */

static int llsm_py_conf_nfrm(llsm_chunk* chunk) {
  if(chunk == NULL || chunk->conf == NULL)
    return -1;
  int* nfrm = (int*)llsm_container_get(chunk->conf, LLSM_CONF_NFRM);
  if(nfrm == NULL)
    return -1;
  return nfrm[0];
}

static llsm_container* llsm_py_frame_at(llsm_chunk* chunk, int i) {
  if(chunk == NULL || chunk->frames == NULL)
    return NULL;
  return chunk->frames[i];
}

static size_t llsm_py_vector_row_bytes(int max_len) {
  return sizeof(int) + (size_t)max_len * sizeof(FP_TYPE);
}

static FP_TYPE* llsm_py_vector_row_ptr(void* block, int row, int max_len) {
  char* base = (char*)block + (size_t)row * llsm_py_vector_row_bytes(max_len);
  return (FP_TYPE*)((int*)base + 1);
}

static void llsm_py_vector_row_set_length(void* block, int row, int max_len, int len) {
  char* base = (char*)block + (size_t)row * llsm_py_vector_row_bytes(max_len);
  ((int*)base)[0] = len;
}

static void* llsm_py_create_vector_block(int nfrm, int max_len) {
  size_t row_bytes = llsm_py_vector_row_bytes(max_len);
  char* block = (char*)calloc((size_t)nfrm, row_bytes > 0 ? row_bytes : sizeof(int));
  if(block == NULL && nfrm > 0)
    return NULL;
  for(int i = 0; i < nfrm; i++)
    llsm_py_vector_row_set_length(block, i, max_len, 0);
  return block;
}

static int llsm_py_copy_owned_scalar_member(llsm_container* dst_frame, llsm_container* src_frame, int member_idx) {
  FP_TYPE* src = src_frame != NULL ? (FP_TYPE*)llsm_container_get(src_frame, member_idx) : NULL;
  if(src == NULL) {
    llsm_container_remove(dst_frame, member_idx);
    return 0;
  }
  FP_TYPE* dst = llsm_create_fp(src[0]);
  if(dst == NULL)
    return -1;
  llsm_container_attach_(dst_frame, member_idx, (void*)dst,
                         (llsm_fdestructor)llsm_delete_fp,
                         (llsm_fcopy)llsm_copy_fp);
  return 0;
}

static int llsm_py_copy_owned_vector_member(llsm_container* dst_frame, llsm_container* src_frame, int member_idx) {
  FP_TYPE* src = src_frame != NULL ? (FP_TYPE*)llsm_container_get(src_frame, member_idx) : NULL;
  if(src == NULL) {
    llsm_container_remove(dst_frame, member_idx);
    return 0;
  }
  int len = llsm_fparray_length(src);
  FP_TYPE* dst = llsm_create_fparray(len);
  if(dst == NULL)
    return -1;
  if(len > 0)
    memcpy(dst, src, (size_t)len * sizeof(FP_TYPE));
  llsm_container_attach_(dst_frame, member_idx, (void*)dst,
                         (llsm_fdestructor)llsm_delete_fparray,
                         (llsm_fcopy)llsm_copy_fparray);
  return 0;
}

int llsm_py_aoptions_resize_nchannel(llsm_aoptions* options, int nchannel) {
  if(options == NULL)
    return -1;
  if(nchannel < 1)
    return -2;
  int old_nfreq = options->nchannel > 0 ? options->nchannel - 1 : 0;
  int new_nfreq = nchannel - 1;
  int alloc_nfreq = new_nfreq > 0 ? new_nfreq : 1;
  FP_TYPE* new_freq = (FP_TYPE*)calloc((size_t)alloc_nfreq, sizeof(FP_TYPE));
  if(new_freq == NULL)
    return -3;
  int copy_nfreq = old_nfreq < new_nfreq ? old_nfreq : new_nfreq;
  if(copy_nfreq > 0 && options->chanfreq != NULL)
    memcpy(new_freq, options->chanfreq, (size_t)copy_nfreq * sizeof(FP_TYPE));
  if(new_nfreq > old_nfreq) {
    FP_TYPE fill = (old_nfreq > 0 && options->chanfreq != NULL) ? options->chanfreq[old_nfreq - 1] : 0.0f;
    for(int i = old_nfreq; i < new_nfreq; i++)
      new_freq[i] = fill;
  }
  free(options->chanfreq);
  options->chanfreq = new_freq;
  options->nchannel = nchannel;
  return 0;
}

int llsm_py_aoptions_set_chanfreq(llsm_aoptions* options, FP_TYPE* src, int nfreq) {
  if(options == NULL)
    return -1;
  if(nfreq < 0)
    return -2;
  int alloc_nfreq = nfreq > 0 ? nfreq : 1;
  FP_TYPE* new_freq = (FP_TYPE*)calloc((size_t)alloc_nfreq, sizeof(FP_TYPE));
  if(new_freq == NULL)
    return -3;
  if(nfreq > 0 && src != NULL)
    memcpy(new_freq, src, (size_t)nfreq * sizeof(FP_TYPE));
  free(options->chanfreq);
  options->chanfreq = new_freq;
  options->nchannel = nfreq + 1;
  return 0;
}

/* pyllsm2 modification: chunk deep-copy in modified vendor/container.c still
 * relies on this hook to re-own shared ndarray-backed frame members in the
 * destination chunk. */
int llsm_py_chunk_prepare_copy(llsm_chunk* dst, llsm_chunk* src) {
  int nfrm = llsm_py_conf_nfrm(src);
  if(nfrm < 0)
    return -1;
  for(int i = 0; i < nfrm; i++) {
    llsm_container* src_frame = llsm_py_frame_at(src, i);
    llsm_container* dst_frame = llsm_py_frame_at(dst, i);
    if(dst_frame == NULL)
      continue;
    if(src->py_frame_f0 != NULL && llsm_py_copy_owned_scalar_member(dst_frame, src_frame, LLSM_FRAME_F0) != 0)
      return -2;
    if(src->py_frame_rd != NULL && llsm_py_copy_owned_scalar_member(dst_frame, src_frame, LLSM_FRAME_RD) != 0)
      return -3;
    if(src->py_vtmagn_block != NULL && llsm_py_copy_owned_vector_member(dst_frame, src_frame, LLSM_FRAME_VTMAGN) != 0)
      return -4;
    if(src->py_vsphse_block != NULL && llsm_py_copy_owned_vector_member(dst_frame, src_frame, LLSM_FRAME_VSPHSE) != 0)
      return -5;
  }
  return 0;
}


/* Build/rebuild a shared frame-scalar buffer with logical shape (nfrm,), where
 * axis 0 is frame index over time, and rebind each frame's F0 pointer into it. */
int llsm_py_chunk_refresh_f0_view(llsm_chunk* chunk) {
  int nfrm = llsm_py_conf_nfrm(chunk);
  if(nfrm < 0)
    return -1;
  FP_TYPE* new_f0 = (FP_TYPE*)calloc((size_t)nfrm, sizeof(FP_TYPE));
  if(new_f0 == NULL && nfrm > 0)
    return -2;
  for(int i = 0; i < nfrm; i++) {
    llsm_container* frame = llsm_py_frame_at(chunk, i);
    FP_TYPE* f0 = frame != NULL ? (FP_TYPE*)llsm_container_get(frame, LLSM_FRAME_F0) : NULL;
    if(f0 != NULL)
      new_f0[i] = f0[0];
  }
  FP_TYPE* old_f0 = chunk->py_frame_f0;
  chunk->py_frame_f0 = new_f0;
  for(int i = 0; i < nfrm; i++) {
    llsm_container* frame = llsm_py_frame_at(chunk, i);
    if(frame != NULL)
      llsm_container_attach_(frame, LLSM_FRAME_F0, (void*)(chunk->py_frame_f0 + i), NULL, NULL);
  }
  free(old_f0);
  return 0;
}

/* Build/rebuild a shared Rd buffer with logical shape (nfrm,), where axis 0 is
 * frame index over time, and rebind each frame's Rd pointer into it. */
int llsm_py_chunk_refresh_rd_view(llsm_chunk* chunk) {
  int nfrm = llsm_py_conf_nfrm(chunk);
  if(nfrm < 0)
    return -1;
  FP_TYPE* new_rd = (FP_TYPE*)calloc((size_t)nfrm, sizeof(FP_TYPE));
  if(new_rd == NULL && nfrm > 0)
    return -2;
  for(int i = 0; i < nfrm; i++) {
    llsm_container* frame = llsm_py_frame_at(chunk, i);
    FP_TYPE* rd = frame != NULL ? (FP_TYPE*)llsm_container_get(frame, LLSM_FRAME_RD) : NULL;
    new_rd[i] = rd != NULL ? rd[0] : NAN;
  }
  FP_TYPE* old_rd = chunk->py_frame_rd;
  chunk->py_frame_rd = new_rd;
  for(int i = 0; i < nfrm; i++) {
    llsm_container* frame = llsm_py_frame_at(chunk, i);
    if(frame != NULL)
      llsm_container_attach_(frame, LLSM_FRAME_RD, (void*)(chunk->py_frame_rd + i), NULL, NULL);
  }
  free(old_rd);
  return 0;
}

/* Build/rebuild shared harmonic matrices with logical shape
 * (nfrm, max_nhar): axis 0 is frame index over time, axis 1 is harmonic index.
 * Each frame's hm->ampl and hm->phse are rebound to its corresponding row. */
int llsm_py_chunk_refresh_harmonics_view(llsm_chunk* chunk, int max_nhar) {
  int nfrm = llsm_py_conf_nfrm(chunk);
  if(nfrm < 0)
    return -1;
  if(max_nhar < 0)
    return -2;
  if(max_nhar == 0) {
    for(int i = 0; i < nfrm; i++) {
      llsm_container* frame = llsm_py_frame_at(chunk, i);
      llsm_hmframe* hm = frame != NULL ? (llsm_hmframe*)llsm_container_get(frame, LLSM_FRAME_HM) : NULL;
      if(hm != NULL && hm->nhar > max_nhar)
        max_nhar = hm->nhar;
    }
  }
  int* new_nhar = (int*)calloc((size_t)nfrm, sizeof(int));
  FP_TYPE* new_ampl = (FP_TYPE*)calloc((size_t)nfrm * (size_t)max_nhar, sizeof(FP_TYPE));
  FP_TYPE* new_phse = (FP_TYPE*)calloc((size_t)nfrm * (size_t)max_nhar, sizeof(FP_TYPE));
  if((new_nhar == NULL || new_ampl == NULL || new_phse == NULL) && (nfrm > 0 || max_nhar > 0)) {
    free(new_nhar);
    free(new_ampl);
    free(new_phse);
    return -3;
  }
  FP_TYPE* old_ampl = chunk->py_hm_ampl;
  FP_TYPE* old_phse = chunk->py_hm_phse;
  int* old_nhar = chunk->py_frame_nhar;
  for(int i = 0; i < nfrm; i++) {
    llsm_container* frame = llsm_py_frame_at(chunk, i);
    if(frame == NULL)
      continue;
    llsm_hmframe* hm = frame != NULL ? (llsm_hmframe*)llsm_container_get(frame, LLSM_FRAME_HM) : NULL;
    if(hm == NULL) {
      hm = llsm_create_hmframe(0);
      if(hm == NULL) {
        free(new_nhar);
        free(new_ampl);
        free(new_phse);
        return -4;
      }
      llsm_container_attach_(frame, LLSM_FRAME_HM, (void*)hm,
                             (llsm_fdestructor)llsm_delete_hmframe,
                             (llsm_fcopy)llsm_copy_hmframe);
    }
    int nhar = hm->nhar;
    if(nhar > max_nhar)
      nhar = max_nhar;
    new_nhar[i] = nhar;
    if(nhar > 0 && hm->ampl != NULL && hm->phse != NULL) {
      memcpy(new_ampl + (size_t)i * (size_t)max_nhar, hm->ampl, (size_t)nhar * sizeof(FP_TYPE));
      memcpy(new_phse + (size_t)i * (size_t)max_nhar, hm->phse, (size_t)nhar * sizeof(FP_TYPE));
    }
    if(hm->owns_vectors) {
      free(hm->ampl);
      free(hm->phse);
    }
    hm->ampl = max_nhar > 0 ? new_ampl + (size_t)i * (size_t)max_nhar : NULL;
    hm->phse = max_nhar > 0 ? new_phse + (size_t)i * (size_t)max_nhar : NULL;
    hm->nhar = nhar;
    hm->owns_vectors = 0;
  }
  chunk->py_frame_nhar = new_nhar;
  chunk->py_hm_ampl = new_ampl;
  chunk->py_hm_phse = new_phse;
  chunk->py_hm_max_nhar = max_nhar;
  free(old_nhar);
  free(old_ampl);
  free(old_phse);
  return 0;
}

/* Build/rebuild shared variable-length row storage with logical values shape
 * (nfrm, max_len): axis 0 is frame index over time, axis 1 is per-frame element
 * index. Each row stores a leading valid length plus the FP_TYPE payload. */
int llsm_py_chunk_refresh_vector_view(llsm_chunk* chunk, int member_idx, int max_len) {
  int nfrm = llsm_py_conf_nfrm(chunk);
  if(nfrm < 0)
    return -1;
  if(max_len < 0)
    return -2;
  if(max_len == 0) {
    for(int i = 0; i < nfrm; i++) {
      llsm_container* frame = llsm_py_frame_at(chunk, i);
      FP_TYPE* vec = frame != NULL ? (FP_TYPE*)llsm_container_get(frame, member_idx) : NULL;
      if(vec != NULL) {
        int len_i = llsm_fparray_length(vec);
        if(len_i > max_len)
          max_len = len_i;
      }
    }
  }
  void* new_block = llsm_py_create_vector_block(nfrm, max_len);
  if(new_block == NULL && nfrm > 0)
    return -3;
  void* old_block = member_idx == LLSM_FRAME_VTMAGN ? chunk->py_vtmagn_block : chunk->py_vsphse_block;
  for(int i = 0; i < nfrm; i++) {
    llsm_container* frame = llsm_py_frame_at(chunk, i);
    FP_TYPE* vec = frame != NULL ? (FP_TYPE*)llsm_container_get(frame, member_idx) : NULL;
    int len_i = 0;
    if(vec != NULL) {
      len_i = llsm_fparray_length(vec);
      if(len_i > max_len)
        len_i = max_len;
    }
    llsm_py_vector_row_set_length(new_block, i, max_len, len_i);
    FP_TYPE* row = llsm_py_vector_row_ptr(new_block, i, max_len);
    if(len_i > 0 && vec != NULL)
      memcpy(row, vec, (size_t)len_i * sizeof(FP_TYPE));
    if(frame != NULL)
      llsm_container_attach_(frame, member_idx, (void*)row, NULL, NULL);
  }
  if(member_idx == LLSM_FRAME_VTMAGN) {
    chunk->py_vtmagn_block = new_block;
    chunk->py_vtmagn_max_len = max_len;
  } else if(member_idx == LLSM_FRAME_VSPHSE) {
    chunk->py_vsphse_block = new_block;
    chunk->py_vsphse_max_len = max_len;
  }
  free(old_block);
  return 0;
}

int llsm_py_chunk_nfrm(llsm_chunk* chunk) {
  return llsm_py_conf_nfrm(chunk);
}



int llsm_py_chunk_fill_nhar(llsm_chunk* chunk, int* dst, int nfrm) {
  int conf_nfrm = llsm_py_conf_nfrm(chunk);
  if(conf_nfrm < 0 || dst == NULL)
    return -1;
  if(conf_nfrm != nfrm)
    return -2;
  for(int i = 0; i < nfrm; i++) {
    dst[i] = 0;
    if(chunk->py_frame_nhar != NULL)
      chunk->py_frame_nhar[i] = 0;
    llsm_container* frame = llsm_py_frame_at(chunk, i);
    llsm_hmframe* hm = frame != NULL ? (llsm_hmframe*)llsm_container_get(frame, LLSM_FRAME_HM) : NULL;
    if(hm != NULL) {
      dst[i] = hm->nhar;
      if(chunk->py_frame_nhar != NULL)
        chunk->py_frame_nhar[i] = hm->nhar;
    }
  }
  return 0;
}

/* pyllsm2 modification: F0 writes now canonicalize through the shared
 * ndarray-backed frame buffer instead of the older per-frame fallback path. */
int llsm_py_chunk_set_f0(llsm_chunk* chunk, FP_TYPE* src, int nfrm) {
  int conf_nfrm = llsm_py_conf_nfrm(chunk);
  if(conf_nfrm < 0 || src == NULL)
    return -1;
  if(conf_nfrm != nfrm)
    return -2;
  if(chunk->py_frame_f0 == NULL) {
    int rc = llsm_py_chunk_refresh_f0_view(chunk);
    if(rc != 0)
      return -3;
  }
  memcpy(chunk->py_frame_f0, src, (size_t)nfrm * sizeof(FP_TYPE));
  return 0;
}



/* pyllsm2 modification: harmonic writes also canonicalize through the
 * shared ndarray-backed harmonic matrix so Python never falls back to the
 * older per-frame realloc workflow. */
int llsm_py_chunk_frame_set_hm(
    llsm_chunk* chunk, int frame_idx, FP_TYPE* ampl, FP_TYPE* phse, int nhar) {
  int nfrm = llsm_py_conf_nfrm(chunk);
  if(nfrm < 0)
    return -1;
  if(frame_idx < 0 || frame_idx >= nfrm)
    return -2;
  if(nhar < 0)
    return -3;
  if(nhar > 0 && (ampl == NULL || phse == NULL))
    return -6;
  if(chunk->py_hm_ampl == NULL || nhar > chunk->py_hm_max_nhar) {
    int target_nhar = nhar > chunk->py_hm_max_nhar ? nhar : chunk->py_hm_max_nhar;
    int rc = llsm_py_chunk_refresh_harmonics_view(chunk, target_nhar);
    if(rc != 0)
      return rc;
  }
  llsm_container* frame = chunk->frames != NULL ? chunk->frames[frame_idx] : NULL;
  llsm_hmframe* hm = frame != NULL ? (llsm_hmframe*)llsm_container_get(frame, LLSM_FRAME_HM) : NULL;
  if(hm == NULL)
    return -4;
  if(hm->ampl != NULL && chunk->py_hm_max_nhar > 0)
    memset(hm->ampl, 0, (size_t)chunk->py_hm_max_nhar * sizeof(FP_TYPE));
  if(hm->phse != NULL && chunk->py_hm_max_nhar > 0)
    memset(hm->phse, 0, (size_t)chunk->py_hm_max_nhar * sizeof(FP_TYPE));
  if(nhar > 0) {
    memcpy(hm->ampl, ampl, (size_t)nhar * sizeof(FP_TYPE));
    memcpy(hm->phse, phse, (size_t)nhar * sizeof(FP_TYPE));
  }
  hm->nhar = nhar;
  if(chunk->py_frame_nhar != NULL)
    chunk->py_frame_nhar[frame_idx] = nhar;
  return 0;
}


int llsm_py_chunk_set_harmonics_matrix(
    llsm_chunk* chunk,
    FP_TYPE* src_ampl,
    FP_TYPE* src_phse,
    int* src_nhar,
    int nfrm,
    int max_nhar) {
  int conf_nfrm = llsm_py_conf_nfrm(chunk);
  if(conf_nfrm < 0 || src_ampl == NULL || src_phse == NULL || src_nhar == NULL)
    return -1;
  if(nfrm != conf_nfrm || max_nhar < 0)
    return -2;
  if(llsm_py_chunk_refresh_harmonics_view(chunk, max_nhar) != 0)
    return -6;
  for(int i = 0; i < nfrm; i++) {
    int nhar = src_nhar[i];
    if(nhar < 0 || nhar > max_nhar)
      return -3;
    llsm_container* frame = llsm_py_frame_at(chunk, i);
    llsm_hmframe* hm = frame != NULL ? (llsm_hmframe*)llsm_container_get(frame, LLSM_FRAME_HM) : NULL;
    if(hm == NULL)
      return -4;
    memset(hm->ampl, 0, (size_t)max_nhar * sizeof(FP_TYPE));
    memset(hm->phse, 0, (size_t)max_nhar * sizeof(FP_TYPE));
    if(nhar > 0) {
      memcpy(hm->ampl, src_ampl + (size_t)i * (size_t)max_nhar, (size_t)nhar * sizeof(FP_TYPE));
      memcpy(hm->phse, src_phse + (size_t)i * (size_t)max_nhar, (size_t)nhar * sizeof(FP_TYPE));
    }
    hm->nhar = nhar;
    if(chunk->py_frame_nhar != NULL)
      chunk->py_frame_nhar[i] = nhar;
  }
  return 0;
}

int llsm_py_chunk_clear_member_mask(llsm_chunk* chunk, int member_idx, unsigned char* mask, int nmask) {
  int nfrm = llsm_py_conf_nfrm(chunk);
  if(nfrm < 0)
    return -1;
  if(mask != NULL && nmask != nfrm)
    return -2;
  for(int i = 0; i < nfrm; i++) {
    if(mask == NULL || mask[i] != 0) {
      llsm_container* frame = llsm_py_frame_at(chunk, i);
      if(frame != NULL)
        llsm_container_remove(frame, member_idx);
    }
  }
  return 0;
}

int llsm_py_chunk_enable_pbp_mask(llsm_chunk* chunk, unsigned char* mask, int nmask, int clear_hm) {
  int nfrm = llsm_py_conf_nfrm(chunk);
  if(nfrm < 0 || mask == NULL)
    return -1;
  if(nmask != nfrm)
    return -2;
  int rc = llsm_py_chunk_clear_member_mask(chunk, LLSM_FRAME_PBPSYN, NULL, 0);
  if(rc != 0)
    return -3;
  if(clear_hm) {
    for(int i = 0; i < nfrm; i++) {
      llsm_container* frame = llsm_py_frame_at(chunk, i);
      if(frame != NULL)
        llsm_container_attach_(frame, LLSM_FRAME_HM, NULL, NULL, NULL);
    }
  }
  for(int i = 0; i < nfrm; i++) {
    if(mask[i] == 0)
      continue;
    llsm_container* frame = llsm_py_frame_at(chunk, i);
    if(frame == NULL)
      continue;
    int* flag = llsm_create_int(1);
    if(flag == NULL)
      return -4;
    llsm_container_attach_(frame, LLSM_FRAME_PBPSYN, (void*)flag,
                           (llsm_fdestructor)llsm_delete_int, (llsm_fcopy)llsm_copy_int);
  }
  return 0;
}

int llsm_py_chunk_get_scalar(
    llsm_chunk* chunk, int member_idx, double default_value, FP_TYPE* dst, int nfrm) {
  int conf_nfrm = llsm_py_conf_nfrm(chunk);
  if(conf_nfrm < 0 || dst == NULL)
    return -1;
  if(conf_nfrm != nfrm)
    return -2;
  for(int i = 0; i < nfrm; i++) {
    dst[i] = (FP_TYPE)default_value;
    llsm_container* frame = llsm_py_frame_at(chunk, i);
    FP_TYPE* ptr = frame != NULL ? (FP_TYPE*)llsm_container_get(frame, member_idx) : NULL;
    if(ptr != NULL)
      dst[i] = ptr[0];
  }
  return 0;
}

/* pyllsm2 modification: RD writes follow the shared ndarray-backed scalar
 * buffer; only non-ndarray scalar members still use per-frame attachments. */
int llsm_py_chunk_set_scalar(llsm_chunk* chunk, int member_idx, FP_TYPE* src, int nfrm) {
  int conf_nfrm = llsm_py_conf_nfrm(chunk);
  if(conf_nfrm < 0 || src == NULL)
    return -1;
  if(conf_nfrm != nfrm)
    return -2;
  if(member_idx == LLSM_FRAME_RD) {
    if(chunk->py_frame_rd == NULL) {
      int rc = llsm_py_chunk_refresh_rd_view(chunk);
      if(rc != 0)
        return -3;
    }
    memcpy(chunk->py_frame_rd, src, (size_t)nfrm * sizeof(FP_TYPE));
    return 0;
  }
  for(int i = 0; i < nfrm; i++) {
    llsm_container* frame = llsm_py_frame_at(chunk, i);
    if(frame == NULL)
      continue;
    FP_TYPE* p = llsm_create_fp(src[i]);
    if(p == NULL)
      return -4;
    llsm_container_attach_(frame, member_idx, (void*)p, (llsm_fdestructor)llsm_delete_fp,
                           (llsm_fcopy)llsm_copy_fp);
  }
  return 0;
}

int llsm_py_chunk_fill_vector_lengths(llsm_chunk* chunk, int member_idx, int* dst, int nfrm) {
  int conf_nfrm = llsm_py_conf_nfrm(chunk);
  if(conf_nfrm < 0 || dst == NULL)
    return -1;
  if(conf_nfrm != nfrm)
    return -2;
  for(int i = 0; i < nfrm; i++) {
    dst[i] = 0;
    llsm_container* frame = llsm_py_frame_at(chunk, i);
    FP_TYPE* vec = frame != NULL ? (FP_TYPE*)llsm_container_get(frame, member_idx) : NULL;
    if(vec != NULL)
      dst[i] = llsm_fparray_length(vec);
  }
  return 0;
}


int llsm_py_chunk_set_vector_matrix(
    llsm_chunk* chunk,
    int member_idx,
    FP_TYPE* src,
    int* src_len,
    int nfrm,
    int max_len) {
  int conf_nfrm = llsm_py_conf_nfrm(chunk);
  if(conf_nfrm < 0 || src == NULL || src_len == NULL)
    return -1;
  if(conf_nfrm != nfrm || max_len < 0)
    return -2;
  if((member_idx == LLSM_FRAME_VTMAGN || member_idx == LLSM_FRAME_VSPHSE) &&
     llsm_py_chunk_refresh_vector_view(chunk, member_idx, max_len) != 0)
    return -5;
  for(int i = 0; i < nfrm; i++) {
    int len_i = src_len[i];
    if(len_i < 0 || len_i > max_len)
      return -3;
    llsm_container* frame = llsm_py_frame_at(chunk, i);
    if(frame == NULL)
      continue;
    if(member_idx == LLSM_FRAME_VTMAGN || member_idx == LLSM_FRAME_VSPHSE) {
      void* block = member_idx == LLSM_FRAME_VTMAGN ? chunk->py_vtmagn_block : chunk->py_vsphse_block;
      FP_TYPE* row = llsm_py_vector_row_ptr(block, i, max_len);
      llsm_py_vector_row_set_length(block, i, max_len, len_i);
      memset(row, 0, (size_t)max_len * sizeof(FP_TYPE));
      if(len_i > 0)
        memcpy(row, src + (size_t)i * (size_t)max_len, (size_t)len_i * sizeof(FP_TYPE));
      continue;
    }
    FP_TYPE* vec = llsm_create_fparray(len_i);
    if(vec == NULL)
      return -4;
    memcpy(vec, src + (size_t)i * (size_t)max_len, (size_t)len_i * sizeof(FP_TYPE));
    llsm_container_attach_(frame, member_idx, (void*)vec,
                           (llsm_fdestructor)llsm_delete_fparray,
                           (llsm_fcopy)llsm_copy_fparray);
  }
  return 0;
}


int llsm_py_chunk_pitch_shift_layer1(
    llsm_chunk* chunk, double ratio, int compensate_vtmagn_db, int clear_harmonics) {
  int nfrm = llsm_py_conf_nfrm(chunk);
  if(nfrm < 0)
    return -1;
  if(ratio <= 0.0)
    return -2;
  FP_TYPE ratiof = (FP_TYPE)ratio;
  FP_TYPE shift = (FP_TYPE)(20.0 * log10(ratio));
  for(int i = 0; i < nfrm; i++) {
    llsm_container* frame = llsm_py_frame_at(chunk, i);
    if(frame == NULL)
      continue;
    if(clear_harmonics)
      llsm_container_remove(frame, LLSM_FRAME_HM);
    FP_TYPE* f0 = (FP_TYPE*)llsm_container_get(frame, LLSM_FRAME_F0);
    if(f0 != NULL)
      f0[0] *= ratiof;
    if(compensate_vtmagn_db) {
      FP_TYPE* vt = (FP_TYPE*)llsm_container_get(frame, LLSM_FRAME_VTMAGN);
      if(vt != NULL) {
        int nspec = llsm_fparray_length(vt);
        for(int j = 0; j < nspec; j++)
          vt[j] -= shift;
      }
    }
  }
  return 0;
}

int llsm_py_harmonic_analysis_matrix(
    FP_TYPE* x,
    int nx,
    double fs,
    FP_TYPE* f0,
    int nfrm,
    double thop,
    double rel_winsize,
    int maxnhar,
    int method,
    double pad_value,
    FP_TYPE* dst_ampl,
    FP_TYPE* dst_phse,
    int* dst_nhar,
    int dst_max_nhar) {
  if(x == NULL || f0 == NULL || dst_ampl == NULL || dst_phse == NULL || dst_nhar == NULL)
    return -1;
  if(nfrm < 0 || dst_max_nhar < 0)
    return -2;
  int* nhar = (int*)calloc((size_t)nfrm, sizeof(int));
  FP_TYPE** ampl_ptrs = (FP_TYPE**)calloc((size_t)nfrm, sizeof(FP_TYPE*));
  FP_TYPE** phse_ptrs = (FP_TYPE**)calloc((size_t)nfrm, sizeof(FP_TYPE*));
  if((nhar == NULL || ampl_ptrs == NULL || phse_ptrs == NULL) && nfrm > 0) {
    free(nhar);
    free(ampl_ptrs);
    free(phse_ptrs);
    return -3;
  }
  size_t total = (size_t)nfrm * (size_t)dst_max_nhar;
  for(size_t i = 0; i < total; i++) {
    dst_ampl[i] = (FP_TYPE)pad_value;
    dst_phse[i] = (FP_TYPE)pad_value;
  }
  llsm_harmonic_analysis(
      x, nx, (FP_TYPE)fs, f0, nfrm, (FP_TYPE)thop, (FP_TYPE)rel_winsize, maxnhar, method, nhar, ampl_ptrs, phse_ptrs);

  int rc = 0;
  for(int i = 0; i < nfrm; i++) {
    int n_i = nhar[i];
    dst_nhar[i] = n_i;
    if(n_i <= 0)
      continue;
    if(n_i > dst_max_nhar) {
      rc = -4;
      break;
    }
    if(ampl_ptrs[i] == NULL || phse_ptrs[i] == NULL) {
      rc = -5;
      break;
    }
    memcpy(dst_ampl + (size_t)i * (size_t)dst_max_nhar, ampl_ptrs[i], (size_t)n_i * sizeof(FP_TYPE));
    memcpy(dst_phse + (size_t)i * (size_t)dst_max_nhar, phse_ptrs[i], (size_t)n_i * sizeof(FP_TYPE));
  }
  for(int i = 0; i < nfrm; i++) {
    free(ampl_ptrs[i]);
    free(phse_ptrs[i]);
  }
  free(nhar);
  free(ampl_ptrs);
  free(phse_ptrs);
  return rc;
}

int llsm_py_coder_reconstruct_chunk(
    llsm_coder* coder, llsm_chunk* src, llsm_chunk* dst0, llsm_chunk* dst1) {
  int nfrm = llsm_py_conf_nfrm(src);
  if(nfrm < 0 || coder == NULL)
    return -1;
  if(dst0 == NULL && dst1 == NULL)
    return -2;
  if(dst0 != NULL && llsm_py_conf_nfrm(dst0) != nfrm)
    return -3;
  if(dst1 != NULL && llsm_py_conf_nfrm(dst1) != nfrm)
    return -4;
  for(int i = 0; i < nfrm; i++) {
    FP_TYPE* enc = llsm_coder_encode(coder, src->frames[i]);
    if(enc == NULL)
      return -5;
    if(dst0 != NULL) {
      llsm_container* fr0 = llsm_coder_decode_layer0(coder, enc);
      if(fr0 == NULL) {
        free(enc);
        return -6;
      }
      if(dst0->frames[i] != NULL)
        llsm_delete_container(dst0->frames[i]);
      dst0->frames[i] = fr0;
    }
    if(dst1 != NULL) {
      llsm_container* fr1 = llsm_coder_decode_layer1(coder, enc);
      if(fr1 == NULL) {
        free(enc);
        return -7;
      }
      if(dst1->frames[i] != NULL)
        llsm_delete_container(dst1->frames[i]);
      dst1->frames[i] = fr1;
    }
    free(enc);
  }
  return 0;
}

int llsm_py_rtsynth_render_chunk_decomposed(
    llsm_rtsynth_buffer* rt,
    llsm_chunk* chunk,
    FP_TYPE* dst_p,
    FP_TYPE* dst_ap,
    int nx,
    int trim_latency,
    int* dst_ny) {
  int nfrm = llsm_py_conf_nfrm(chunk);
  if(rt == NULL || chunk == NULL || dst_p == NULL || dst_ap == NULL || dst_ny == NULL)
    return -1;
  if(nx <= 0 || nfrm < 0)
    return -2;
  memset(dst_p, 0, (size_t)nx * sizeof(FP_TYPE));
  memset(dst_ap, 0, (size_t)nx * sizeof(FP_TYPE));
  int latency = llsm_rtsynth_buffer_getlatency(rt);
  int count = 0;
  FP_TYPE tmp_p = 0.0f;
  FP_TYPE tmp_ap = 0.0f;
  for(int i = 0; i < nfrm; i++) {
    llsm_rtsynth_buffer_feed(rt, chunk->frames[i]);
    while(count < nx) {
      int status = llsm_rtsynth_buffer_fetch_decomposed(rt, &tmp_p, &tmp_ap);
      if(status == 0)
        break;
      if(count >= latency) {
        dst_p[count - latency] = tmp_p;
        dst_ap[count - latency] = tmp_ap;
      }
      count++;
    }
  }
  while(count < nx) {
    int status = llsm_rtsynth_buffer_fetch_decomposed(rt, &tmp_p, &tmp_ap);
    if(status == 0)
      break;
    if(count >= latency) {
      dst_p[count - latency] = tmp_p;
      dst_ap[count - latency] = tmp_ap;
    }
    count++;
  }
  if(trim_latency) {
    int ny = nx - latency;
    if(ny < 1)
      ny = 1;
    dst_ny[0] = ny;
  } else {
    dst_ny[0] = nx;
  }
  return 0;
}
