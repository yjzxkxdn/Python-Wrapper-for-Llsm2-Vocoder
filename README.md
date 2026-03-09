# pyllsm2

`pyllsm2` is a Python wrapper for `libllsm2`, focused on a clear,
NumPy-friendly API for speech analysis and synthesis.

The package now has two explicit API layers:

- Python-first API at `pyllsm2`
- Raw CFFI API at `pyllsm2.raw`

Most users should start with `pyllsm2`.

## Additional Documentation

Detailed repository structure and API documentation are available in:

- `docs/README.zh-CN.md`
- `docs/project-structure.zh-CN.md`
- `docs/api-python.zh-CN.md`
- `docs/api-raw.zh-CN.md`
- `docs/development.zh-CN.md`


## Python Compatibility Layer

To improve NumPy integration and reduce Python-side loops, this package adds:

- `src/pyllsm2/python_compat.c`

This file exists only for Python-side integration helpers that bind high-level
feature containers to C-backed memory layouts. It is not part of upstream
`libllsm2` core sources.

## C Source Modifications

This repository intentionally modifies a small part of the bundled C sources to
support direct NumPy-style memory sharing for high-level feature arrays.

Modified files:

- `src/pyllsm2/python_compat.c`
- `vendor/libllsm2/llsm.h`
- `vendor/libllsm2/container.c`
- `vendor/libllsm2/frame.c`

Purpose of these modifications:

- make `Layer0Features` harmonic arrays C-backed and directly viewable as NumPy arrays
- make `Layer1Features` scalar/vector arrays C-backed and directly viewable as NumPy arrays
- remove reliance on Python-side auto-commit array wrappers for common ndarray workflows
- remove high-level legacy getter/setter style feature access in favor of direct ndarray properties
- keep chunk copy/free logic consistent when frame members are rebound to shared array storage

In the modified C/C header files, comments marked with `pyllsm2 modification`
identify the added compatibility code paths.

Internal audit note for `src/pyllsm2/python_compat.c`:

- retired legacy helper exports that only served the pre-ndarray compat path
- kept only helpers still used by the current high-level wrapper or raw regression tests
- canonicalized F0 / Rd / harmonic writes through shared ndarray-backed buffers
- stopped relying on the old `llsm_py_chunk_resample_linear_f0` path because it no longer matches the post-refactor chunk ownership model

## Scope

- Wraps the public `libllsm2` APIs from:
  - `llsm.h`
  - `dsputils.h`
  - `llsmrt.h`
  - `llsmutils.h`
- Exposes selected `ciglet` APIs used with LLSM2:
  - `qifft`
  - `spec2env`
  - `lfmodel_*`
  - `ifdetector_*`
  - `filterbank_*`

`pyin` and `gvps` are intentionally not included in this package build.

## Install

Editable install:

```bash
pip install -e ./pyllsm2
```

If your environment cannot create an isolated build environment:

```bash
pip install -e ./pyllsm2 --no-build-isolation
```

Build note: package build requires a `libllsm2` source tree at one of:

- `vendor/libllsm2`
- `libllsm2`
- `../libllsm2`

## Quick Start

```python
import pyllsm2

analysis_options = pyllsm2.AnalysisOptions()
analysis_options.thop = 128 / 16000.0
analysis_options.maxnhar = 120
analysis_options.maxnhar_e = 5
analysis_options.npsd = 128

synthesis_options = pyllsm2.SynthesisOptions(16000)

# f0 should come from an external F0 extractor such as librosa or parselmouth
layer0 = pyllsm2.analyze(analysis_options, x, 16000, f0)
layer1 = pyllsm2.to_layer1(layer0, 1024)
output = pyllsm2.synthesize(synthesis_options, layer1)
y = output.y
```

## Analysis Flow

The Python-first workflow is intentionally split into two feature containers:

- `Layer0Features`: harmonic-plus-noise features produced directly by analysis
- `Layer1Features`: source-filter style features derived from layer 0

Typical flow:

```python
import pyllsm2

layer0 = pyllsm2.analyze(analysis_options, x, fs, f0)
layer1 = pyllsm2.to_layer1(layer0, 1024)
resynth0 = pyllsm2.synthesize(synthesis_options, layer0)
resynth1 = pyllsm2.synthesize(synthesis_options, layer1)
```

## Python-First API

The top-level `pyllsm2` module is intentionally small and discoverable.

### Core classes

- `AnalysisOptions`
- `SynthesisOptions`
- `Layer0Features`
- `Layer1Features`
- `Chunk`
- `Coder`
- `RTSynthBuffer`
- `Output`

### Core functions

- `analyze(...)`
- `to_layer1(...)`
- `to_layer0(...)`
- `synthesize(...)`
- `analyze_chunk(...)`
- `synthesize_output(...)`
- `warp_frequency(...)`
- `spectral_mean(...)`
- `spectrum_from_envelope(...)`
- `harmonic_analysis(...)`
- `synthesize_harmonic_frame(...)`
- `qifft(...)`
- `spec2env(...)`
- `lfmodel_from_rd(...)`
- `lfmodel_spectrum(...)`
- `lfmodel_period(...)`
- `ifdetector_estimate(...)`

### Layer-0 container

`Layer0Features` is the main analysis result. Common operations include:

- `layer0.f0`
- `layer0.ampl`
- `layer0.phse`
- `layer0.nhar`
- `np.log(layer0.ampl + 1e-6)`
- `layer0.ampl = np.zeros((nfrm, max_nhar), dtype=np.float32)`
- `layer0.phasepropagate(...)`
- `layer0.phasesync_rps(...)`
- `layer0.resample_linear_f0(...)`
- `layer0.to_layer1(...)`

Array shapes and axis meanings:

- `layer0.f0`: shape `(nfrm,)`; axis 0 is frame index over time
- `layer0.nhar`: shape `(nfrm,)`; axis 0 is frame index over time
- `layer0.ampl`: shape `(nfrm, max_nhar)`; axis 1 is harmonic index
- `layer0.phse`: shape `(nfrm, max_nhar)`; axis 1 is harmonic index

### Layer-1 container

`Layer1Features` is the derived source-filter representation. Common operations include:

- `layer1.f0`
- `layer1.rd`
- `layer1.vtmagn`
- `layer1.vsphse`
- `layer1.vsphse_lengths`
- `np.mean(layer1.vtmagn, axis=0)`
- `layer1.vsphse = np.zeros((nfrm, max_nhar), dtype=np.float32)`
- `layer1.pitch_shift(...)`
- `layer1.to_layer0()`

Array shapes and axis meanings:

- `layer1.f0`: shape `(nfrm,)`; axis 0 is frame index over time
- `layer1.rd`: shape `(nfrm,)`; axis 0 is frame index over time
- `layer1.vtmagn`: shape `(nfrm, nspec)`; axis 1 is spectral-bin index
- `layer1.vsphse`: shape `(nfrm, max_nhar)`; axis 1 is source-harmonic index
- `layer1.vsphse_lengths`: shape `(nfrm,)`; axis 0 is frame index over time

### Advanced chunk API

`Chunk` remains available for advanced or low-level workflows, but it is no longer
the primary user-facing feature container.

Direct object-style usage is supported in normal workflows. Context-manager
usage is also available when you want explicit resource cleanup.

When you do need `Chunk`, prefer layer-specific allocation and direct ndarray
properties instead of mixed member-style access:

- `Chunk.allocate_layer0(nfrm, fs=..., max_nhar=...)`
- `Chunk.allocate_layer1(nfrm, fs=..., nspec=..., max_nhar=...)`
- `chunk.f0`, `chunk.ampl`, `chunk.phse`, `chunk.nhar`
- `chunk.rd`, `chunk.vtmagn`, `chunk.vsphse`, `chunk.vsphse_lengths`

## Raw API

Low-level access is available under `pyllsm2.raw`.

Use this layer only if you need direct CFFI access to native pointers, raw
constants, or one-to-one mappings of `libllsm2` symbols.

```python
import pyllsm2

ffi = pyllsm2.raw.ffi
lib = pyllsm2.raw.lib

opt = lib.llsm_create_aoptions()
method = pyllsm2.raw.LLSM_AOPTION_HMPP
```

Examples of what lives under `pyllsm2.raw`:

- `ffi`
- `lib`
- `LLSM_*` constants
- `llsm_*` native symbols
- `container_attach(...)`
- pointer-copy helpers such as `copy_fp_ptr(...)`

## Design Notes

- Top-level exports are Python-first and intentionally avoid raw symbol spam.
- Top-level analysis flow is centered on `Layer0Features -> Layer1Features`.
- Legacy aliases such as `AOptions`, `SOptions`, `analyze_frames`, and
  `synthesize_audio` are no longer part of the public top-level API.
- Raw `libllsm2` access remains available, but is now clearly separated under
  `pyllsm2.raw`.

## Tests

Tests are now grouped by intent inside `tests/`:

- `tests/unit/`: lightweight unit tests for high-level ndarray-oriented APIs.
- `tests/learning/`: practical, example-like tests that replace the old `examples/` folder.
- `tests/raw/`: focused tests for `pyllsm2.raw` helpers, constants, pointers, and interop.
- `tests/integration/`: functional/integration coverage, including ports of `libllsm2/test/*`.

Generated audio from the learning and integration suites is written to `tests/audio_outputs/`.

Install test dependencies:

```bash
pip install -e ./pyllsm2[test] --no-build-isolation
```

Run tests:

```bash
pytest tests -q
```

## Packaging

`pyllsm2` uses:

- `pyproject.toml`
- `setuptools`
- `cffi`
- `src/` layout

## License

`pyllsm2` is licensed under **GPL-3.0-or-later**.

- Main license text: `LICENSE`
- Third-party notices: `THIRD_PARTY_NOTICES.md`

