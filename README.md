# pyllsm2

`pyllsm2` is a CFFI wrapper package for `libllsm2`.

## Scope

- Exposes all public `llsm2` APIs from:
  - `llsm.h`
  - `dsputils.h`
  - `llsmrt.h`
  - `llsmutils.h`
- Exposes selected high-level `ciglet` APIs used with LLSM2:
  - `qifft`
  - `spec2env`
  - `lfmodel_*`
  - `ifdetector_*`
  - `filterbank_*`

`pyin` / `gvps` are intentionally not included in this package build.

## Install

```bash
pip install -e ./pyllsm2
```

If your environment cannot create an isolated build env, use:

```bash
pip install -e ./pyllsm2 --no-build-isolation
```

## Use

```python
import pyllsm2

opt = pyllsm2.llsm_create_aoptions()
```

## Tests

Tests are ports/adaptations of `libllsm2/test/*` (not the legacy `libllsm` tests),
including `test-structs`, `test-dsputils`, `test-harmonic`, and a layer0 analyze/synthesize flow.

`libpyin`-related flow is replaced by Python-side alternatives.
Default test flow uses Python-generated F0; optional smoke test can use `librosa.pyin`.

Install test dependencies:

```bash
pip install -e ./pyllsm2[test] --no-build-isolation
```

Run:

```bash
pytest pyllsm2/tests -q
```

Run optional `librosa.pyin` smoke test:

```bash
PYLLSM2_TEST_LIBROSA=1 pytest pyllsm2/tests/test_layer0_pipeline_from_libllsm2.py -q
```

## PyPI Auto Publish (GitHub Actions)

Workflow file:

- `.github/workflows/publish-pypi.yml`

Behavior:

- triggers on GitHub Release `published` (and manual `workflow_dispatch`)
- builds `sdist`
- publishes to PyPI via trusted publishing (`id-token`)

You need to configure PyPI Trusted Publisher for this GitHub repository/project.

Build note: package build requires `libllsm2` source tree at one of:

- `vendor/libllsm2`
- `libllsm2`
- `../libllsm2`

## License

`pyllsm2` is licensed under **GPL-3.0-or-later**.

- Main license text: `LICENSE`
- Third-party notices: `THIRD_PARTY_NOTICES.md`
