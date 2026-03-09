"""Python-first interface for the :mod:`pyllsm2` package.

The top-level package intentionally exposes only the high-level, NumPy-friendly
API. Raw CFFI bindings and native constants live under :mod:`pyllsm2.raw`.

Typical usage:

    >>> import pyllsm2
    >>> options = pyllsm2.AnalysisOptions()
    >>> layer0 = pyllsm2.analyze(options, x, fs, f0)
    >>> layer1 = pyllsm2.to_layer1(layer0, 1024)
    >>> output = pyllsm2.synthesize(pyllsm2.SynthesisOptions(fs), layer1)

For low-level access:

    >>> import pyllsm2
    >>> pyllsm2.raw.lib.llsm_create_aoptions()
"""

from . import raw
from ._wrapper import (
    AnalysisOptions,
    Chunk,
    Coder,
    FP_DTYPE,
    Layer0Features,
    Layer1Features,
    Output,
    RTSynthBuffer,
    SynthesisOptions,
    analyze,
    analyze_chunk,
    harmonic_analysis,
    ifdetector_estimate,
    lfmodel_from_rd,
    lfmodel_period,
    lfmodel_spectrum,
    qifft,
    spec2env,
    spectral_mean,
    synthesize,
    spectrum_from_envelope,
    synthesize_harmonic_frame,
    synthesize_output,
    to_layer0,
    to_layer1,
    warp_frequency,
)

__all__ = [
    "AnalysisOptions",
    "SynthesisOptions",
    "Layer0Features",
    "Layer1Features",
    "Chunk",
    "Coder",
    "RTSynthBuffer",
    "Output",
    "FP_DTYPE",
    "analyze",
    "analyze_chunk",
    "to_layer1",
    "to_layer0",
    "synthesize",
    "synthesize_output",
    "warp_frequency",
    "spectral_mean",
    "spectrum_from_envelope",
    "harmonic_analysis",
    "synthesize_harmonic_frame",
    "qifft",
    "spec2env",
    "lfmodel_from_rd",
    "lfmodel_spectrum",
    "lfmodel_period",
    "ifdetector_estimate",
    "raw",
]
