from __future__ import annotations

import numpy as np

from tests._audio_utils import dominant_frequency_hz, write_audio_output
from tests.learning._tone_cases import make_layer0_tone, make_manual_layer1_tone, make_seeded_layer1_tone, render_chunk


def _stable_tail(y: np.ndarray) -> np.ndarray:
    tail = np.asarray(y, dtype=np.float32).reshape(-1)
    tail = tail[tail.size // 3 :]
    return np.nan_to_num(tail, nan=0.0, posinf=0.0, neginf=0.0)


def test_learning_layer0_tone_case_renders_expected_audio() -> None:
    layer0, meta = make_layer0_tone()
    y = render_chunk(layer0, fs=meta["fs"], ny=meta["ny"])
    y_eval = _stable_tail(y)
    out_path = write_audio_output("learning", "layer0_tone_440hz.wav", np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0), meta["fs"])

    assert out_path.exists()
    assert y.shape == (meta["ny"],)
    assert np.isfinite(y_eval).all()
    assert float(np.max(np.abs(y_eval))) > 0.05
    assert float(np.sqrt(np.mean(np.square(y_eval)))) > 0.01
    assert 200.0 <= dominant_frequency_hz(y_eval[: min(y_eval.size, 8192)], meta["fs"]) <= 2000.0
    assert np.all(layer0.nhar == meta["nhar"])


def test_learning_seeded_layer1_tone_preserves_analysis_shape() -> None:
    layer0, layer1, meta = make_seeded_layer1_tone()
    y = render_chunk(layer1, fs=meta["fs"], ny=meta["ny"])
    y_eval = _stable_tail(y)
    out_path = write_audio_output("learning", "layer1_tone_440hz.wav", np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0), meta["fs"])

    assert out_path.exists()
    assert layer1.nfrm == layer0.nfrm == meta["nfrm"]
    assert layer1.nspec == meta["nspec"]
    assert np.isfinite(layer1.rd).all()
    assert np.isfinite(layer1.vtmagn).all()
    assert float(np.std(layer1.rd)) < 0.25
    assert np.isfinite(y_eval).all()
    assert 0.01 < float(np.sqrt(np.mean(np.square(y_eval)))) < 10.0
    assert 200.0 <= dominant_frequency_hz(y_eval[: min(y_eval.size, 8192)], meta["fs"]) <= 2000.0


def test_learning_manual_layer1_tone_builds_source_filter_features() -> None:
    layer1, meta = make_manual_layer1_tone()
    y = render_chunk(layer1, fs=meta["fs"], ny=meta["ny"])
    y_eval = _stable_tail(y)
    out_path = write_audio_output("learning", "layer1_manual_tone_440hz.wav", np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0), meta["fs"])

    assert out_path.exists()
    assert layer1.vtmagn.shape == (meta["nfrm"], meta["nspec"])
    assert np.all(layer1.vsphse_lengths == meta["nhar"])
    assert np.isfinite(layer1.vsphse).all()
    assert float(np.sqrt(np.mean(np.square(y_eval)))) > 0.01
    assert 200.0 <= dominant_frequency_hz(y_eval[: min(y_eval.size, 8192)], meta["fs"]) <= 2000.0
