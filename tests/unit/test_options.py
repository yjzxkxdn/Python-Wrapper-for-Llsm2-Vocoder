from __future__ import annotations

import numpy as np
import pytest

import pyllsm2 as m


def test_analysis_options_support_init_kwargs_and_chanfreq_view() -> None:
    opt = m.AnalysisOptions(thop=0.01, maxnhar=48, chanfreq=[1200.0, 3600.0], hm_method=1)

    assert opt.thop == pytest.approx(0.01)
    assert opt.maxnhar == 48
    assert opt.nchannel == 3
    assert opt.hm_method == 1
    np.testing.assert_allclose(opt.chanfreq, np.array([1200.0, 3600.0], dtype=np.float32))

    opt.chanfreq[:] = np.array([1500.0, 4200.0], dtype=np.float32)
    np.testing.assert_allclose(opt.chanfreq, np.array([1500.0, 4200.0], dtype=np.float32))


def test_analysis_options_resize_nchannel_and_copy_chanfreq() -> None:
    opt = m.AnalysisOptions(chanfreq=[1000.0, 3000.0, 6000.0])
    opt.nchannel = 5

    assert opt.nchannel == 5
    np.testing.assert_allclose(opt.chanfreq[:3], np.array([1000.0, 3000.0, 6000.0], dtype=np.float32))
    np.testing.assert_allclose(opt.chanfreq[3:], np.array([6000.0], dtype=np.float32))

    chunk = m.Chunk.allocate_layer0(2, fs=16000, max_nhar=4, options=opt)
    assert chunk.nfrm == 2


def test_analysis_options_reject_inconsistent_nchannel_and_chanfreq() -> None:
    with pytest.raises(ValueError, match="inconsistent"):
        m.AnalysisOptions(nchannel=4, chanfreq=[1000.0, 3000.0])


def test_synthesis_options_support_init_kwargs() -> None:
    opt = m.SynthesisOptions(fs=22050, use_iczt=0, use_l1=1, iczt_param_a=0.4, iczt_param_b=2.0)

    assert opt.fs == pytest.approx(22050)
    assert opt.use_iczt == 0
    assert opt.use_l1 == 1
    assert opt.iczt_param_a == pytest.approx(0.4)
    assert opt.iczt_param_b == pytest.approx(2.0)
