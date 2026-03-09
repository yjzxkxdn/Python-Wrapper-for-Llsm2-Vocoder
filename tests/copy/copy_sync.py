import numpy as np
import parselmouth as pm

def extract_f0_parselmouth(
    x: np.ndarray,
    sampling_rate: int,
    hop_length: int,
    target_len: int,
    f0_min: float = 50.0,
) -> np.ndarray:
    l_pad = int(np.ceil(1.5 / f0_min * sampling_rate))
    r_pad = hop_length * ((len(x) - 1) // hop_length + 1) - len(x) + l_pad + 1
    s = pm.Sound(np.pad(x, (l_pad, r_pad)), sampling_rate).to_pitch_ac(
        time_step=hop_length / sampling_rate,
        voicing_threshold=0.6,
        pitch_floor=f0_min,
        pitch_ceiling=1100,
    )
    f0 = np.asarray(s.selected_array["frequency"], dtype=np.float32)
    if f0.size < target_len:
        f0 = np.pad(f0, (0, target_len - f0.size))
    f0 = f0[:target_len]
    return np.ascontiguousarray(f0, dtype=np.float32)

import pyllsm2 as m
import soundfile as sf

input_file = r""
input_wav, fs = sf.read(input_file)

nhop = 256

# 创建分析和合成配置类
opt_a = m.AnalysisOptions()
opt_a.thop = nhop / float(fs)
opt_a.npsd = 128
opt_a.maxnhar = 120
opt_a.maxnhar_e = 5

opt_s = m.SynthesisOptions(fs)

# 计算基频
nfrm = max(1, input_wav.size // nhop)
f0 = extract_f0_parselmouth(input_wav, fs, nhop, nfrm)

layer0 = m.analyze(opt_a, input_wav, float(fs), f0)
out0 = m.synthesize(opt_s, layer0).y
sf.write("output0.wav", out0, fs)

layer0.ampl*=0.5
out0_ = m.synthesize(opt_s, layer0).y
sf.write("output0_0.5.wav", out0_, fs)

layer1 = m.to_layer1(layer0, 2048)
layer1.phasesync_rps(layer1_based=True)
pbp_mask = (np.arange(layer1.nfrm, dtype=np.int32) % 100) > 50
layer1.enable_pulse_by_pulse(pbp_mask, clear_harmonics=True)
layer1.phasepropagate(sign=1)
out1 = m.synthesize(opt_s, layer1).y

sf.write("output1.wav", out1, fs)















