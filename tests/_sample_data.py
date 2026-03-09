from __future__ import annotations

import numpy as np


def mask_from_lengths(lengths: np.ndarray, width: int) -> np.ndarray:
    return np.arange(width, dtype=np.int32)[None, :] < np.asarray(lengths, dtype=np.int32)[:, None]


def sample_layer0_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    f0 = np.array([120.0, 150.0, 180.0, 210.0], dtype=np.float32)
    nhar = np.array([4, 3, 5, 2], dtype=np.int32)
    width = int(nhar.max())
    ampl = np.full((f0.size, width), 1e-3, dtype=np.float32)
    phse = np.zeros((f0.size, width), dtype=np.float32)
    for i, n_i in enumerate(nhar):
        ampl[i, :n_i] = np.linspace(0.2, 1.0, int(n_i), dtype=np.float32)
        phse[i, :n_i] = np.linspace(0.1, 0.6, int(n_i), dtype=np.float32)
    return f0, ampl, phse, nhar


def sample_layer1_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    f0 = np.array([120.0, 150.0, 180.0, 210.0], dtype=np.float32)
    rd = np.array([1.5, 1.7, 1.9, 2.1], dtype=np.float32)
    lengths = np.array([4, 3, 5, 2], dtype=np.int32)
    vtmagn = np.stack(
        [np.linspace(-20.0 - i, -8.0 - i, 8, dtype=np.float32) for i in range(f0.size)],
        axis=0,
    )
    vsphse = np.zeros((f0.size, int(lengths.max())), dtype=np.float32)
    for i, n_i in enumerate(lengths):
        vsphse[i, :n_i] = np.linspace(0.05, 0.45, int(n_i), dtype=np.float32)
    return f0, rd, vtmagn, vsphse, lengths
