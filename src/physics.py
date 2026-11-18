"""Ladder step B: hand-made physical features of one CSI window (PyTorch, ONNX-safe).

The same function builds the training table and runs inside the ONNX model, so training
and test always compute identical numbers. Only simple operations are used (means, sums,
matrix products) so the export is safe; the Doppler spectrum is a fixed matrix product
(a DFT) instead of an FFT call.

Input: cleaned window x (600, 3, 1024, 2)  ->  15 numbers.
Working signal: complex CSI averaged to 150 time steps (50 Hz) x 3 Rx x 126 subcarrier groups.
"""

import math

import torch

from models import N_GROUPS, T_DOWN

FS = 200.0 / T_DOWN                  # 50 Hz after averaging 4 frames
T = 600 // T_DOWN                    # 150 time steps

FEATURE_NAMES = [
    # power ("RSSI"): bodies absorb and scatter energy
    "log_power", "freq_selectivity",
    # motion: how much and how fast things move
    "log_dyn_ratio", "doppler_0.3-2Hz", "doppler_2-5Hz", "doppler_5-10Hz", "doppler_10-25Hz",
    "very_slow_share", "log_amp_cv",
    # phase difference between antennas (sensitive to small movements)
    "log_phase01_var", "log_phase02_var",
    # structure: how many independent things change
    "participation_ratio", "log_frame_change", "dyn_kurtosis", "rx_agreement",
]


def _dft_matrices():
    """Hann-windowed DFT for 150 time steps, positive frequencies only (cos and sin parts)."""
    t = torch.arange(T, dtype=torch.float32)
    k = torch.arange(T // 2 + 1, dtype=torch.float32)
    hann = 0.5 - 0.5 * torch.cos(2 * math.pi * t / T)
    ang = 2 * math.pi * k[:, None] * t[None, :] / T
    return torch.cos(ang) * hann, torch.sin(ang) * hann, k * FS / T     # (76, 150) x2, freqs (76,)


_COS, _SIN, FREQS = _dft_matrices()


def _band(power, lo, hi):
    m = ((FREQS >= lo) & (FREQS < hi)).float()
    return (power * m).sum()


def physics_features(x):
    """x: cleaned window (600, 3, 1024, 2) float -> (15,) float."""
    g = x[:, :, 8:1016].reshape(T, T_DOWN, 3, N_GROUPS, 8, 2).mean(dim=(1, 4))     # (150, 3, 126, 2)
    re, im = g[..., 0], g[..., 1]
    amp = torch.sqrt(re ** 2 + im ** 2) + 1e-6                                    # (150, 3, 126)

    # --- power
    static_amp = amp.mean(dim=0)                                                  # (3, 126)
    log_power = torch.log(static_amp.mean())
    freq_sel = (static_amp.std(dim=1, unbiased=False) / static_amp.mean(dim=1)).mean()

    # --- motion: complex dynamic part (what changes during the window)
    dre = re - re.mean(dim=0, keepdim=True)
    dim_ = im - im.mean(dim=0, keepdim=True)
    dyn_e = (dre ** 2 + dim_ ** 2).mean()
    stat_e = (re.mean(dim=0) ** 2 + im.mean(dim=0) ** 2).mean()
    log_dyn = torch.log(dyn_e / stat_e + 1e-12)

    # Doppler power spectrum of the dynamic part, summed over Rx and subcarriers
    dr = dre.reshape(T, -1); di = dim_.reshape(T, -1)
    # DFT of a complex signal: (a + ib)(cos - i sin) -> real = a cos + b sin, imag = b cos - a sin
    spec_re = _COS.to(x) @ dr + _SIN.to(x) @ di
    spec_im = _COS.to(x) @ di - _SIN.to(x) @ dr
    power = (spec_re ** 2 + spec_im ** 2).sum(dim=1)                             # (76,)
    total = power[1:].sum() + 1e-12
    bands = [_band(power, lo, hi) / total for lo, hi in [(0.3, 2), (2, 5), (5, 10), (10, 25.1)]]
    very_slow = _band(power, 0.3, 0.7) / total                                    # breathing-like
    amp_cv = torch.log((amp.std(dim=0, unbiased=False) / amp.mean(dim=0)).mean() + 1e-9)

    # --- phase difference Rx0-Rx1 and Rx0-Rx2 as unit arrows; how much they wobble
    cos_d = (re[:, :1] * re[:, 1:] + im[:, :1] * im[:, 1:]) / (amp[:, :1] * amp[:, 1:])
    sin_d = (im[:, :1] * re[:, 1:] - re[:, :1] * im[:, 1:]) / (amp[:, :1] * amp[:, 1:])
    wobble = (cos_d.var(dim=0, unbiased=False) + sin_d.var(dim=0, unbiased=False)).mean(dim=1)   # (2,)
    ph01, ph02 = torch.log(wobble[0] + 1e-9), torch.log(wobble[1] + 1e-9)

    # --- structure of the amplitude change
    a = (amp - amp.mean(dim=0, keepdim=True)).reshape(T, -1)                     # (150, 378)
    C = a.T @ a / T
    pr = torch.diagonal(C).sum() ** 2 / ((C ** 2).sum() + 1e-12) / a.shape[1]     # effective rank share
    flat = amp.reshape(T, -1)
    frame_change = torch.log(((flat[1:] - flat[:-1]) ** 2).mean() / (flat ** 2).mean() + 1e-12)
    z = a / (a.std(dim=0, unbiased=False, keepdim=True) + 1e-9)
    kurt = (z ** 4).mean()
    d = amp - amp.mean(dim=0, keepdim=True)                                       # (150, 3, 126)
    dn = d / (d.std(dim=0, unbiased=False, keepdim=True) + 1e-9)
    rx_agree = (dn[:, 0] * dn[:, 1]).mean()

    return torch.stack([log_power, freq_sel, log_dyn, *bands, very_slow, amp_cv, ph01, ph02,
                        pr, frame_change, kurt, rx_agree])


def fingerprint(x):
    """Room 'fingerprint' of a cleaned window (600, 3, 1024, 2) -> (378,).

    The frozen picture: average signal strength per Rx and subcarrier group (log scale),
    shifted and scaled to mean 0 / spread 1 so overall power does not matter, only its shape.
    Two fingerprints are compared with a dot product / 378 (cosine similarity, -1..1).
    """
    g = x[:, :, 8:1016].reshape(T, T_DOWN, 3, N_GROUPS, 8, 2).mean(dim=(1, 4))
    amp = torch.sqrt(g[..., 0] ** 2 + g[..., 1] ** 2) + 1e-6
    p = torch.log(amp.mean(dim=0)).reshape(-1)
    return (p - p.mean()) / (p.std(unbiased=False) + 1e-9)
