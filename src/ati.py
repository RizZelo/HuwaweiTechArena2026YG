"""Frame-timing (ATI) features: how busy the WiFi channel is during a window.

Normally a frame arrives every ~5 ms. When the channel is busy (other devices, e.g.
people's phones, transmitting), frames are delayed (gaps of 10-190 ms) and then a few
frames come quickly to catch up. These numbers describe that behaviour.

Two versions with identical results:
    ati_features_np(intervals)     NumPy, for building tables and reading
    ati_features_torch(intervals)  PyTorch, goes inside the ONNX model
intervals: (600,) microseconds, last value is padding (0) and is ignored.
"""

import numpy as np
import torch

FEATURE_NAMES = [
    "median_ms", "log_mean", "log_std", "p10", "p90", "log_p99", "log_max", "mad",
    "frac_gt_1.5x", "frac_gt_3x", "frac_gt_10x", "frac_lt_0.8x",
    "lag1_autocorr", "log_duration", "gap_events",
]
QUANTILES = {"p10": 0.10, "p90": 0.90, "p99": 0.99}


def _q_index(n, q):
    """Same rank rule in NumPy and torch: the value at position round(q * (n - 1)) of the sorted list."""
    return int(round(q * (n - 1)))


def ati_features_np(intervals):
    a = np.asarray(intervals[:-1], dtype=np.float32) / 1000.0          # ms, padding dropped
    n = len(a)
    s = np.sort(a)
    med = s[_q_index(n, 0.5)]
    p10, p90, p99 = (s[_q_index(n, q)] for q in QUANTILES.values())
    mad = np.sort(np.abs(a - med))[_q_index(n, 0.5)]
    c = a - a.mean()
    ac = (c[:-1] * c[1:]).mean() / ((c * c).mean() + 1e-9)
    gap = a > 1.5 * med
    return np.array([
        med, np.log1p(a.mean()), np.log1p(a.std()), p10, p90, np.log1p(p99), np.log1p(s[-1]), mad,
        gap.mean(), (a > 3 * med).mean(), (a > 10 * med).mean(), (a < 0.8 * med).mean(),
        ac, np.log(a.sum() + 1e-9), (gap[1:] & ~gap[:-1]).sum() / n,
    ], dtype=np.float32)


def ati_features_torch(intervals):
    a = intervals[:-1] / 1000.0
    n = a.shape[0]
    s = torch.sort(a).values
    med = s[_q_index(n, 0.5)]
    p10, p90, p99 = (s[_q_index(n, q)] for q in QUANTILES.values())
    mad = torch.sort(torch.abs(a - med)).values[_q_index(n, 0.5)]
    c = a - a.mean()
    ac = (c[:-1] * c[1:]).mean() / ((c * c).mean() + 1e-9)
    gap = (a > 1.5 * med).float()
    return torch.stack([
        med, torch.log1p(a.mean()), torch.log1p(a.std(unbiased=False)), p10, p90, torch.log1p(p99),
        torch.log1p(s[-1]), mad,
        gap.mean(), (a > 3 * med).float().mean(), (a > 10 * med).float().mean(), (a < 0.8 * med).float().mean(),
        ac, torch.log(a.sum() + 1e-9), (gap[1:] * (1 - gap[:-1])).sum() / n,
    ])
