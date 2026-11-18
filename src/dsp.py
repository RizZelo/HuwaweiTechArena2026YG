"""Physics-motivated signal processing on one CSI window (plain NumPy).

Multipath model:   H(f, t) = sum_k a_k * exp(-j 2 pi f tau_k(t))
                            = H_static(f)  +  H_dynamic(f, t)
Walls / furniture give the static part (the room fingerprint -> overfitting risk).
People give the dynamic part (motion, breathing) and also change the static part
a little by blocking / reflecting paths.

All functions take csi of shape (T, 3, S) complex (T frames, 3 Rx, S subcarriers)
unless stated otherwise. Later these will be rewritten in torch for ONNX export,
so keep them to simple array operations.
"""

import numpy as np

from data import FS, VALID_SC


# ---------------------------------------------------------------- basic cleaning

def clean(csi):
    """Drop the always-zero guard subcarriers -> (T, 3, 1013)."""
    return csi[:, :, VALID_SC]


def normalize_gain(csi):
    """Divide by the mean amplitude so overall power / distance to router cancels."""
    return csi / (np.abs(csi).mean() + 1e-9)


def group_subcarriers(x, n_groups=64):
    """Average neighbouring subcarriers (they are strongly correlated) -> (T, 3, n_groups)."""
    T, R, S = x.shape
    S_used = (S // n_groups) * n_groups
    return x[:, :, :S_used].reshape(T, R, n_groups, -1).mean(axis=3)


# ---------------------------------------------------------------- cleaning

def _shifted(x, k):
    """k copies of x shifted in time (edges repeated) -> (k, T, ...). Frame t of copy i is x[t + i - k//2]."""
    h = k // 2
    pad = [(h, h)] + [(0, 0)] * (x.ndim - 1)
    p = np.pad(x, pad, mode="edge")
    return np.stack([p[i:i + len(x)] for i in range(k)])


def despike(csi, k=5):
    """Remove one-frame interference bursts with a median filter over time.

    For every frame and subcarrier, keep the middle value of the k frames around it
    (done on real and imaginary parts separately). A burst lasting 1-2 frames is ignored;
    human motion (much slower than 5 ms per frame) passes through unchanged.
    """
    re = np.median(_shifted(csi.real, k), axis=0)
    im = np.median(_shifted(csi.imag, k), axis=0)
    return (re + 1j * im).astype(csi.dtype)


def smooth(csi, k=5):
    """Moving average over k frames: random per-frame noise averages out, slow motion stays."""
    return _shifted(csi, k).mean(axis=0).astype(csi.dtype)


def clean_window(csi):
    """Full cleaning used before every model: despike, then smooth. Shape is unchanged."""
    return smooth(despike(csi))


# ---------------------------------------------------------------- static / dynamic

def split_static_dynamic(csi):
    """Static = complex mean over time (walls, furniture); dynamic = what is left (people).

    Done on the complex values, not amplitudes: a person moving changes the phase of
    their reflected path, which amplitude alone partly hides.
    """
    static = csi.mean(axis=0, keepdims=True)
    return static, csi - static


def static_profile(csi):
    """Room 'fingerprint': mean amplitude per (Rx, subcarrier), z-scored -> (3 * S,).

    Two profiles can be compared with a dot product / length (cosine similarity).
    """
    p = np.abs(csi).mean(axis=0).ravel()
    return (p - p.mean()) / (p.std() + 1e-12)


def spike_fraction(csi, factor=5.0):
    """Share of frames hit by a short interference burst, per 20 MHz sub-band -> (4,).

    The 80 MHz channel is made of four 20 MHz sub-bands (256 subcarriers each). A
    neighbouring WiFi network on one of them shows up as single frames that jump much
    more than usual on that sub-band only. Expects all 1024 subcarriers.
    """
    jump = np.abs(np.diff(csi, axis=0)).mean(axis=1)          # (T-1, 1024) frame-to-frame change
    bands = [jump[:, max(6, 256 * k):min(1019, 256 * (k + 1))].mean(axis=1) for k in range(4)]
    B = np.stack(bands, axis=1)                               # (T-1, 4)
    return (B > factor * np.median(B, axis=0)).mean(axis=0)


def dynamic_ratio(csi):
    """Energy of the dynamic part relative to the static part (scale free)."""
    static, dyn = split_static_dynamic(csi)
    return (np.abs(dyn) ** 2).mean() / ((np.abs(static) ** 2).mean() + 1e-12)


def csi_ratio(csi, a=0, b=1):
    """H_a / H_b between two Rx antennas -> (T, S).

    Hardware phase / gain errors are common to all antennas and cancel in the ratio.
    (In this dataset the raw phase already looks corrected, so this mainly gives a
    second, differently-normalised view.)
    """
    return csi[:, a] / (csi[:, b] + 1e-9)


# ---------------------------------------------------------------- time-frequency

def doppler_spectrum(csi, fs=FS):
    """Power vs Doppler frequency of the dynamic part, averaged over Rx and subcarriers.

    A body moving at speed v shifts frequencies by about 2 v / wavelength
    (5 GHz -> 1 m/s gives ~33 Hz). Returns (freqs, power) with freqs in [-fs/2, fs/2).
    Assumes uniform sampling (ATI is not perfectly uniform; good enough to look at).
    """
    _, dyn = split_static_dynamic(csi)
    T = len(dyn)
    win = np.hanning(T)[:, None, None]
    spec = np.fft.fftshift(np.fft.fft(dyn * win, axis=0), axes=0)
    power = (np.abs(spec) ** 2).mean(axis=(1, 2))
    freqs = np.fft.fftshift(np.fft.fftfreq(T, d=1 / fs))
    return freqs, power


def doppler_spectrogram(csi, fs=FS, nperseg=128, hop=16):
    """Short-time Doppler spectrum -> (freqs, times, power[time, freq]).

    This is the "image" used by many WiFi-sensing papers: it shows how bodies move,
    not what the room looks like.
    """
    _, dyn = split_static_dynamic(csi)
    win = np.hanning(nperseg)[:, None, None]
    starts = range(0, len(dyn) - nperseg + 1, hop)
    rows = []
    for s in starts:
        seg = dyn[s:s + nperseg]
        seg = seg - seg.mean(axis=0, keepdims=True)
        spec = np.fft.fftshift(np.fft.fft(seg * win, axis=0), axes=0)
        rows.append((np.abs(spec) ** 2).mean(axis=(1, 2)))
    freqs = np.fft.fftshift(np.fft.fftfreq(nperseg, d=1 / fs))
    times = (np.array(list(starts)) + nperseg / 2) / fs
    return freqs, times, np.array(rows)


# ---------------------------------------------------------------- structure

def covariance_eigenvalues(csi, n_groups=64, k=10):
    """Top-k eigenvalues of the dynamic-part covariance, as fractions of total variance.

    Treat each (Rx, subcarrier group) as a sensor and each frame as an observation.
    Independent movers tend to add independent directions of variation, so the way
    the variance spreads over eigenvalues relates to "how many things move".
    """
    x = group_subcarriers(csi, n_groups)
    _, dyn = split_static_dynamic(x)
    X = dyn.reshape(len(dyn), -1)                       # (T, 3 * n_groups)
    C = (X.conj().T @ X) / len(X)                       # Hermitian covariance
    ev = np.linalg.eigvalsh(C)[::-1].real
    return ev[:k] / (ev.sum() + 1e-12)


def delay_profile(csi):
    """Power-delay profile: IFFT over subcarriers, averaged over time and Rx.

    Separates paths by travel distance; at 80 MHz one bin ~ 12.5 ns ~ 3.75 m.
    Peak positions depend on room geometry, so use its shape/statistics, not the raw bins.
    Returns (delays_ns, power).
    """
    S = csi.shape[2]
    cir = np.fft.ifft(csi, axis=2)
    power = (np.abs(cir) ** 2).mean(axis=(0, 1))
    delays_ns = np.arange(S) / 80e6 * 1e9
    return delays_ns, power


# ---------------------------------------------------------------- timing

def ati_stats(ati):
    """Summary of frame-interval timing. Busy channel (more devices) -> longer / irregular gaps."""
    a = np.asarray(ati[:-1], dtype=np.float64)          # last value is padding (0)
    med = np.median(a)
    return {
        "ati_median": med,
        "ati_std": a.std(),
        "ati_p99": np.percentile(a, 99),
        "ati_gap_frac": (a > 2 * med).mean(),           # fraction of delayed frames
        "ati_duration_s": a.sum() / 1e6,
    }
