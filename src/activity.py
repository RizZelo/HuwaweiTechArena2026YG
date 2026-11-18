"""Activity score: how much moves in a window, compared with the window's own noise floor.

Rebuilt in plain NumPy from the team's 0.41 model (ONNX analysed 2026-10-26), step by step:

    1. pool      1006 valid subcarriers -> 128 groups (complex average)
    2. phase     remove the phase tilt across frequency and each antenna's phase offset
    3. median3   remove one-frame glitches (median of 3 frames, first/last frame kept)
    4. relative  (signal - its average over time) / its average strength, per antenna and group
    5. spectrum  Doppler power over the 600 frames (Hann window), summed over antennas and groups
    6. score     log( mean power over all non-zero frequencies / mean power at 57-100 Hz )

57-100 Hz is far too fast for any human, so it is pure noise: dividing by it makes the score
independent of how loud or quiet a room is. No smoothing: smoothing would erase that band.

Shapes: csi window (600, 3, 1024) complex  ->  one float.
"""

import numpy as np

T = 600
INVALID = set(range(0, 7)) | set(range(510, 515)) | set(range(1018, 1024))   # guard band + DC
VALID = np.array([k for k in range(1024) if k not in INVALID])               # 1006 subcarriers
GROUPS = np.array_split(VALID, 128)                                           # 110 x 8 and 18 x 7
CENTRES = np.array([g.mean() for g in GROUPS])                                # subcarrier index of each group
NOISE_BINS = np.arange(171, 430)                                              # |f| = 57..100 Hz at 200 Hz
HANN = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(T) / T)
STILL_THRESHOLD = 3.0                                                         # the 0.41 model's "empty" cut


def pool(csi):
    """(T, 3, 1024) -> (T, 3, 128): average the complex values inside each group."""
    return np.stack([csi[:, :, g].mean(axis=2) for g in GROUPS], axis=2)


def detrend_phase(z):
    """Rotate the whole window so the average signal has no phase tilt and no antenna offset."""
    m = z.mean(axis=0)                                                        # (3, 128) average signal
    step = np.angle(np.sum(m[:, 1:] * np.conj(m[:, :-1])))                    # phase change per group
    tilt = step / 8 * CENTRES                                                 # per subcarrier, at each centre
    offset = np.angle(np.sum(m * np.exp(-1j * tilt), axis=1))                 # (3,) per antenna
    return z * np.exp(-1j * (tilt[None, :] + offset[:, None]))[None]


def median3(z):
    """Median of 3 consecutive frames, separately for real and imaginary parts; edges unchanged."""
    out = z.copy()
    for part in (np.real, np.imag):
        p = part(z)
        med = np.median(np.stack([p[:-2], p[1:-1], p[2:]]), axis=0)
        if part is np.real:
            out.real[1:-1] = med
        else:
            out.imag[1:-1] = med
    return out


def relative_change(z):
    """(signal - average) / average strength, per antenna and group (walls and loudness removed)."""
    mu = z.mean(axis=0, keepdims=True)
    strength = np.abs(mu)
    return (z - mu) / (strength + 1e-3 * strength.mean())


def doppler_power(d):
    """Power at each of the 600 Doppler frequencies, summed over antennas and groups."""
    spec = np.fft.fft(d * HANN[:, None, None], axis=0)
    return (np.abs(spec) ** 2).sum(axis=(1, 2))


def front_end(csi):
    """Steps 1-4: raw window -> relative change (600, 3, 128) complex."""
    return relative_change(median3(detrend_phase(pool(csi.astype(np.complex128)))))


def activity_score(csi):
    """Steps 1-6: one number. Below STILL_THRESHOLD the 0.41 model treats the window as empty."""
    power = doppler_power(front_end(csi))
    return float(np.log(power[1:].mean() / power[NOISE_BINS].mean() + 1e-6))


# ---------------------------------------------------------------- variants (Phase 1)

FREQ = np.minimum(np.arange(T), T - np.arange(T)) * (200.0 / T)               # |Doppler| of each bin, Hz


def _log_ratio(power, bins):
    """log( mean power in these bins / mean power in the noise band )."""
    return float(np.log(power[bins].mean() / power[NOISE_BINS].mean() + 1e-6))


def score_variants(csi):
    """Several activity-like scores of one window (same front-end), as a dict of floats.

    act         the 0.41 model's score (all non-zero frequencies)
    act_slow    only 0.3-5 Hz   (walking, shifting, gestures)
    act_fast    only 5-25 Hz    (quick movements)
    act_breath  only 0.3-0.7 Hz (breathing-like)
    act_rx0..2  the score seen by each antenna alone; act_rx_med = their median
    spread      share of the 128 subcarrier groups whose own score is above the still threshold
    """
    d = front_end(csi)
    spec = np.abs(np.fft.fft(d * HANN[:, None, None], axis=0)) ** 2           # (600, 3, 128)
    p_all = spec.sum(axis=(1, 2))
    out = {
        "act": _log_ratio(p_all, slice(1, None)),
        "act_slow": _log_ratio(p_all, (FREQ >= 0.3) & (FREQ < 5)),
        "act_fast": _log_ratio(p_all, (FREQ >= 5) & (FREQ < 25)),
        "act_breath": _log_ratio(p_all, (FREQ >= 0.3) & (FREQ < 0.7)),
    }
    p_rx = spec.sum(axis=2)                                                   # (600, 3)
    rx = [_log_ratio(p_rx[:, r], slice(1, None)) for r in range(3)]
    out.update({f"act_rx{r}": rx[r] for r in range(3)})
    out["act_rx_med"] = float(np.median(rx))
    g = np.log(spec[1:].mean(axis=0) / spec[NOISE_BINS].mean(axis=0) + 1e-6)   # (3, 128) per group
    out["spread"] = float((g > STILL_THRESHOLD).mean())
    return out
