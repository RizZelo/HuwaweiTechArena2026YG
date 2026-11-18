"""Loading the dataset: index files, full traces, and 600-frame windows.

Everything here is plain NumPy / pandas. Shapes used across the project:

    csi  : (T, 3, 1024) complex64   time x Rx antenna x subcarrier (Tx dim dropped)
    ati  : (T,)         float32     microseconds between frame i and i+1
"""

from pathlib import Path

import blosc2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIRS = {
    "labeled": ROOT / "Data" / "FutureNetwork-data-labeled" / "labeled",
    "unlabeled": ROOT / "Data" / "FutureNetwork-data-unlabeled" / "unlabeled",
}
CACHE = ROOT / "cache"

WINDOW = 600            # frames per test sample (~3 s)
FS = 200.0              # nominal sampling rate, Hz
N_SUBCARRIERS = 1024
# Subcarriers 0-5 and 1019-1023 are always zero (guard band) -> drop them.
VALID_SC = slice(6, 1019)


def load_index(which="labeled"):
    """samples.csv as a DataFrame. Labeled headcounts are converted to int."""
    df = pd.read_csv(DATA_DIRS[which] / "samples.csv")
    if which == "labeled":
        df["headcount"] = df["headcount"].astype(int)
    return df


def load_trace(identifier, which="labeled"):
    """Full trace: csi (12000, 3, 1024) complex64, ati (12000,) float32."""
    folder = DATA_DIRS[which]
    csi = blosc2.load_array(str(folder / f"CSI_{identifier}.b2"))[:, 0]
    ati = blosc2.load_array(str(folder / f"ATI_{identifier}.b2"))
    return csi, ati


def load_ati(identifier, which="labeled"):
    """Only the frame intervals (12000,) float32 - much faster than load_trace."""
    return blosc2.load_array(str(DATA_DIRS[which] / f"ATI_{identifier}.b2"))


def windows(csi, ati, step=WINDOW, length=WINDOW):
    """Yield (start, csi_window, ati_window) like the test samples.

    The last interval of every window is set to 0, exactly as in the test set.
    """
    for s in range(0, len(csi) - length + 1, step):
        a = ati[s:s + length].copy()
        a[-1] = 0
        yield s, csi[s:s + length], a


def to_model_input(csi_window, ati_window):
    """Convert a window to the exact tensors the platform feeds the ONNX model.

    frames    : (600, 1, 3, 1024, 2) float32  (real, imag)
    intervals : (600,) float32
    """
    frames = np.ascontiguousarray(csi_window).view(np.float32)
    frames = frames.reshape(len(csi_window), 1, 3, N_SUBCARRIERS, 2)
    return frames, ati_window.astype(np.float32)
