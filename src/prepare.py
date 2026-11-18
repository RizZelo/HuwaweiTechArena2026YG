"""Cut every trace into 600-frame windows and cache the model input.

    python src/prepare.py --clean              # step-2 CNN input, cleaned (amplitude groups)
    python src/prepare.py                      # step-2 CNN input, raw (as the old notebooks did)
    python src/prepare.py --rep image          # step-3 image (always cleaned), labelled only
    python src/prepare.py --rep image --unlabeled   # ... also the unlabelled set (for step 4)

Writes cache/windows_L_<raw|clean|image>.npz (labelled, 20 windows per trace)
   and cache/windows_U_<raw|clean|image>.npz (unlabelled, 10 windows per trace).
"""

import argparse

import numpy as np
import torch

import data
from models import despike, smooth, window_to_groups, window_to_image

STEPS = {"labeled": 600, "unlabeled": 1200}     # same window spacing as the old caches


def represent(frames, rep, clean):
    """One window in platform format (600, 1, 3, 1024, 2) -> cached array."""
    x = torch.from_numpy(frames[:, 0])
    with torch.no_grad():
        if rep == "image":
            return window_to_image(smooth(despike(x))).numpy()
        return window_to_groups(x, clean).numpy()


def build(which, rep, clean):
    idx = data.load_index(which)
    W, Y, L, T = [], [], [], []
    for n, r in enumerate(idx.itertuples()):
        csi, ati = data.load_trace(r.identifier, which)
        for s, w, _ in data.windows(csi, ati, step=STEPS[which]):
            frames, _ = data.to_model_input(w, ati[s:s + data.WINDOW])
            W.append(represent(frames, rep, clean).astype(np.float16))
            Y.append(int(r.headcount) if which == "labeled" else -1)
            L.append(r.layout); T.append(r.identifier)
        if (n + 1) % 10 == 0 or n + 1 == len(idx):
            print(f"{which}: {n + 1}/{len(idx)} traces", flush=True)
    tag = "image" if rep == "image" else ("clean" if clean else "raw")
    out = data.CACHE / f"windows_{which[0].upper()}_{tag}.npz"
    np.savez(out, W=np.array(W), y=np.array(Y), layouts=np.array(L), traces=np.array(T))
    print("saved", out, len(W), "windows")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", default="groups", choices=["groups", "image"])
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--unlabeled", action="store_true", help="for --rep image: also build the unlabelled cache")
    args = ap.parse_args()
    sets = ["labeled", "unlabeled"] if (args.rep == "groups" or args.unlabeled) else ["labeled"]
    for which in sets:
        build(which, args.rep, args.clean)
