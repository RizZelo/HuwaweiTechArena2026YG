"""Ladder step B: headcount from hand-made physical CSI features only (logistic regression).

    python src/train_physics.py --output submissions/vB_physics_only.onnx

Same recipe as step A (train_ati.py) with different features: no randomness, equal weight
per headcount. The feature table is cached (cache/physics_{L,U}.npz) for steps C and D.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

import data
from export import export_and_check
from models import despike, smooth
from physics import FEATURE_NAMES, physics_features
from train_ati import fit, predict

STEPS = {"labeled": 600, "unlabeled": 1200}


def feature_table(which):
    out = data.CACHE / f"physics_{which[0].upper()}.npz"
    if out.exists():
        z = np.load(out, allow_pickle=True)
        return z["X"], pd.DataFrame(z["meta"].tolist())
    rows, X = [], []
    idx = data.load_index(which)
    for n, r in enumerate(idx.itertuples()):
        csi, ati = data.load_trace(r.identifier, which)
        for s, w, a in data.windows(csi, ati, step=STEPS[which]):
            frames, _ = data.to_model_input(w, a)
            with torch.no_grad():
                X.append(physics_features(smooth(despike(torch.from_numpy(frames[:, 0])))).numpy())
            rows.append({"trace": r.identifier, "layout": r.layout, "start": s,
                         "headcount": int(r.headcount) if which == "labeled" else -1})
        if (n + 1) % 25 == 0 or n + 1 == len(idx):
            print(f"  {which}: {n + 1}/{len(idx)} traces", flush=True)
    X = np.array(X)
    np.savez(out, X=X, meta=np.array(rows, dtype=object))
    return X, pd.DataFrame(rows)


class PhysicsModel(nn.Module):
    """Platform tensors in -> clean -> 15 physics features -> standardise -> linear -> 5 logits."""

    def __init__(self, mean, std, clf):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))
        self.register_buffer("W", torch.tensor(clf.coef_, dtype=torch.float32))
        self.register_buffer("b", torch.tensor(clf.intercept_, dtype=torch.float32))

    def forward(self, frames, intervals):
        z = (physics_features(smooth(despike(frames[:, 0]))) - self.mean) / self.std
        # intervals is not used; the tiny term keeps it as a declared input of the ONNX file
        return self.W @ z + self.b + 0.0 * intervals[0]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    XL, L = feature_table("labeled")
    XU, U = feature_table("unlabeled")
    yL = L.headcount.values.astype(int)
    print(f"labelled windows {len(XL)}, unlabelled windows {len(XU)}, features {XL.shape[1]}")

    print("local check (hold out one room):")
    for h in ["a", "c", "g"]:
        te = (L.layout == h).values
        p = predict(fit(XL[~te], yL[~te]), XL[te])
        print(f"  held {h}: acc {(p == yL[te]).mean():.2f}  predicted {np.bincount(p, minlength=5).tolist()}"
              f"  true {np.bincount(yL[te], minlength=5).tolist()}")

    model = fit(XL, yL)
    pU = predict(model, XU)
    print("unlabelled (test-like) windows predicted 0..4:", (np.bincount(pU, minlength=5) / len(pU)).round(2).tolist())
    print("most common answer per unlabelled room:",
          {r: f"{np.bincount(pU[U.layout == r], minlength=5).argmax()} "
              f"({np.bincount(pU[U.layout == r], minlength=5).max() / (U.layout == r).sum():.0%})"
           for r in sorted(U.layout.unique())})
    coef = pd.DataFrame(model[2].coef_, columns=FEATURE_NAMES, index=[f"hc{k}" for k in range(5)])
    print("weights (standardised features; + means 'pushes towards this headcount'):")
    print(coef.round(2).T.to_string())

    export_and_check(PhysicsModel(*model).eval(), args.output)
