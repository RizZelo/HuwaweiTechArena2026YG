"""Ladder step A: headcount from frame timing (ATI) only, with logistic regression.

    python src/train_ati.py --output submissions/vA_ati_only.onnx

Logistic regression has no randomness: same data -> same model, so its leaderboard score
depends only on the features. Every headcount gets equal weight (assumption: the test has
roughly equal numbers of each class).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from torch import nn

import data
from ati import FEATURE_NAMES, ati_features_np, ati_features_torch
from export import export_and_check

STEPS = {"labeled": 300, "unlabeled": 600}      # window spacing (overlap is fine, model is simple)


def feature_table(which):
    rows, X = [], []
    for r in data.load_index(which).itertuples():
        a = data.load_ati(r.identifier, which)
        for s in range(0, len(a) - data.WINDOW + 1, STEPS[which]):
            w = a[s:s + data.WINDOW].copy(); w[-1] = 0
            X.append(ati_features_np(w))
            rows.append({"trace": r.identifier, "layout": r.layout,
                         "headcount": int(r.headcount) if which == "labeled" else -1})
    return np.array(X), pd.DataFrame(rows)


def fit(X, y):
    mean, std = X.mean(0), X.std(0) + 1e-6
    clf = LogisticRegression(class_weight="balanced", max_iter=5000).fit((X - mean) / std, y)
    return mean, std, clf


def predict(model, X):
    mean, std, clf = model
    return clf.predict((X - mean) / std)


class ATIModel(nn.Module):
    """Platform tensors in -> ATI features -> standardise -> linear -> 5 logits."""

    def __init__(self, mean, std, clf):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))
        self.register_buffer("W", torch.tensor(clf.coef_, dtype=torch.float32))
        self.register_buffer("b", torch.tensor(clf.intercept_, dtype=torch.float32))

    def forward(self, frames, intervals):
        z = (ati_features_torch(intervals) - self.mean) / self.std
        # frames is not used; the tiny term keeps it as a declared input of the ONNX file
        return self.W @ z + self.b + 0.0 * frames.reshape(-1)[0]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    XL, L = feature_table("labeled")
    XU, U = feature_table("unlabeled")
    yL = L.headcount.values
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

    export_and_check(ATIModel(*model).eval(), args.output)
