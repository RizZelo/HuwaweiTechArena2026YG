"""Ladder step D: physics features measured relative to each room's usual values.

    python src/train_room_norm.py --output submissions/vD_physics_roomnorm.onnx

Exactly step B (same 15 features, same logistic regression, equal weights) with ONE change:
every feature becomes "how many usual spreads above / below this room's average".

1. Room fingerprints: average fingerprint of every known room (labelled a b c e g,
   unlabelled d h i j u v). "unknown" recordings are not used as rooms.
2. Room normal values: mean and spread of the 15 features over all windows of that room,
   computed WITHOUT labels (the test rooms get theirs from the unlabelled data).
3. Soft match: a window is compared with every room fingerprint; the most similar rooms get
   the most weight. If no room is similar, a "global" average over all rooms takes over.
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
from physics import FEATURE_NAMES, fingerprint, physics_features
from train_ati import fit, predict
from train_physics import STEPS, feature_table

TAU = 30.0            # how sharply the soft match prefers the most similar room
GLOBAL_SCORE = 0.5    # similarity at which the "global" fallback is as strong as a room


def fingerprint_table(which, meta):
    out = data.CACHE / f"fingerprint_{which[0].upper()}.npz"
    if out.exists():
        return np.load(out)["F"]
    F, idx = [], data.load_index(which)
    for n, r in enumerate(idx.itertuples()):
        csi, ati = data.load_trace(r.identifier, which)
        for s, w, a in data.windows(csi, ati, step=STEPS[which]):
            frames, _ = data.to_model_input(w, a)
            with torch.no_grad():
                F.append(fingerprint(smooth(despike(torch.from_numpy(frames[:, 0])))).numpy())
        if (n + 1) % 25 == 0 or n + 1 == len(idx):
            print(f"  fingerprints {which}: {n + 1}/{len(idx)} traces", flush=True)
    F = np.array(F)
    assert len(F) == len(meta), "fingerprint and feature tables must list the same windows"
    np.savez(out, F=F)
    return F


def room_tables(X, F, room):
    """Per room: fingerprint centroid, feature mean, feature spread. Last row = global fallback."""
    names = sorted(r for r in set(room) if r != "unknown")
    C = np.array([F[room == r].mean(0) for r in names])
    C = (C - C.mean(1, keepdims=True)) / C.std(1, keepdims=True)
    MU = np.array([X[room == r].mean(0) for r in names] + [X.mean(0)])
    SD = np.array([X[room == r].std(0) + 1e-6 for r in names] + [X.std(0) + 1e-6])
    return names, C, MU, SD


def match_weights(F, C):
    """Soft room match: softmax over (cosine similarity to every room, plus the global score)."""
    s = F @ C.T / F.shape[1]
    s = np.concatenate([s, np.full((len(F), 1), GLOBAL_SCORE)], axis=1)
    e = np.exp(TAU * (s - s.max(1, keepdims=True)))
    return e / e.sum(1, keepdims=True)


def normalise(X, F, C, MU, SD):
    w = match_weights(F, C)
    return (X - w @ MU) / (w @ SD)


def matcher_check(F, room, trace):
    """Hold out each recording, rebuild its room's fingerprint without it, see where it lands."""
    names = sorted(r for r in set(room) if r != "unknown")
    hits = []
    for t in np.unique(trace[room != "unknown"]):
        m = trace == t
        keep = ~m
        C = np.array([F[keep & (room == r)].mean(0) if (keep & (room == r)).any() else F[room == r].mean(0)
                      for r in names])
        C = (C - C.mean(1, keepdims=True)) / C.std(1, keepdims=True)
        w = match_weights(F[m], C).mean(0)
        best = w.argmax()
        hits.append({"room": room[m][0], "matched": names[best] if best < len(names) else "global",
                     "weight": w[best]})
    h = pd.DataFrame(hits)
    h["ok"] = h.room == h.matched
    return h


class RoomNormModel(nn.Module):
    """frames -> clean -> features + fingerprint -> soft room match -> relative features -> linear."""

    def __init__(self, C, MU, SD, mean, std, clf):
        super().__init__()
        f32 = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)
        for k, v in dict(C=C, MU=MU, SD=SD, mean=mean, std=std, W=clf.coef_, b=clf.intercept_).items():
            self.register_buffer(k, f32(v))

    def forward(self, frames, intervals):
        x = smooth(despike(frames[:, 0]))
        f, fp = physics_features(x), fingerprint(x)
        s = torch.cat([self.C @ fp / fp.shape[0], torch.full((1,), GLOBAL_SCORE)])
        w = torch.softmax(TAU * s, dim=0)
        rel = (f - w @ self.MU) / (w @ self.SD)
        z = (rel - self.mean) / self.std
        return self.W @ z + self.b + 0.0 * intervals[0]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    XL, L = feature_table("labeled")
    XU, U = feature_table("unlabeled")
    FL, FU = fingerprint_table("labeled", L), fingerprint_table("unlabeled", U)
    yL = L.headcount.values.astype(int)
    X, F = np.concatenate([XL, XU]), np.concatenate([FL, FU])
    room = np.concatenate([("L:" + L.layout).where(L.layout != "unknown", "unknown").values,
                           ("U:" + U.layout).where(U.layout != "unknown", "unknown").values])
    trace = np.concatenate([L.trace.values, U.trace.values])

    h = matcher_check(F, room, trace)
    print(f"room matcher, each recording held out: {h.ok.mean():.0%} matched to the right room "
          f"({h.ok.sum()}/{len(h)}); typical weight on the chosen room {h.weight.median():.2f}")
    wrong = h[~h.ok].groupby(["room", "matched"]).size()
    if len(wrong):
        print("  mismatches:", wrong.to_dict())

    names, C, MU, SD = room_tables(X, F, room)
    print("rooms:", names)
    RL = normalise(XL, FL, C, MU, SD)
    RU = normalise(XU, FU, C, MU, SD)

    print("local check (hold out one room; its normal values come from its windows WITHOUT labels):")
    for hroom in ["a", "c", "g"]:
        te = (L.layout == hroom).values
        p = predict(fit(RL[~te], yL[~te]), RL[te])
        print(f"  held {hroom}: acc {(p == yL[te]).mean():.2f}  predicted {np.bincount(p, minlength=5).tolist()}"
              f"  true {np.bincount(yL[te], minlength=5).tolist()}")

    model = fit(RL, yL)
    pU = predict(model, RU)
    print("unlabelled (test-like) windows predicted 0..4:", (np.bincount(pU, minlength=5) / len(pU)).round(2).tolist())
    print("most common answer per unlabelled room:",
          {r: f"{np.bincount(pU[U.layout == r], minlength=5).argmax()} "
              f"({np.bincount(pU[U.layout == r], minlength=5).max() / (U.layout == r).sum():.0%})"
           for r in sorted(U.layout.unique())})
    coef = pd.DataFrame(model[2].coef_, columns=FEATURE_NAMES, index=[f"hc{k}" for k in range(5)])
    print("weights (relative features; + means 'pushes towards this headcount'):")
    print(coef.round(2).T.to_string())

    export_and_check(RoomNormModel(C, MU, SD, *model).eval(), args.output)
