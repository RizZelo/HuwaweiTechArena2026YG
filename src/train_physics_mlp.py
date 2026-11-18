"""Ladder step B-MLP: the 15 physics features of step B, but a small neural network instead of
logistic regression. Only change vs. B. Labelled data only (the unlabelled data comes next).

    python src/train_physics_mlp.py --output submissions/vB_mlp.onnx

Network: 15 -> 32 -> 5 (one hidden layer, ReLU, dropout). Equal weight per headcount, like B.
A network has randomness, so 5 seeds are trained and averaged inside the ONNX file.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from export import export_and_check
from models import despike, smooth
from physics import physics_features
from train_physics import feature_table

SEEDS = 5


def make_mlp(n_in=15, hidden=32):
    return nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden, 5))


def train_mlp(Z, y, seed, epochs=150, bs=64, lr=3e-3):
    """Z: standardised features (N, 15) float32, y: (N,) int64."""
    torch.manual_seed(seed); np.random.seed(seed)
    net = make_mlp(Z.shape[1])
    counts = torch.bincount(y, minlength=5).float()
    weight = torch.where(counts > 0, counts.sum() / (5 * counts.clamp(min=1)), torch.zeros_like(counts))
    opt = torch.optim.AdamW(net.parameters(), lr, weight_decay=1e-2)
    for ep in range(epochs):
        net.train()
        for idx in torch.randperm(len(Z)).split(bs):
            loss = F.cross_entropy(net(Z[idx]), y[idx], weight=weight)
            opt.zero_grad(); loss.backward(); opt.step()
    return net.eval()


def ensemble_probs(nets, Z):
    with torch.no_grad():
        return torch.stack([F.softmax(n(Z), dim=1) for n in nets]).mean(0)


class MLPModel(nn.Module):
    """frames -> clean -> 15 physics features -> standardise -> average of 5 MLPs -> log-probabilities."""

    def __init__(self, mean, std, nets):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))
        self.nets = nn.ModuleList(nets)

    def forward(self, frames, intervals):
        z = ((physics_features(smooth(despike(frames[:, 0]))) - self.mean) / self.std).unsqueeze(0)
        p = torch.stack([F.softmax(n(z), dim=1) for n in self.nets]).mean(0)
        return torch.log(p + 1e-9)[0] + 0.0 * intervals[0]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    XL, L = feature_table("labeled")
    XU, U = feature_table("unlabeled")
    yL = torch.from_numpy(L.headcount.values.astype(np.int64))

    def standardise(Xtr, *others):
        m, s = Xtr.mean(0), Xtr.std(0) + 1e-6
        return m, s, [torch.from_numpy(((X - m) / s).astype(np.float32)) for X in (Xtr, *others)]

    print("local check (hold out one room; per seed and ensemble):")
    for h in ["a", "c", "g"]:
        te = (L.layout == h).values
        _, _, (Ztr, Zte) = standardise(XL[~te], XL[te])
        nets = [train_mlp(Ztr, yL[~te], s) for s in range(SEEDS)]
        accs = [(ensemble_probs([n], Zte).argmax(1) == yL[te]).float().mean().item() for n in nets]
        p = ensemble_probs(nets, Zte).argmax(1)
        print(f"  held {h}: seeds {' '.join(f'{a:.2f}' for a in accs)} | ensemble {(p == yL[te]).float().mean().item():.2f}"
              f"  predicted {torch.bincount(p, minlength=5).tolist()}  true {torch.bincount(yL[te], minlength=5).tolist()}")

    mean, std, (ZL, ZU) = standardise(XL, XU)
    nets = [train_mlp(ZL, yL, s) for s in range(SEEDS)]
    pU = ensemble_probs(nets, ZU).argmax(1).numpy()
    print("unlabelled (test-like) windows predicted 0..4:", (np.bincount(pU, minlength=5) / len(pU)).round(2).tolist())
    print("most common answer per unlabelled room:",
          {r: f"{np.bincount(pU[U.layout == r], minlength=5).argmax()} "
              f"({np.bincount(pU[U.layout == r], minlength=5).max() / (U.layout == r).sum():.0%})"
           for r in sorted(U.layout.unique())})

    export_and_check(MLPModel(mean, std, nets).eval(), args.output)
