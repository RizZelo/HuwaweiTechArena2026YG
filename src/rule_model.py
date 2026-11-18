"""Rule model "activity - power": no training, every number fixed in advance.

    python src/rule_model.py --power-weight 0.5 --output submissions/vR_act_minus_power.onnx
    python src/rule_model.py --power-weight 0   --output submissions/vR_act_only.onnx   (= the team's 0.40 rule)

For one window:
    1. activity  = the team's score (activity.py), computed in torch
       activity < 3.0  -> still -> 0 people
    2. power     = log of the average signal strength (physics.py "log_power"), measured RELATIVE
                   to the room's usual power (room found by fingerprint, as in step D)
    3. combined  = z(activity) - W * z(relative power)
                   physics: bodies absorb energy, so less power than usual = more people
    4. active windows are cut into 4 equal groups by the combined score -> 1, 2, 3, 4
       (cut points = quartiles over the active unlabelled windows; the test is ~equal per class)

All statistics (z-scores, cut points, room usual power) come from the unlabelled windows,
without any labels. W is fixed by hand (0.5 = power nudges, activity leads); W = 0 gives
the pure activity rule.
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

import activity as A
import data
from export import export_and_check
from models import despike, smooth
from physics import fingerprint, physics_features
from train_room_norm import GLOBAL_SCORE, TAU, match_weights

# ---------------------------------------------------------------- activity score in torch

_POOL = np.zeros((1024, 128), np.float32)
for _j, _g in enumerate(A.GROUPS):
    _POOL[_g, _j] = 1.0 / len(_g)
_K = np.arange(A.T)
_COS = (A.HANN[:, None] * np.cos(2 * np.pi * _K[:, None] * _K[None, :] / A.T)).astype(np.float32)
_SIN = (A.HANN[:, None] * np.sin(2 * np.pi * _K[:, None] * _K[None, :] / A.T)).astype(np.float32)
_NOISE = np.isin(_K, A.NOISE_BINS).astype(np.float32) / len(A.NOISE_BINS)


def _med3(a, b, c):
    return torch.max(torch.min(a, b), torch.min(torch.max(a, b), c))


class Activity(nn.Module):
    """Torch twin of activity.activity_score (raw window (600, 3, 1024, 2) -> one number)."""

    def __init__(self):
        super().__init__()
        for k, v in dict(pool=_POOL, centres=A.CENTRES.astype(np.float32), cos=_COS, sin=_SIN, noise=_NOISE).items():
            self.register_buffer(k, torch.tensor(v))

    def forward(self, x):
        re, im = x[..., 0] @ self.pool, x[..., 1] @ self.pool                        # (600, 3, 128)
        mr, mi = re.mean(0), im.mean(0)
        s_re = (mr[:, 1:] * mr[:, :-1] + mi[:, 1:] * mi[:, :-1]).sum()
        s_im = (mi[:, 1:] * mr[:, :-1] - mr[:, 1:] * mi[:, :-1]).sum()
        tilt = torch.atan2(s_im, s_re) / 8 * self.centres                          # (128,)
        a_re = (mr * torch.cos(tilt) + mi * torch.sin(tilt)).sum(1)
        a_im = (mi * torch.cos(tilt) - mr * torch.sin(tilt)).sum(1)
        psi = tilt[None, :] + torch.atan2(a_im, a_re)[:, None]                     # (3, 128)
        zr = re * torch.cos(psi) + im * torch.sin(psi)
        zi = im * torch.cos(psi) - re * torch.sin(psi)
        zr = torch.cat([zr[:1], _med3(zr[:-2], zr[1:-1], zr[2:]), zr[-1:]])
        zi = torch.cat([zi[:1], _med3(zi[:-2], zi[1:-1], zi[2:]), zi[-1:]])
        ur, ui = zr.mean(0, keepdim=True), zi.mean(0, keepdim=True)
        strength = torch.sqrt(ur ** 2 + ui ** 2)
        scale = strength + 1e-3 * strength.mean()
        dr, di = ((zr - ur) / scale).permute(1, 2, 0), ((zi - ui) / scale).permute(1, 2, 0)   # (3, 128, 600)
        sr = dr @ self.cos + di @ self.sin
        si = di @ self.cos - dr @ self.sin
        power = (sr ** 2 + si ** 2).sum(dim=(0, 1))                                # (600,)
        return torch.log(power[1:].mean() / (power @ self.noise) + 1e-6)


# ---------------------------------------------------------------- the rule model

class RuleModel(nn.Module):
    def __init__(self, C, room_power, stats, cuts, w, soft=False):
        super().__init__()
        self.soft = soft
        self.activity = Activity()
        f32 = lambda a: torch.tensor(np.asarray(a, dtype=np.float32))
        self.register_buffer("C", f32(C))                       # room fingerprints (R, 378)
        self.register_buffer("room_power", f32(room_power))     # usual log power per room + global (R+1,)
        self.register_buffer("stats", f32(stats))               # act mean, act std, rel mean, rel std
        self.register_buffer("cuts", f32(cuts))                 # 3 quartile cut points
        self.w = float(w)

    def forward(self, frames, intervals):
        raw = frames[:, 0]
        act = self.activity(raw)
        clean = smooth(despike(raw))
        fp = fingerprint(clean)
        s = torch.cat([self.C @ fp / fp.shape[0], torch.full((1,), GLOBAL_SCORE)])
        rel = physics_features(clean)[0] - torch.softmax(TAU * s, dim=0) @ self.room_power
        comb = (act - self.stats[0]) / self.stats[1] - self.w * (rel - self.stats[2]) / self.stats[3]
        if self.soft:
            return self._soft_log_probs(act, comb) + 0.0 * intervals[0]
        group = 1.0 + (comb > self.cuts).float().sum()
        cls = torch.where(act < A.STILL_THRESHOLD, torch.zeros_like(group), group)
        k = torch.arange(5, dtype=torch.float32)
        return -10.0 * torch.abs(k - cls) + 0.0 * intervals[0]                   # argmax = cls


    def _soft_log_probs(self, act, comb):
        """Soft version (for averaging with other models): log-probabilities instead of one class.

        still/active: the same soft switch as the team's 0.41 model, sigmoid((act - 3) / 0.25)
        active 1..4:  soft cut points, P(group > j) = sigmoid((comb - cut_j) / SOFT_TAU)
        """
        p_active = torch.sigmoid(torch.clamp((act - A.STILL_THRESHOLD) / 0.25, -8.0, 8.0))
        above = torch.sigmoid((comb - self.cuts) / SOFT_TAU)                    # (3,) decreasing
        groups = torch.stack([1 - above[0], above[0] - above[1], above[1] - above[2], above[2]])
        probs = torch.cat([(1 - p_active).reshape(1), p_active * groups])
        return torch.log(probs + 1e-6)


SOFT_TAU = 0.3      # width of the soft cut points (combined score has spread ~1); fixed, not tuned


def build(w):
    """All fixed numbers of the rule, from unlabelled windows only (no labels used)."""
    t = pd.read_csv(data.CACHE / "score_table.csv")
    # fingerprints exist for labelled windows every 600 frames and unlabelled every 1200 frames
    fps = []
    for which, tag in [("labeled", "L"), ("unlabeled", "U")]:
        meta = pd.DataFrame(np.load(data.CACHE / f"physics_{tag}.npz", allow_pickle=True)["meta"].tolist())
        F = np.load(data.CACHE / f"fingerprint_{tag}.npz")["F"]
        fps.append(meta.assign(set=tag, fp=list(F)))
    fp = pd.concat(fps)[["set", "trace", "start", "fp"]]
    t = t.merge(fp, on=["set", "trace", "start"], how="inner")
    t["room"] = np.where(t.layout == "unknown", "unknown", t.set + ":" + t.layout)

    names = sorted(r for r in t.room.unique() if r != "unknown")
    Fall = np.stack(t.fp.values)
    C = np.array([Fall[(t.room == r).values].mean(0) for r in names])
    C = (C - C.mean(1, keepdims=True)) / C.std(1, keepdims=True)
    room_power = np.array([t.log_power[t.room == r].median() for r in names] + [t.log_power.median()])
    t["rel"] = t.log_power.values - match_weights(Fall, C) @ room_power

    act_U = t[(t.set == "U") & (t.act >= A.STILL_THRESHOLD)]
    stats = [act_U.act.mean(), act_U.act.std(), act_U.rel.mean(), act_U.rel.std()]
    comb = (act_U.act - stats[0]) / stats[1] - w * (act_U.rel - stats[2]) / stats[3]
    cuts = np.quantile(comb, [0.25, 0.5, 0.75])
    print(f"rooms {names}\nstats {np.round(stats, 3).tolist()} | cuts {np.round(cuts, 3).tolist()} | "
          f"active unlabelled windows used: {len(act_U)}")
    return C, room_power, stats, cuts, t


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--power-weight", type=float, default=0.5)
    ap.add_argument("--soft", action="store_true", help="output log-probabilities (for combining models)")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    C, room_power, stats, cuts, t = build(args.power_weight)
    model = RuleModel(C, room_power, stats, cuts, args.power_weight, soft=args.soft).eval()

    # offline prediction mix on all unlabelled windows that have a fingerprint (same rule in NumPy)
    U = t[t.set == "U"]
    comb = (U.act - stats[0]) / stats[1] - args.power_weight * (U.rel - stats[2]) / stats[3]
    cls = np.where(U.act < A.STILL_THRESHOLD, 0, 1 + np.searchsorted(cuts, comb))
    print("unlabelled windows predicted 0..4:", (np.bincount(cls, minlength=5) / len(cls)).round(2).tolist())
    export_and_check(model, args.output)
