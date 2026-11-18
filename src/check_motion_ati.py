"""Two questions about the teammate's pipeline, answered without uploads.

    python src/check_motion_ati.py

1. Does OUR motion score (activity.py `act`, team rule 0.40-0.41) add anything to HIS cues
   (activity E and Doppler bandwidth, computed from antenna-pair products)?
2. Where does frame timing (ATI) carry signal: empty vs occupied, or how many people?

Windows: every 600 frames (cache/score_table.csv: 960 labelled + 3000 unlabelled).
His cues are recomputed here with his CountCues code (copied from model_ahmed.ipynb).
His presence decision for unlabelled windows comes from cache/taps_model_ahmed_w.pkl.
"""

import math

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata, spearmanr

import data
from ati import FEATURE_NAMES, ati_features_np

T, FS, NFD = 600, 200.0, 299
ACTIVE = slice(6, 1019)
PAIRS = ((0, 1), (0, 2), (1, 2))
ATI_USE = ["log_std", "p90", "log_p99", "mad", "frac_gt_1.5x", "frac_gt_3x", "frac_gt_10x",
           "frac_lt_0.8x", "lag1_autocorr", "gap_events"]          # relative / shape features, not the device's median


def block_matrix(k=ACTIVE.stop - ACTIVE.start, nb=32):
    m = torch.zeros(k, nb)
    edges = torch.linspace(0, k, nb + 1).round().long()
    for b in range(nb):
        m[edges[b]:edges[b + 1], b] = 1.0 / float(edges[b + 1] - edges[b])
    return m


class CountCues(torch.nn.Module):
    """His Stage B cues: frames (600,1,3,1024,2) -> [activity E, Doppler bandwidth]"""

    def __init__(self):
        super().__init__()
        t = torch.arange(T, dtype=torch.float32) / FS
        f = torch.arange(1, NFD + 1, dtype=torch.float32) * FS / T
        win = torch.hann_window(T, periodic=False)
        self.cosb = torch.cos(2 * math.pi * t[:, None] * f[None]) * win[:, None]
        self.sinb = torch.sin(2 * math.pi * t[:, None] * f[None]) * win[:, None]
        self.blk, self.f = block_matrix(), f
        self.m_act, self.m_noise = (f <= 40.0).float(), (f >= 60.0).float()
        self.m_bw = ((f >= 6.0) & (f <= 94.0)).float()

    def forward(self, frames):
        re, im = frames[:, 0, :, ACTIVE, 0], frames[:, 0, :, ACTIVE, 1]
        xr, xi = [], []
        for i, j in PAIRS:
            pr = re[:, i] * re[:, j] + im[:, i] * im[:, j]
            pi = im[:, i] * re[:, j] - re[:, i] * im[:, j]
            mr, mi = pr.mean(0, keepdim=True), pi.mean(0, keepdim=True)
            scale = torch.sqrt(mr * mr + mi * mi) + 1e-6
            xr.append(((pr - mr) / scale) @ self.blk); xi.append(((pi - mi) / scale) @ self.blk)
        xr, xi = torch.cat(xr, 1), torch.cat(xi, 1)
        xr, xi = xr - xr.mean(0, keepdim=True), xi - xi.mean(0, keepdim=True)
        a_re, a_im = xr.T @ self.cosb + xi.T @ self.sinb, xi.T @ self.cosb - xr.T @ self.sinb
        b_re, b_im = xr.T @ self.cosb - xi.T @ self.sinb, xi.T @ self.cosb + xr.T @ self.sinb
        S = (a_re ** 2 + a_im ** 2 + b_re ** 2 + b_im ** 2).sum(0) + 1e-12
        E = torch.log((S * self.m_act).sum() / self.m_act.sum()) - torch.log((S * self.m_noise).sum() / self.m_noise.sum())
        w = S * self.m_bw
        mu = (w * self.f).sum() / w.sum()
        return torch.stack([E, torch.sqrt((w * (self.f - mu) ** 2).sum() / w.sum())])


def build_table():
    path = data.CACHE / "motion_ati_table.pkl"
    if path.exists():
        return pd.read_pickle(path)
    t = pd.read_csv(data.CACHE / "score_table.csv")
    cc = CountCues().eval()
    rows = []
    for (st, tr), g in t.groupby(["set", "trace"], sort=False):
        csi, ati = data.load_trace(tr, "labeled" if st == "L" else "unlabeled")
        for s in g.start:
            w, a = csi[s:s + T], ati[s:s + T].copy(); a[-1] = 0
            f, i = data.to_model_input(w, a)
            with torch.no_grad():
                E, bw = cc(torch.from_numpy(f)).numpy()
            rows.append(dict(set=st, trace=tr, start=s, E=E, bw=bw, **dict(zip(FEATURE_NAMES, ati_features_np(i)))))
    t = t.merge(pd.DataFrame(rows), on=["set", "trace", "start"])
    taps = pd.read_pickle(data.CACHE / "taps_model_ahmed_w.pkl")[["trace", "start", "room", "cls", "fused"]]
    t = t.merge(taps.assign(set="U"), on=["set", "trace", "start"], how="left")
    t["room"] = t.room.fillna(t.trace.str.split("_").str[1])          # labelled rooms: a1, c1, ...
    t.to_pickle(path)
    return t


def reliability(g, col):
    """share of the variation that is BETWEEN recordings (a recording has one headcount)"""
    within = g.groupby("trace")[col].var().mean(); between = g.groupby("trace")[col].mean().var()
    return between / (between + within)


def auc(score, pos):
    r = rankdata(score); n1 = pos.sum(); n0 = len(pos) - n1
    return (r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def pair_acc(g, col):
    """inside one room: for windows with different headcounts, how often the larger count scores higher"""
    v, y = g[col].values, g.headcount.values
    hi, lo = np.where(y[:, None] > y[None, :])
    return float(np.mean((v[hi] > v[lo]) + 0.5 * (v[hi] == v[lo])))


if __name__ == "__main__":
    t = build_table()
    L, U = t[t.set == "L"], t[t.set == "U"]
    U_occ = U[U.cls > 0]
    pd.set_option("display.width", 200)

    print("=== 1. OUR motion score vs HIS cues ===")
    print("\nunlabelled occupied windows, Spearman inside each room (1 = same ordering):")
    rows = []
    for r, g in U_occ.groupby("room"):
        rows.append(dict(room=r, n=len(g), act_vs_E=spearmanr(g.act, g.E)[0], act_vs_bw=spearmanr(g.act, g.bw)[0],
                         E_vs_bw=spearmanr(g.E, g.bw)[0], act_vs_fused=spearmanr(g.act, g.fused)[0],
                         E_vs_fused=spearmanr(g.E, g.fused)[0]))
    print(pd.DataFrame(rows).round(2).to_string(index=False))
    print("\nsteadiness inside a recording (between-recording share; higher = steadier):")
    print({c: round(np.mean([reliability(g, c) for _, g in U_occ.groupby("room")]), 2) for c in ["act", "E", "bw", "fused"]})

    print("\nlabelled rooms with 1-4 people, occupied windows (a1 runs backwards: crowded = still):")
    rows = []
    for r in ["a1", "c1"]:
        g = L[(L.room == r) & (L.headcount > 0)].copy()
        for c in ["E", "bw", "act"]:
            g["rk_" + c] = rankdata(g[c]) / len(g)
        g["EB"] = (g.rk_E + g.rk_bw) / 2
        g["EB_act"] = (g.rk_E + g.rk_bw + g.rk_act) / 3
        med = g.groupby("trace")[["act", "E", "bw", "EB", "EB_act", "headcount"]].median()
        for c in ["act", "E", "bw", "EB", "EB_act"]:
            rows.append(dict(room=r, score=c, pair_acc=pair_acc(g, c), recording_spearman=spearmanr(med[c], med.headcount)[0]))
    print(pd.DataFrame(rows).pivot(index="score", columns="room").round(2).to_string())

    print("\nempty vs occupied AUC in the mixed labelled rooms (0.5 = chance; >0.5 = higher when occupied):")
    rows = []
    for r in ["e1", "g1"]:
        g = L[L.room == r]
        rows.append(dict(room=r, **{c: auc(g[c].values, (g.headcount > 0).values) for c in ["act", "E", "bw"]}))
    print(pd.DataFrame(rows).round(2).to_string(index=False))

    print("\n=== 2. Frame timing (ATI): where is the signal? ===")
    rows = []
    for c in ATI_USE:
        d = dict(feature=c)
        for r in ["e1", "g1"]:
            g = L[L.room == r]; d[f"presence_auc_{r}"] = auc(g[c].values, (g.headcount > 0).values)
        for r in ["a1", "c1"]:
            g = L[(L.room == r) & (L.headcount > 0)]
            med = g.groupby("trace")[[c, "headcount"]].median()
            d[f"count_rho_{r}"] = spearmanr(med[c], med.headcount)[0]
        d["U_auc_vs_his_presence"] = np.mean([auc(g[c].values, (g.cls > 0).values) for _, g in U.groupby("room")])
        d["U_rho_vs_his_fused"] = np.mean([spearmanr(g[c], g.fused)[0] for _, g in U_occ.groupby("room")])
        d["U_rho_vs_act"] = np.mean([spearmanr(g[c], g.act)[0] for _, g in U.groupby("room")])
        d["U_steadiness"] = np.mean([reliability(g, c) for _, g in U.groupby("room")])
        rows.append(d)
    print(pd.DataFrame(rows).round(2).to_string(index=False))
    print("\nper-room median of frame interval (ms) and share of frames delayed > 1.5x:")
    print(t.groupby("room")[["median_ms", "frac_gt_1.5x"]].median().round(3).T.to_string())
