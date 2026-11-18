"""Two label-free stress tests of the 0.47 model (submissions/vF_ahmed_w_cuts_fixed.onnx).

    python src/check_flaws.py

1. UNSEEN ROOM: for each unlabelled room, remove its fingerprint from the model's room bank
   and run that room's windows again. The model then has to treat the room as new (it borrows
   another room's statistics or falls back to the global ones) - this is what happens on the
   private test. We compare with the answers given when the room is known.
2. STEADIER BANDWIDTH: the Doppler bandwidth cue is computed from ONE spectrum of the 3 s window.
   Averaging the spectra of 5 overlapping 1 s pieces (Welch's method) should make it less noisy.
   Checked with the steadiness inside a recording (a recording has one headcount).
"""

import math

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import torch
from onnx import numpy_helper
from scipy.stats import spearmanr

import data
from check_motion_ati import ACTIVE, FS, PAIRS, block_matrix, reliability
from fix_ahmed_cuts import NAMED, UNLABELLED, taps, with_taps

MODEL = data.ROOT / "submissions" / "vF_ahmed_w_cuts_fixed.onnx"


def unseen_room():
    path = data.CACHE / "unseen_room.pkl"
    if path.exists():
        return pd.read_pickle(path)
    base = onnx.load(str(MODEL)); names = taps(base)
    known = pd.read_pickle(data.CACHE / "taps_vF_ahmed_w_cuts_fixed.pkl")
    thr = {t.name: numpy_helper.to_array(t) for t in base.graph.initializer}
    rows = []
    for r in UNLABELLED:
        m = with_taps(base)
        for t in m.graph.initializer:
            if t.name == "bank_fp":
                fp = numpy_helper.to_array(t).copy(); fp[NAMED.index(r)] = 1e3      # this room can never be the nearest
                t.CopyFrom(numpy_helper.from_array(fp, "bank_fp"))
        sess = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
        for tr, g in known[known.room == r].groupby("trace", sort=False):
            csi, ati = data.load_trace(tr, "unlabeled")
            for s in g.start:
                a = ati[s:s + 600].copy(); a[-1] = 0
                f, i = data.to_model_input(csi[s:s + 600], a)
                lo, fused, onehot, kn = sess.run(names, {"frames": f, "intervals": i})
                rows.append(dict(trace=tr, start=s, new_cls=int(lo.argmax()),
                                 new_room=NAMED[int(onehot.argmax())] if float(kn) > 0.5 else "global"))
    out = known.merge(pd.DataFrame(rows), on=["trace", "start"])
    cut = np.stack([thr["thr"][NAMED.index(r)] if r != "global" else thr["thr_g"] for r in out.recognised])
    out["cls"] = np.where(out.cls > 0, 1 + (out.fused.values[:, None] > cut).sum(1), 0)   # answers of the fixed file
    out.to_pickle(path)
    return out


class Bandwidth(torch.nn.Module):
    """His antenna-pair streams -> Doppler bandwidth, from one 600-frame spectrum or from Welch pieces."""

    def __init__(self, seg, hop):
        super().__init__()
        self.seg, self.hop = seg, hop
        t = torch.arange(seg, dtype=torch.float32) / FS
        f = torch.arange(1, seg // 2, dtype=torch.float32) * FS / seg
        win = torch.hann_window(seg, periodic=False)
        self.cosb = torch.cos(2 * math.pi * t[:, None] * f[None]) * win[:, None]
        self.sinb = torch.sin(2 * math.pi * t[:, None] * f[None]) * win[:, None]
        self.f, self.blk = f, block_matrix()
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
        S = 0.0
        for s in range(0, 600 - self.seg + 1, self.hop):
            r, q = xr[s:s + self.seg], xi[s:s + self.seg]
            r, q = r - r.mean(0, keepdim=True), q - q.mean(0, keepdim=True)
            a_re, a_im = r.T @ self.cosb + q.T @ self.sinb, q.T @ self.cosb - r.T @ self.sinb
            b_re, b_im = r.T @ self.cosb - q.T @ self.sinb, q.T @ self.cosb + r.T @ self.sinb
            S = S + (a_re ** 2 + a_im ** 2 + b_re ** 2 + b_im ** 2).sum(0)
        w = (S + 1e-12) * self.m_bw
        mu = (w * self.f).sum() / w.sum()
        return torch.sqrt((w * (self.f - mu) ** 2).sum() / w.sum())


def welch_table():
    path = data.CACHE / "welch_bw.pkl"
    if path.exists():
        return pd.read_pickle(path)
    known = pd.read_pickle(data.CACHE / "taps_vF_ahmed_w_cuts_fixed.pkl")
    variants = {"bw_1x3s": Bandwidth(600, 600), "bw_5x1s": Bandwidth(200, 100), "bw_11x0.5s": Bandwidth(100, 50)}
    rows = []
    for tr, g in known.groupby("trace", sort=False):
        csi, ati = data.load_trace(tr, "unlabeled")
        for s in g.start:
            f, _ = data.to_model_input(csi[s:s + 600], ati[s:s + 600])
            x = torch.from_numpy(f)
            with torch.no_grad():
                rows.append(dict(trace=tr, start=s, **{k: float(v(x)) for k, v in variants.items()}))
    out = known.merge(pd.DataFrame(rows), on=["trace", "start"])
    out.to_pickle(path)
    return out


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    u = unseen_room()
    print("=== 1. the room is removed from the model's bank (simulates a private-test room) ===")
    rows = []
    for r, g in u.groupby("room"):
        both = g[(g.cls > 0) & (g.new_cls > 0)]
        rows.append(dict(room=r, windows=len(g), treated_as=g.new_room.value_counts(normalize=True).round(2).to_dict(),
                         same_answer=(g.cls == g.new_cls).mean(), same_empty_decision=((g.cls > 0) == (g.new_cls > 0)).mean(),
                         called_empty=(g.new_cls == 0).mean(), same_count=(both.cls == both.new_cls).mean(),
                         count_off_by_2plus=(abs(both.cls - both.new_cls) >= 2).mean()))
    print(pd.DataFrame(rows).round(2).to_string(index=False))
    print(f"all rooms together: same answer {(u.cls == u.new_cls).mean():.2f} | "
          f"same empty decision {((u.cls > 0) == (u.new_cls > 0)).mean():.2f}")
    print("class shares when the room is unseen (0..4):")
    print(u.groupby("room").new_cls.value_counts(normalize=True).unstack(fill_value=0).round(2).to_string())

    w = welch_table()
    occ = w[w.cls > 0]
    print("\n=== 2. Doppler bandwidth from averaged spectra (occupied unlabelled windows) ===")
    rows = []
    for c in ["bw_1x3s", "bw_5x1s", "bw_11x0.5s"]:
        rows.append(dict(version=c, steadiness=np.mean([reliability(g, c) for _, g in occ.groupby("room")]),
                         same_order_as_current=np.mean([spearmanr(g[c], g.bw_1x3s)[0] for _, g in occ.groupby("room")])))
    print(pd.DataFrame(rows).round(2).to_string(index=False))
    print("(steadiness = share of the variation that is between recordings; the current cue is bw_1x3s)")
