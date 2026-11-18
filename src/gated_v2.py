"""v2 (the 0.39 CNN) combined with the activity score's "still" switch. No training.

    python src/gated_v2.py --mode argmax  --output submissions/vG_v2_gate.onnx
    python src/gated_v2.py --mode ranking --output submissions/vG_v2_gate_rank.onnx

argmax   still (activity < 3) -> 0, otherwise v2's own answer
         (the way the team's 0.41 model combines its CNN with the switch)
ranking  still -> 0, otherwise v2's expected headcount sum(k * p_k) is used as a ranking score:
         active windows are cut into 4 equal groups -> 1..4 (cut points = quartiles over active
         unlabelled windows). Removes v2's "say 1" bias; tests whether v2 ORDERS people well.
"""

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

import data
from activity import STILL_THRESHOLD
from export import export_and_check
from models import Submission, make_net
from rule_model import Activity


class GatedV2(nn.Module):
    def __init__(self, v2, cuts=None):
        super().__init__()
        self.activity = Activity()
        self.v2 = v2
        self.ranking = cuts is not None
        self.register_buffer("cuts", torch.tensor(np.zeros(3) if cuts is None else cuts, dtype=torch.float32))

    def forward(self, frames, intervals):
        act = self.activity(frames[:, 0])
        logits = self.v2(frames, intervals)
        if self.ranking:
            p = torch.softmax(logits, dim=0)
            expected = (p * torch.arange(5, dtype=torch.float32)).sum()
            group = 1.0 + (expected > self.cuts).float().sum()
        else:
            group = torch.argmax(logits).float()
        cls = torch.where(act < STILL_THRESHOLD, torch.zeros_like(group), group)
        return -10.0 * torch.abs(torch.arange(5, dtype=torch.float32) - cls) + 0.0 * intervals[0]


def ranking_cuts(v2_onnx, step=1200):
    """Quartiles of v2's expected headcount over ACTIVE unlabelled windows (no labels used)."""
    v2 = ort.InferenceSession(str(v2_onnx), providers=["CPUExecutionProvider"])
    act = ort.InferenceSession(str(data.ROOT / "submissions" / "vR_act_minus_power.onnx"), providers=["CPUExecutionProvider"])
    E = []
    for r in data.load_index("unlabeled").itertuples():
        csi, ati = data.load_trace(r.identifier, "unlabeled")
        for s, w, a in data.windows(csi, ati, step=step):
            f, i = data.to_model_input(w, a)
            if act.run(["logits"], {"frames": f, "intervals": i})[0].argmax() == 0:       # still window
                continue
            lo = v2.run(["logits"], {"frames": f, "intervals": i})[0]
            p = np.exp(lo - lo.max()); p /= p.sum()
            E.append((p * np.arange(5)).sum())
    E = np.array(E)
    cuts = np.quantile(E, [0.25, 0.5, 0.75])
    print(f"active unlabelled windows: {len(E)} | v2 expected headcount: min {E.min():.2f} "
          f"median {np.median(E):.2f} max {E.max():.2f} | cuts {cuts.round(3).tolist()}")
    return cuts


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["argmax", "ranking"], required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    net = make_net()
    net.load_state_dict(torch.load(data.CACHE / "cnn_ssl.pt", map_location="cpu"))
    v2 = Submission(net.eval(), clean=False).eval()
    cuts = ranking_cuts(data.ROOT / "submissions" / "v2_ssl_cnn.onnx") if args.mode == "ranking" else None
    export_and_check(GatedV2(v2, cuts).eval(), args.output)
