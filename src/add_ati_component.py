"""Add frame timing (ATI) to the teammate's count score: one extra fusion component.

    python src/add_ati_component.py --model submissions/vF_ahmed_w_cuts_fixed.onnx --weight 0.25 \
                                    --output submissions/vA2_ahmed_w_ati025.onnx

late  = share of frame intervals longer than 1.5x the window's median interval
        (the channel was busy: hypothesis "more people -> more phones -> more delays")
rank  = position of `late` inside the recognised room (mid-rank against a grid of 99 quantiles of
        that room's occupied unlabelled windows; ties count half, because most windows have 0-2 late frames)
fused = (1 - W) * his fused score + W * rank          (direction and W fixed in advance)

The cut points are then recomputed as the quartiles of the NEW fused score, per recognised room
(unlabelled rooms with >= 60 occupied windows; other rooms use the mean, as in his notebook).
Empty/occupied decision, his weights and networks: unchanged.
"""

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import torch
from onnx import helper, numpy_helper

import data
from fix_ahmed_cuts import NAMED, UNLABELLED, run_unlabelled, taps

PGRID = np.linspace(0.01, 0.99, 99)


def late_share_np(intervals):
    a = np.asarray(intervals[:-1], np.float32)
    return float((a > 1.5 * np.sort(a)[(len(a) - 1) // 2]).mean())


class AtiComponent(torch.nn.Module):
    """(intervals, room one-hot, room-known flag, his fused score) -> new fused score"""

    def __init__(self, grid, grid_g, w):
        super().__init__()
        self.register_buffer("grid", torch.tensor(grid, dtype=torch.float32))       # (11, 99)
        self.register_buffer("grid_g", torch.tensor(grid_g, dtype=torch.float32))   # (99,)
        self.w = float(w)

    def forward(self, intervals, onehot, known, fused):
        a = intervals[:-1]
        late = (a > 1.5 * torch.sort(a).values[(a.shape[0] - 1) // 2]).float().mean()
        g = known * (onehot @ self.grid) + (1.0 - known) * self.grid_g
        rank = (g < late).float().mean() + 0.5 * (g == late).float().mean()
        return (1.0 - self.w) * fused + self.w * rank


def splice(model, comp, fused_name, onehot_name, known_name):
    """Insert `comp` between his fused score and everything that used it."""
    path = data.CACHE / "_ati_component.onnx"
    ex = (torch.full((600,), 5085.0), torch.zeros(11), torch.tensor(1.0), torch.tensor(0.5))
    torch.onnx.export(comp, ex, str(path), input_names=["c_intervals", "c_onehot", "c_known", "c_fused"],
                      output_names=["c_out"], opset_version=18, dynamo=False)
    sub = onnx.load(str(path)).graph
    rename = {"c_intervals": "intervals", "c_onehot": onehot_name, "c_known": known_name, "c_fused": fused_name}
    pre = lambda n: rename[n] if n in rename else (n if n == "" else "ati_" + n)

    m = onnx.ModelProto(); m.CopyFrom(model)
    for n in m.graph.node:                                              # his consumers now read the new score
        n.input[:] = ["ati_c_out" if i == fused_name else i for i in n.input]
    new_nodes = []
    for n in sub.node:
        c = onnx.NodeProto(); c.CopyFrom(n)
        c.input[:] = [pre(i) for i in n.input]; c.output[:] = [pre(o) for o in n.output]; c.name = "ati_" + n.name
        new_nodes.append(c)
    for t in sub.initializer:
        c = onnx.TensorProto(); c.CopyFrom(t); c.name = pre(t.name); m.graph.initializer.append(c)
    # place the new nodes right before the first node that reads ati_c_out (keeps the graph in order)
    first = next(k for k, n in enumerate(m.graph.node) if "ati_c_out" in n.input)
    nodes = list(m.graph.node)
    del m.graph.node[:]
    m.graph.node.extend(nodes[:first] + new_nodes + nodes[first:])
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--weight", type=float, default=0.25)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    base = onnx.load(str(args.model))
    names = taps(base)

    # his fused score + recognised room on every unlabelled window (fused does not depend on the cut points)
    df = run_unlabelled(base, data.CACHE / f"taps_{args.model.stem}.pkl")
    late = []
    for tr, g in df.groupby("trace", sort=False):
        _, ati = data.load_trace(tr, "unlabeled")
        for s in g.start:
            a = ati[s:s + 600].astype(np.float32).copy(); a[-1] = 0
            late.append(((tr, s), late_share_np(a)))
    df["late"] = df.set_index(["trace", "start"]).index.map(dict(late))

    # grids from each unlabelled room's occupied windows (his rule: true room, >= 60 windows)
    occ = df[df.cls > 0]
    grid = np.zeros((len(NAMED), 99)); ok = np.zeros(len(NAMED), bool)
    for k, r in enumerate(NAMED):
        v = occ.late[occ.room == r]
        if r in UNLABELLED and len(v) >= 60:
            grid[k] = np.quantile(v, PGRID); ok[k] = True
    grid_g = grid[ok].mean(0); grid[~ok] = grid_g
    print("late frames per window (unlabelled):", (df.late * 599).round().value_counts().sort_index().head(6).to_dict())

    def rank_of(v, g):
        return (g < v).mean() + 0.5 * (g == v).mean()
    gsel = [grid[NAMED.index(r)] if r != "global" else grid_g for r in df.recognised]
    df["ati_rank"] = [rank_of(v, g) for v, g in zip(df.late, gsel)]
    df["new_fused"] = (1 - args.weight) * df.fused + args.weight * df.ati_rank

    init = {t.name: numpy_helper.to_array(t) for t in base.graph.initializer}
    thr = init["thr"].copy(); okt = np.zeros(len(NAMED), bool)
    for k, r in enumerate(NAMED):
        v = df.new_fused[(df.recognised == r) & (df.cls > 0)]
        if r in UNLABELLED and len(v) >= 60:
            thr[k] = np.quantile(v, [0.25, 0.5, 0.75]); okt[k] = True
    thr_g = thr[okt].mean(0); thr[~okt] = thr_g

    m = splice(base, AtiComponent(grid, grid_g, args.weight), names[1], names[2], names[3])
    for t in m.graph.initializer:
        if t.name in ("thr", "thr_g"):
            t.CopyFrom(numpy_helper.from_array((thr if t.name == "thr" else thr_g).astype(np.float32), t.name))
    onnx.checker.check_model(m)
    onnx.save(m, str(args.output))

    cut = np.stack([thr[NAMED.index(r)] if r != "global" else thr_g for r in df.recognised])
    df["expect"] = np.where(df.cls > 0, 1 + (df.new_fused.values[:, None] > cut).sum(1), 0)
    base_cut = np.stack([init["thr"][NAMED.index(r)] if r != "global" else init["thr_g"] for r in df.recognised])
    df["before"] = np.where(df.cls > 0, 1 + (df.fused.values[:, None] > base_cut).sum(1), 0)
    print(f"answers changed vs {args.model.name}: {(df.expect != df.before).mean():.1%} of unlabelled windows")
    print("class shares among occupied windows per room:")
    print(df[df.expect > 0].groupby("room").expect.value_counts(normalize=True).unstack(fill_value=0).round(2).to_string())

    opts = ort.SessionOptions(); opts.intra_op_num_threads = 4
    sess = ort.InferenceSession(str(args.output), opts, providers=["CPUExecutionProvider"])
    check, same, times = df.sample(150, random_state=1), 0, []
    from time import perf_counter
    for r in check.itertuples():
        csi, ati = data.load_trace(r.trace, "unlabeled")
        w, a = csi[r.start:r.start + 600], ati[r.start:r.start + 600].copy(); a[-1] = 0
        f, i = data.to_model_input(w, a)
        t0 = perf_counter(); out = sess.run(["logits"], {"frames": f, "intervals": i})[0]; times.append(perf_counter() - t0)
        assert out.shape == (5,) and out.dtype == np.float32
        same += int(out.argmax() == r.expect)
    print(f"saved {args.output} ({args.output.stat().st_size:,} bytes) | ONNX == hand-computed answer on {same}/{len(check)} windows")
    print(f"CPU time per sample: mean {np.mean(times[1:]):.3f} s, max {np.max(times[1:]):.3f} s (limit 0.7 s)")
