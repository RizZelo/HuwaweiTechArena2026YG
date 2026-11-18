"""Presence probe: keep the 0.47 model's empty/occupied decision, replace its 1..4 answer by noise.

    python src/presence_probe.py --model submissions/vF_ahmed_w_cuts_fixed.onnx \
                                 --output submissions/vQ_presence_probe.onnx

Output class:
    empty (the model's own argmax is 0) -> 0
    otherwise                           -> 1 + floor(4 * frac(S))     S = sum of one subcarrier's real part over the 600 frames
A pseudo-random number computed from the data bytes: the same window always gives the same answer
(as the platform requires), but it has no link with the headcount, so it is right 1 time in 4.

On a balanced test (20% per class) with p = share of truly empty windows that the model calls empty
and exactly 20% of windows called empty:  score = 0.15 + 0.25 * p.
"""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
from onnx import TensorProto, helper, numpy_helper
from scipy.stats import spearmanr

import data

SUB = (100, 101)          # subcarrier slice used for the pseudo-random number


def probe_model(model):
    m = onnx.ModelProto(); m.CopyFrom(model)
    g = m.graph
    for n in g.node:                                              # keep his answer inside, under another name
        n.output[:] = ["team_logits" if o == "logits" else o for o in n.output]
    g.output.pop()
    c = lambda name, v, dt=np.float32: g.initializer.append(numpy_helper.from_array(np.asarray(v, dt), name))
    c("pp_starts", [0, 0, 0, SUB[0], 0], np.int64); c("pp_ends", [600, 1, 1, SUB[1], 1], np.int64)
    c("pp_axes", [0, 1, 2, 3, 4], np.int64); c("pp_one", 1.0); c("pp_four", 4.0); c("pp_zero", 0.0); c("pp_three", 3.0)
    c("pp_k", np.arange(5)); c("pp_m10", -10.0); c("pp_zero_i", 0, np.int64)
    g.node.extend([
        helper.make_node("Slice", ["frames", "pp_starts", "pp_ends", "pp_axes"], ["pp_x"]),
        helper.make_node("ReduceSum", ["pp_x"], ["pp_s"], keepdims=0),
        helper.make_node("Abs", ["pp_s"], ["pp_abs"]),
        helper.make_node("Mod", ["pp_abs", "pp_one"], ["pp_frac"], fmod=1),
        helper.make_node("Mul", ["pp_frac", "pp_four"], ["pp_f4"]),
        helper.make_node("Floor", ["pp_f4"], ["pp_fl"]),
        helper.make_node("Min", ["pp_fl", "pp_three"], ["pp_fl3"]),
        helper.make_node("Add", ["pp_fl3", "pp_one"], ["pp_rand"]),                           # 1..4
        helper.make_node("ArgMax", ["team_logits"], ["pp_arg"], axis=0, keepdims=0),
        helper.make_node("Equal", ["pp_arg", "pp_zero_i"], ["pp_empty"]),
        helper.make_node("Where", ["pp_empty", "pp_zero", "pp_rand"], ["pp_cls"]),
        helper.make_node("Sub", ["pp_k", "pp_cls"], ["pp_d"]),
        helper.make_node("Abs", ["pp_d"], ["pp_ad"]),
        helper.make_node("Mul", ["pp_ad", "pp_m10"], ["pp_scores"]),
        helper.make_node("Mul", ["team_logits", "pp_zero"], ["pp_keep"]),
        helper.make_node("Add", ["pp_scores", "pp_keep"], ["logits"]),
    ])
    g.output.append(helper.make_tensor_value_info("logits", TensorProto.FLOAT, [5]))
    onnx.checker.check_model(m)
    return m


def rand_class_np(frames):
    s = np.float32(frames[:, 0, 0, SUB[0]:SUB[1], 0].sum(dtype=np.float32))
    return 1 + min(int(np.floor(np.mod(np.abs(s), np.float32(1.0)) * 4)), 3)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    base = onnx.load(str(args.model))
    onnx.save(probe_model(base), str(args.output))

    k = pd.read_pickle(data.CACHE / "taps_vF_ahmed_w_cuts_fixed.pkl")
    sess = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    rows, same, times = [], 0, []
    for tr, g in k.groupby("trace", sort=False):
        csi, ati = data.load_trace(tr, "unlabeled")
        for r in g.itertuples():
            a = ati[r.start:r.start + 600].copy(); a[-1] = 0
            f, i = data.to_model_input(csi[r.start:r.start + 600], a)
            t0 = perf_counter(); out = sess.run(["logits"], {"frames": f, "intervals": i})[0]; times.append(perf_counter() - t0)
            assert out.shape == (5,) and out.dtype == np.float32
            expect = 0 if r.cls == 0 else rand_class_np(f)
            same += int(out.argmax() == expect)
            rows.append(dict(room=r.room, cls=r.cls, probe=int(out.argmax()), fused=r.fused))
    d = pd.DataFrame(rows)
    print(f"saved {args.output} ({args.output.stat().st_size:,} bytes)")
    print(f"ONNX == hand rule on {same}/{len(d)} unlabelled windows | CPU time per sample: mean {np.mean(times[1:]):.3f} s, max {np.max(times[1:]):.3f} s")
    print("answers 0..4 on unlabelled windows:", (np.bincount(d.probe, minlength=5) / len(d)).round(3).tolist())
    o = d[d.cls > 0]
    print("among occupied windows, share of random answers 1..4:", (np.bincount(o.probe, minlength=5)[1:] / len(o)).round(3).tolist())
    print("per room (1..4):", {r: (np.bincount(g.probe, minlength=5)[1:] / len(g)).round(2).tolist() for r, g in o.groupby("room")})
    print("link with his count score (should be ~0): Spearman %.3f" % spearmanr(o.probe, o.fused)[0])
    print("is the random answer the same as his own answer? %.3f (chance 0.25)" % (o.probe == o.cls).mean())
