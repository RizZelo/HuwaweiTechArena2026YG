"""Change the fusion weights of the 0.47 model: halve the noisiest component (Doppler bandwidth).

    python src/reweight_components.py --bandwidth-scale 0.5 --output submissions/vW_bw_half.onnx

component_steadiness.py showed that bandwidth is the noisiest of the four in-room ranks (steadiness 0.25 against
0.56-0.75 for the others) while carrying the largest weight (0.363). Rule fixed in advance: bandwidth weight
x scale, the other three weights rescaled by the same factor so that the weights still sum to 1.
The cut points are recomputed as the quartiles of the NEW fused score of the occupied unlabelled windows
recognised as each room (his notebook rule). Presence, networks and room recognition are unchanged.
"""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

import data
from check_motion_ati import reliability
from component_steadiness import TENSORS, W_NOW, component_table
from fix_ahmed_cuts import NAMED, UNLABELLED

MUL_NODES = {"activity": "/Mul_20", "bandwidth": "/Mul_23", "encdec": "/Mul_26", "rank": "/Mul_29"}


def new_weights(scale):
    bw = W_NOW["bandwidth"] * scale
    f = (1.0 - bw) / (1.0 - W_NOW["bandwidth"])
    return {k: (bw if k == "bandwidth" else v * f) for k, v in W_NOW.items()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bandwidth-scale", type=float, default=0.5)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    W = new_weights(args.bandwidth_scale)
    print("weights now -> new:", {k: (round(W_NOW[k], 3), round(W[k], 3)) for k in W})

    t = component_table()
    t["fused_new"] = sum(W[c] * t[c] for c in W)
    occ = t[t.cls > 0]
    base = onnx.load(str(data.ROOT / "submissions" / "vF_ahmed_w_cuts_fixed.onnx"))
    init = {x.name: numpy_helper.to_array(x) for x in base.graph.initializer}
    thr = init["thr"].copy(); ok = np.zeros(len(NAMED), bool)
    for k, r in enumerate(NAMED):
        v = occ.fused_new[(occ.recognised == r)]
        if r in UNLABELLED and len(v) >= 60:
            thr[k] = np.quantile(v, [0.25, 0.5, 0.75]); ok[k] = True
    thr_g = thr[ok].mean(0); thr[~ok] = thr_g

    m = onnx.ModelProto(); m.CopyFrom(base)
    byout = {o: n for n in m.graph.node for o in n.output}
    for comp, mul in MUL_NODES.items():
        node = next(n for n in m.graph.node if n.name == mul)
        const = next(byout[i] for i in node.input if i in byout and byout[i].op_type == "Constant")
        const.attribute[0].t.CopyFrom(numpy_helper.from_array(np.asarray(W[comp], np.float32)))
    for x in m.graph.initializer:
        if x.name in ("thr", "thr_g"):
            x.CopyFrom(numpy_helper.from_array((thr if x.name == "thr" else thr_g).astype(np.float32), x.name))
    onnx.checker.check_model(m)
    onnx.save(m, str(args.output))

    cut = np.stack([thr[NAMED.index(r)] if r != "global" else thr_g for r in t.recognised])
    t["expect"] = np.where(t.cls > 0, 1 + (t.fused_new.values[:, None] > cut).sum(1), 0)
    base_cut = np.stack([init["thr"][NAMED.index(r)] if r != "global" else init["thr_g"] for r in t.recognised])
    t["before"] = np.where(t.cls > 0, 1 + (t.fused.values[:, None] > base_cut).sum(1), 0)
    print(f"answers changed vs the 0.47 file: {(t.expect != t.before).mean():.1%} of unlabelled windows "
          f"| changed by 2+ classes: {(abs(t.expect - t.before) >= 2).mean():.1%}")
    o = t[(t.cls > 0) & (t.room != 'unknown')]
    print("steadiness of the fused score (mean over rooms): before %.2f -> after %.2f" % (
        np.mean([reliability(g, "fused") for _, g in o.groupby("room")]),
        np.mean([reliability(g, "fused_new") for _, g in o.groupby("room")])))
    print("class shares among occupied windows per room:")
    print(t[t.expect > 0].groupby("room").expect.value_counts(normalize=True).unstack(fill_value=0).round(2).to_string())

    opts = ort.SessionOptions(); opts.intra_op_num_threads = 4
    sess = ort.InferenceSession(str(args.output), opts, providers=["CPUExecutionProvider"])
    same, times = 0, []
    chk = t.sample(150, random_state=3)
    for r in chk.itertuples():
        csi, ati = data.load_trace(r.trace, "unlabeled")
        a = ati[r.start:r.start + 600].copy(); a[-1] = 0
        f, i = data.to_model_input(csi[r.start:r.start + 600], a)
        t0 = perf_counter(); out = sess.run(["logits"], {"frames": f, "intervals": i})[0]; times.append(perf_counter() - t0)
        assert out.shape == (5,) and out.dtype == np.float32
        same += int(out.argmax() == r.expect)
    print(f"saved {args.output} ({args.output.stat().st_size:,} bytes) | ONNX == hand-computed answer on {same}/{len(chk)} windows")
    print(f"CPU time per sample: mean {np.mean(times[1:]):.3f} s, max {np.max(times[1:]):.3f} s (limit 0.7 s)")
