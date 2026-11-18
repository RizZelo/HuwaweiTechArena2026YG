"""Check (and repair) the class cut points stored inside the teammate's ONNX file.

    python src/fix_ahmed_cuts.py --model path/to/team_model.onnx \
                                 --output submissions/vF_ahmed_cuts_fixed.onnx

Inside his model:
    fused score        the tensor compared with the cut points (/Add_27 in model_ahmed, /Add_18 in model_ahmed_w)
    /Div_1_output_0    recognised room (one-hot over 11 rooms)
    /Mul_2_output_0    1 if a room was recognised, 0 if the global statistics are used
    thr  (11, 3)       per-room cut points: fused score -> class 1 / 2 / 3 / 4

His notebook computes `thr` with the wrong room for every window (`rid_all` is overwritten in
the MAE cell). This script runs the file on every unlabelled window (his windows: every 600
frames), prints the class shares per room, and writes a copy whose `thr` rows are the real
quartiles of the fused score over each unlabelled room's occupied windows. Nothing else changes.
"""

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
from onnx import TensorProto, helper, numpy_helper

import data

NAMED = ["a1", "b1", "c1", "d1", "e1", "g1", "h1", "i1", "j1", "u1", "v1"]     # order of his room bank
UNLABELLED = ["d1", "h1", "i1", "j1", "u1", "v1"]


def taps(model):
    """logits, fused score (the tensor compared with the cut points), recognised room, 'room known' flag"""
    fused = next(n.input[1] for n in model.graph.node if n.op_type == "Sub" and n.input[0].startswith("/Concat"))
    return ["logits", fused, "/Div_1_output_0", "/Mul_2_output_0"]


def with_taps(model):
    m = onnx.ModelProto(); m.CopyFrom(model)
    for n in taps(model)[1:]:
        m.graph.output.append(helper.make_tensor_value_info(n, TensorProto.FLOAT, None))
    return m


def run_unlabelled(model, cache):
    if cache.exists():
        return pd.read_pickle(cache)
    opts = ort.SessionOptions(); opts.intra_op_num_threads = 4
    sess = ort.InferenceSession(with_taps(model).SerializeToString(), opts, providers=["CPUExecutionProvider"])
    names = taps(model)
    rows = []
    for r in data.load_index("unlabeled").itertuples():
        csi, ati = data.load_trace(r.identifier, "unlabeled")
        for s, w, a in data.windows(csi, ati, step=600):
            f, i = data.to_model_input(w, a)
            lo, fused, onehot, known = sess.run(names, {"frames": f, "intervals": i})
            rows.append(dict(trace=r.identifier, room=r.room, start=s, cls=int(lo.argmax()), fused=float(fused.ravel()[0]),
                             recognised=NAMED[int(onehot.ravel().argmax())] if float(known.ravel()[0]) > 0.5 else "global"))
    df = pd.DataFrame(rows); df.to_pickle(cache)
    return df


def shares(df):
    occ = df[df.cls > 0]
    t = occ.groupby("room").cls.value_counts(normalize=True).unstack(fill_value=0).round(2)
    t["empty"] = df.groupby("room").cls.apply(lambda c: (c == 0).mean()).round(2)
    t["windows"] = df.groupby("room").size()
    return t


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    model = onnx.load(str(args.model))
    df = run_unlabelled(model, data.CACHE / f"taps_{args.model.stem}.pkl")

    print("class shares among occupied windows per unlabelled room (design: 0.25 each; empty: 0.20)")
    print(shares(df).to_string())
    print("\nrecognised room vs true room (share of windows):")
    print(pd.crosstab(df.room, df.recognised, normalize="index").round(2).to_string())

    init = {t.name: t for t in model.graph.initializer}
    thr = numpy_helper.to_array(init["thr"]).copy()
    # a row of `thr` is applied to the windows RECOGNISED as that room, so its quartiles come from those windows
    print("\nroom   stored cut points       real quartiles of the occupied windows recognised as this room")
    for k, r in enumerate(NAMED):
        occ = df[(df.recognised == r) & (df.cls > 0)]
        if r in UNLABELLED and len(occ) >= 60:          # same rule as his notebook; labelled rooms keep the global row
            q = np.quantile(occ.fused, [0.25, 0.5, 0.75])
            print(f"{r:5s}  {np.round(thr[k], 3).tolist()!s:22s}  {np.round(q, 3).tolist()}")
            thr[k] = q
    fixed = onnx.ModelProto(); fixed.CopyFrom(model)
    for t in fixed.graph.initializer:
        if t.name == "thr":
            t.CopyFrom(numpy_helper.from_array(thr.astype(np.float32), "thr"))
    onnx.checker.check_model(fixed)
    onnx.save(fixed, str(args.output))
    print(f"\nsaved {args.output} ({args.output.stat().st_size:,} bytes): only `thr` changed")

    # expected answers of the fixed file (same presence decision, new cut points), then check the file itself
    thr_g = numpy_helper.to_array(init["thr_g"])
    cut = np.stack([thr[NAMED.index(r)] if r != "global" else thr_g for r in df.recognised])
    df["fixed"] = np.where(df.cls > 0, 1 + (df.fused.values[:, None] > cut).sum(1), 0)
    print("\nclass shares among occupied windows AFTER the fix:")
    print(df[df.fixed > 0].groupby("room").fixed.value_counts(normalize=True).unstack(fill_value=0).round(2).to_string())
    print(f"answers changed vs the original file: {(df.fixed != df.cls).mean():.1%} of unlabelled windows")

    sess = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    check = df.sample(150, random_state=0)
    same = 0
    for r in check.itertuples():
        csi, ati = data.load_trace(r.trace, "unlabeled")
        w, a = csi[r.start:r.start + 600], ati[r.start:r.start + 600].copy(); a[-1] = 0
        f, i = data.to_model_input(w, a)
        same += int(sess.run(["logits"], {"frames": f, "intervals": i})[0].argmax() == r.fixed)
    print(f"fixed ONNX file == expected answer on {same}/{len(check)} random unlabelled windows")
