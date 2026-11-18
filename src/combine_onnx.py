"""Combine several finished ONNX models into one file that averages their probabilities.

    python src/combine_onnx.py --models cache/rule_soft.onnx path/to/team_model.onnx \
                               --output submissions/vC_rule_plus_team.onnx

Each model is copied unchanged (its names get a prefix m0_, m1_, ...). All of them read the
same `frames` and `intervals`; their outputs go through softmax, are averaged (equal weights
unless --weights is given) and the log of the average is the new `logits`.

Checks: the combined file equals the average of the separate models on real windows,
output float32 (5,), size < 100 MB, CPU time per sample.
"""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, compose, helper, numpy_helper

from export import real_samples


def combine(paths, weights):
    graphs = [compose.add_prefix(onnx.load(str(p)), prefix=f"m{i}_").graph for i, p in enumerate(paths)]
    nodes, inits, infos = [], [], []
    for i, g in enumerate(graphs):
        nodes += [helper.make_node("Identity", ["frames"], [f"m{i}_frames"]),
                  helper.make_node("Identity", ["intervals"], [f"m{i}_intervals"])]
        nodes += list(g.node); inits += list(g.initializer); infos += list(g.value_info)
        nodes.append(helper.make_node("Softmax", [f"m{i}_logits"], [f"m{i}_prob"], axis=0))
        inits.append(numpy_helper.from_array(np.array(weights[i], np.float32), f"m{i}_weight"))
        nodes.append(helper.make_node("Mul", [f"m{i}_prob", f"m{i}_weight"], [f"m{i}_wprob"]))
    nodes.append(helper.make_node("Sum", [f"m{i}_wprob" for i in range(len(graphs))], ["avg_prob"]))
    inits.append(numpy_helper.from_array(np.array(1e-9, np.float32), "eps"))
    nodes.append(helper.make_node("Add", ["avg_prob", "eps"], ["avg_prob_eps"]))
    nodes.append(helper.make_node("Log", ["avg_prob_eps"], ["logits"]))
    graph = helper.make_graph(
        nodes, "combined",
        inputs=[helper.make_tensor_value_info("frames", TensorProto.FLOAT, [600, 1, 3, 1024, 2]),
                helper.make_tensor_value_info("intervals", TensorProto.FLOAT, [600])],
        outputs=[helper.make_tensor_value_info("logits", TensorProto.FLOAT, [5])],
        initializer=inits, value_info=infos)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    return model


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=Path, nargs="+", required=True)
    ap.add_argument("--weights", type=float, nargs="+")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    w = args.weights or [1.0 / len(args.models)] * len(args.models)
    onnx.save(combine(args.models, w), str(args.output))

    opts = ort.SessionOptions(); opts.intra_op_num_threads = 4; opts.inter_op_num_threads = 1
    comb = ort.InferenceSession(str(args.output), sess_options=opts, providers=["CPUExecutionProvider"])
    parts = [ort.InferenceSession(str(p), providers=["CPUExecutionProvider"]) for p in args.models]
    soft = lambda v: np.exp(v - v.max()) / np.exp(v - v.max()).sum()
    diffs, times, preds = [], [], []
    for f, i in real_samples():
        t = perf_counter(); out = comb.run(["logits"], {"frames": f, "intervals": i})[0]; times.append(perf_counter() - t)
        assert out.shape == (5,) and out.dtype == np.float32
        ref = sum(wk * soft(s.run(["logits"], {"frames": f, "intervals": i})[0]) for wk, s in zip(w, parts))
        diffs.append(np.abs(np.exp(out) - ref).max()); preds.append(int(out.argmax()))
    print(f"saved {args.output} ({args.output.stat().st_size:,} bytes, limit 100,000,000) | weights {w}")
    print(f"combined file vs average of the separate models: max probability difference {max(diffs):.2e}")
    print(f"CPU time per sample: mean {np.mean(times[1:]):.3f} s, max {np.max(times[1:]):.3f} s (limit 0.7 s)")
    print("predictions on test windows:", preds)
