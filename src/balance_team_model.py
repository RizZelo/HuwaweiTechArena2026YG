"""Balanced version of the team's 0.41 model: same still switch, same CNN, fixed class correction.

    python src/balance_team_model.py --model path/to/team_model.onnx \
                                     --output submissions/vT_team_balanced.onnx

Inside the team model there are two intermediate results:
    log_4   the activity score            (still if < 3.0)
    add_51  the CNN's 5 scores before the switch

New output, added to a copy of the graph (the original file is not changed):
    still  -> 0
    active -> argmax over classes 1..4 of (CNN score + correction)     (never 0 when moving)
The 4 corrections are chosen so that, on the ACTIVE unlabelled windows, the answers are
1, 2, 3, 4 at 25% each (the test is class-balanced and still windows are the empty ones).
The CNN's ordering is kept; only its preference for some classes is removed.
"""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

import data
from export import real_samples

STILL = 3.0


def with_outputs(model, names):
    m = onnx.ModelProto(); m.CopyFrom(model)
    for n in names:
        m.graph.output.append(helper.make_tensor_value_info(n, TensorProto.FLOAT, None))
    return m


def active_cnn_scores(model, step=1200):
    """CNN scores (classes 1..4) of every ACTIVE unlabelled window."""
    sess = ort.InferenceSession(with_outputs(model, ["log_4", "add_51"]).SerializeToString(),
                                providers=["CPUExecutionProvider"])
    Z = []
    for r in data.load_index("unlabeled").itertuples():
        csi, ati = data.load_trace(r.identifier, "unlabeled")
        for s, w, a in data.windows(csi, ati, step=step):
            f, i = data.to_model_input(w, a)
            act, z = sess.run(["log_4", "add_51"], {"frames": f, "intervals": i})
            if act[0] >= STILL:
                Z.append(z[0, 1:])
    return np.array(Z)


def fit_correction(Z, iters=2000, lr=0.05):
    """Shift the 4 scores until argmax picks each class 25% of the time."""
    b = np.zeros(4)
    for _ in range(iters):
        share = np.bincount((Z + b).argmax(1), minlength=4) / len(Z)
        b -= lr * np.log((share + 1e-3) / 0.25)
        b -= b[0]                                   # only differences matter
    return b, np.bincount((Z + b).argmax(1), minlength=4) / len(Z)


def balanced_model(model, b):
    m = onnx.ModelProto(); m.CopyFrom(model)
    g = m.graph
    old_out = g.output[0].name                      # "logits" of the original model
    for n in g.node:                                # rename the original output, keep it inside
        n.output[:] = ["team_logits" if o == old_out else o for o in n.output]
    g.output.pop()
    c = lambda name, v, dt=np.float32: g.initializer.append(numpy_helper.from_array(np.asarray(v, dt), name))
    c("bal_bias", b.reshape(1, 4)); c("bal_starts", [1], np.int64); c("bal_ends", [5], np.int64)
    c("bal_axes", [1], np.int64); c("bal_one", 1.0); c("bal_zero", 0.0); c("bal_still", STILL)
    c("bal_k", np.arange(5)); c("bal_minus10", -10.0)
    g.node.extend([
        helper.make_node("Slice", ["add_51", "bal_starts", "bal_ends", "bal_axes"], ["bal_z"]),          # (1, 4)
        helper.make_node("Add", ["bal_z", "bal_bias"], ["bal_zb"]),
        helper.make_node("ArgMax", ["bal_zb"], ["bal_arg"], axis=1, keepdims=0),                       # (1,)
        helper.make_node("Cast", ["bal_arg"], ["bal_argf"], to=TensorProto.FLOAT),
        helper.make_node("Add", ["bal_argf", "bal_one"], ["bal_active_cls"]),                         # 1..4
        helper.make_node("Less", ["log_4", "bal_still"], ["bal_is_still"]),
        helper.make_node("Where", ["bal_is_still", "bal_zero", "bal_active_cls"], ["bal_cls"]),        # (1,)
        helper.make_node("Sub", ["bal_k", "bal_cls"], ["bal_d"]),
        helper.make_node("Abs", ["bal_d"], ["bal_ad"]),
        helper.make_node("Mul", ["bal_ad", "bal_minus10"], ["bal_scores"]),
        helper.make_node("Mul", ["team_logits", "bal_zero"], ["bal_keep"]),                            # keeps both inputs used
        helper.make_node("Add", ["bal_scores", "bal_keep"], ["logits"]),
    ])
    g.output.append(helper.make_tensor_value_info("logits", TensorProto.FLOAT, [5]))
    onnx.checker.check_model(m)
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    team = onnx.load(str(args.model))

    Z = active_cnn_scores(team)
    before = np.bincount(Z.argmax(1), minlength=4) / len(Z)
    b, after = fit_correction(Z)
    print(f"active unlabelled windows: {len(Z)}")
    print(f"CNN answers 1..4 on them, before: {before.round(2).tolist()} | after correction: {after.round(2).tolist()}")
    print(f"correction added to the scores of classes 1..4: {b.round(3).tolist()}")

    onnx.save(balanced_model(team, b), str(args.output))
    sess = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    check = ort.InferenceSession(with_outputs(team, ["log_4", "add_51"]).SerializeToString(), providers=["CPUExecutionProvider"])
    same, times = 0, []
    S = real_samples()
    for f, i in S:
        t = perf_counter(); out = sess.run(["logits"], {"frames": f, "intervals": i})[0]; times.append(perf_counter() - t)
        assert out.shape == (5,) and out.dtype == np.float32
        _, act, z = check.run(None, {"frames": f, "intervals": i})
        expect = 0 if act[0] < STILL else 1 + int((z[0, 1:] + b).argmax())
        same += int(out.argmax()) == expect
    print(f"saved {args.output} ({args.output.stat().st_size:,} bytes) | ONNX answer == rule computed by hand: {same}/{len(S)}")
    print(f"CPU time per sample: mean {np.mean(times[1:]):.3f} s, max {np.max(times[1:]):.3f} s (limit 0.7 s)")
