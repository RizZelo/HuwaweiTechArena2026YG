"""Flaw 2: is the per-room 20% empty quota the best way to turn the presence score into empty / occupied?

    python src/check_presence_threshold.py

Inside the 0.47 file: /Add_6_output_0 = presence score (high = looks empty), /Add_7_output_0 = the threshold of the
recognised room (or the global one). Empty if score > threshold.

Label-free test: a recording has ONE headcount, so a good decision calls each recording all-empty or
all-occupied. purity = mean over recordings of max(share empty, share occupied) (1.0 = perfect), at the same
overall empty share (20%) for every variant, so a variant cannot win by calling everything occupied.
Variants:
  per-room quota   threshold of the recognised room (the current model)
  global           one threshold for all windows
  shrunk           the average of the two
  true-room quota  the quota of the TRUE room (not available at test time: a reference)
"""

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
from onnx import TensorProto, helper

import data

MODEL = data.ROOT / "submissions" / "vF_ahmed_w_cuts_fixed.onnx"
TAPS = ["/Add_6_output_0", "/Add_7_output_0", "/Div_1_output_0", "/Mul_2_output_0"]
EMPTY = 0.20


def presence_table():
    path = data.CACHE / "presence_scores.pkl"
    if path.exists():
        return pd.read_pickle(path)
    m = onnx.load(str(MODEL))
    for n in TAPS:
        m.graph.output.append(helper.make_tensor_value_info(n, TensorProto.FLOAT, None))
    sess = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
    known = pd.read_pickle(data.CACHE / "taps_vF_ahmed_w_cuts_fixed.pkl")[["trace", "start", "room", "cls", "recognised"]]
    rows = []
    for tr, g in known.groupby("trace", sort=False):
        csi, ati = data.load_trace(tr, "unlabeled")
        for s in g.start:
            a = ati[s:s + 600].copy(); a[-1] = 0
            f, i = data.to_model_input(csi[s:s + 600], a)
            sc, tau, _, _ = sess.run(["logits"] + TAPS, {"frames": f, "intervals": i})[1:5]
            rows.append(dict(trace=tr, start=s, score=float(sc.ravel()[0]), tau=float(tau.ravel()[0])))
    out = known.merge(pd.DataFrame(rows), on=["trace", "start"])
    out.to_pickle(path)
    return out


def purity(df, empty):
    p = empty.groupby(df.trace).mean()
    return float(np.maximum(p, 1 - p).mean()), int(((p > 0.1) & (p < 0.9)).sum()), p


if __name__ == "__main__":
    t = presence_table()
    print(f"model's own decision == (score > tau): {((t.score > t.tau) == (t.cls == 0)).mean():.3f}")
    tau_g = np.quantile(t.score, 1 - EMPTY)
    # quota of the TRUE room: quantile of that room's own scores
    q_true = t.groupby("room").score.transform(lambda s: np.quantile(s, 1 - EMPTY))
    variants = {"per-room quota (current)": t.score > t.tau, "global": t.score > tau_g}
    sh = (t.tau + tau_g) / 2
    variants["shrunk (half room, half global)"] = t.score > sh + np.quantile(t.score - sh, 1 - EMPTY)
    variants["true-room quota (reference)"] = t.score > q_true
    rows = []
    for name, e in variants.items():
        pur, mixed, _ = purity(t, e)
        rows.append(dict(variant=name, empty_share=e.mean(), purity=pur, mixed_recordings=mixed))
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print(f"({t.trace.nunique()} recordings; purity 1.0 = every recording called all-empty or all-occupied)")

    print("\nper room: share of windows called empty, recordings, mixed recordings (10-90% empty), all-empty recordings")
    for name in ["per-room quota (current)", "global"]:
        e = variants[name]; rows = {}
        for r, g in t.groupby("room"):
            p = e[g.index].groupby(g.trace).mean()
            rows[r] = dict(empty=e[g.index].mean(), recordings=len(p), mixed=int(((p > .1) & (p < .9)).sum()), all_empty=int((p >= .9).sum()))
        print(name); print(pd.DataFrame(rows).round(2).to_string())
