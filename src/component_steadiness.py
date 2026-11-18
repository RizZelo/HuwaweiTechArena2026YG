"""How noisy is each component of the 0.47 model, and does a different fixed weighting steady the count score?

    python src/component_steadiness.py

Components (in-room ranks, 0..1) read from inside vF_ahmed_w_cuts_fixed.onnx:
    activity (weight 0.204), Doppler bandwidth (0.363), encdec (0.205), rank (0.228)
Label-free checks on the occupied unlabelled windows:
    steadiness     = share of the variation that is between recordings (a recording has one headcount)
    agreement      = Spearman between the recording averages of two components (do they say the same thing?)
and the same steadiness for other fixed weightings of the four ranks.
"""

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
from onnx import TensorProto, helper
from scipy.stats import spearmanr

import data
from check_motion_ati import reliability
from fix_ahmed_cuts import taps

MODEL = data.ROOT / "submissions" / "vF_ahmed_w_cuts_fixed.onnx"
TENSORS = {"activity": "/ReduceMean_2_output_0", "bandwidth": "/ReduceMean_3_output_0",
           "encdec": "/ReduceMean_4_output_0", "rank": "/ReduceMean_5_output_0"}
W_NOW = {"activity": 0.204, "bandwidth": 0.363, "encdec": 0.205, "rank": 0.228}


def component_table():
    path = data.CACHE / "components_w.pkl"
    if path.exists():
        return pd.read_pickle(path)
    m = onnx.load(str(MODEL))
    for n in TENSORS.values():
        m.graph.output.append(helper.make_tensor_value_info(n, TensorProto.FLOAT, None))
    sess = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
    known = pd.read_pickle(data.CACHE / "taps_vF_ahmed_w_cuts_fixed.pkl")
    rows = []
    for tr, g in known.groupby("trace", sort=False):
        csi, ati = data.load_trace(tr, "unlabeled")
        for s in g.start:
            a = ati[s:s + 600].copy(); a[-1] = 0
            f, i = data.to_model_input(csi[s:s + 600], a)
            out = sess.run(list(TENSORS.values()), {"frames": f, "intervals": i})
            rows.append(dict(trace=tr, start=s, **{k: float(v.ravel()[0]) for k, v in zip(TENSORS, out)}))
    out = known.merge(pd.DataFrame(rows), on=["trace", "start"])
    out.to_pickle(path)
    return out


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    t = component_table()
    U = t[(t.cls > 0) & (t.room != "unknown")].copy()
    rooms = sorted(U.room.unique())
    comps = list(TENSORS)
    chk = sum(W_NOW[c] * U[c] for c in comps)
    print(f"occupied windows {len(U)} | rebuilt fused == model's fused: max difference {np.abs(chk - U.fused).max():.1e}\n")

    print("steadiness of each component (between-recording share, per room; higher = steadier)")
    st = pd.DataFrame({c: {r: reliability(g, c) for r, g in U.groupby("room")} for c in comps + ["fused"]})
    print(st.round(2).to_string()); print("mean:", st.mean().round(2).to_dict())

    print("\nagreement of recording averages between components (Spearman, mean over rooms)")
    R = U.groupby(["room", "trace"])[comps].mean().reset_index()
    ag = pd.DataFrame({a: {b: np.mean([spearmanr(g[a], g[b])[0] for _, g in R.groupby("room")]) for b in comps} for a in comps})
    print(ag.round(2).to_string())

    print("\nsteadiness of other fixed weightings (activity, bandwidth, encdec, rank)")
    rows = []
    for name, w in {"current (learned)": W_NOW, "original 0.46": dict(activity=.335, bandwidth=.33, encdec=.1675, rank=.1675),
                    "equal": dict(activity=.25, bandwidth=.25, encdec=.25, rank=.25),
                    "no bandwidth": dict(activity=.3, bandwidth=0.0, encdec=.35, rank=.35),
                    "half bandwidth": dict(activity=.25, bandwidth=.18, encdec=.285, rank=.285)}.items():
        U["x"] = sum(w[c] * U[c] for c in comps)
        s = np.mean([reliability(g, "x") for _, g in U.groupby("room")])
        same = np.mean([spearmanr(g.x, g.fused)[0] for _, g in U.groupby("room")])
        rows.append(dict(weights=name, steadiness=s, same_order_as_current=same))
    print(pd.DataFrame(rows).round(2).to_string(index=False))
