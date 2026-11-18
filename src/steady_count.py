"""Steadier count score: a student that predicts a RECORDING's average count score from ONE window.

    python src/steady_count.py

Why: a recording has one headcount, but the teammate's count score (`fused`) jumps around between the
20 windows of a recording (about half of its variation is noise inside a recording). The average of
the whole recording is much steadier. A student model that sees one window and is trained to output the
average of its recording learns which window features carry the recording-level signal, and drops the
noisy part. No labels are used for training.

Check (leave one room out, no labels):
  - how well the output of ONE window predicts the average of the OTHER windows of its recording
    (compared with the raw fused score of that window);
  - steadiness = share of the variation that is between recordings;
  - the ordering of recordings must stay the same as the raw score (the student must not invent a new ordering).
Sanity check with labels, only for information: pair accuracy / recording Spearman in a1 and c1.
"""

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

import data
from check_motion_ati import build_table, pair_acc, reliability
from fix_ahmed_cuts import taps, with_taps

MODEL = data.ROOT / "submissions" / "vF_ahmed_w_cuts_fixed.onnx"
ALL = ["act", "act_slow", "act_fast", "act_breath", "act_rx0", "act_rx1", "act_rx2", "act_rx_med", "spread",
       "log_power", "freq_selectivity", "log_dyn_ratio", "doppler_0.3-2Hz", "doppler_2-5Hz", "doppler_5-10Hz",
       "doppler_10-25Hz", "very_slow_share", "log_amp_cv", "log_phase01_var", "log_phase02_var",
       "participation_ratio", "log_frame_change", "dyn_kurtosis", "rx_agreement", "E", "bw"]
LEVELS = ["log_power", "freq_selectivity"]                       # absolute levels: room-specific, risky for new rooms
SETS = {"all": ALL, "without_levels": [c for c in ALL if c not in LEVELS], "motion_only": ["act", "act_slow", "act_fast", "E", "bw"]}


def labelled_fused(t):
    """his fused score on the labelled windows of the table (cached)"""
    path = data.CACHE / "labelled_fused.pkl"
    if path.exists():
        return pd.read_pickle(path)
    m = onnx.load(str(MODEL)); names = taps(m)
    sess = ort.InferenceSession(with_taps(m).SerializeToString(), providers=["CPUExecutionProvider"])
    rows = []
    for tr, g in t[t.set == "L"].groupby("trace", sort=False):
        csi, ati = data.load_trace(tr, "labeled")
        for s in g.start:
            a = ati[s:s + 600].copy(); a[-1] = 0
            f, i = data.to_model_input(csi[s:s + 600], a)
            lo, fused, _, _ = sess.run(names, {"frames": f, "intervals": i})
            rows.append(dict(trace=tr, start=s, fused_l=float(fused.ravel()[0])))
    out = pd.DataFrame(rows); out.to_pickle(path)
    return out


def make_model(kind):
    return Ridge(alpha=30.0) if kind == "ridge" else HistGradientBoostingRegressor(max_depth=3, max_iter=150, learning_rate=0.05, min_samples_leaf=40)


def loo_mean(g, col):
    """average of the OTHER windows of the same recording"""
    s = g.groupby("trace")[col].transform("sum"); n = g.groupby("trace")[col].transform("count")
    return (s - g[col]) / (n - 1)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    t = build_table()
    U = t[(t.set == "U") & (t.cls > 0)].copy()                    # occupied unlabelled windows (his gate)
    U["target"] = U.groupby("trace").fused.transform("mean")      # the recording's average count score
    U["loo"] = loo_mean(U, "fused")
    rooms = ["d1", "h1", "i1", "j1", "u1", "v1"]
    print(f"occupied unlabelled windows: {len(U)} in {U.trace.nunique()} recordings\n")

    print("=== leave one room out (no labels) ===")
    print("per held-out room: Spearman of ONE window's output with the average of the OTHER windows of its recording")
    rows = []
    for kind in ["ridge", "boost"]:
        for sname, feats in SETS.items():
            pred = pd.Series(np.nan, index=U.index)
            for r in rooms:
                tr, te = U[U.room != r], U[U.room == r]
                mu, sd = tr[feats].mean(), tr[feats].std() + 1e-9
                mdl = make_model(kind).fit(((tr[feats] - mu) / sd).clip(-6, 6), tr.target)
                pred[te.index] = mdl.predict(((te[feats] - mu) / sd).clip(-6, 6))
            U["pred"] = pred
            d = {"model": kind, "features": sname}
            d["raw_fused_vs_loo"] = np.mean([spearmanr(g.fused, g.loo)[0] for _, g in U[U.room.isin(rooms)].groupby("room")])
            d["student_vs_loo"] = np.mean([spearmanr(g.pred, g.loo)[0] for _, g in U[U.room.isin(rooms)].groupby("room")])
            d["steady_raw"] = np.mean([reliability(g, "fused") for _, g in U[U.room.isin(rooms)].groupby("room")])
            d["steady_student"] = np.mean([reliability(g, "pred") for _, g in U[U.room.isin(rooms)].groupby("room")])
            d["recording_order_kept"] = np.mean([spearmanr(g.groupby("trace").pred.mean(), g.groupby("trace").fused.mean())[0]
                                                 for _, g in U[U.room.isin(rooms)].groupby("room")])
            rows.append(d)
            U[f"pred_{kind}_{sname}"] = pred
    res = pd.DataFrame(rows)
    print(res.round(2).to_string(index=False))
    print("(raw_fused_vs_loo is what the current model has; student_vs_loo should be clearly higher; recording_order_kept should stay high)")

    print("\nper room, best setting by student_vs_loo:")
    best = res.sort_values("student_vs_loo", ascending=False).iloc[0]
    col = f"pred_{best.model}_{best.features}"
    print(f"best: {best.model} / {best.features}")
    print(pd.DataFrame({r: dict(raw=spearmanr(g.fused, g.loo)[0], student=spearmanr(g[col], g.loo)[0],
                                steady_raw=reliability(g, "fused"), steady_student=reliability(g.assign(p=g[col]), "p"))
                        for r, g in U[U.room.isin(rooms)].groupby("room")}).round(2).to_string())

    print("\n=== sanity with labels (information only): student trained on ALL unlabelled occupied windows ===")
    Lf = labelled_fused(t)
    L = t[t.set == "L"].merge(Lf, on=["trace", "start"])
    feats = SETS[best.features]
    mu, sd = U[feats].mean(), U[feats].std() + 1e-9
    mdl = make_model(best.model).fit(((U[feats] - mu) / sd).clip(-6, 6), U.target)
    L["student"] = mdl.predict(((L[feats] - mu) / sd).clip(-6, 6))
    rows = []
    for r in ["a1", "c1"]:
        g = L[(L.room == r) & (L.headcount > 0)]
        med = g.groupby("trace")[["fused_l", "student", "headcount"]].median()
        for c in ["fused_l", "student"]:
            rows.append(dict(room=r, score=c, pair_acc=pair_acc(g, c), recording_spearman=spearmanr(med[c], med.headcount)[0]))
    print(pd.DataFrame(rows).pivot(index="score", columns="room").round(2).to_string())
