"""Phase 1: compare candidate scores on the unlabelled (test-like) recordings, without labels.

    python src/phase1_checks.py

The rule used by every candidate (same as the team's 0.40 rule):
    act < 3.0 (still)  -> 0 people
    otherwise          -> rank by the candidate score, cut into 4 equal groups -> 1, 2, 3, 4
    (cut points = quartiles of the score over the ACTIVE unlabelled windows)

Label-free checks, on active unlabelled windows only (that is where the ranking matters):
    reliability   share of the variation that is BETWEEN recordings of the same room
                  (a recording's headcount is constant, so a good score is stable inside it)
    agree_rule    share of a recording's windows that get the recording's majority class
    corr_act      rank correlation with `act` across recordings (1.0 = same ranking, no new info)
And a sanity look at the labelled rooms a and c (known to be misleading, not a criterion):
    corr_hc_a/c   rank correlation of the score with the true headcount inside room a / c
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import data
from activity import STILL_THRESHOLD
from physics import FEATURE_NAMES

VARIANTS = ["act", "act_slow", "act_fast", "act_breath", "act_rx0", "act_rx1", "act_rx2", "act_rx_med", "spread"]


def rule_classes(score, active, cuts):
    """0 if still, else 1..4 by which quartile of the active scores it falls in."""
    return np.where(active, 1 + np.searchsorted(cuts, score), 0)


def reliability(df, col):
    between, within, n = 0.0, 0.0, 0
    for _, g in df.groupby("layout"):
        means = g.groupby("trace")[col].mean()
        if len(means) < 3:
            continue
        between += means.var() * len(means)
        within += g.groupby("trace")[col].var().mean() * len(means)
        n += len(means)
    return between / (between + within)


def check(t, col):
    U = t[(t.set == "U") & (t.layout != "unknown")]
    A = U[U.act >= STILL_THRESHOLD]
    A = A[A.groupby("trace").trace.transform("size") >= 5]                    # recordings with >= 5 active windows
    cuts = np.quantile(A[col], [0.25, 0.5, 0.75])
    cls = rule_classes(A[col].values, np.ones(len(A), bool), cuts)
    agree = pd.Series(cls, index=A.index).groupby(A.trace).agg(lambda c: np.bincount(c).max() / len(c)).mean()
    rec = A.groupby("trace")[list(dict.fromkeys([col, "act"]))].mean()
    L = t[t.set == "L"]
    hc = {room: spearmanr(L[L.layout == room][col], L[L.layout == room].headcount).statistic for room in ["a", "c"]}
    return {"score": col, "reliability": reliability(A, col), "agree_rule": agree,
            "corr_act": spearmanr(rec[col], rec["act"]).statistic, "corr_hc_a": hc["a"], "corr_hc_c": hc["c"]}


if __name__ == "__main__":
    t = pd.read_csv(data.CACHE / "score_table.csv")
    U = t[t.set == "U"]
    still = U.act < STILL_THRESHOLD
    rec_still = U.groupby("trace").act.apply(lambda a: (a < STILL_THRESHOLD).mean())
    print(f"unlabelled windows: {len(U)}, still {still.mean():.0%} | recordings mostly still (>50% of windows): "
          f"{(rec_still > 0.5).sum()} of {len(rec_still)}")

    res = pd.DataFrame([check(t, c) for c in VARIANTS + FEATURE_NAMES])
    pd.set_option("display.width", 200)
    print("\nCandidates (active unlabelled windows). Want: high reliability, high agree_rule, corr_act < 1 = new info")
    print(res.sort_values("reliability", ascending=False).round(2).to_string(index=False))
    res.to_csv(data.CACHE / "phase1_checks.csv", index=False)
