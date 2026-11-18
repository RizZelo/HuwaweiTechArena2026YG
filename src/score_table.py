"""Phase 1 table: every 600-frame window of every recording, with all scores side by side.

    python src/score_table.py        # -> cache/score_table.csv  (about 10 minutes)

Columns: set (L/U), layout, trace, start, headcount (-1 if unlabelled),
         activity variants (activity.score_variants) and the 15 physics features (physics.py).
All scores are computed on exactly the same windows, so they can be compared directly.
"""

import numpy as np
import pandas as pd
import torch

import data
from activity import score_variants
from models import despike, smooth
from physics import FEATURE_NAMES, physics_features

STEP = 600          # 20 non-overlapping windows per 60 s recording

if __name__ == "__main__":
    rows = []
    for which in ["labeled", "unlabeled"]:
        idx = data.load_index(which)
        for n, r in enumerate(idx.itertuples()):
            csi, ati = data.load_trace(r.identifier, which)
            for s, w, a in data.windows(csi, ati, step=STEP):
                row = {"set": which[0].upper(), "layout": r.layout, "trace": r.identifier, "start": s,
                       "headcount": int(r.headcount) if which == "labeled" else -1}
                row.update(score_variants(w))
                frames, _ = data.to_model_input(w, a)
                with torch.no_grad():
                    f = physics_features(smooth(despike(torch.from_numpy(frames[:, 0])))).numpy()
                row.update(dict(zip(FEATURE_NAMES, f.tolist())))
                rows.append(row)
            if (n + 1) % 25 == 0 or n + 1 == len(idx):
                print(f"{which}: {n + 1}/{len(idx)} recordings", flush=True)
    out = data.CACHE / "score_table.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("saved", out, len(rows), "windows")
