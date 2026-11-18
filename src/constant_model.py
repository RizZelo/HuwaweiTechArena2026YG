"""A model that always gives the same answer (a baseline / probe).

    python src/constant_model.py 1 --output submissions/const_1.onnx

Its leaderboard score is exactly the share of test samples with that headcount.
The inputs are still read (multiplied by 0) so the ONNX file keeps both input names.
"""

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

import data


class Constant(nn.Module):
    def __init__(self, k):
        super().__init__()
        logits = torch.zeros(5)
        logits[k] = 10.0
        self.register_buffer("scores", logits)

    def forward(self, frames, intervals):
        return self.scores + 0.0 * (frames.mean() + intervals.mean())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("headcount", type=int, choices=range(5))
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    csi, ati = data.load_trace(data.load_index("labeled").identifier.iloc[0])
    frames, intervals = data.to_model_input(csi[:600], ati[:600])
    torch.onnx.export(Constant(args.headcount).eval(), (torch.from_numpy(frames), torch.from_numpy(intervals)),
                      str(args.output), input_names=["frames", "intervals"], output_names=["logits"],
                      opset_version=18, external_data=False)
    sess = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    out = sess.run(["logits"], {"frames": frames, "intervals": intervals})[0]
    assert out.shape == (5,) and out.dtype == np.float32
    print("inputs:", [i.name for i in sess.get_inputs()], "| logits:", out.tolist(), "| prediction:", int(out.argmax()))
    print(f"saved {args.output} ({args.output.stat().st_size:,} bytes)")
