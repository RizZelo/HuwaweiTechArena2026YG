"""Export a trained CNN (+ its preprocessing) to ONNX and check it like the platform would.

    python src/export.py --weights cache/cnn_ssl_clean.pt --clean --output submissions/v3_ssl_cnn_clean.onnx
    python src/export.py --arch image --weights cache/image_cnn_seed*.pt --output submissions/v4_image_cnn.onnx

--arch image accepts several weight files and averages their predictions inside the ONNX file.
--prior p0 p1 p2 p3 p4 shifts the scores from the training class balance to this test balance
(public test, measured by the team with constant submissions: 0.20 0.35 0.36 0.045 0.045).

Checks: output shape/dtype, file size < 100 MB, ONNX == PyTorch on real windows from
several rooms, and average CPU time per sample (limit 0.7 s).
"""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import onnxruntime as ort
import torch

import data
from torch import nn

from models import ImageCNN, ImageSubmission, Submission, make_net


class PriorShift(nn.Module):
    """Prior correction: log p_test(c) = log p_model(c) + log(test_prior(c) / train_prior(c)) + const."""

    def __init__(self, model, train_prior, test_prior):
        super().__init__()
        self.model = model
        self.register_buffer("bias", torch.log(torch.tensor(test_prior) / torch.tensor(train_prior)).float())

    def forward(self, frames, intervals):
        return torch.log_softmax(self.model(frames, intervals), dim=0) + self.bias


def train_prior():
    """Class balance of the labelled training windows (same for every model)."""
    y = np.load(data.CACHE / "windows_L_image.npz")["y"]
    return (np.bincount(y, minlength=5) / len(y)).tolist()


def real_samples(n_per_room=2):
    """A few real windows (platform format) from different labelled rooms."""
    idx = data.load_index("labeled").groupby("layout").head(1)
    out = []
    for r in idx.itertuples():
        csi, ati = data.load_trace(r.identifier)
        for s, w, a in list(data.windows(csi, ati, step=4800))[:n_per_room]:
            out.append(data.to_model_input(w, a))
    return out


def export_and_check(model, output):
    """Write the ONNX file and check it like the platform would (shape, size, ONNX == torch, time)."""
    samples = real_samples()
    frames, intervals = samples[0]
    torch.onnx.export(model, (torch.from_numpy(frames), torch.from_numpy(intervals)), str(output),
                      input_names=["frames", "intervals"], output_names=["logits"],
                      opset_version=18, external_data=False)

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(output), sess_options=opts, providers=["CPUExecutionProvider"])

    max_diff, max_pdiff, same, times, preds = 0.0, 0.0, 0, [], []
    for frames, intervals in samples:
        t = perf_counter()
        logits = sess.run(["logits"], {"frames": frames, "intervals": intervals})[0]
        times.append(perf_counter() - t)
        assert logits.shape == (5,) and logits.dtype == np.float32, "output must be float32 (5,)"
        with torch.no_grad():
            ref = model(torch.from_numpy(frames), torch.from_numpy(intervals)).numpy()
        max_diff = max(max_diff, np.abs(logits - ref).max())
        soft = lambda v: np.exp(v - v.max()) / np.exp(v - v.max()).sum()
        max_pdiff = max(max_pdiff, np.abs(soft(logits) - soft(ref)).max())
        same += int(logits.argmax() == ref.argmax())
        preds.append(int(logits.argmax()))

    size = output.stat().st_size
    print(f"saved {output} ({size:,} bytes, limit 100,000,000)")
    print(f"ONNX vs PyTorch: same prediction {same}/{len(samples)}, max probability difference {max_pdiff:.2e} "
          f"(max raw score difference {max_diff:.2e}; large values only matter if probabilities differ)")
    print(f"CPU time per sample: mean {np.mean(times[1:]):.3f} s, max {np.max(times[1:]):.3f} s (limit 0.7 s)")
    print("predictions on test windows:", preds)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="ssl_cnn", choices=["ssl_cnn", "image"])
    ap.add_argument("--weights", type=Path, nargs="+", required=True)
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--prior", type=float, nargs=5, help="test class balance for prior correction")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    def load(net, path):
        net.load_state_dict(torch.load(path, map_location="cpu"))
        return net.eval()

    if args.arch == "image":
        model = ImageSubmission([load(ImageCNN(), w) for w in args.weights]).eval()
    else:
        model = Submission(load(make_net(), args.weights[0]), clean=args.clean).eval()
    print("models in file:", len(args.weights))
    if args.prior:
        model = PriorShift(model, train_prior(), args.prior).eval()
        print("prior correction, bias:", [round(b, 2) for b in model.bias.tolist()])

    export_and_check(model, args.output)
