"""Step 3: the image CNN, trained on labelled windows only (no pre-training yet).

    python src/train_image_cnn.py                 # 5 seeds
    python src/train_image_cnn.py --seeds 3 --epochs 30

1. Local check: hold out room a, c, g in turn; train every seed on the other rooms; report each
   seed's accuracy and the accuracy of their average (the ensemble). Only a sanity check.
2. Final: train every seed on all labelled windows -> cache/image_cnn_seed<k>.pt
   (export.py --arch image averages them inside one ONNX file).
"""

import argparse

import numpy as np
import torch
import torch.nn.functional as F

import data
from models import DEVICE, ImageCNN, augment_image


def load():
    z = np.load(data.CACHE / "windows_L_image.npz")
    return torch.from_numpy(z["W"]).to(DEVICE), torch.from_numpy(z["y"]).to(DEVICE), z["layouts"]


def train(W, y, seed, epochs, bs=32, lr=1e-3):
    torch.manual_seed(seed); np.random.seed(seed)
    net = ImageCNN().to(DEVICE)
    # every class counts equally, even though 0 and 1 people have far more windows
    counts = torch.bincount(y, minlength=5).float()
    weight = torch.where(counts > 0, counts.sum() / (5 * counts.clamp(min=1)), torch.zeros_like(counts))
    opt = torch.optim.AdamW(net.parameters(), lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for ep in range(epochs):
        net.train()
        for idx in torch.randperm(len(W), device=DEVICE).split(bs):
            if len(idx) < 2:
                continue
            loss = F.cross_entropy(net(augment_image(W[idx].float())), y[idx], weight=weight)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    return net.eval()


def probs(net, W, bs=128):
    with torch.no_grad():
        return torch.cat([F.softmax(net(W[i:i + bs].float()), dim=1) for i in range(0, len(W), bs)])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--skip-check", action="store_true")
    args = ap.parse_args()
    W, y, lay = load()
    print("device:", DEVICE, "| windows:", len(W), "| class counts:", torch.bincount(y, minlength=5).tolist())

    if not args.skip_check:
        print("local check (hold out one room):")
        for held in ["a", "c", "g"]:
            te = torch.from_numpy(lay == held).to(DEVICE); tr = ~te
            P = [probs(train(W[tr], y[tr], s, args.epochs), W[te]) for s in range(args.seeds)]
            accs = [(p.argmax(1) == y[te]).float().mean().item() for p in P]
            ens = torch.stack(P).mean(0).argmax(1)
            print(f"  held {held}: seeds {' '.join(f'{a:.2f}' for a in accs)} | ensemble "
                  f"{(ens == y[te]).float().mean().item():.2f}  predicted {torch.bincount(ens, minlength=5).tolist()}"
                  f"  true {torch.bincount(y[te], minlength=5).tolist()}", flush=True)

    for s in range(args.seeds):
        net = train(W, y, s, args.epochs)
        torch.save(net.state_dict(), data.CACHE / f"image_cnn_seed{s}.pt")
    print(f"saved cache/image_cnn_seed0..{args.seeds - 1}.pt")
