"""The 0.39 recipe: SimCLR pre-training on unlabelled + labelled windows, then fine-tuning.

    python src/train_ssl_cnn.py --tag clean          # uses cache/windows_*_clean.npz
    python src/train_ssl_cnn.py --tag raw            # same recipe on uncleaned windows

1. Local check: for held-out rooms a, c, g, fine-tune on the other rooms and report accuracy
   (weak signal: only a sanity check, the leaderboard decides).
2. Final model: fine-tune on all labelled windows -> cache/cnn_ssl_<tag>.pt
"""

import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import data
from models import DEVICE, augment, make_net, nt_xent, prep


def load(which, tag):
    z = np.load(data.CACHE / f"windows_{which}_{tag}.npz")
    return torch.from_numpy(z["W"]).to(DEVICE), torch.from_numpy(z["y"]).to(DEVICE), z["layouts"]


def pretrain(net, Wpre, epochs=20, bs=128):
    encoder = net[:8]                              # everything up to the 32-number summary
    proj = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 32)).to(DEVICE)
    opt = torch.optim.AdamW(list(encoder.parameters()) + list(proj.parameters()), 1e-3, weight_decay=1e-2)
    for ep in range(epochs):
        encoder.train(); tot = 0
        for idx in torch.randperm(len(Wpre)).split(bs):
            if len(idx) < 8:
                continue
            g = Wpre[idx].float()
            loss = nt_xent(proj(encoder(prep(augment(g)))), proj(encoder(prep(augment(g)))))
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        print(f"  pretrain epoch {ep}: loss {tot / max(1, len(Wpre) // bs):.3f}", flush=True)


def finetune(init_state, W, y, epochs=30, lr=3e-4, bs=32):
    net = make_net().to(DEVICE)
    net.load_state_dict(init_state)
    opt = torch.optim.AdamW(net.parameters(), lr, weight_decay=1e-2)
    for ep in range(epochs):
        net.train()
        for idx in torch.randperm(len(W)).split(bs):
            loss = F.cross_entropy(net(prep(W[idx].float())), y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    return net.eval()


def predict(net, W, bs=256):
    with torch.no_grad():
        return torch.cat([net(prep(W[i:i + bs].float())).argmax(1) for i in range(0, len(W), bs)])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="clean", choices=["clean", "raw"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    Wl, yl, lay = load("L", args.tag)
    Wu, _, _ = load("U", args.tag)

    print("device:", DEVICE)
    print("pre-training on", len(Wu) + len(Wl), "windows")
    net = make_net().to(DEVICE)
    pretrain(net, torch.cat([Wu, Wl]))
    pre_state = {k: v.clone() for k, v in net.state_dict().items()}
    torch.save(pre_state, data.CACHE / f"cnn_pretrained_{args.tag}.pt")

    print("local check (hold out one room):")
    for held in ["a", "c", "g"]:
        te = torch.from_numpy(lay == held); tr = ~te
        m = finetune(pre_state, Wl[tr], yl[tr])
        pred = predict(m, Wl[te])
        acc = (pred == yl[te]).float().mean().item()
        print(f"  held {held}: acc {acc:.2f}  predicted counts {torch.bincount(pred, minlength=5).tolist()}"
              f"  true counts {torch.bincount(yl[te], minlength=5).tolist()}", flush=True)

    final = finetune(pre_state, Wl, yl)
    out = data.CACHE / f"cnn_ssl_{args.tag}.pt"
    torch.save(final.state_dict(), out)
    print("saved", out)
