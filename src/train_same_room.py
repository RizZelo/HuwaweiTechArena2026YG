"""Step 4: "same-room contrastive" pre-training of the image CNN, then fine-tuning.

Idea: two windows from the SAME recording must look alike (same headcount, same room);
they must look different from windows of OTHER recordings OF THE SAME ROOM. Recognising
the room never helps, because every comparison is made inside one room. To win, the
model has to notice what changes between recordings of a room: the people.

    python src/train_same_room.py pretrain           # -> cache/image_samroom_encoder.pt
    python src/train_same_room.py evaluate           # embedding check vs old SimCLR / random
    python src/train_same_room.py finetune           # 5 seeds -> cache/image_ssl_seed<k>.pt

Needs cache/windows_{L,U}_image.npz (python src/prepare.py --rep image --unlabeled).
"""

import argparse

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

import data
from models import DEVICE, ImageCNN, augment_image, make_net, prep

ENCODER = data.CACHE / "image_samroom_encoder.pt"


def load_all():
    """All image windows (labelled + unlabelled) with their recording id and room key."""
    zl = np.load(data.CACHE / "windows_L_image.npz")
    zu = np.load(data.CACHE / "windows_U_image.npz")
    W = torch.from_numpy(np.concatenate([zl["W"], zu["W"]])).to(DEVICE)
    traces = np.concatenate([zl["traces"], zu["traces"]])
    layouts = np.concatenate([zl["layouts"], zu["layouts"]])
    sets = np.array(["L"] * len(zl["W"]) + ["U"] * len(zu["W"]))
    # "unknown" recordings may come from different rooms -> each one is its own room
    rooms = np.where(layouts == "unknown", traces, np.char.add(np.char.add(sets, ":"), layouts))
    y = np.concatenate([zl["y"], zu["y"]])
    return W, traces, rooms, y, sets, layouts


# ---------------------------------------------------------------- pre-training

class RoomSampler:
    """Batch = a few rooms x several recordings per room x 2 windows per recording."""

    def __init__(self, traces, rooms, rooms_per_batch=4, traces_per_room=8):
        self.windows = {t: np.where(traces == t)[0] for t in np.unique(traces)}
        room_traces = {}
        for t in self.windows:
            room_traces.setdefault(rooms[self.windows[t][0]], []).append(t)
        self.room_traces = {r: ts for r, ts in room_traces.items() if len(ts) >= 2}   # need negatives
        names = list(self.room_traces)
        w = np.sqrt([len(self.room_traces[r]) for r in names])                         # big rooms less dominant
        self.names, self.p = names, w / w.sum()
        self.R, self.K = rooms_per_batch, traces_per_room

    def batch(self):
        idx_a, idx_b, room_of = [], [], []
        for ri, r in enumerate(np.random.choice(self.names, self.R, replace=False, p=self.p)):
            ts = self.room_traces[r]
            for t in np.random.choice(ts, min(self.K, len(ts)), replace=False):
                a, b = np.random.choice(self.windows[t], 2, replace=False)
                idx_a.append(a); idx_b.append(b); room_of.append(ri)
        return np.array(idx_a), np.array(idx_b), np.array(room_of)


def same_room_loss(za, zb, room_of, tau=0.1):
    """Each window must pick its partner among all windows of the SAME room (not other rooms)."""
    z = F.normalize(torch.cat([za, zb]), dim=1)
    n = za.shape[0]
    room = torch.from_numpy(np.concatenate([room_of, room_of])).to(z.device)
    sim = z @ z.T / tau
    allowed = (room[:, None] == room[None, :]) & ~torch.eye(2 * n, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(~allowed, -1e9)
    target = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, target)


def pretrain(steps=3000, lr=1e-3, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    W, traces, rooms, *_ = load_all()
    sampler = RoomSampler(traces, rooms)
    print(f"device {DEVICE} | {len(W)} windows | {len(sampler.names)} rooms with >= 2 recordings")
    net = ImageCNN().to(DEVICE)
    proj = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 32)).to(DEVICE)
    opt = torch.optim.AdamW(list(net.parameters()) + list(proj.parameters()), lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    tot = 0
    for step in range(steps):
        net.train()
        a, b, room_of = sampler.batch()
        za = proj(net.embed(augment_image(W[a].float())))
        zb = proj(net.embed(augment_image(W[b].float())))
        loss = same_room_loss(za, zb, room_of)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        tot += loss.item()
        if (step + 1) % 250 == 0:
            print(f"  step {step + 1}: loss {tot / 250:.3f}", flush=True); tot = 0
    torch.save(net.state_dict(), ENCODER)
    print("saved", ENCODER)


# ---------------------------------------------------------------- embedding check

def embed_image(net, W, bs=128):
    net.eval()
    with torch.no_grad():
        return torch.cat([net.embed(W[i:i + bs].float()) for i in range(0, len(W), bs)]).cpu().numpy()


def embedding_report(name, E, y, layouts):
    """Room should be hard to guess, headcount of an unseen room easy (labelled windows only)."""
    knn = KNeighborsClassifier(5)
    room_acc = cross_val_score(knn, E, layouts, cv=3).mean()
    hc_knn = cross_val_score(knn, E, y, cv=3).mean()
    held = []
    for h in ["a", "c", "g"]:
        te = layouts == h
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(E[~te], y[~te])
        held.append(clf.score(E[te], y[te]))
    print(f"  {name:28s} room {room_acc:.2f} (lower=better) | headcount kNN {hc_knn:.2f} | "
          f"unseen room a/c/g {' '.join(f'{x:.2f}' for x in held)} (mean {np.mean(held):.2f})")


def evaluate():
    W, traces, rooms, y, sets, layouts = load_all()
    L = sets == "L"
    yl, ll = y[L], layouts[L]
    print("embedding check on labelled windows:")
    torch.manual_seed(0)
    embedding_report("random image CNN", embed_image(ImageCNN().to(DEVICE), W[L]), yl, ll)
    net = ImageCNN().to(DEVICE); net.load_state_dict(torch.load(ENCODER))
    embedding_report("same-room contrastive (new)", embed_image(net, W[L]), yl, ll)
    # the old SimCLR encoder behind the 0.39 model works on amplitude groups
    old = make_net().to(DEVICE); old.load_state_dict(torch.load(data.CACHE / "cnn_pretrained.pt"))
    zr = np.load(data.CACHE / "windows_L_raw.npz")
    with torch.no_grad():
        Wr = torch.from_numpy(zr["W"]).to(DEVICE)
        Eo = torch.cat([old[:8](prep(Wr[i:i + 128].float())) for i in range(0, len(Wr), 128)]).cpu().numpy()
    embedding_report("old SimCLR (0.39 model)", Eo, zr["y"], zr["layouts"])


# ---------------------------------------------------------------- fine-tuning

def finetune(Wl, yl, seed, epochs=30, bs=32):
    torch.manual_seed(seed); np.random.seed(seed)
    net = ImageCNN().to(DEVICE)
    net.load_state_dict(torch.load(ENCODER))
    net.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(64, 5)).to(DEVICE)       # fresh classifier
    opt = torch.optim.AdamW([{"params": net.features.parameters(), "lr": 3e-4},
                             {"params": net.head.parameters(), "lr": 1e-3}], weight_decay=1e-2)
    for ep in range(epochs):
        net.train()
        for idx in torch.randperm(len(Wl), device=DEVICE).split(bs):
            if len(idx) < 2:
                continue
            loss = F.cross_entropy(net(augment_image(Wl[idx].float())), yl[idx])   # no class weights
            opt.zero_grad(); loss.backward(); opt.step()
    return net.eval()


def probs(net, W, bs=128):
    with torch.no_grad():
        return torch.cat([F.softmax(net(W[i:i + bs].float()), dim=1) for i in range(0, len(W), bs)])


def run_finetune(seeds=5):
    W, traces, rooms, y, sets, layouts = load_all()
    L = torch.from_numpy(sets == "L").to(DEVICE)
    Wl, yl = W[L], torch.from_numpy(y[sets == "L"]).to(DEVICE)
    ll = layouts[sets == "L"]
    print("local check (hold out one room, ensemble of seeds):")
    for h in ["a", "c", "g"]:
        te = torch.from_numpy(ll == h).to(DEVICE)
        P = torch.stack([probs(finetune(Wl[~te], yl[~te], s), Wl[te]) for s in range(seeds)]).mean(0)
        pred = P.argmax(1)
        print(f"  held {h}: acc {(pred == yl[te]).float().mean().item():.2f}  predicted "
              f"{torch.bincount(pred, minlength=5).tolist()}  true {torch.bincount(yl[te], minlength=5).tolist()}",
              flush=True)

    nets = [finetune(Wl, yl, s) for s in range(seeds)]
    for s, n in enumerate(nets):
        torch.save(n.state_dict(), data.CACHE / f"image_ssl_seed{s}.pt")
    print(f"saved cache/image_ssl_seed0..{seeds - 1}.pt")

    # does it still give one answer per room on the test-like unlabelled rooms?
    U = sets == "U"
    P = torch.stack([probs(n, W[torch.from_numpy(U).to(DEVICE)]) for n in nets]).mean(0).argmax(1).cpu().numpy()
    print("unlabelled windows predicted 0..4:", (np.bincount(P, minlength=5) / len(P)).round(2).tolist())
    lu = layouts[U]
    print("most common answer per unlabelled room:",
          {r: f"{np.bincount(P[lu == r], minlength=5).argmax()} ({np.bincount(P[lu == r], minlength=5).max() / (lu == r).sum():.0%})"
           for r in sorted(set(lu))})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["pretrain", "evaluate", "finetune"])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    if args.stage == "pretrain":
        pretrain(args.steps)
    elif args.stage == "evaluate":
        evaluate()
    else:
        run_finetune(args.seeds)
