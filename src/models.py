"""PyTorch pieces: preprocessing that goes inside the ONNX file, the CNN, and SSL helpers.

The preprocessing here is the torch twin of the NumPy code in dsp.py. The same functions
are used to build the training caches (prepare.py) and inside the exported model
(export.py), so training and test see exactly the same input.

Shapes:
    x       (T, 3, 1024, 2)  float  one window, real/imag in the last axis
    groups  (T, 3, 64)       float  amplitude averaged over 16 neighbouring subcarriers
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# Train on the NVIDIA GPU when PyTorch can see one, otherwise on the CPU.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------- preprocessing (in ONNX)

def _shifted(x, k):
    """k copies of x shifted in time with repeated edges (same as dsp._shifted)."""
    h = k // 2
    p = torch.cat([x[:1]] * h + [x] + [x[-1:]] * h, dim=0)
    T = x.shape[0]
    return [p[i:i + T] for i in range(k)]


def _median3(a, b, c):
    return torch.max(torch.min(a, b), torch.min(torch.max(a, b), c))


def despike(x):
    """Median of 5 frames over time (= dsp.despike), written with min/max only.

    torch.median / sort is ~13x slower in ONNX Runtime; this gives identical numbers.
    """
    a, b, c, d, e = _shifted(x, 5)
    f = torch.max(torch.min(a, b), torch.min(c, d))
    g = torch.min(torch.max(a, b), torch.max(c, d))
    return _median3(e, f, g)


def smooth(x, k=5):
    """Moving average over k frames (= dsp.smooth)."""
    return torch.stack(_shifted(x, k), dim=0).mean(dim=0)


def amp_groups(x):
    """Amplitude, averaged over groups of 16 subcarriers -> (T, 3, 64)."""
    amp = torch.sqrt(x[..., 0] ** 2 + x[..., 1] ** 2)
    return amp.reshape(x.shape[0], 3, 64, 16).mean(dim=3)


def window_to_groups(x, clean):
    """Full preprocessing of one window up to the cached representation."""
    if clean:
        x = smooth(despike(x))
    return amp_groups(x)


def prep(g):
    """Cached groups (N, T, 3, 64) -> CNN input (N, 192, T): remove gain and static part."""
    N, T = g.shape[0], g.shape[1]
    g = g / (g.mean(dim=(1, 2, 3), keepdim=True) + 1e-6)     # remove overall power (room/distance)
    g = g - g.mean(dim=1, keepdim=True)                       # remove the frozen picture
    return g.reshape(N, T, 192).permute(0, 2, 1)


# ---------------------------------------------------------------- the CNN (same as the 0.39 model)

def make_net():
    """3 conv layers over time; the 192 (Rx, subcarrier group) series are input channels."""
    return nn.Sequential(
        nn.Conv1d(192, 32, 9, stride=2, padding=4), nn.ReLU(),
        nn.Conv1d(32, 32, 9, stride=2, padding=4), nn.ReLU(),
        nn.Conv1d(32, 32, 9, stride=2, padding=4), nn.ReLU(),
        nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(32, 5))


class Submission(nn.Module):
    """Wraps a trained net with the preprocessing: platform tensors in, 5 logits out."""

    def __init__(self, net, clean):
        super().__init__()
        self.net = net
        self.clean = clean

    def forward(self, frames, intervals):
        g = window_to_groups(frames[:, 0], self.clean).unsqueeze(0)   # (1, 600, 3, 64)
        return self.net(prep(g))[0]


# ---------------------------------------------------------------- self-supervised (SimCLR) helpers

def augment(g):
    """Random changes that should not change the headcount. g: (B, 600, 3, 64)."""
    B = g.shape[0]
    L = np.random.randint(300, 601); s = np.random.randint(0, 601 - L)
    g = g[:, s:s + L]                                          # random time crop
    g = g * torch.exp(0.2 * torch.randn(B, 1, 3, 1, device=g.device))   # random gain per antenna
    g = g * (torch.rand(B, 1, 1, 64, device=g.device) > 0.15).float()   # drop some subcarrier groups
    return g * (1 + 0.05 * torch.randn_like(g))                # small multiplicative noise


def nt_xent(a, b, tau=0.2):
    """Contrastive loss: two augmented views of the same window should match each other."""
    z = F.normalize(torch.cat([a, b]), dim=1)
    sim = z @ z.T / tau
    sim.fill_diagonal_(-1e9)
    n = a.shape[0]
    target = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(a.device)
    return F.cross_entropy(sim, target)


# ================================================================ step 3: the "image" CNN
#
# Input image per window: 7 channels x 150 time steps x 126 subcarrier groups
#   channels 0-2 : amplitude change of Rx0, Rx1, Rx2          (how strong the signal is)
#   channels 3-6 : phase-difference change Rx0-Rx1, Rx0-Rx2   (cos and sin parts)
# Every channel keeps only what changes over time (frozen picture removed) and is scaled
# to unit size, so the absolute power / noise level of a room cannot be used.

N_GROUPS = 126           # subcarriers 8..1015 (guard band dropped), 8 per group
T_DOWN = 4               # average 4 frames -> 150 time steps (motion < 25 Hz kept)


def window_to_image(x):
    """Cleaned window (600, 3, 1024, 2) -> image (7, 150, 126)."""
    T = x.shape[0] // T_DOWN
    x = x[:, :, 8:1016]                                                     # (600, 3, 1008, 2)
    x = x.reshape(T, T_DOWN, 3, N_GROUPS, 8, 2).mean(dim=(1, 4))            # complex average -> (150, 3, 126, 2)
    re, im = x[..., 0], x[..., 1]
    amp = torch.sqrt(re ** 2 + im ** 2) + 1e-6                              # (150, 3, 126)

    a = amp / amp.mean()
    a = a - a.mean(dim=0, keepdim=True)                                     # amplitude change

    # unit-length phase difference between Rx0 and Rx1 / Rx2: z0 * conj(zk) / (|z0| |zk|)
    pr = (re[:, :1] * re[:, 1:] + im[:, :1] * im[:, 1:]) / (amp[:, :1] * amp[:, 1:])   # cos (150, 2, 126)
    pi = (im[:, :1] * re[:, 1:] - re[:, :1] * im[:, 1:]) / (amp[:, :1] * amp[:, 1:])   # sin
    p = torch.cat([pr, pi], dim=1)
    p = p - p.mean(dim=0, keepdim=True)                                     # phase-difference change

    img = torch.cat([a, p], dim=1).permute(1, 0, 2)                         # (7, 150, 126)
    std = img.std(dim=(1, 2), keepdim=True) + 1e-6
    return img / std


class ImageCNN(nn.Module):
    """Small 2D CNN over (time, subcarrier) with weights shared across subcarriers.

    The same small filters slide over every subcarrier; at the end we average / max over the
    whole subcarrier axis, so the model cannot learn "subcarrier 37 matters" (a room detail).
    """

    def __init__(self, ch=7, width=32, n_classes=5):
        super().__init__()

        def block(cin, cout, k):
            return nn.Sequential(nn.Conv2d(cin, cout, k, padding=(k[0] // 2, k[1] // 2)),
                                 nn.BatchNorm2d(cout), nn.ReLU())
        self.features = nn.Sequential(
            block(ch, 16, (7, 3)), nn.MaxPool2d((2, 2)),            # -> 75 x 63
            block(16, width, (7, 3)), nn.MaxPool2d((2, 2)),         # -> 37 x 31
            block(width, width, (5, 3)), nn.MaxPool2d((2, 1)),      # -> 18 x 31
            block(width, width, (5, 3)))
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(2 * width, n_classes))

    def embed(self, img):
        """img (N, 7, T, 126) -> summary (N, 2 * width): mean and max over subcarriers and time."""
        f = self.features(img)
        return torch.cat([f.mean(dim=(2, 3)), f.amax(dim=(2, 3))], dim=1)

    def forward(self, img):
        return self.head(self.embed(img))


def augment_image(img):
    """Changes that should not change the headcount. img: (B, 7, T, 126)."""
    B, C, T, S = img.shape
    dev = img.device
    L = np.random.randint(T // 2, T + 1); s = np.random.randint(0, T - L + 1)
    img = img[:, :, s:s + L]                                                       # random time crop
    flip = torch.rand(B, 1, 1, 1, device=dev) < 0.5
    img = torch.where(flip, img.flip(3), img)                                      # mirror subcarrier axis
    keep = torch.ones(B, 1, 1, S, device=dev)
    band = torch.randint(0, 4, (B,), device=dev)                                   # drop one 20 MHz quarter
    drop = torch.rand(B, device=dev) < 0.3
    sc = torch.arange(S, device=dev)
    q = (sc[None] // (S // 4 + 1)) == band[:, None]
    keep[(drop[:, None] & q)[:, None, None, :]] = 0
    img = img * keep
    img = img * torch.exp(0.2 * torch.randn(B, C, 1, 1, device=dev))              # random channel gain
    return img + 0.05 * torch.randn_like(img)                                      # small noise


class ImageSubmission(nn.Module):
    """Platform tensors in, 5 logits out: clean -> image -> average of several trained CNNs."""

    def __init__(self, nets):
        super().__init__()
        self.nets = nn.ModuleList(nets)

    def forward(self, frames, intervals):
        img = window_to_image(smooth(despike(frames[:, 0]))).unsqueeze(0)
        probs = torch.stack([F.softmax(n(img), dim=1) for n in self.nets]).mean(dim=0)
        return torch.log(probs + 1e-9)[0]
