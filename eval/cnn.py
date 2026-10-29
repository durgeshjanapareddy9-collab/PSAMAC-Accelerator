"""CNN inference with approximate multipliers: MNIST top-1 accuracy (M2.9).

Steps
  1. Train a small LeNet-style CNN (2 conv + 2 FC layers) on MNIST in float
     with PyTorch [LeCun et al. 1998]. Training is seeded; the trained weights
     are cached in results/cnn_lenet_mnist.pt so every later run uses the same
     network.
  2. Post-training quantization (as in Jacob et al. 2018, simplified):
       activations (input pixels, post-ReLU outputs): uint8 0..255, zero-point 0,
           scale = largest value seen on 2,000 training images / 255;
       weights: symmetric int8 per layer, w_q = round(w / s_w), s_w = max|w| / 127;
       biases: int32 at scale s_in * s_w.
  3. Integer inference in numpy, where EVERY multiplication goes through a
     design's unsigned 8x8 LUT -- exactly what the MAC array would compute:
       product = sign(w_q) * LUT[activation, |w_q|]      (activation on port a,
                                                           |weight| on port b)
       acc     = sum of products + bias                  (int32/int64, no overflow)
       next activation = clip(round(acc * s_in * s_w / s_out), 0, 255)
     The requantization scale is applied in floating point here (real hardware
     would use an integer multiply and shift); it is the same for all designs.
  4. Top-1 accuracy on all 10,000 MNIST test images for every design, compared
     with the exact int8 network (the N = 0 LUT: primary reference) and with
     the float network.

How a layer is computed quickly: a layer with K inputs per output is
    acc[n, o] = sum_k G[k, a[n, k], o],   G[k, v, o] = sign(W[o,k]) * LUT[v, |W[o,k]|]
which is a sparse "one-hot" matrix (one 1 per (n, k), at column k*256 + a[n,k])
times the table G reshaped to (K*256, outputs). All values are integers below
2^53, so doing it in float64 is exact. For N = 0 the result is checked against
a plain integer matrix multiply.

Usage:  python -m eval.cnn     (via `make cnn`; needs torch, see requirements-cnn.txt)
Writes results/cnn_accuracy.csv (and results/cnn_lenet_mnist.pt on the first run).
"""

import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets

from model import designs
from model.approx_mul import REPO, RESULTS

DATA = REPO / "build" / "data"
WEIGHTS = RESULTS / "cnn_lenet_mnist.pt"
SEED = 0


# ---- 1. Float network ---------------------------------------------------------------

class LeNet(nn.Module):
    """28x28 -> conv5x5(6, pad 2) -> pool -> conv5x5(16) -> pool -> fc 120 -> fc 10."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 10)

    def forward(self, x, record=None):
        """x: float images in [0, 1]. record: dict to store post-ReLU outputs."""
        x = F.relu(self.conv1(x))
        if record is not None: record["conv1"] = x
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        if record is not None: record["conv2"] = x
        x = F.max_pool2d(x, 2).flatten(1)
        x = F.relu(self.fc1(x))
        if record is not None: record["fc1"] = x
        return self.fc2(x)


def mnist():
    """(train images uint8 N x 28 x 28, train labels, test images, test labels)."""
    tr = datasets.MNIST(DATA, train=True, download=True)
    te = datasets.MNIST(DATA, train=False, download=True)
    return tr.data.numpy(), tr.targets.numpy(), te.data.numpy(), te.targets.numpy()


def to_float(images_u8):
    return torch.from_numpy(images_u8).float().div(255).unsqueeze(1)


def train(xtr, ytr, epochs=3):
    torch.manual_seed(SEED)
    model = LeNet()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x, y = to_float(xtr), torch.from_numpy(ytr).long()
    g = torch.Generator().manual_seed(SEED)
    for ep in range(epochs):
        perm = torch.randperm(len(x), generator=g)
        for i in range(0, len(x), 128):
            idx = perm[i:i + 128]
            opt.zero_grad()
            loss = F.cross_entropy(model(x[idx]), y[idx])
            loss.backward()
            opt.step()
        print(f"  epoch {ep + 1}: last batch loss {loss.item():.4f}")
    return model


def float_accuracy(model, x, y):
    with torch.no_grad():
        return float((model(to_float(x)).argmax(1).numpy() == y).mean())


# ---- 2. Quantization ------------------------------------------------------------------

def quantize(model, xcal):
    """Integer parameters and scales for each layer."""
    rec = {}
    with torch.no_grad():
        model(to_float(xcal), record=rec)
    act_scale = {"input": 1 / 255, **{k: float(v.max()) / 255 for k, v in rec.items()}}
    layers, s_in = [], act_scale["input"]
    for name, mod, s_out in [("conv1", model.conv1, act_scale["conv1"]),
                             ("conv2", model.conv2, act_scale["conv2"]),
                             ("fc1", model.fc1, act_scale["fc1"]),
                             ("fc2", model.fc2, None)]:
        w = mod.weight.detach().numpy().reshape(mod.weight.shape[0], -1)   # outputs x K
        s_w = float(np.abs(w).max()) / 127
        wq = np.clip(np.round(w / s_w), -127, 127).astype(np.int64)
        bq = np.round(mod.bias.detach().numpy() / (s_in * s_w)).astype(np.int64)
        # Requantize to the next layer's uint8 scale (None = final logits, kept as int).
        mult = None if s_out is None else s_in * s_w / s_out
        layers.append({"name": name, "wq": wq, "bq": bq, "mult": mult})
        s_in = s_out
    return layers


# ---- 3. Integer inference through a LUT -------------------------------------------------

def im2col(x, k, pad):
    """x: n x C x H x W uint8 -> (n*Ho*Wo) x (C*k*k), same order as torch weights."""
    if pad:
        x = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    n, c, h, w = x.shape
    ho, wo = h - k + 1, w - k + 1
    cols = np.lib.stride_tricks.sliding_window_view(x, (k, k), axis=(2, 3))   # n,c,ho,wo,k,k
    return cols.transpose(0, 2, 3, 1, 4, 5).reshape(n * ho * wo, c * k * k), (n, ho, wo)


def lut_matmul(acts, wq, table):
    """acc[n, o] = sum_k sign(wq[o,k]) * table[acts[n,k], |wq[o,k]|]  (exact integers)."""
    n, K = acts.shape
    # G[k, v, o] = sign(w) * LUT[v, |w|]  -> (K*256) x outputs
    G = (np.sign(wq).T[:, None, :] * table[:, np.abs(wq).T].transpose(1, 0, 2))
    G = G.reshape(K * 256, -1).astype(np.float64)
    cols = (np.arange(K) * 256 + acts).ravel()
    X = sp.csr_matrix((np.ones(n * K), cols, np.arange(0, n * K + 1, K)), shape=(n, K * 256))
    return np.rint(X @ G).astype(np.int64)


def requant(acc, mult):
    """int accumulator -> uint8 activation of the next layer (ReLU = clip at 0)."""
    return np.clip(np.rint(acc * mult), 0, 255).astype(np.int64)


def maxpool2(x):
    n, c, h, w = x.shape
    return x.reshape(n, c, h // 2, 2, w // 2, 2).max(axis=(3, 5))


def int_inference(images, layers, table, check_exact=False):
    """Predicted classes for uint8 images (n x 28 x 28) using LUT `table`."""
    L = {l["name"]: l for l in layers}

    def layer(acts, l):
        acc = lut_matmul(acts, l["wq"], table) + l["bq"]
        if check_exact:   # N = 0: must equal a plain integer matrix multiply
            assert np.array_equal(acc, acts @ l["wq"].T + l["bq"]), l["name"]
        return acc

    x = images[:, None].astype(np.int64)                                  # n,1,28,28
    cols, (n, ho, wo) = im2col(x, 5, 2)
    a = requant(layer(cols, L["conv1"]), L["conv1"]["mult"])
    x = maxpool2(a.reshape(n, ho, wo, 6).transpose(0, 3, 1, 2))           # n,6,14,14
    cols, (n, ho, wo) = im2col(x, 5, 0)
    a = requant(layer(cols, L["conv2"]), L["conv2"]["mult"])
    x = maxpool2(a.reshape(n, ho, wo, 16).transpose(0, 3, 1, 2))          # n,16,5,5
    x = x.reshape(n, -1)                                                  # same order as torch flatten
    x = requant(layer(x, L["fc1"]), L["fc1"]["mult"])
    return layer(x, L["fc2"]).argmax(axis=1)


# ---- 4. Evaluate every design -------------------------------------------------------------

def main():
    torch.set_num_threads(4)
    xtr, ytr, xte, yte = mnist()
    if WEIGHTS.exists():
        model = LeNet()
        model.load_state_dict(torch.load(WEIGHTS))
        print(f"loaded {WEIGHTS.relative_to(REPO)}")
    else:
        print("training LeNet on MNIST (float, 3 epochs, seed 0) ...")
        model = train(xtr, ytr)
        torch.save(model.state_dict(), WEIGHTS)
    model.eval()
    acc_float = float_accuracy(model, xte, yte)
    layers = quantize(model, xtr[:2000])

    rows = []
    for d in designs.DESIGNS:
        t = time.time()
        # Batches of 1,000 images keep the sparse matrices small (conv2: 150 M entries
        # for all 10,000 images at once would not fit in memory).
        pred = np.concatenate([int_inference(xte[i:i + 1000], layers, d.lut(),
                                             check_exact=(d.name == "trunc_N0"))
                               for i in range(0, len(xte), 1000)])
        rows.append({"design": d.name, "family": d.family, "N": d.N,
                     "top1": float((pred == yte).mean())})
        print(f"  {d.name:16s} top-1 {rows[-1]['top1'] * 100:6.2f} %   ({time.time() - t:.1f} s)")
    df = pd.DataFrame(rows)
    acc_int8 = float(df.loc[df.design == "trunc_N0", "top1"].iloc[0])
    df["delta_vs_int8_pp"] = (df["top1"] - acc_int8) * 100      # percentage points
    df["delta_vs_float_pp"] = (df["top1"] - acc_float) * 100
    df["float_top1"] = acc_float
    df.to_csv(RESULTS / "cnn_accuracy.csv", index=False)
    print(f"\nfloat top-1 {acc_float * 100:.2f} %, exact int8 top-1 {acc_int8 * 100:.2f} %")
    print("wrote results/cnn_accuracy.csv")


if __name__ == "__main__":
    main()
