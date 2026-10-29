"""Image filtering with approximate multipliers: PSNR and SSIM (M2.3, M2.8).

Image filtering is a convolution: every output pixel is a weighted sum of the
pixels around it, i.e. a chain of multiply-accumulates, exactly what the MAC
array computes. Here every multiplication is done through a design's LUT
(lut[pixel, weight]), so the result is bit-identical to what the hardware
would produce (eval/../model/systolic.py checks this on real image patches
through the RTL array).

    raw[r, c]  = sum over the kernel taps (u, v) of LUT[pixel(r+u, c+v), w(u, v)]
    out[r, c]  = raw[r, c] >> 8          (weights sum to about 256 = 1.0)

Image borders are handled by mirroring the image ("reflect" padding). The
kernel is applied without flipping (correlation); all our kernels are
symmetric, so this is the same as convolution.

Workloads (8-bit grayscale):
    camera_gauss5     skimage "camera" (512x512), 5x5 Gaussian (sigma 1)
    astronaut_gauss5  skimage "astronaut" converted to gray (512x512), same kernel
    rather_denoise    the case used by Rather et al. 2025: cameraman at 256x256,
                      Gaussian noise of variance 0.01 added, then a 3x3
                      averaging filter (all nine weights 28, sum 252)

Quality: PSNR (dB) and SSIM [Wang et al. 2004] of each design's output
against the SAME pipeline run with the exact multiplier (lut N=0). So PSNR is
infinite and SSIM is 1 for an exact design.

Usage:  python -m eval.image_conv    (via `make eval`; needs `make baselines`)
Writes results/image_quality.csv and figures/image_samples.png.
"""

from functools import cache

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import color, data, transform, util
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from model import designs
from model.approx_mul import REPO, RESULTS, lut

FIGURES = REPO / "figures"
SEED = 0                       # noise seed for the Rather et al. case (repeatable)


# ---- Images and kernels --------------------------------------------------------

def to_u8(x):
    """Float image in [0, 1] -> uint8 0..255 (rounded)."""
    return np.clip(np.round(x * 255), 0, 255).astype(np.uint8)


@cache
def camera():
    return data.camera()                                    # 512x512 uint8


@cache
def astronaut_gray():
    return to_u8(color.rgb2gray(data.astronaut()))          # 512x512 uint8


@cache
def rather_noisy_camera():
    """256x256 cameraman with Gaussian noise of variance 0.01 (Rather et al. setup)."""
    small = transform.resize(camera(), (256, 256), anti_aliasing=True)   # float 0..1
    noisy = util.random_noise(small, mode="gaussian", var=0.01, rng=SEED)
    return to_u8(noisy)


def gaussian_kernel(size=5, sigma=1.0, total=256):
    """Gaussian weights scaled to sum to about `total` and rounded to integers."""
    r = np.arange(size) - size // 2
    g = np.exp(-(r[:, None] ** 2 + r[None, :] ** 2) / (2 * sigma ** 2))
    return np.round(g / g.sum() * total).astype(np.int64)


def average_kernel(size=3):
    """Averaging filter: equal weights, floor(256 / size^2) each (28 for 3x3)."""
    return np.full((size, size), 256 // (size * size), dtype=np.int64)


def filter_bank():
    """Four unsigned 5x5 kernels, used as the four columns of B in the array test.
    Kernel 0 is the main Gaussian used for the image results."""
    center = np.full((5, 5), 4, dtype=np.int64)
    center[2, 2] = 156                                     # 24*4 + 156 = 252
    return [gaussian_kernel(5, 1.0), np.full((5, 5), 10, dtype=np.int64),
            gaussian_kernel(5, 2.0), center]


WORKLOADS = {
    "camera_gauss5": (camera, lambda: gaussian_kernel(5, 1.0)),
    "astronaut_gauss5": (astronaut_gray, lambda: gaussian_kernel(5, 1.0)),
    "rather_denoise": (rather_noisy_camera, lambda: average_kernel(3)),
}


# ---- The convolution ---------------------------------------------------------------

def conv_raw(img, kernel, table):
    """Sum of LUT[pixel, weight] over the kernel window, for every pixel (int64)."""
    k = kernel.shape[0]
    pad = np.pad(img, k // 2, mode="reflect")
    h, w = img.shape
    raw = np.zeros((h, w), dtype=np.int64)
    for u in range(k):
        for v in range(k):
            # One multiply per pixel for this tap: look up LUT[pixel, w(u, v)].
            raw += table[pad[u:u + h, v:v + w], kernel[u, v]]
    return raw


def filter_image(img, kernel, table):
    """Filtered 8-bit image: raw >> 8, clipped to 0..255."""
    return np.clip(conv_raw(img, kernel, table) >> 8, 0, 255).astype(np.uint8)


def patches(img, coords, k):
    """im2col: the k x k window around each (row, col), flattened row by row,
    with the same reflect padding as conv_raw. Returns len(coords) x k*k."""
    pad = np.pad(img, k // 2, mode="reflect")
    return np.stack([pad[r:r + k, c:c + k].ravel() for r, c in coords])


# ---- Evaluation ----------------------------------------------------------------------

def evaluate():
    exact = lut(0)
    rows, outputs = [], {}
    for wname, (get_img, get_kernel) in WORKLOADS.items():
        img, kernel = get_img(), get_kernel()
        ref = filter_image(img, kernel, exact)
        outputs[(wname, "input")] = img
        for d in designs.DESIGNS:
            out = filter_image(img, kernel, d.lut())
            outputs[(wname, d.name)] = out
            same = np.array_equal(out, ref)
            rows.append({
                "workload": wname, "design": d.name, "family": d.family, "N": d.N,
                "psnr_db": np.inf if same else peak_signal_noise_ratio(ref, out, data_range=255),
                "ssim": 1.0 if same else structural_similarity(ref, out, data_range=255),
                "mean_abs_diff": float(np.mean(np.abs(out.astype(int) - ref))),
                "max_abs_diff": int(np.max(np.abs(out.astype(int) - ref))),
                "mean_bias": float(np.mean(out.astype(int) - ref)),
            })
    return pd.DataFrame(rows), outputs


def plot_samples(outputs, path):
    """Exact output next to a few approximate ones, camera + Gaussian."""
    show = [("input", "Input"), ("trunc_N0", "Exact (N=0)"), ("trunc_N6", "Trunc N=6"),
            ("trunc_N8", "Trunc N=8"), ("trunc_N10", "Trunc N=10"),
            ("trunc_N12", "Trunc N=12"), ("evo_1446", "EvoApprox 1446"),
            ("evo_JQQ", "EvoApprox JQQ")]
    fig, axes = plt.subplots(2, 4, figsize=(13, 7), constrained_layout=True)
    for ax, (key, title) in zip(axes.ravel(), show):
        crop = outputs[("camera_gauss5", key)][100:300, 150:350]    # zoom on the face/camera
        ax.imshow(crop, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle("5x5 Gaussian filter on 'camera' through each multiplier (200x200 crop)",
                 fontsize=11, x=0.01, ha="left")
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    df, outputs = evaluate()
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "image_quality.csv", index=False)
    plot_samples(outputs, FIGURES / "image_samples.png")
    pd.set_option("display.width", 140)
    print(df.pivot(index="design", columns="workload", values="psnr_db")
          .reindex([d.name for d in designs.DESIGNS]).round(2).to_string())
    print("\nwrote results/image_quality.csv and figures/image_samples.png")


if __name__ == "__main__":
    main()
