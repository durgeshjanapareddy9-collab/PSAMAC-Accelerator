"""Regenerate every plot in figures/ from the CSVs in results/ (M2.12).

Usage:  python -m eval.figures     (via `make figures`)

Figures
  error_vs_N.png        error metrics vs N                 (model/metrics.py)
  image_samples.png     filtered images per design         (eval/image_conv.py)
  ppa_vs_N.png          multiplier area, delay, power vs N
  array_vs_N.png        4x4 array area, delay, power, energy per MAC vs N
  quality_vs_N.png      image PSNR and SSIM vs N
  cnn_vs_N.png          MNIST top-1 accuracy vs N
  tradeoff.png          accuracy vs area / power / delay, baselines marked
  metric_correlation.png  Spearman rho of each error metric vs quality loss

Colour = design family, always in this fixed order (blue = ours, orange =
EvoApproxLib, aqua = TruMD, yellow = a*b reference, magenta = reconstruction),
and every family also has its own marker shape, so colour is never the only cue.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model import designs, metrics
from model.approx_mul import REPO, RESULTS

FIG = REPO / "figures"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
FAMILY = {   # colour, marker, legend name
    "trunc": ("#2a78d6", "o", "Ours: truncated Dadda (N)"),
    "evo":   ("#eb6834", "s", "EvoApproxLib"),
    "trumd": ("#1baf7a", "^", "TruMD"),
    "behav": ("#eda100", "D", "a*b (Yosys)"),
    "recon": ("#e87ba4", "*", "Recon. Rather D1"),
}
NS = list(range(13))
PSNR_CAP = 70.0           # identical-to-exact outputs (PSNR = inf) are drawn here


def style():
    plt.rcParams.update({
        "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "legend.frameon": False,
        "lines.linewidth": 2, "lines.markersize": 6,
    })


def tidy(ax, title=None, xlabel=None, ylabel=None):
    if title:
        ax.set_title(title, fontsize=10, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def save(fig, name, title):
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK)
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)
    print(f"wrote figures/{name}")


def ours(df, col="design"):
    """Rows of our N sweep, ordered by N (works for ppa_mul and the eval CSVs)."""
    key = {r: n for n in NS for r in (f"approx_mul_N{n}", f"trunc_N{n}")}
    d = df[df[col].isin(key)].copy()
    d["N"] = d[col].map(key)
    return d.sort_values("N")


# ---- Hardware vs N --------------------------------------------------------------------

def fig_ppa():
    p = pd.read_csv(RESULTS / "ppa_mul.csv")
    o, behav = ours(p), p[p.design == "mul_behav"].iloc[0]
    base = o.iloc[0]
    c, m, _ = FAMILY["trunc"]
    cb = FAMILY["behav"][0]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), constrained_layout=True)
    ax = axes[0]
    ax.plot(o.N, 100 * o.area_mul_um2 / base.area_mul_um2, color=c, marker=m)
    ax.axhline(100 * behav.area_mul_um2 / base.area_mul_um2, color=cb, ls="--", lw=1.5)
    ax.text(12, 100 * behav.area_mul_um2 / base.area_mul_um2 + 2, "a*b", color=MUTED, ha="right")
    ax.set_ylim(0, 115)
    tidy(ax, f"Multiplier area (% of N=0 = {base.area_mul_um2:.0f} um²)", "Truncation depth N", "%")
    ax = axes[1]
    ax.plot(o.N, o.min_period_ns, color=c, marker=m)
    ax.axhline(behav.min_period_ns, color=cb, ls="--", lw=1.5)
    ax.text(12, behav.min_period_ns + 0.08, "a*b", color=MUTED, ha="right")
    ax.set_ylim(0, None)
    tidy(ax, "Minimum clock period (ns), flop to flop", "Truncation depth N", "ns")
    ax = axes[2]
    for col, ls, lab in [("power_image_comb_uW", "-", "image data"),
                         ("power_random_comb_uW", ":", "uniform random")]:
        ax.plot(o.N, 100 * o[col] / base[col], color=c, marker=m, ls=ls, label=lab)
    ax.set_ylim(0, 110)
    ax.legend(loc="upper right")
    tidy(ax, "Multiplier logic power (% of N=0)", "Truncation depth N", "%")
    for ax in axes:
        ax.set_xticks(NS)
    save(fig, "ppa_vs_N.png", "8x8 multiplier vs N - Yosys + OpenSTA, sky130_fd_sc_hd tt 25C 1.80V, "
         "pre-layout, 10 ns clock")


def fig_array():
    a = pd.read_csv(RESULTS / "ppa_array.csv").sort_values("N")
    base = a.iloc[0]
    c, m, _ = FAMILY["trunc"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), constrained_layout=True)
    panels = [("area_total_um2", "Array area (% of N=0 = {:,.0f} um²)"),
              ("power_image_total_uW", "Array power, image tiles (% of N=0 = {:,.0f} uW)"),
              ("energy_image_pJ_per_op", "Energy per useful MAC (% of N=0 = {:.2f} pJ)")]
    for ax, (col, title) in zip(axes, panels):
        if col not in a or a[col].isna().all():
            ax.text(0.5, 0.5, "not measured yet (make array)", ha="center", transform=ax.transAxes)
            continue
        title = title.format(base[col])
        ax.plot(a.N, 100 * a[col] / base[col], color=c, marker=m)
        ax.set_ylim(0, 110)
        ax.set_xticks(NS)
        tidy(ax, title, "Truncation depth N", "%")
    save(fig, "array_vs_N.png", "4x4 output-stationary systolic array (16 MAC PEs, 24-bit "
         "accumulators) vs N - same flow and corner")


# ---- Application quality vs N ---------------------------------------------------------

WORKLOAD_STYLE = {"camera_gauss5": ("-", "camera, 5x5 Gaussian"),
                  "astronaut_gauss5": ("--", "astronaut, 5x5 Gaussian"),
                  "rather_denoise": (":", "noisy cameraman, 3x3 mean (Rather et al. case)")}


def fig_quality():
    q = pd.read_csv(RESULTS / "image_quality.csv")
    c, m, _ = FAMILY["trunc"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), constrained_layout=True)
    for w, (ls, lab) in WORKLOAD_STYLE.items():
        o = ours(q[q.workload == w])
        o = o[o.N > 0]
        axes[0].plot(o.N, o.psnr_db.clip(upper=PSNR_CAP), color=c, marker=m, ls=ls, label=lab)
        axes[1].plot(o.N, o.ssim, color=c, marker=m, ls=ls, label=lab)
    axes[0].set_ylim(0, PSNR_CAP + 5)
    tidy(axes[0], f"PSNR vs exact output (dB; identical output = inf, drawn at {PSNR_CAP:.0f})",
         "Truncation depth N", "dB")
    axes[1].set_ylim(0, 1.05)
    tidy(axes[1], "SSIM vs exact output (1 = identical)", "Truncation depth N")
    axes[1].legend(loc="lower left", fontsize=9)
    for ax in axes:
        ax.set_xticks(NS[1:])
    save(fig, "quality_vs_N.png", "Image filtering through the truncated multiplier (bit-exact LUT model)")


def fig_cnn():
    d = pd.read_csv(RESULTS / "cnn_accuracy.csv")
    o = ours(d)
    c, m, _ = FAMILY["trunc"]
    fig, ax = plt.subplots(figsize=(7.5, 3.8), constrained_layout=True)
    ax.plot(o.N, 100 * o.top1, color=c, marker=m)
    ax.axhline(100 * d.float_top1.iloc[0], color=MUTED, ls=":", lw=1.2)
    ax.text(0, 100 * d.float_top1.iloc[0] - 6, f"float {100 * d.float_top1.iloc[0]:.2f} %; "
            f"exact int8 (N=0) {100 * o.top1.iloc[0]:.2f} %", color=MUTED)
    for _, r in o[o.N >= 8].iterrows():
        ax.annotate(f"{100 * r.top1:.1f}", (r.N, 100 * r.top1), textcoords="offset points",
                    xytext=(6, 4), fontsize=8, color=INK)
    ax.set_ylim(0, 105)
    ax.set_xticks(NS)
    tidy(ax, "MNIST top-1 accuracy, 10,000 test images (%)", "Truncation depth N", "%")
    save(fig, "cnn_vs_N.png", "LeNet-style CNN, int8 inference, every multiply through the LUT")


# ---- Trade-off curves with baselines ----------------------------------------------------

def fig_tradeoff():
    ppa = pd.concat([pd.read_csv(RESULTS / "ppa_mul.csv"),
                     pd.read_csv(RESULTS / "ppa_baselines.csv")], ignore_index=True)
    ppa["name"] = ppa.design.str.replace("approx_mul_N", "trunc_N", regex=False)
    base = ppa[ppa.name == "trunc_N0"].iloc[0]
    ppa["area_pct"] = 100 * ppa.area_mul_um2 / base.area_mul_um2
    ppa["power_pct"] = 100 * ppa.power_image_comb_uW / base.power_image_comb_uW
    ppa = ppa.set_index("name")
    img = pd.read_csv(RESULTS / "image_quality.csv")
    img = img[img.workload == "camera_gauss5"].set_index("design")
    cnn = pd.read_csv(RESULTS / "cnn_accuracy.csv").set_index("design")
    ys = [("PSNR, camera 5x5 Gaussian (dB)", img.psnr_db.clip(upper=PSNR_CAP)),
          ("MNIST top-1 (%)", 100 * cnn.top1)]
    xs = [("area_pct", "Multiplier area (% of exact N=0)"),
          ("power_pct", "Multiplier logic power, image data (% of N=0)"),
          ("min_period_ns", "Minimum clock period (ns)")]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for i, (ylab, y) in enumerate(ys):
        for j, (xcol, xlab) in enumerate(xs):
            ax = axes[i, j]
            o = [f"trunc_N{n}" for n in NS]
            # Delay is not monotonic in N (area-oriented mapping), so a line through
            # the delay points would zig-zag: markers only there.
            ax.plot(ppa.loc[o, xcol], y.loc[o], color=FAMILY["trunc"][0],
                    marker=FAMILY["trunc"][1], label=FAMILY["trunc"][2], zorder=2,
                    ls="none" if xcol == "min_period_ns" else "-")
            for n in (0, 4, 8, 10, 12):
                ax.annotate(f"N={n}", (ppa.loc[f"trunc_N{n}", xcol], y.loc[f"trunc_N{n}"]),
                            textcoords="offset points", xytext=(5, -12), fontsize=8, color=MUTED)
            for fam in ("evo", "trumd"):
                names = [d.name for d in designs.DESIGNS if d.family == fam]
                col, mk, lab = FAMILY[fam]
                ax.scatter(ppa.loc[names, xcol], y.loc[names], color=col, marker=mk, s=70,
                           edgecolor=SURFACE, linewidth=1.5, label=lab, zorder=3)
                if fam == "evo" and i == 0:     # labels only once (CNN row is crowded)
                    for nme in names:
                        ax.annotate(nme.replace("evo_", ""), (ppa.loc[nme, xcol], y.loc[nme]),
                                    textcoords="offset points", xytext=(6, 4), fontsize=8, color=INK)
            b = ppa.loc["mul_behav"]
            if xcol != "power_pct" or not np.isnan(b[xcol]):
                ax.scatter([b[xcol]], [y.loc["trunc_N0"]], color=FAMILY["behav"][0],
                           marker=FAMILY["behav"][1], s=60, edgecolor=SURFACE, linewidth=1.5,
                           label=FAMILY["behav"][2], zorder=3)
            tidy(ax, None, xlab, ylab if j == 0 else None)
            if i == 0:
                ax.set_ylim(0, PSNR_CAP + 5)
    axes[0, 0].legend(loc="lower right", fontsize=9)
    save(fig, "tradeoff.png", "Accuracy vs hardware cost: our N sweep (line) and published "
         f"baselines (points), identical flow. PSNR of identical output drawn at {PSNR_CAP:.0f} dB")


# ---- Metric correlation ----------------------------------------------------------------

def fig_correlation():
    c = pd.read_csv(RESULTS / "metric_correlation.csv")
    shown = [("camera_gauss5:PSNR_loss", "Camera Gaussian: PSNR loss"),
             ("rather_denoise:PSNR_loss", "Rather et al. case: PSNR loss"),
             ("mnist_cnn:top1_drop_pp", "MNIST CNN: top-1 drop")]
    order = ["ER", "abs_ME", "MED", "MRED", "MSE", "EDmax", "EDmax_count", "EDmax_freq",
             "wMED", "wMSE", "wER"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), constrained_layout=True, sharey=True)
    for ax, (q, title) in zip(axes, shown):
        d = c[c.quality_loss == q].set_index("metric").loc[order]
        cols = [FAMILY["evo"][0] if m.startswith("w") else FAMILY["trunc"][0] for m in order]
        ax.barh(range(len(order)), d.spearman_rho, color=cols, height=0.7)
        for k, v in enumerate(d.spearman_rho):
            ax.text(v + (0.03 if v >= 0 else -0.03), k, f"{v:.2f}", va="center",
                    ha="left" if v >= 0 else "right", fontsize=8, color=INK)
        ax.axvline(0, color=MUTED, lw=1)
        ax.set_xlim(-1.15, 1.15)
        ax.set_yticks(range(len(order)), order)
        ax.invert_yaxis()
        tidy(ax, title, "Spearman rho (1 = ranks designs perfectly)")
    from matplotlib.patches import Patch
    axes[2].legend(handles=[Patch(color=FAMILY["trunc"][0], label="metric over all 65,536 pairs"),
                            Patch(color=FAMILY["evo"][0], label="workload-aware (w*) metric")],
                   loc="upper left", fontsize=8)
    n = int(c.n_designs.iloc[0])
    save(fig, "metric_correlation.png", f"Which error metric ranks {n} designs like their quality "
         "loss? (ours N=1..12 + 4 EvoApproxLib + Rather D1 reconstruction)")


def main():
    FIG.mkdir(exist_ok=True)
    style()
    df = metrics.sweep()
    metrics.plot(df, FIG / "error_vs_N.png")
    print("wrote figures/error_vs_N.png")
    style()
    from eval import image_conv
    _, outputs = image_conv.evaluate()
    image_conv.plot_samples(outputs, FIG / "image_samples.png")
    print("wrote figures/image_samples.png")
    style()
    fig_ppa()
    fig_array()
    fig_quality()
    fig_cnn()
    fig_tradeoff()
    fig_correlation()


if __name__ == "__main__":
    main()
