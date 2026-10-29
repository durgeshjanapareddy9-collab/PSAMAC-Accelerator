"""Error metrics for 8x8 approximate multipliers, computed over all 65,536 pairs.

Usage:  python -m model.metrics              our N sweep
        python -m model.metrics --baselines  the baseline designs (needs `make baselines`)
Writes results/error_metrics.csv and figures/error_vs_N.png, or
results/error_metrics_baselines.csv.

Definitions (following the surveys in docs/REFERENCES.md, refs 9-11), with
    exact  = a * b
    approx = the design's output
    error  = approx - exact          (signed; negative = result too small)
    ED     = |error|                 (error distance)

    ER          error rate: fraction of pairs with error != 0
    ME          mean error (signed) -- the bias. Truncation only removes value,
                so ME < 0 for our designs and ME = -MED.
    MED         mean error distance: mean of ED
    NMED        MED / (255 * 255), i.e. relative to the largest exact product
    MRED        mean relative error distance: mean of ED / exact, skipping the
                pairs whose exact product is 0 (a or b is 0)
    EDmax       largest ED
    EDmax_count number of pairs (out of 65,536) whose ED equals EDmax
    MSE         mean squared error (PSNR, an image-quality measure, is based on it)

metrics_for_lut() works on any 256x256 table, so the baseline designs use the
same code.
"""

import sys

import matplotlib

matplotlib.use("Agg")                    # draw to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model.approx_mul import N_MAX, REPO, RESULTS, exact_lut, lut

FIGURES = REPO / "figures"
EXACT = exact_lut()


def metrics_for_lut(table):
    """Return a dict of error metrics for a 256x256 LUT (table[a, b])."""
    table = np.asarray(table, dtype=np.int64)
    error = table - EXACT
    ed = np.abs(error)
    nonzero = EXACT != 0
    ed_max = int(ed.max())
    return {
        "ER": float(np.mean(error != 0)),
        "ME": float(error.mean()),
        "MED": float(ed.mean()),
        "NMED": float(ed.mean() / (255 * 255)),
        "MRED": float(np.mean(ed[nonzero] / EXACT[nonzero])),
        "EDmax": ed_max,
        "EDmax_count": int(np.count_nonzero(ed == ed_max)) if ed_max > 0 else 0,
        "MSE": float(np.mean(error.astype(np.float64) ** 2)),
    }


def sweep():
    """Metrics for our truncated multiplier at every N."""
    rows = []
    for n in range(N_MAX + 1):
        rows.append({"design": f"trunc_N{n}", "N": n, **metrics_for_lut(lut(n))})
    return pd.DataFrame(rows)


def plot(df, path):
    """Four small panels (one per metric) sharing the N axis.

    Separate panels instead of one chart with two y-axes, because the metrics
    have very different scales. Log scale where values span several decades.
    """
    color, ink, muted, grid = "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df"
    plt.rcParams.update({
        "font.size": 10, "axes.edgecolor": grid, "axes.labelcolor": muted,
        "xtick.color": muted, "ytick.color": muted, "axes.titlecolor": ink,
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    })
    d = df[df["N"] > 0]          # N = 0 has zero error and cannot go on a log axis
    panels = [
        ("MED", "Mean error distance (MED)", True),
        ("EDmax", "Maximum error distance (EDmax)", True),
        ("EDmax_count", "Pairs hitting EDmax (of 65,536)", True),
        ("ER", "Error rate (ER, %)", False),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4), constrained_layout=True)
    for ax, (col, title, log) in zip(axes, panels):
        y = d[col] * (100 if col == "ER" else 1)
        ax.plot(d["N"], y, color=color, linewidth=2, marker="o", markersize=5)
        if log:
            ax.set_yscale("log")
        else:
            ax.set_ylim(0, 100)          # a percentage axis starts at 0
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_xlabel("Truncation depth N")
        ax.set_xticks(range(1, N_MAX + 1))
        ax.grid(True, color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.suptitle("Error of the truncated 8x8 Dadda multiplier vs N "
                 "(all 65,536 input pairs; N = 0 is exact)", x=0.01, ha="left",
                 fontsize=11, color=ink)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def baseline_metrics():
    """Metrics for the published baselines and the reconstruction."""
    from model import designs
    rows = [{"design": d.name, "family": d.family, **metrics_for_lut(d.lut())}
            for d in designs.baselines()]
    return pd.DataFrame(rows)


def main():
    if "--baselines" in sys.argv:
        df = baseline_metrics()
        csv = RESULTS / "error_metrics_baselines.csv"
        df.to_csv(csv, index=False)
        with pd.option_context("display.width", 140, "display.float_format", "{:.6g}".format):
            print(df.to_string(index=False))
        print(f"\nwrote {csv.relative_to(REPO)}")
        return
    df = sweep()
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    csv = RESULTS / "error_metrics.csv"
    df.to_csv(csv, index=False)
    plot(df, FIGURES / "error_vs_N.png")
    with pd.option_context("display.width", 120, "display.float_format", "{:.6g}".format):
        print(df.to_string(index=False))
    print(f"\nwrote {csv.relative_to(REPO)} and figures/error_vs_N.png")


if __name__ == "__main__":
    main()
