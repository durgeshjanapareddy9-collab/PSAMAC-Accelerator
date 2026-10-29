"""Which error metric predicts application quality? (M2.10)

Hypothesis under test (Rather et al. 2025, ref. 1): image quality tracks the
maximum error EDmax and how often it occurs, not just the mean error MED/NMED.

Method
  * Pool of designs: our truncated multiplier at N = 1..12, the four
    EvoApproxLib designs, and the functional reconstruction of Rather et al.'s
    Design 1 (17 designs). Left out: the exact N = 0 (every metric is 0), and
    TruMD, whose outputs are identical to our N = 2, 4, 6, 8 (verified on all
    65,536 pairs) -- including them would count the same design twice.
  * Error metrics, all over the 65,536 input pairs (model/metrics.py):
    ER, |ME| (bias), MED, MRED, MSE, EDmax, EDmax_count, and
    EDmax_freq = EDmax x EDmax_count / 65,536 (how much the worst case adds
    to the mean error; one number for "EDmax and how often it occurs").
    NMED is MED / 65,025, so it ranks designs exactly like MED.
  * Quality loss per design (higher = worse):
      image workloads: PSNR loss = -PSNR, SSIM loss = 1 - SSIM
      CNN: top-1 drop in percentage points vs the exact int8 network
  * Spearman rank correlation (rho) between each metric and each quality
    loss: +1 means the metric orders the designs exactly like the quality loss.
    Designs whose output is identical to the exact output are tied at the best
    rank. With only 17 designs the p-values are rough; we report them anyway.

Within our own family alone every metric rises with N, so every metric would
get rho = 1; the baselines, whose error profiles differ, are what make the
test meaningful.

Additional analysis (ours, beyond the hypothesis): workload-aware metrics.
The metrics above assume every (a, b) pair is equally likely. A real workload
uses only some pairs (e.g. filter weights below 64). We therefore also compute
MED and MSE weighted by how often each (a, b) pair actually occurs in each
workload (wMED, wMSE), and test those the same way.

Writes results/metric_correlation.csv, results/matched_pairs.csv and
results/workload_metrics.csv.
Usage:  python -m eval.analysis     (via `make analysis`, after eval and cnn)
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from eval import image_conv
from model import designs
from model.approx_mul import RESULTS, exact_lut

METRICS = ["ER", "abs_ME", "MED", "MRED", "MSE", "EDmax", "EDmax_count", "EDmax_freq"]
EXACT = exact_lut()


def pool():
    """Names of the designs in the correlation pool."""
    return ([f"trunc_N{n}" for n in range(1, 13)]
            + [d.name for d in designs.DESIGNS if d.family in ("evo", "recon")])


def error_table():
    e = pd.concat([pd.read_csv(RESULTS / "error_metrics.csv"),
                   pd.read_csv(RESULTS / "error_metrics_baselines.csv")], ignore_index=True)
    e["abs_ME"] = e["ME"].abs()
    e["EDmax_freq"] = e["EDmax"] * e["EDmax_count"] / 65536
    return e.set_index("design")


def quality_table():
    """One column per quality-loss measure, one row per design."""
    img = pd.read_csv(RESULTS / "image_quality.csv")
    q = {}
    for w in img["workload"].unique():
        d = img[img["workload"] == w].set_index("design")
        q[f"{w}:PSNR_loss"] = -d["psnr_db"].replace(np.inf, 1000.0)   # identical output = best
        q[f"{w}:SSIM_loss"] = 1 - d["ssim"]
    cnn = pd.read_csv(RESULTS / "cnn_accuracy.csv").set_index("design")
    q["mnist_cnn:top1_drop_pp"] = -cnn["delta_vs_int8_pp"]
    return pd.DataFrame(q)


# ---- Workload-aware metrics -----------------------------------------------------------

def image_pair_counts(workload):
    """How often each (pixel a, weight b) pair is used by an image workload."""
    get_img, get_kernel = image_conv.WORKLOADS[workload]
    img, kernel = get_img(), get_kernel()
    k = kernel.shape[0]
    pad = np.pad(img, k // 2, mode="reflect")
    h, w = img.shape
    counts = np.zeros((256, 256))
    for u in range(k):
        for v in range(k):
            hist = np.bincount(pad[u:u + h, v:v + w].ravel(), minlength=256)
            counts[:, kernel[u, v]] += hist
    return counts


def cnn_pair_counts():
    """How often each (activation a, |weight| b) pair is used by the exact int8 CNN.

    For a layer, count[a, m] = sum over inputs k of (#samples with act_k = a)
    x (#outputs whose |w[o, k]| = m), summed over the layers.
    """
    import torch
    from eval import cnn
    model = cnn.LeNet()
    model.load_state_dict(torch.load(cnn.WEIGHTS))
    model.eval()
    xtr, _, xte, _ = cnn.mnist()
    layers = cnn.quantize(model, xtr[:2000])
    L = {l["name"]: l for l in layers}
    counts = np.zeros((256, 256))

    def add(acts, l):
        K = acts.shape[1]
        hist = np.zeros((K, 256))
        np.add.at(hist, (np.repeat(np.arange(K)[None], len(acts), 0), acts), 1)
        wc = np.zeros((K, 256))
        np.add.at(wc, (np.repeat(np.arange(K)[:, None], l["wq"].shape[0], 1),
                       np.abs(l["wq"]).T), 1)
        counts[:] += hist.T @ wc

    table = EXACT
    for i in range(0, len(xte), 1000):          # same steps as cnn.int_inference, exact LUT
        x = xte[i:i + 1000][:, None].astype(np.int64)
        cols, (n, ho, wo) = cnn.im2col(x, 5, 2)
        add(cols, L["conv1"])
        a = cnn.requant(cnn.lut_matmul(cols, L["conv1"]["wq"], table) + L["conv1"]["bq"], L["conv1"]["mult"])
        x = cnn.maxpool2(a.reshape(n, ho, wo, 6).transpose(0, 3, 1, 2))
        cols, (n, ho, wo) = cnn.im2col(x, 5, 0)
        add(cols, L["conv2"])
        a = cnn.requant(cnn.lut_matmul(cols, L["conv2"]["wq"], table) + L["conv2"]["bq"], L["conv2"]["mult"])
        x = cnn.maxpool2(a.reshape(n, ho, wo, 16).transpose(0, 3, 1, 2)).reshape(n, -1)
        add(x, L["fc1"])
        x = cnn.requant(cnn.lut_matmul(x, L["fc1"]["wq"], table) + L["fc1"]["bq"], L["fc1"]["mult"])
        add(x, L["fc2"])
    return counts


def workload_metrics(names):
    """wMED and wMSE of every design under each workload's (a, b) distribution."""
    workloads = {w: image_pair_counts(w) for w in image_conv.WORKLOADS}
    workloads["mnist_cnn"] = cnn_pair_counts()
    rows = []
    for w, c in workloads.items():
        p = c / c.sum()
        for name in names:
            err = (designs.get_lut(name) - EXACT).astype(np.float64)
            rows.append({"workload": w, "design": name,
                         "wMED": float((p * np.abs(err)).sum()),
                         "wMSE": float((p * err ** 2).sum()),
                         "wER": float(p[err != 0].sum()),
                         "pairs_used_pct": float(100 * np.count_nonzero(c) / 65536)})
    return pd.DataFrame(rows)


# ---- Correlations -----------------------------------------------------------------------------

def correlations(e, q, wm, names):
    rows = []
    for qcol in q.columns:
        workload = qcol.split(":")[0]
        y = q.loc[names, qcol]
        cand = {m: e.loc[names, m] for m in METRICS}
        wmw = wm[wm["workload"] == workload].set_index("design")
        for m in ("wMED", "wMSE", "wER"):
            cand[m] = wmw.loc[names, m]
        for m, x in cand.items():
            rho, pval = spearmanr(x, y)
            rows.append({"quality_loss": qcol, "metric": m, "spearman_rho": rho,
                         "p_value": pval, "n_designs": len(names),
                         "kind": "workload-aware" if m.startswith("w") else "uniform inputs"})
    return pd.DataFrame(rows)


def matched_pairs(e, q):
    """Each EvoApproxLib design next to our N with the closest MED."""
    rows = []
    ours = e.loc[[f"trunc_N{n}" for n in range(1, 13)]]
    for d in [x.name for x in designs.DESIGNS if x.family in ("evo", "recon")]:
        twin = (np.log(ours["MED"]) - np.log(e.loc[d, "MED"])).abs().idxmin()
        for name in (d, twin):
            rows.append({"pair": d, "design": name, "MED": e.loc[name, "MED"],
                         "EDmax": e.loc[name, "EDmax"], "EDmax_count": e.loc[name, "EDmax_count"],
                         "ER": e.loc[name, "ER"], "MSE": e.loc[name, "MSE"],
                         **{c: q.loc[name, c] for c in q.columns}})
    return pd.DataFrame(rows)


def main():
    names = pool()
    e, q = error_table(), quality_table()
    wm = workload_metrics(names)
    wm.to_csv(RESULTS / "workload_metrics.csv", index=False)
    c = correlations(e, q, wm, names)
    c.to_csv(RESULTS / "metric_correlation.csv", index=False)
    mp = matched_pairs(e, q)
    mp.to_csv(RESULTS / "matched_pairs.csv", index=False)

    pd.set_option("display.width", 200)
    piv = c.pivot(index="metric", columns="quality_loss", values="spearman_rho")
    piv = piv.reindex(METRICS + ["wMED", "wMSE", "wER"])
    print(f"Spearman rho, metric vs quality loss ({len(names)} designs)\n")
    print(piv.round(3).to_string())
    print("\nBest uniform-input metric per quality measure:")
    u = c[c["kind"] == "uniform inputs"]
    for qcol, g in u.groupby("quality_loss"):
        best = g.loc[g["spearman_rho"].idxmax()]
        print(f"  {qcol:32s} {best['metric']:12s} rho = {best['spearman_rho']:.3f}")
    print("\nwrote results/metric_correlation.csv, results/matched_pairs.csv, "
          "results/workload_metrics.csv")


if __name__ == "__main__":
    main()
