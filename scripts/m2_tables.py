"""Print the Milestone 2 result tables as markdown, straight from results/*.csv.

Every number in docs/report.md and docs/project_guide.md is copied from this
script's output, so the documents never contain hand-typed results.

Usage:  python scripts/m2_tables.py            (all tables)
        python scripts/m2_tables.py ppa_mul    (one table, by name)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
sys.path.insert(0, str(REPO / "scripts"))
from md_tables import md, pct   # noqa: E402  (same helpers as Milestone 1)


def read(name):
    return pd.read_csv(RESULTS / name)


def t_verification():
    v = read("baseline_verification.csv")
    rows = [{"design": r.design, "reference model": r.reference_model,
             "result": "PASS 65,536/65,536" if r["pass"] is True or r["pass"] == "True"
             else ("software only" if not r.has_rtl else f"FAIL ({r.mismatches})")}
            for _, r in v.iterrows()]
    return "Baseline verification (results/baseline_verification.csv)", md(pd.DataFrame(rows))


def t_ppa_mul():
    p = read("ppa_mul.csv")
    base = p[p.design == "approx_mul_N0"].iloc[0]
    rows = []
    for _, r in p.iterrows():
        rows.append({
            "design": r.design.replace("approx_mul_", ""),
            "area (um²)": f"{r.area_mul_um2:.0f}",
            "area vs N=0": pct(r.area_mul_um2, base.area_mul_um2),
            "cells": r.cells_mul,
            "min period (ns)": f"{r.min_period_ns:.2f}",
            "logic power, image (uW)": f"{r.power_image_comb_uW:.1f}",
            "vs N=0": pct(r.power_image_comb_uW, base.power_image_comb_uW),
            "logic power, random (uW)": f"{r.power_random_comb_uW:.1f}",
            "vs N=0 ": pct(r.power_random_comb_uW, base.power_random_comb_uW),
            "total incl. flops, image (uW)": f"{r.power_image_total_uW:.1f}",
        })
    return ("Multiplier (in mul_wrap) vs N (results/ppa_mul.csv). Logic power = the "
            "multiplier's combinational cells; total adds the 32 wrapper flip-flops"), md(pd.DataFrame(rows))


def t_ppa_baselines():
    p = pd.concat([read("ppa_mul.csv"), read("ppa_baselines.csv")], ignore_index=True)
    base = p[p.design == "approx_mul_N0"].iloc[0]
    show = ["approx_mul_N0", "mul_behav", "evo_1446", "approx_mul_N4", "trumd_4", "evo_GS2",
            "approx_mul_N5", "evo_185Q", "approx_mul_N6", "trumd_6", "approx_mul_N7",
            "approx_mul_N8", "trumd_8", "evo_JQQ", "approx_mul_N9", "trumd_2", "approx_mul_N2"]
    rows = []
    for d in show:
        r = p[p.design == d].iloc[0]
        rows.append({"design": d.replace("approx_mul_", "ours "),
                     "area (um²)": f"{r.area_mul_um2:.0f}",
                     "area vs N=0": pct(r.area_mul_um2, base.area_mul_um2),
                     "min period (ns)": f"{r.min_period_ns:.2f}",
                     "logic power, image (uW)": f"{r.power_image_comb_uW:.1f}",
                     "vs N=0": pct(r.power_image_comb_uW, base.power_image_comb_uW),
                     "energy/multiply, image (pJ, incl. flops)": f"{r.energy_image_pJ_per_op:.3f}"})
    return ("Baselines through the identical flow, next to our closest N "
            "(results/ppa_baselines.csv, results/ppa_mul.csv)"), md(pd.DataFrame(rows))


def t_array():
    a = read("ppa_array.csv").sort_values("N")
    base = a.iloc[0]
    rows = []
    for _, r in a.iterrows():
        row = {"N": r.N, "area (um²)": f"{r.area_total_um2:,.0f}",
               "area vs N=0": pct(r.area_total_um2, base.area_total_um2),
               "logic / flops (um²)": f"{r.area_logic_um2:,.0f} / {r.area_flops_um2:,.0f}",
               "flops": r.flops, "min period (ns)": f"{r.min_period_ns:.2f}",
               "fmax (MHz)": f"{r.fmax_mhz:.0f}"}
        if "power_image_total_uW" in a:
            row.update({"power, image (uW)": f"{r.power_image_total_uW:,.0f}",
                        "power vs N=0": pct(r.power_image_total_uW, base.power_image_total_uW),
                        "energy/MAC (pJ)": f"{r.energy_image_pJ_per_op:.3f}"})
        rows.append(row)
    return "4x4 systolic array vs N (results/ppa_array.csv)", md(pd.DataFrame(rows))


def t_image():
    q = read("image_quality.csv")
    order = list(dict.fromkeys(q.design))
    rows = []
    for d in order:
        g = q[q.design == d].set_index("workload")
        rows.append({"design": d, **{
            f"{w} PSNR (dB) / SSIM": ("identical" if np.isinf(g.loc[w, "psnr_db"])
                                      else f"{g.loc[w, 'psnr_db']:.1f} / {g.loc[w, 'ssim']:.4f}")
            for w in g.index}})
    return ("Image quality vs the exact pipeline (results/image_quality.csv); "
            "'identical' = bit-identical output"), md(pd.DataFrame(rows))


def t_cnn():
    c = read("cnn_accuracy.csv")
    rows = [{"design": r.design, "top-1 (%)": f"{100 * r.top1:.2f}",
             "change vs exact int8 (pp)": f"{r.delta_vs_int8_pp:+.2f}",
             "change vs float (pp)": f"{r.delta_vs_float_pp:+.2f}"} for _, r in c.iterrows()]
    f = c.float_top1.iloc[0]
    return (f"MNIST top-1, 10,000 test images (results/cnn_accuracy.csv); float network "
            f"{100 * f:.2f} %"), md(pd.DataFrame(rows))


def t_errors():
    e = pd.concat([read("error_metrics.csv"), read("error_metrics_baselines.csv")], ignore_index=True)
    rows = [{"design": r.design, "ER (%)": f"{100 * r.ER:.1f}", "ME": f"{r.ME:.2f}",
             "MED": f"{r.MED:.2f}", "NMED": f"{r.NMED:.2e}", "MRED": f"{r.MRED:.2e}",
             "MSE": f"{r.MSE:.4g}", "EDmax": r.EDmax, "pairs at EDmax": r.EDmax_count}
            for _, r in e.iterrows()]
    return "Error metrics, all 65,536 pairs (results/error_metrics*.csv)", md(pd.DataFrame(rows))


def t_correlation():
    c = read("metric_correlation.csv")
    piv = c.pivot(index="metric", columns="quality_loss", values="spearman_rho")
    order = ["ER", "abs_ME", "MED", "MRED", "MSE", "EDmax", "EDmax_count", "EDmax_freq",
             "wMED", "wMSE", "wER"]
    piv = piv.reindex(order).map(lambda x: f"{x:+.2f}")
    piv.columns = [x.replace(":", " ") for x in piv.columns]
    n = int(c.n_designs.iloc[0])
    return (f"Spearman rho between metric and quality loss, {n} designs "
            "(results/metric_correlation.csv)"), md(piv.reset_index())


def t_pvalues():
    c = read("metric_correlation.csv")
    piv = c.pivot(index="metric", columns="quality_loss", values="p_value")
    order = ["ER", "MED", "MSE", "EDmax", "EDmax_freq", "wMED"]
    piv = piv.reindex(order).map(lambda x: f"{x:.1e}")
    piv.columns = [x.replace(":", " ") for x in piv.columns]
    return "p-values of selected correlations (results/metric_correlation.csv)", md(piv.reset_index())


def t_pairs():
    m = read("matched_pairs.csv")
    rows = [{"pair": r.pair, "design": r.design, "MED": f"{r.MED:.1f}", "EDmax": r.EDmax,
             "pairs at EDmax": r.EDmax_count, "ER (%)": f"{100 * r.ER:.1f}",
             "camera PSNR loss (dB)": "0 (identical)" if r["camera_gauss5:PSNR_loss"] <= -999
             else f"{r['camera_gauss5:PSNR_loss']:.1f}",
             "CNN top-1 drop (pp)": f"{r['mnist_cnn:top1_drop_pp']:+.2f}"}
            for _, r in m.iterrows()]
    return ("Each baseline next to our N with the closest MED (results/matched_pairs.csv); "
            "PSNR loss = -PSNR"), md(pd.DataFrame(rows))


def t_workload():
    w = read("workload_metrics.csv")
    piv = w.pivot(index="design", columns="workload", values="wMED")
    piv = piv.reindex(list(dict.fromkeys(w.design))).map(lambda x: f"{x:.3g}")
    used = w.groupby("workload").pairs_used_pct.first().map(lambda x: f"{x:.1f} %")
    return ("Workload-aware mean error wMED (results/workload_metrics.csv). Share of the 65,536 "
            "(a, b) pairs each workload uses: " + ", ".join(f"{k} {v}" for k, v in used.items())), \
        md(piv.reset_index())


TABLES = {"verification": t_verification, "errors": t_errors, "ppa_mul": t_ppa_mul,
          "ppa_baselines": t_ppa_baselines, "array": t_array, "image": t_image, "cnn": t_cnn,
          "correlation": t_correlation, "pvalues": t_pvalues, "pairs": t_pairs,
          "workload": t_workload}


def main():
    names = sys.argv[1:] or list(TABLES)
    for n in names:
        title, table = TABLES[n]()
        print(f"### {title}\n\n{table}\n")


if __name__ == "__main__":
    main()
