"""Area and delay of multipliers: Yosys synthesis + OpenSTA timing.

For each design (a multiplier module and a value of N) this script
  1. synthesizes rtl/mul_wrap.sv around that multiplier (flow/synth.tcl),
  2. reads the cell counts from Yosys and the cell areas from the liberty file,
     and splits the area into flip-flops (the wrapper) and the multiplier,
  3. runs OpenSTA (flow/sta.tcl) for the minimum clock period, setup and
     hold slack (WNS/TNS, violations) and the max/min path delays,
  4. records the Yosys structural check and warning counts,
and writes one CSV row per design.

Area and delay come from the same netlist. Area is also reported for the
multiplier alone (total minus the 32 wrapper flip-flops: 8 + 8 inputs, 16 outputs), which is the number
we compare across N.

Design names (--design, repeatable):
  approx_mul:<N>   our multiplier with truncation depth N, in mul_wrap
  mul_behav:0      plain a*b, in mul_wrap
  <baseline>       a baseline from model/designs.py (e.g. evo_1446, trumd_4), in mul_wrap
  array:<N>        the 4x4 systolic array (rtl/systolic_array.sv) with every PE at depth N.
                   The array has its own registers, so it is timed as it is, without
                   mul_wrap; its area is split into flip-flops and logic.

Usage (normally via the Makefile, which passes the tool paths):
  python flow/ppa.py --out results/ppa_n0.csv --design approx_mul:0 --design mul_behav:0 \
      --lib <lib> --synth-lib <filtered lib> --yosys yosys --sta <sta>
"""

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SYNTH_OUT = REPO / "results" / "synth"
sys.path.insert(0, str(REPO))
from model import designs   # noqa: E402  (registry: RTL files of every design)


def resolve(spec):
    """Turn a --design spec into what synthesis needs.

    Returns dict: tag (file prefix), design (CSV name), module, N, top, srcs, defines.
    """
    if spec.startswith("array:"):
        n = int(spec.split(":")[1])
        return {"tag": f"array_N{n}", "design": f"array_N{n}", "module": "systolic_array",
                "N": n, "top": "systolic_array",
                "srcs": designs.MUL_SRCS + ["rtl/mac_pe.sv", "rtl/systolic_array.sv"],
                "defines": []}
    if ":" in spec:
        mul, n = spec.split(":")
        n = int(n)
        srcs = designs.MUL_SRCS if mul == "approx_mul" else ["rtl/mul_behav.sv"]
        return {"tag": f"wrap_{mul}_N{n}", "module": mul, "N": n, "top": "mul_wrap",
                "design": f"{mul}_N{n}" if mul == "approx_mul" else mul,
                "srcs": srcs + ["rtl/mul_wrap.sv"], "defines": [f"MUL={mul}"]}
    d = designs.get(spec)
    if not d.has_rtl:
        sys.exit(f"ERROR: {spec} has no RTL (software-only model)")
    return {"tag": f"wrap_{d.name}", "design": d.name, "module": d.module, "N": d.N,
            "top": "mul_wrap", "srcs": d.srcs + ["rtl/mul_wrap.sv"],
            "defines": [f"MUL={d.module}"] + d.defines}


def liberty_areas(lib_path):
    """Map cell name -> area (um^2), read from the liberty file."""
    areas, cell = {}, None
    cell_re = re.compile(r'^\s*cell\s*\(\s*"?([^")\s]+)"?\s*\)')
    area_re = re.compile(r'^\s*area\s*:\s*([0-9.eE+-]+)')
    with open(lib_path) as f:
        for line in f:
            m = cell_re.match(line)
            if m:
                cell = m.group(1)
                continue
            m = area_re.match(line)
            if m and cell and cell not in areas:
                areas[cell] = float(m.group(1))
    return areas


def parse_stat(stat_path):
    """Return (cell counts dict, total chip area) from a Yosys `stat` report."""
    counts, area = {}, None
    for line in Path(stat_path).read_text().splitlines():
        m = re.match(r"^\s+(sky130_\S+)\s+(\d+)\s*$", line)
        if m:
            counts[m.group(1)] = int(m.group(2))
        m = re.search(r"Chip area for module .*:\s*([0-9.]+)", line)
        if m:
            area = float(m.group(1))
    if area is None:
        raise RuntimeError(f"no 'Chip area' line in {stat_path}")
    return counts, area


def is_flop(cell):
    """True for sky130 flip-flop cells (dfxtp, dfrtp, ...)."""
    return re.match(r"sky130_fd_sc_hd__(s?df|edf)", cell) is not None


def run(cmd, env, log_path):
    """Run a tool, save its full output to a log, stop with the error if it fails."""
    res = subprocess.run(cmd, cwd=REPO, env={**os.environ, **env},
                         capture_output=True, text=True)
    Path(log_path).write_text(res.stdout + res.stderr)
    if res.returncode != 0:
        tail = "\n".join((res.stdout + res.stderr).splitlines()[-20:])
        sys.exit(f"ERROR: {' '.join(cmd)} failed (see {log_path}):\n{tail}")
    return res.stdout


def ppa_one(spec, args, areas):
    r = resolve(spec)
    tag, n = r["tag"], r["N"]
    SYNTH_OUT.mkdir(parents=True, exist_ok=True)
    env = {"TOP": r["top"], "N": str(n), "SRCS": " ".join(r["srcs"]), "LIB": args.synth_lib,
           "OUT": str(SYNTH_OUT), "DEFINES": " ".join(f"-D{x}" for x in r["defines"])}
    env["TAG"] = tag

    # 1. Synthesis.
    run([args.yosys, "-q", "-c", "flow/synth.tcl"], env, SYNTH_OUT / f"{tag}.log")

    # 2. Area split: flip-flops (wrapper) vs everything else (multiplier).
    counts, total_area = parse_stat(SYNTH_OUT / f"{tag}_stat.txt")
    missing = [c for c in counts if c not in areas]
    if missing:
        sys.exit(f"ERROR: cells not found in liberty: {missing}")
    flop_area = sum(areas[c] * k for c, k in counts.items() if is_flop(c))
    flop_cells = sum(k for c, k in counts.items() if is_flop(c))
    all_cells = sum(counts.values())

    # 3. Timing.
    out = run([args.sta, "-no_init", "-no_splash", "-exit", "flow/sta.tcl"],
              {"NETLIST": str(SYNTH_OUT / f"{tag}.v"), "LIB": args.lib,
               "OUT": str(SYNTH_OUT / tag), "TOP": r["top"]},
              SYNTH_OUT / f"{tag}_sta.log")
    sta = dict(re.findall(r"^([A-Z0-9_]+) (-?[0-9.]+)$", out, re.M))
    keys = ["MIN_PERIOD_NS", "SETUP_SLACK_NS", "SETUP_WNS_NS", "SETUP_TNS_NS",
            "SETUP_VIOLATIONS", "HOLD_SLACK_NS", "HOLD_SLACK_R2R_NS", "HOLD_WNS_NS",
            "HOLD_TNS_NS", "HOLD_VIOLATIONS", "ENDPOINTS", "MAX_PATH_ARRIVAL_NS",
            "MAX_PATH_CELL_DELAY_NS", "MAX_PATH_NET_DELAY_NS", "MAX_PATH_CELLS",
            "MIN_PATH_ARRIVAL_NS", "MIN_PATH_CELL_DELAY_NS", "MIN_PATH_NET_DELAY_NS"]
    missing = [k for k in keys if k not in sta]
    if missing:
        sys.exit(f"ERROR: {missing} not in OpenSTA output for {tag}:\n{out}")
    min_period = float(sta["MIN_PERIOD_NS"])
    # check_setup prints nothing when the constraints have no problems.
    sta_issues = len((SYNTH_OUT / f"{tag}.checks.txt").read_text().splitlines())

    # 4. Yosys structural check and warnings.
    check = (SYNTH_OUT / f"{tag}_check.txt").read_text()
    m = re.search(r"Found and reported (\d+) problems", check)
    if not m:
        sys.exit(f"ERROR: no problem count in {tag}_check.txt:\n{check}")
    warnings = [l for l in (SYNTH_OUT / f"{tag}.log").read_text().splitlines()
                if l.startswith("Warning:")]
    # Warnings about cells the liberty parser cannot model (scan/enable flops,
    # which this design never uses) vs anything else.
    lib_warnings = [l for l in warnings if "in pin attribute of cell" in l]

    if r["top"] == "mul_wrap":
        area = {"area_mul_um2": round(total_area - flop_area, 4),
                "cells_mul": all_cells - flop_cells,
                "area_wrap_um2": round(total_area, 4)}
    else:   # the array: report all of it, split into logic and flip-flops
        area = {"area_total_um2": round(total_area, 4),
                "area_logic_um2": round(total_area - flop_area, 4),
                "area_flops_um2": round(flop_area, 4),
                "cells_total": all_cells, "cells_logic": all_cells - flop_cells}
    return {
        "design": r["design"],
        "module": r["module"],
        "N": n,
        **area,
        "flops": flop_cells,
        "min_period_ns": min_period,
        "fmax_mhz": round(1000.0 / min_period, 2),
        # Yosys
        "cell_types": len(counts),
        "yosys_check_problems": int(m.group(1)),
        "yosys_warnings": len(warnings),
        "yosys_liberty_warnings": len(lib_warnings),
        # OpenSTA, slacks at the 10 ns clock in flow/mul_wrap.sdc
        "setup_slack_ns": float(sta["SETUP_SLACK_NS"]),
        "setup_wns_ns": float(sta["SETUP_WNS_NS"]),
        "setup_tns_ns": float(sta["SETUP_TNS_NS"]),
        "setup_violations": int(sta["SETUP_VIOLATIONS"]),
        "hold_slack_ns": float(sta["HOLD_SLACK_NS"]),
        "hold_slack_r2r_ns": float(sta["HOLD_SLACK_R2R_NS"]),
        "hold_wns_ns": float(sta["HOLD_WNS_NS"]),
        "hold_tns_ns": float(sta["HOLD_TNS_NS"]),
        "hold_violations": int(sta["HOLD_VIOLATIONS"]),
        "endpoints": int(sta["ENDPOINTS"]),
        "sta_check_issues": sta_issues,
        "max_path_ns": float(sta["MAX_PATH_ARRIVAL_NS"]),
        "max_path_cell_ns": float(sta["MAX_PATH_CELL_DELAY_NS"]),
        "max_path_net_ns": float(sta["MAX_PATH_NET_DELAY_NS"]),
        "max_path_cells": int(sta["MAX_PATH_CELLS"]),
        "min_path_ns": float(sta["MIN_PATH_ARRIVAL_NS"]),
        "min_path_cell_ns": float(sta["MIN_PATH_CELL_DELAY_NS"]),
        "min_path_net_ns": float(sta["MIN_PATH_NET_DELAY_NS"]),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--design", action="append", required=True,
                   help="module:N, e.g. approx_mul:4 (repeatable)")
    p.add_argument("--out", required=True)
    p.add_argument("--lib", required=True)
    p.add_argument("--synth-lib", required=True)
    p.add_argument("--yosys", default="yosys")
    p.add_argument("--sta", required=True)
    args = p.parse_args()

    areas = liberty_areas(args.lib)
    rows = []
    for d in args.design:
        row = ppa_one(d, args, areas)
        print(", ".join(f"{k}={v}" for k, v in row.items()))
        rows.append(row)

    fields = list(dict.fromkeys(k for row in rows for k in row))   # union, in order
    with open(REPO / args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
