# PROGRESS: AxMAC

Status key: [ ] todo · [~] in progress · [x] done (check passed and output shown)

Update this file at the end of every session: tick the boxes and add a line to the session
log. If you fall behind, cut the CNN evaluation first. Keep the baseline comparison.

## Milestone 1: due Thu Oct 1, 2026

**Day 1: Sat Sep 26**
- [x] M1.1 Set up the environment and record tool versions; create the repo and Makefile skeleton; copy REFERENCES.md into docs/; first commit
- [x] M1.2 Flow smoke test: synthesize mul_behav (a*b) to sky130_fd_sc_hd and print its area
- [x] M1.3 Python model, LUT and pytest tests: the closed-form MED check, plus N=4 giving EDmax 49 occurring for 256 pairs

**Day 2: Sun Sep 27**
- [x] M1.4 Write gen_dadda.py, generate rtl/dadda8_reduce.sv, and add the full/half adders
- [x] M1.5 Build approx_mul.sv with the N mask; the exhaustive testbench passes for N=0 (65,536/65,536)

**Day 3: Mon Sep 28**
- [x] M1.6 `make test` passes for all N = 0..12
- [x] M1.7 metrics.py: MED, NMED, MRED, ER, ME, and EDmax with its frequency, written to a CSV and plotted
- [x] M1.8 (optional) Functional cross-check against TruMD at N = 2, 4, 6, 8 (done in M2.6: TruMD trunc-k = our N = k on all 65,536 pairs)

**Day 4: Tue Sep 29**
- [x] M1.9 Build mul_wrap; run Yosys and OpenSTA to get area and delay for N=0 and for a*b
- [x] M1.10 Power for N=0: simulation VCD, then OpenSTA report_power (random stimulus, documented)

**Day 5: Wed Sep 30**
- [x] M1.11 (if time) Area and delay for every N, written to results/ppa_mul.csv
- [x] M1.12 Write the README and docs/milestone1.md
- [x] M1.13 Clean-clone check (`make test metrics synth power` works), then push to Git (clean-clone check passed, results byte-identical; pushed to https://github.com/durgeshjanapareddy9-collab/PSAMAC-Accelerator-, main = 4780d32)

**What we submit on Oct 1** (contents of docs/milestone1.md):
1. Repo link, containing the RTL, testbenches, model and scripts.
2. A description of the multiplier and the N knob.
3. The verification result: RTL matches the model on all 65,536 inputs for N = 0..12.
4. The error-metrics table and plot (MED, NMED, MRED, ER, and max error with its frequency).
5. Area, delay and power for the exact N=0 design, compared against a*b.
6. The plan for Milestone 2.

## Milestone 2: due Thu Oct 29, 2026

**Week 1 (Oct 2-8): MAC, array and first image results**
- [x] M2.1 mac_pe.sv and its testbench (bit-exact against the LUT golden model)
- [x] M2.2 New 4x4 systolic_array.sv, built from scratch, and its testbench (random matrices; all tested N pass)
- [x] M2.3 image_conv.py: PSNR and SSIM across our N sweep (software only), including the Rather et al. cameraman case (Gaussian noise with variance 0.01, 3x3 averaging)

**Week 2 (Oct 9-15): PPA sweep and baselines**
- [x] M2.4 Realistic image stimulus, and the power flow working with it
- [x] M2.5 Area, delay and power vs N for the multiplier and the array (results/ppa_*.csv)
- [x] M2.6 Fetch 3-4 EvoApproxLib designs and TruMD; verify each against its own model; build their LUTs. Optional: add a functional-only reconstruction of Rather et al. Design 1 (N=4 +12)
- [x] M2.7 Run the baselines through the identical flow (results/ppa_baselines.csv)

**Week 3 (Oct 16-22): Application accuracy and analysis**
- [x] M2.8 Image evaluation for every design: our N points plus the baselines
- [x] M2.9 CNN: train on MNIST, quantize, run LUT-based inference, and get top-1 accuracy for every design
- [x] M2.10 analysis.py: test the Rather et al. observation of which metric (mean error, or EDmax and its frequency) best predicts quality loss
- [x] M2.11 An image patch run through the RTL array matches the Python model

**Week 4 (Oct 23-28): Figures, report and release**
- [x] M2.12 Figures:
  - error vs N;
  - area, delay and power vs N;
  - PSNR and SSIM vs N;
  - top-1 accuracy vs N;
  - accuracy vs area/delay/power curves (with the baselines marked);
  - the metric-correlation chart.
- [x] M2.13 docs/report.md: motivation, design, verification, results, baselines, analysis, limitations, prior work (cited from docs/REFERENCES.md), and a comparison with Rather et al.
- [~] M2.14 `make all` in a clean clone reproduces every figure; final README; tag v1.0 (README done; clean-copy `make all` passed: 259/259 regenerated results and figures byte-identical; commit + tag v1.0 by the team)
- [ ] M2.15 (optional) LibreLane layout for N=0 and one approximate N (not started)

**What we submit on Oct 29:**
1. Repo (tagged v1.0), with a README that reproduces every figure from scratch.
2. The final report, including:
   - the accuracy-vs-area/delay/power curves;
   - image PSNR/SSIM;
   - the CNN top-1 accuracy change;
   - the baseline comparison;
   - the error-metric analysis.
3. Slides, if the course asks for them.

## After the course (not due Oct 29)
- [ ] A1 Add a second approximation family (partial-product perforation, or Mitchell logarithmic)
- [ ] A2 Integrate the multiplier into the RV32IM core, with CSR-controlled exact/approximate mode
- [ ] A3 Spike co-simulation showing that exact mode is bit-accurate

## Team
| Area | Owner |
|---|---|
| Flow (Yosys/OpenSTA/power) and baselines | |
| RTL and testbenches | |
| Python model, metrics, image evaluation, CNN | |
| Report and README | everyone |

## Session log
<!-- One line per session: date · who · what was done · checks run · next step · blockers -->
2026-09-27 · Durgesh + Claude · M1.1-M1.12 done (OpenSTA 3.1.0 built from source; Yosys 0.33; sky130A tt corner) · make test 13/13 N PASS, 42 pytest pass, gate-level power sim 2008/2008 correct, clean clone reproduces results byte-identical · next: push to GitHub (M1.13), then M2.1 · open: ABC delay non-monotonic in N (decide timing-driven mapping for M2.5); CLAUDE.local.md not committed (team decision)
2026-10-10 · Durgesh + Claude · Detailed Yosys/OpenSTA results added to docs/milestone1.md §5: constraints moved to flow/mul_wrap.sdc; sta.tcl now reports setup/hold slack, WNS/TNS, violations, max/min path (cell vs net delay), check_setup; synth.tcl runs Yosys `check` (needed read_liberty -lib, otherwise 16 false "undriven" outputs) · make synth power sweep rerun: netlists byte-identical, all earlier CSV columns unchanged, 0 setup/hold violations and 0 check problems for N=0..12 and a*b · next: M2.1 · open: M2.1-M2.3 (week 1) not started yet
2026-10-10 · Durgesh + Claude · Added flow/report.py + `make report` → results/flow_report.md (Yosys/OpenSTA/power results, summary computed from the CSVs); md_tables.py split into per-table functions (output unchanged) · make report twice gives byte-identical files · next: M2.1
2026-10-22 · Durgesh + Claude · M2.1-M2.13 done: mac_pe + 4x4 systolic_array (test-mac 13/13 N PASS, incl. stalls, wrap, im2col tiles = image_conv), baselines (4 EvoApproxLib vendored MIT, TruMD fetched at pinned commit; 8/8 RTL vs own model PASS; TruMD = our N=k), power with image stimulus for every N + array + baselines (every gate-level netlist output vs LUT matches), image eval, CNN (float 98.42 %, int8 98.45 %, N=8 -0.22 pp), analysis (ER best uniform metric for images, EDmax-freq negative; workload-aware wMED rho>=0.99), figures, docs/report.md, docs/project_guide.md, README · open: timing-driven ABC tried and dropped (Yosys 0.33 &nf ignores -D; documented as limitation); array power uses 50 tiles (gate-level sim slow) · next: team commits + tags v1.0, optional M2.15 layout
2026-10-22 · Durgesh + Claude · M2.14 check: copy of the working tree (tracked + new files), all generated results/figures deleted, `make venv venv-cnn all JOBS=13` -> exit 0 in 19 min; 259 generated files byte-identical to the working tree, 0 different · next: team reviews docs, commits, tags v1.0
2026-10-22 · Durgesh + Claude · Deliverables audit: `make figures` now also runs `make diagrams` (yosys_* figures regenerate byte-identical); M1.13 confirmed pushed (origin/main = HEAD 4780d32), repo link added to milestone1.md and README · open: commit + push M2 work, tag v1.0
