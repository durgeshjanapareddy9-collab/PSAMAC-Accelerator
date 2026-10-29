# Makefile for AxMAC -- one command per project step.
#
#   make venv       create .venv/ and install Python packages
#   make versions   record tool versions in results/tool_versions.txt
#   make gen        regenerate rtl/dadda8_reduce.sv from scripts/gen_dadda.py
#   make test       RTL vs Python model, all 65,536 input pairs, N = 0..12
#   make metrics    error metrics (MED, NMED, MRED, ER, ME, EDmax)
#   make synth      Yosys synthesis to sky130 (area)
#   make power      OpenSTA power from simulation activity
#   make test-mac   MAC PE and 4x4 systolic array vs the golden model, N = 0..12
#   make sweep      area/delay/power for every N
#   make array      area/delay/power of the 4x4 systolic array for every N
#   make report     results/flow_report.md: all Yosys/OpenSTA/power results in one file
#   make diagrams   Graphviz circuit diagrams from Yosys `show` -> figures/yosys_*
#   make baselines  EvoApproxLib / TruMD: verify, metrics, same flow
#   make eval       image convolution quality (PSNR, SSIM), every design
#   make cnn        MNIST CNN top-1 accuracy, every design (needs make venv-cnn)
#   make analysis   which error metric predicts quality loss
#   make figures    regenerate every figure in figures/ (plots + circuit diagrams)
#   make all        everything above, in order (about 20 minutes on a 16-core laptop)
#
# Paths to tools and the liberty file live in flow/config.mk.

include flow/config.mk

# Truncation depths covered by the sweep.
NS := 0 1 2 3 4 5 6 7 8 9 10 11 12

# Designs simulated in parallel by the power flow (the array is slow).
JOBS ?= 8

.PHONY: venv venv-cnn versions gen test test-model test-mac metrics synth synth-one power sweep array report diagrams baselines eval cnn \
        analysis figures all clean

# ---- Setup -------------------------------------------------------------------

venv: $(VENV)/bin/activate

# "python -m pip" instead of .venv/bin/pip: this folder's path contains a
# space, which breaks the #! line of the pip script.
$(VENV)/bin/activate: requirements.txt
	python3 -m venv $(VENV)
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	touch $@

# torch and torchvision (CPU builds), only for the CNN step.
venv-cnn: $(VENV)/.cnn-installed
$(VENV)/.cnn-installed: requirements-cnn.txt | venv
	$(PYTHON) -m pip install -r requirements-cnn.txt
	touch $@

versions: venv
	bash scripts/tool_versions.sh "$(YOSYS)" "$(STA)" "$(IVERILOG)" "$(VERILATOR)" "$(PYTHON)" "$(LIB)" "$(OPENSTA_SRC)"

# Regenerate the Dadda reduction tree (the output file is committed).
gen: venv
	$(PYTHON) scripts/gen_dadda.py

# ---- Project steps (filled in as the tasks in PROGRESS.md are done) ----------
# Each placeholder fails on purpose, so nobody mistakes it for a passing step.

# RTL files of the approximate multiplier.
MUL_SRCS := rtl/full_adder.sv rtl/half_adder.sv rtl/dadda8_reduce.sv rtl/approx_mul.sv

# Full test: Python model tests, then RTL vs model for every N.
test: test-model $(addprefix sim-N,$(NS))
	$(PYTHON) -m model.check_rtl $(NS)

# Simulate one N and write results/rtl_dump_N<n>.txt:  make sim-N4
sim-N%: venv
	@mkdir -p build results
	$(IVERILOG) -g2012 -Wall -Wno-timescale -P tb_approx_mul_exhaustive.N=$* -o build/tb_mul_N$*.vvp \
	    tb/tb_approx_mul_exhaustive.sv $(MUL_SRCS)
	$(VVP) -n build/tb_mul_N$*.vvp

# Simulate and check a single N:  make check-N4
check-N%: sim-N%
	$(PYTHON) -m model.check_rtl $*

# MAC PE and 4x4 systolic array vs the LUT golden model, every N (M2.1, M2.2, M2.11).
MAC_SRCS := $(MUL_SRCS) rtl/mac_pe.sv rtl/systolic_array.sv
test-mac: $(addprefix pe-N,$(NS)) $(addprefix array-N,$(NS))

# One N:  make pe-N4  /  make array-N4
pe-N%: venv
	@mkdir -p build results
	$(PYTHON) -m model.systolic gen-pe $*
	$(IVERILOG) -g2012 -Wall -Wno-timescale -P tb_mac_pe.N=$* -o build/tb_pe_N$*.vvp tb/tb_mac_pe.sv $(MAC_SRCS)
	$(VVP) -n build/tb_pe_N$*.vvp
	$(PYTHON) -m model.systolic check-pe $*

array-N%: venv
	@mkdir -p build results
	$(PYTHON) -m model.systolic gen-array $*
	$(IVERILOG) -g2012 -Wall -Wno-timescale -P tb_systolic_array.N=$* -o build/tb_array_N$*.vvp \
	    tb/tb_systolic_array.sv $(MAC_SRCS)
	$(VVP) -n build/tb_array_N$*.vvp
	$(PYTHON) -m model.systolic check-array $*

# Python model unit tests, and build the cached LUTs in results/.
test-model: venv
	$(PYTHON) -m pytest -q model/tests
	$(PYTHON) -m model.approx_mul

# Error metrics for every N -> results/error_metrics.csv, figures/error_vs_N.png
metrics: venv
	$(PYTHON) -m model.metrics

# Liberty copy without "do not use" cells, for Yosys only (see flow/make_synth_lib.py).
SYNTH_LIB := build/sky130_fd_sc_hd_synth.lib
$(SYNTH_LIB): $(LIB) flow/make_synth_lib.py
	@mkdir -p build
	python3 flow/make_synth_lib.py $(LIB) $@

# Area and delay (Yosys + OpenSTA) for the exact Dadda multiplier (N=0) and
# the reference a*b, both inside the registered wrapper -> results/ppa_n0.csv
PPA = $(PYTHON) flow/ppa.py --lib $(LIB) --synth-lib $(SYNTH_LIB) --yosys $(YOSYS) --sta $(STA)
synth: venv $(SYNTH_LIB)
	$(PPA) --out results/ppa_n0.csv --design approx_mul:0 --design mul_behav:0

# Synthesize one design:  make synth-one TOP=<module> N=<n> SRCS="<files>"
SYNTH_OUT := results/synth
synth-one: $(SYNTH_LIB)
	@mkdir -p $(SYNTH_OUT)
	TOP=$(TOP) N=$(N) SRCS="$(SRCS)" LIB=$(SYNTH_LIB) OUT=$(SYNTH_OUT) \
	    $(YOSYS) -q -l $(SYNTH_OUT)/$(TOP)_N$(N).log -c flow/synth.tcl

# Power (gate-level simulation + OpenSTA) for N=0 and a*b -> results/power_n0.csv
# Uses the netlists from `make synth`.
power: synth
	$(PYTHON) flow/power.py --out results/power_n0.csv --design approx_mul:0 --design mul_behav:0 \
	    --lib $(LIB) --sta $(STA) --cell-models $(CELL_MODELS) --iverilog $(IVERILOG) --vvp $(VVP)

# Area, delay and power for every N, plus a*b -> results/ppa_mul.csv (M2.5)
# Power is measured twice: uniform random inputs and image data (M2.4);
# the details are in results/power_mul_<stimulus>.csv.
POWER = $(PYTHON) flow/power.py --lib $(LIB) --sta $(STA) --cell-models $(CELL_MODELS) \
        --iverilog $(IVERILOG) --vvp $(VVP)
MUL_DESIGNS := $(foreach n,$(NS),--design approx_mul:$(n)) --design mul_behav:0
sweep: venv $(SYNTH_LIB)
	$(PPA) --out results/ppa_mul.csv $(MUL_DESIGNS)
	$(POWER) --stimulus random --out results/power_mul_random.csv --merge-into results/ppa_mul.csv $(MUL_DESIGNS)
	$(POWER) --stimulus image --out results/power_mul_image.csv --merge-into results/ppa_mul.csv $(MUL_DESIGNS)

# The 4x4 systolic array at every N: area, delay, power (image tiles) -> results/ppa_array.csv
ARRAY_DESIGNS := $(foreach n,$(NS),--design array:$(n))
array: venv $(SYNTH_LIB)
	$(PPA) --out results/ppa_array.csv $(ARRAY_DESIGNS)
	$(POWER) --stimulus image --jobs $(JOBS) --out results/power_array_image.csv --merge-into results/ppa_array.csv $(ARRAY_DESIGNS)

# One readable report of the flow results -> results/flow_report.md
# (reads the CSVs and reports written by `make synth power sweep`).
report: venv
	$(PYTHON) flow/report.py --out results/flow_report.md

# Circuit diagrams (Yosys `show` + Graphviz dot) -> figures/yosys_*.svg/.png
# The gate-level views use the netlists from `make sweep`.
diagrams: venv
	$(PYTHON) flow/diagrams.py --lib $(LIB) --yosys $(YOSYS) --dot $(DOT)

# Baselines (M2.6, M2.7): fetch TruMD, verify every baseline's RTL against its own
# model on all 65,536 pairs and build its LUT, error metrics, then the identical
# flow (same wrapper, clock, corner and stimulus) -> results/ppa_baselines.csv
BASE_DESIGNS := --design evo_1446 --design evo_GS2 --design evo_185Q --design evo_JQQ \
                --design trumd_2 --design trumd_4 --design trumd_6 --design trumd_8
baselines: venv $(SYNTH_LIB)
	$(PYTHON) baselines/verify.py --iverilog $(IVERILOG) --vvp $(VVP)
	$(PYTHON) -m model.metrics --baselines
	$(PPA) --out results/ppa_baselines.csv $(BASE_DESIGNS)
	$(POWER) --stimulus random --out results/power_baselines_random.csv --merge-into results/ppa_baselines.csv $(BASE_DESIGNS)
	$(POWER) --stimulus image --out results/power_baselines_image.csv --merge-into results/ppa_baselines.csv $(BASE_DESIGNS)

# Image filtering quality (PSNR, SSIM) for every design -> results/image_quality.csv (M2.3, M2.8)
eval: venv
	$(PYTHON) -m eval.image_conv

# MNIST CNN top-1 for every design -> results/cnn_accuracy.csv (M2.9).
# Needs torch + torchvision (CPU): make venv-cnn
cnn: venv-cnn
	$(PYTHON) -m eval.cnn

# Which error metric predicts quality loss -> results/metric_correlation.csv (M2.10)
analysis: venv-cnn
	$(PYTHON) -m eval.analysis

# Every figure in figures/ (M2.12): the plots from the CSVs in results/, plus
# the Yosys circuit diagrams (which read the netlists written by `make sweep`).
figures: venv diagrams
	$(PYTHON) -m eval.figures

# Everything, in dependency order.
all: test test-mac metrics synth power sweep array report baselines eval cnn analysis figures

# Remove generated files (keeps .venv/ and the trained CNN weights).
clean:
	rm -rf results/*.npy results/rtl_dump_*.txt results/pe_dump_*.txt results/array_dump_*.txt \
	       results/*.vcd results/*.saif results/synth build
