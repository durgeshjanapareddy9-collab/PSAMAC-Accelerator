# flow/power.tcl -- power of a synthesized mul_wrap netlist from simulation activity.
#
# Run with OpenSTA (normally from flow/power.py):
#   NETLIST=... LIB=... VCD=... SCOPE=tb_power/dut CLK_NS=10 OUT=... \
#       sta -no_init -exit flow/power.tcl
#
# Method:
#   1. The gate-level netlist was simulated (tb/tb_power.sv) and every net's
#      switching was recorded in a VCD file.
#   2. read_vcd attaches that activity (toggle rate and time-at-1) to every
#      cell pin (the VCD includes cell pins, see tb/tb_power.sv).
#   3. report_power adds up, per cell: internal power (inside the cell),
#      switching power (charging the capacitance it drives) and leakage.
#   Corner: the liberty file in LIB (tt, 25 C, 1.80 V). Pre-layout: no wire
#   capacitance, ideal clock. Zero-delay simulation: glitches are not counted.

read_liberty $::env(LIB)
read_verilog $::env(NETLIST)
# Top module: mul_wrap, or systolic_array for the array (env TOP).
link_design  [expr {[info exists ::env(TOP)] ? $::env(TOP) : "mul_wrap"}]

# Same constraints as timing; the clock period comes from CLK_NS.
read_sdc     [file join [file dirname [info script]] mul_wrap.sdc]

read_vcd -scope $::env(SCOPE) $::env(VCD)

# How many cell pins got their activity from the simulation ("vcd") and how
# many did not ("unannotated"). flow/power.py requires unannotated = 0.
report_activity_annotation

report_power -digits 6 > $::env(OUT).power.txt
report_power -digits 6
