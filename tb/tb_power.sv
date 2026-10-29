// tb_power.sv -- drive the synthesized mul_wrap netlist to record switching
// activity for power analysis.
//
// This simulates the GATE-LEVEL netlist from Yosys (sky130 cells), not the
// RTL, so the VCD contains the activity of every internal wire. OpenSTA then
// reads the VCD (read_vcd) and computes power from it.
//
// Stimulus, a new pair every clock cycle, either
//   STIM_FILE = ""      : uniform random 8-bit a and b (Milestone 1), or
//   STIM_FILE = <file>  : pairs "a b" read from a file, one per line. flow/stimulus.py
//                         writes image data there: a = pixel, b = filter weight, in
//                         the order one PE of the array consumes them (Milestone 2).
//
// If DUMP_FILE is set, every output is also written as "a b p" (inputs and the
// product that came out two cycles later), so flow/power.py can check the
// gate-level netlist of ANY design, exact or approximate, against its LUT.
//
// Timeline:
//   - clock period CLK_NS (10 ns = 100 MHz), same for every design
//   - WARMUP cycles to fill the pipeline, then CYCLES cycles recorded in the VCD
//
// If CHECK_EXACT = 1 (exact designs only), the testbench also checks every
// output against a*b. Inputs change on the falling clock edge. mul_wrap has
// two register stages, so inputs applied at falling edge t are captured into
// a_q/b_q at the next rising edge, their product reaches p_out at the rising
// edge after that, and we read it at falling edge t+2.

`timescale 1ns/1ps

module tb_power;
    parameter int    CYCLES      = 5000;       // cycles recorded for power
    parameter int    WARMUP      = 10;         // cycles before recording starts
    parameter real   CLK_NS      = 10.0;       // clock period in ns
    parameter int    SEED        = 1;          // random seed (fixed => repeatable)
    parameter int    CHECK_EXACT = 0;          // 1: compare p_out with a*b
    parameter string VCD_FILE    = "build/power.vcd";
    parameter string STIM_FILE   = "";         // "" = uniform random
    parameter string DUMP_FILE   = "";         // "" = no output dump

    logic        clk = 0;
    logic [7:0]  a_in = 0, b_in = 0;
    logic [15:0] p_out;

    // Device under test: the synthesized netlist (module name mul_wrap).
    mul_wrap dut (.clk(clk), .a_in(a_in), .b_in(b_in), .p_out(p_out));

    always #(CLK_NS / 2) clk = ~clk;

    // exp_prev = product of the inputs applied one falling edge ago. When we
    // check at edge t+2, it holds the product of the inputs from edge t.
    logic [15:0] exp_prev = 0;
    logic [7:0]  a_prev = 0, b_prev = 0;       // the inputs that exp_prev belongs to
    integer seed = SEED;
    integer errors = 0, checked = 0;
    integer cyc;
    integer fstim = 0, fdump = 0, va, vb, r;

    initial begin
        if (STIM_FILE != "") begin
            fstim = $fopen(STIM_FILE, "r");
            if (fstim == 0) begin
                $display("ERROR: cannot open %s", STIM_FILE);
                $finish;
            end
        end
        if (DUMP_FILE != "") fdump = $fopen(DUMP_FILE, "w");
        for (cyc = 0; cyc < WARMUP + CYCLES; cyc++) begin
            if (cyc == WARMUP) begin
                // Start recording. Depth 2 = the netlist's top level plus the
                // ports of every cell instance (e.g. _217_/A, _217_/X). OpenSTA
                // matches VCD signals to cell pins by these names; with depth 1
                // only the top-level ports would match.
                $dumpfile(VCD_FILE);
                $dumpvars(2, dut);
            end
            @(negedge clk);                         // change inputs away from the clock edge
            if (CHECK_EXACT && cyc >= 2) begin      // pipeline is full after 2 cycles
                checked++;
                if (p_out !== exp_prev) begin
                    errors++;
                    if (errors <= 5)
                        $display("MISMATCH cycle %0d: p_out=%0d expected=%0d", cyc, p_out, exp_prev);
                end
            end
            if (fdump != 0 && cyc >= 2)
                $fdisplay(fdump, "%0d %0d %0d", a_prev, b_prev, p_out);
            exp_prev = a_in * b_in;                 // product of the inputs applied at cyc-1
            a_prev   = a_in;
            b_prev   = b_in;
            if (fstim != 0) begin
                r = $fscanf(fstim, "%d %d\n", va, vb);
                if (r != 2) begin
                    $display("ERROR: %s ran out of stimulus at cycle %0d", STIM_FILE, cyc);
                    $finish;
                end
                a_in = va[7:0];
                b_in = vb[7:0];
            end else begin
                a_in = $random(seed);
                b_in = $random(seed);
            end
        end
        if (fdump != 0) $fclose(fdump);
        $dumpflush;
        if (CHECK_EXACT)
            $display("tb_power: gate-level check %0d/%0d outputs correct",
                     checked - errors, checked);
        $display("tb_power: recorded %0d cycles at %0.1f ns", CYCLES, CLK_NS);
        $finish;
    end
endmodule
