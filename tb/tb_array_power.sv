// tb_array_power.sv -- drive the synthesized systolic_array NETLIST with a
// schedule file and record switching activity (VCD) for power analysis.
//
// Same schedule format as tb_systolic_array.sv ("C clear en a0..a3 b0..b3",
// "D" = dump all 16 sums), written by flow/stimulus.py from image tiles. The
// sums are dumped to DUMP_FILE so flow/power.py can check that the gate-level
// netlist still computes the right answers.
//
// The netlist has no parameters left (N was fixed at synthesis), so this
// testbench instantiates it without any.

`timescale 1ns/1ps

module tb_array_power;
    parameter real   CLK_NS    = 10.0;
    parameter int    WARMUP    = 8;              // cycles before the VCD starts
    parameter string STIM_FILE = "build/stim_array_image.txt";
    parameter string DUMP_FILE = "build/array_power_dump.txt";
    parameter string VCD_FILE  = "build/array_power.vcd";
    localparam int ROWS = 4, COLS = 4, ACC_W = 24;

    logic                       clk = 0, clear = 0, en = 0;
    logic [ROWS*8-1:0]          a_left = 0;
    logic [COLS*8-1:0]          b_top = 0;
    logic [ROWS*COLS*ACC_W-1:0] c_out;

    systolic_array dut (.clk(clk), .clear(clear), .en(en),
                        .a_left(a_left), .b_top(b_top), .c_out(c_out));

    always #(CLK_NS / 2) clk = ~clk;

    integer fin, fout, r, cl, e, v, i, cycles = 0, recorded = 0;
    logic   more = 1;
    string  op;

    initial begin
        fin  = $fopen(STIM_FILE, "r");
        fout = $fopen(DUMP_FILE, "w");
        if (fin == 0 || fout == 0) begin
            $display("ERROR: cannot open %s or %s", STIM_FILE, DUMP_FILE);
            $finish;
        end
        @(negedge clk);
        while (more) begin
            r = $fscanf(fin, "%s", op);
            if (r != 1) begin
                more = 0;
            end else if (op == "C") begin
                if (cycles == WARMUP) begin
                    // Depth 2: top level + every cell's pins (see tb_power.sv).
                    $dumpfile(VCD_FILE);
                    $dumpvars(2, dut);
                end
                r = $fscanf(fin, "%d %d", cl, e);
                clear = cl[0];
                en    = e[0];
                for (i = 0; i < ROWS; i++) begin
                    r = $fscanf(fin, "%d", v);
                    a_left[i*8 +: 8] = v[7:0];
                end
                for (i = 0; i < COLS; i++) begin
                    r = $fscanf(fin, "%d", v);
                    b_top[i*8 +: 8] = v[7:0];
                end
                @(negedge clk);
                cycles++;
                if (cycles > WARMUP) recorded++;
            end else if (op == "D") begin
                for (i = 0; i < ROWS*COLS; i++)
                    $fwrite(fout, "%0d%s", c_out[i*ACC_W +: ACC_W], (i == ROWS*COLS-1) ? "\n" : " ");
            end
        end
        $dumpflush;
        $fclose(fin);
        $fclose(fout);
        $display("tb_array_power: recorded %0d cycles at %0.1f ns", recorded, CLK_NS);
        $finish;
    end
endmodule
