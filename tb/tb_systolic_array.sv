// tb_systolic_array.sv -- play a cycle-by-cycle schedule into systolic_array
// and dump the 16 accumulators whenever the schedule says so.
//
//   input : build/array_vectors_N<N>.txt (python -m model.systolic gen-array <N>)
//           "C clear en a0 a1 a2 a3 b0 b1 b2 b3"   apply for one clock cycle
//           "D"                                     dump all of c_out
//   output: results/array_dump_N<N>.txt, one line of 16 sums per "D"
//           C[0][0] C[0][1] ... C[3][3]  (checked by model.systolic check-array)
//
// The schedule (made in Python) contains the whole protocol: clearing,
// skew-free data input, random stall cycles and the drain cycles. See
// rtl/systolic_array.sv and model/systolic.py.

`timescale 1ns/1ps

module tb_systolic_array;
    parameter int N     = 0;
    parameter int ROWS  = 4;
    parameter int COLS  = 4;
    parameter int ACC_W = 24;

    logic                       clk = 0, clear = 0, en = 0;
    logic [ROWS*8-1:0]          a_left = 0;
    logic [COLS*8-1:0]          b_top = 0;
    logic [ROWS*COLS*ACC_W-1:0] c_out;

    systolic_array #(.N(N), .ROWS(ROWS), .COLS(COLS), .ACC_W(ACC_W)) dut (
        .clk(clk), .clear(clear), .en(en), .a_left(a_left), .b_top(b_top), .c_out(c_out));

    always #5 clk = ~clk;

    integer fin, fout, r, cl, e, v, i, cycles = 0, dumps = 0;
    logic   more = 1;                    // 0 once the vector file is used up
    string  op;

    initial begin
        fin  = $fopen($sformatf("build/array_vectors_N%0d.txt", N), "r");
        fout = $fopen($sformatf("results/array_dump_N%0d.txt", N), "w");
        if (fin == 0 || fout == 0) begin
            $display("ERROR: cannot open the vector or dump file");
            $finish;
        end
        @(negedge clk);
        while (more) begin
            r = $fscanf(fin, "%s", op);
            if (r != 1) begin
                more = 0;                    // end of file
            end else if (op == "C") begin
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
                @(negedge clk);              // one rising edge
                cycles++;
            end else if (op == "D") begin
                for (i = 0; i < ROWS*COLS; i++)
                    $fwrite(fout, "%0d%s", c_out[i*ACC_W +: ACC_W], (i == ROWS*COLS-1) ? "\n" : " ");
                dumps++;
            end
        end
        $fclose(fin);
        $fclose(fout);
        $display("tb_systolic_array: N=%0d, %0d cycles, %0d results dumped", N, cycles, dumps);
        $finish;
    end
endmodule
