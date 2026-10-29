// systolic_array.sv -- ROWS x COLS output-stationary systolic array of mac_pe.
//
// What it computes:  C = A x B, with A of size ROWS x K and B of size K x COLS.
//   C[i][j] = sum over k of A[i][k] * B[k][j]   (each product from approx_mul #(N))
// K can be any length: the array just keeps accumulating while data flows in.
//
// "Output-stationary": PE(i, j) owns C[i][j]. Its accumulator stays put while
// the A values flow left-to-right along row i and the B values flow
// top-to-bottom along column j. Every PE does one MAC per cycle, so the 16 PEs
// do 16 multiplies per cycle with no extra memory traffic -- each input byte
// is reused ROWS or COLS times as it travels across the array. This reuse is
// why accelerators such as the TPU [Jouppi et al. 2017] use systolic arrays.
//
// Data enters unskewed: at cycle k the caller puts column k of A on a_left
// (one byte per row) and row k of B on b_top (one byte per column).
//
// Skewing (done here, inside the array): row i of A is delayed by i cycles
// and column j of B by j cycles, by small shift registers. Because each PE
// also delays its inputs by one cycle, PE(i, j) then sees, at cycle t,
//     A[i][t - i - j]   and   B[t - i - j][j]
// i.e. the SAME k = t - i - j on both inputs -- exactly the pair it must
// multiply. Without the skew, PE(i, j) would multiply mismatched elements.
//
//   cycle t:      b_top[0]  b_top[1]  b_top[2]  b_top[3]
//                    |      [1 reg]   [2 regs]  [3 regs]      (column skew)
//   a_left[0] ----> PE00 --> PE01 --> PE02 --> PE03
//   a_left[1] [1]-> PE10 --> PE11 --> PE12 --> PE13
//   a_left[2] [2]-> PE20 --> ...                              (row skew)
//   a_left[3] [3]-> PE30 --> ...
//
// Timing: the last pair (k = K-1) reaches the far corner PE(ROWS-1, COLS-1)
// at cycle K-1 + (ROWS-1) + (COLS-1). So the caller feeds K cycles of data,
// then ROWS+COLS-2 more cycles of zeros (0 x anything = 0 adds nothing), and
// then every C[i][j] is ready on c_out.
//
// Starting a new matrix: hold clear = 1 (with en = 1 and zero inputs) for
// ROWS + COLS cycles. This empties the skew and pass-through registers (fills
// them with zeros) and sets every accumulator to 0.

module systolic_array #(
    parameter int N     = 0,           // truncation depth used by EVERY PE
    parameter int ROWS  = 4,
    parameter int COLS  = 4,
    parameter int ACC_W = 24           // accumulator width of each PE
) (
    input  logic                         clk,
    input  logic                         clear,   // clears all accumulators
    input  logic                         en,      // 1 = run, 0 = stall everything
    input  logic [ROWS*8-1:0]            a_left,  // a_left[i*8 +: 8] = A[i][k] at cycle k
    input  logic [COLS*8-1:0]            b_top,   // b_top[j*8 +: 8]  = B[k][j] at cycle k
    output logic [ROWS*COLS*ACC_W-1:0]   c_out    // c_out[(i*COLS+j)*ACC_W +: ACC_W] = C[i][j]
);
    // Wires between PEs, kept as flat packed vectors (a simple subset that
    // both Icarus and Yosys accept). Byte (i*(COLS+1) + j) of a_h is the A
    // value entering PE(i, j) from the left; byte (i*COLS + j) of b_v is the
    // B value entering PE(i, j) from above. The extra last position in each
    // row/column (j = COLS or i = ROWS) is what leaves the array, unused.
    logic [ROWS*(COLS+1)*8-1:0] a_h;
    logic [(ROWS+1)*COLS*8-1:0] b_v;

    genvar i, j;
    generate
        // ---- Row skew: row i of A goes through i registers ----
        for (i = 0; i < ROWS; i++) begin : g_skew_a
            if (i == 0) begin : g_none
                assign a_h[0 +: 8] = a_left[0 +: 8];
            end else begin : g_delay
                // Shift register of i bytes: new byte enters at the bottom,
                // the oldest byte (top) leaves i cycles later.
                logic [i*8-1:0] sr;
                always_ff @(posedge clk) begin
                    if (en) sr <= (i == 1) ? a_left[i*8 +: 8]
                                           : {sr, a_left[i*8 +: 8]};   // shift up one byte
                end
                assign a_h[(i*(COLS+1))*8 +: 8] = sr[(i-1)*8 +: 8];
            end
        end

        // ---- Column skew: column j of B goes through j registers ----
        for (j = 0; j < COLS; j++) begin : g_skew_b
            if (j == 0) begin : g_none
                assign b_v[0 +: 8] = b_top[0 +: 8];
            end else begin : g_delay
                logic [j*8-1:0] sr;
                always_ff @(posedge clk) begin
                    if (en) sr <= (j == 1) ? b_top[j*8 +: 8]
                                           : {sr, b_top[j*8 +: 8]};
                end
                assign b_v[j*8 +: 8] = sr[(j-1)*8 +: 8];
            end
        end

        // ---- The grid of processing elements ----
        for (i = 0; i < ROWS; i++) begin : g_row
            for (j = 0; j < COLS; j++) begin : g_col
                mac_pe #(.N(N), .ACC_W(ACC_W)) u_pe (
                    .clk   (clk),
                    .clear (clear),
                    .en    (en),
                    .a_in  (a_h[(i*(COLS+1) + j)*8 +: 8]),
                    .b_in  (b_v[(i*COLS + j)*8 +: 8]),
                    .a_out (a_h[(i*(COLS+1) + j + 1)*8 +: 8]),   // to the PE on the right
                    .b_out (b_v[((i+1)*COLS + j)*8 +: 8]),       // to the PE below
                    .acc   (c_out[(i*COLS + j)*ACC_W +: ACC_W])
                );
            end
        end
    endgenerate
endmodule
