//============================================================================
// softmax_pipe_fast.v -- IMPROVED-TIMING pipelined unit (MODEs 0-3).
// vs softmax_pipelined.v:
//  (1) S1 max/min via 3-level TOURNAMENT TREE (log2 depth) instead of a
//      7-deep serial compare chain -> shorter S1 critical path.
//  (2) S2 index = (sh-lo)>>>3 (multiply removed).
// Same interface: in_valid -> 5 cycles -> out_valid. Module name kept
// 'softmax_pipelined' so tb_pipelined.v works unchanged.
//============================================================================
module softmax_pipelined #(
    parameter integer WIN=8, W=16, MODE=1, ETA_SHIFT=1
)(
    input  wire              clk,
    input  wire              rst_n,
    input  wire              in_valid,
    input  wire [WIN*W-1:0]  scores_flat,
    output reg  [WIN*W-1:0]  weights_flat,
    output reg               out_valid
);
    reg signed [15:0] exp_lut [0:255];
    initial begin
        if (MODE==0) $readmemh("exp_lut_swat.mem",   exp_lut);
        else         $readmemh("exp_lut_stable.mem", exp_lut);
    end
    localparam signed [16:0] LO_SWAT=-17'sd1024, LO_STABLE=-17'sd2048;

    integer k;
    // ---------- S1: latch + TREE max/min ----------
    reg [WIN*W-1:0]   s1_scores;  reg s1_valid;
    reg signed [15:0] s1_max, s1_min;
    reg signed [15:0] a [0:7];
    reg signed [15:0] mx4 [0:3], mn4 [0:3], mx2 [0:1], mn2 [0:1];
    always @(posedge clk) begin
        if (!rst_n) s1_valid<=0;
        else begin
            s1_valid  <= in_valid;
            s1_scores <= scores_flat;
            for (k=0;k<8;k=k+1) a[k] = $signed(scores_flat[k*W +: W]);
            for (k=0;k<4;k=k+1) begin              // level 1
                mx4[k] = (a[2*k]>a[2*k+1]) ? a[2*k] : a[2*k+1];
                mn4[k] = (a[2*k]<a[2*k+1]) ? a[2*k] : a[2*k+1];
            end
            for (k=0;k<2;k=k+1) begin              // level 2
                mx2[k] = (mx4[2*k]>mx4[2*k+1]) ? mx4[2*k] : mx4[2*k+1];
                mn2[k] = (mn4[2*k]<mn4[2*k+1]) ? mn4[2*k] : mn4[2*k+1];
            end
            s1_max <= (mx2[0]>mx2[1]) ? mx2[0] : mx2[1];   // level 3
            s1_min <= (mn2[0]<mn2[1]) ? mn2[0] : mn2[1];
        end
    end
    // ---------- S2: thr, subtract, >>3 index, exp lookup ----------
    reg signed [15:0] s2_ev [0:WIN-1];  reg s2_valid;
    reg signed [15:0] thr;  reg signed [15:0] sck;
    reg signed [16:0] sh, d;  reg [7:0] idx;  reg keepf;
    always @(posedge clk) begin
        if (!rst_n) s2_valid<=0;
        else begin
            s2_valid <= s1_valid;
            thr = s1_max - ((s1_max - s1_min) >>> ETA_SHIFT);
            for (k=0;k<WIN;k=k+1) begin
                sck = $signed(s1_scores[k*W +: W]);
                if (MODE==2)      keepf = (sck >= thr);
                else if (MODE==3) keepf = ((s1_max-s1_min) < 16'sd3072) ? 1'b1 : (sck >= thr);
                else              keepf = 1'b1;
                if (keepf) begin
                    sh = (MODE==0)? {sck[15],sck} : ({sck[15],sck} - {s1_max[15],s1_max});
                    d  = sh - ((MODE==0)? LO_SWAT : LO_STABLE);
                    if (d[16])              idx = 8'd0;
                    else if (d > 17'sd2047) idx = 8'd255;
                    else                    idx = d[10:3];
                    s2_ev[k] <= exp_lut[idx];
                end else s2_ev[k] <= 0;
            end
        end
    end
    // ---------- S3: sum ----------
    reg signed [15:0] s3_ev [0:WIN-1];  reg [19:0] s3_sum;  reg s3_valid;
    reg [19:0] acc;
    always @(posedge clk) begin
        if (!rst_n) s3_valid<=0;
        else begin
            s3_valid <= s2_valid;  acc = 0;
            for (k=0;k<WIN;k=k+1) begin s3_ev[k]<=s2_ev[k]; acc = acc + {4'd0,s2_ev[k]}; end
            s3_sum <= (acc==0) ? 20'd1 : acc;
        end
    end
    // ---------- S4: reciprocal ----------
    reg signed [15:0] s4_ev [0:WIN-1];  reg [16:0] s4_recip;  reg s4_valid;
    always @(posedge clk) begin
        if (!rst_n) s4_valid<=0;
        else begin
            s4_valid <= s3_valid;
            for (k=0;k<WIN;k=k+1) s4_ev[k]<=s3_ev[k];
            s4_recip <= 17'd65536 / s3_sum;
        end
    end
    // ---------- S5: multiply ----------
    reg [32:0] prod;
    always @(posedge clk) begin
        if (!rst_n) out_valid<=0;
        else begin
            out_valid <= s4_valid;
            for (k=0;k<WIN;k=k+1) begin
                prod = s4_ev[k]*s4_recip;
                weights_flat[k*W +: W] <= prod[23:8];
            end
        end
    end
endmodule
