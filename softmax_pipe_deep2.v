//============================================================================
// softmax_pipe_deep.v -- DEEP-PIPELINED unit (MODEs 0-3): pipef front-end
// + the single-cycle divider REPLACED by a 17-stage unrolled restoring
// divider, so no stage contains more than a few gate-levels of logic.
//
//   S1 tree max/min -> S2 thr/sub/>>3/exp -> S3 sum -> D0..D16 divider
//   -> SM multiply   (latency = 21 cycles, throughput = 1 window/cycle)
//
// The divider computes floor(65536/sum) BIT-EXACTLY (same truncating
// division as "17'd65536 / s3_sum"), one quotient bit per stage:
//   numerator N = 65536 = bit16..bit0 = 1,0,0,...,0
//   stage i: rem = (rem<<1) | N[16-i]; if rem>=div {rem-=div; q=1}
// The 8 ev values ride alongside in a delay line so they arrive at the
// multiplier in the same cycle as their own quotient.
//============================================================================
module softmax_pipe_deep2 #(
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

    integer k, i;
   // ---------- S1a: latch + Level 1 & 2 of max/min tournament ----------
    reg [WIN*W-1:0]   s1a_scores;  reg s1a_valid;
    reg signed [15:0] a [0:7];
    reg signed [15:0] mx4 [0:3], mn4 [0:3];
    reg signed [15:0] mx2 [0:1], mn2 [0:1];
    always @(posedge clk) begin
        if (!rst_n) s1a_valid<=0;
        else begin
            s1a_valid  <= in_valid;
            s1a_scores <= scores_flat;
            for (k=0;k<8;k=k+1) a[k] = $signed(scores_flat[k*W +: W]);
            for (k=0;k<4;k=k+1) begin
                mx4[k] = (a[2*k]>a[2*k+1]) ? a[2*k] : a[2*k+1];
                mn4[k] = (a[2*k]<a[2*k+1]) ? a[2*k] : a[2*k+1];
            end
            for (k=0;k<2;k=k+1) begin
                mx2[k] <= (mx4[2*k]>mx4[2*k+1]) ? mx4[2*k] : mx4[2*k+1];
                mn2[k] <= (mn4[2*k]<mn4[2*k+1]) ? mn4[2*k] : mn4[2*k+1];
            end
        end
    end
    // ---------- S1b: Level 3 (final round) + re-sync scores ----------
    reg [WIN*W-1:0]   s1_scores;  reg s1_valid;
    reg signed [15:0] s1_max, s1_min;
    always @(posedge clk) begin
        if (!rst_n) s1_valid<=0;
        else begin
            s1_valid  <= s1a_valid;
            s1_scores <= s1a_scores;      // <-- delayed one extra tick to match
            s1_max    <= (mx2[0]>mx2[1]) ? mx2[0] : mx2[1];
            s1_min    <= (mn2[0]<mn2[1]) ? mn2[0] : mn2[1];
        end
    end
    // ---------- S2: thr, subtract, >>3 index, exp lookup (identical) ----
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
    // ---------- S3: sum (identical) ----------
    reg [WIN*W-1:0] s3_evf;  reg [19:0] s3_sum;  reg s3_valid;
    reg [19:0] acc;
    always @(posedge clk) begin
        if (!rst_n) s3_valid<=0;
        else begin
            s3_valid <= s2_valid;  acc = 0;
            for (k=0;k<WIN;k=k+1) begin
                s3_evf[k*W +: W] <= s2_ev[k];
                acc = acc + {4'd0,s2_ev[k]};
            end
            s3_sum <= (acc==0) ? 20'd1 : acc;
        end
    end
    // ---------- D0..D16: 17-stage unrolled restoring divider ----------
    // Each stage: tiny shift + 21-bit compare/subtract. Payload per stage:
    // remainder, partial quotient, divisor, delayed ev bank, valid.
    localparam integer DS = 17;
    reg [20:0]        dv_rem [0:DS-1];
    reg [16:0]        dv_q   [0:DS-1];
    reg [19:0]        dv_div [0:DS-1];
    reg [WIN*W-1:0]   dv_evf [0:DS-1];
    reg [DS-1:0]      dv_v;
    reg [20:0] trem;
    always @(posedge clk) begin
        if (!rst_n) dv_v <= {DS{1'b0}};
        else begin
            // stage 0: consumes S3. rem starts 0; shift in N[16]=1 -> rem=1
            trem = 21'd1;
            if (trem >= {1'b0,s3_sum}) begin
                dv_rem[0] <= trem - {1'b0,s3_sum};  dv_q[0] <= 17'd1;
            end else begin
                dv_rem[0] <= trem;                   dv_q[0] <= 17'd0;
            end
            dv_div[0] <= s3_sum;
            dv_evf[0] <= s3_evf;
            dv_v[0]   <= s3_valid;
            // stages 1..16: shift in N[16-i]=0
            for (i=1;i<DS;i=i+1) begin
                trem = {dv_rem[i-1][19:0], 1'b0};
                if (trem >= {1'b0,dv_div[i-1]}) begin
                    dv_rem[i] <= trem - {1'b0,dv_div[i-1]};
                    dv_q[i]   <= {dv_q[i-1][15:0], 1'b1};
                end else begin
                    dv_rem[i] <= trem;
                    dv_q[i]   <= {dv_q[i-1][15:0], 1'b0};
                end
                dv_div[i] <= dv_div[i-1];
                dv_evf[i] <= dv_evf[i-1];
                dv_v[i]   <= dv_v[i-1];
            end
        end
    end
    // ---------- SM-A / SM-B: TWO-STAGE multiply (timing fix) --------------
    // The old single stage pushed data through the ENTIRE DSP multiplier
    // combinationally (~7 ns room = the Fmax wall at ~124 MHz). Splitting
    // into two registered stages lets Vivado retime the registers INTO the
    // DSP's built-in MREG/PREG, cutting the room roughly in half.
    // Latency 21 -> 23 cycles; throughput still 1 window/cycle; the product
    // and slice are unchanged, so results stay BIT-IDENTICAL.
    reg signed [32:0] prod_r [0:WIN-1];   // SM-A: registered raw products
    reg               smv;
    always @(posedge clk) begin
        if (!rst_n) begin smv<=0; out_valid<=0; end
        else begin
            // SM-A: multiply, register the full product (absorbed into MREG)
            smv <= dv_v[DS-1];
            for (k=0;k<WIN;k=k+1)
                prod_r[k] <= $signed(dv_evf[DS-1][k*W +: W]) * $signed({1'b0,dv_q[DS-1]});
            // SM-B: slice the registered product (register absorbed as PREG)
            out_valid <= smv;
            for (k=0;k<WIN;k=k+1)
                weights_flat[k*W +: W] <= prod_r[k][23:8];
        end
    end
endmodule
