//============================================================================
// softmax_unit_opt_reg.v -- REDUCED-LUT version (registered I/O, MODEs 0-3)
// LUT-reduction methods applied vs softmax_unit_reg.v:
//  (1) index math:  ((sh-lo)*255)>>>11  -->  (sh-lo)>>>3
//      (span=2048, 256 entries -> 8 units/entry: the multiply disappears)
//  (2) right-sized widths: every signal cut to its true range instead of
//      blanket 32-bit (comparators/adders shrink accordingly)
//      sc,maxv,minv,thr: 16b | sh,d: 17b | idx:8b | ev:16b | sumv:17b
//      recip:17b | prod:32b
// Same interface & modes as softmax_unit_reg.v. Q8.8 in, Q8.8 weights out.
//============================================================================
module softmax_unit #(
    parameter integer WIN=8, W=16, MODE=1, ETA_SHIFT=1
)(
    input  wire              clk,
    input  wire              rst_n,
    input  wire              start,
    input  wire [WIN*W-1:0]  scores_flat,
    output reg  [WIN*W-1:0]  weights_flat,
    output reg               done
);
    (* rom_style="distributed", dont_touch="true" *)
    reg [15:0] exp_lut [0:255];
    initial begin
        if (MODE==0) $readmemh("exp_lut_swat.mem",   exp_lut);
        else         $readmemh("exp_lut_stable.mem", exp_lut);
    end
    localparam signed [16:0] LO_SWAT=-17'sd1024, LO_STABLE=-17'sd2048;
    localparam signed [15:0] GATE=16'sd3072;          // 12.0 in Q8.8

    reg [WIN*W-1:0] scores_r;  reg start_r;
    always @(posedge clk) begin
        if (!rst_n) begin scores_r<=0; start_r<=0; end
        else begin scores_r<=scores_flat; start_r<=start; end
    end

    integer k;
    reg signed [15:0] sc [0:WIN-1];
    reg signed [15:0] maxv, minv, thr;
    reg signed [16:0] sh, d;
    reg        [7:0]  idx;
    reg        [15:0] ev [0:WIN-1];
    reg        [16:0] sumv;
    reg        [16:0] recip;
    reg        [31:0] prod;
    reg               keepf;
    reg [WIN*W-1:0]   w_comb;

    always @(*) begin
        for (k=0;k<WIN;k=k+1) sc[k]=$signed(scores_r[k*W +: W]);
        maxv=sc[0]; minv=sc[0];
        for (k=1;k<WIN;k=k+1) begin
            if (sc[k]>maxv) maxv=sc[k];
            if (sc[k]<minv) minv=sc[k];
        end
        thr = maxv - ((maxv-minv)>>>ETA_SHIFT);
        sumv=0;
        for (k=0;k<WIN;k=k+1) begin
            if (MODE==2)      keepf=(sc[k]>=thr);
            else if (MODE==3) keepf=((maxv-minv)<GATE)?1'b1:(sc[k]>=thr);
            else              keepf=1'b1;
            if (keepf) begin
                sh = (MODE==0)? {sc[k][15],sc[k]} : ({sc[k][15],sc[k]} - {maxv[15],maxv});
                d  = sh - ((MODE==0)? LO_SWAT : LO_STABLE);
                if (d[16])            idx = 8'd0;          // negative -> clip low
                else if (d > 17'sd2047) idx = 8'd255;      // beyond span -> clip
                else                  idx = d[10:3];       // (d >> 3)
                ev[k] = exp_lut[idx];
            end else ev[k]=0;
            sumv = sumv + ev[k];
        end
        if (sumv==0) sumv=1;
        recip = 17'd65536 / sumv;
        for (k=0;k<WIN;k=k+1) begin
            prod = ev[k]*recip;
            w_comb[k*W +: W] = prod[23:8];
        end
    end

    always @(posedge clk) begin
        if (!rst_n) begin weights_flat<=0; done<=0; end
        else begin weights_flat<=w_comb; done<=start_r; end
    end
endmodule
