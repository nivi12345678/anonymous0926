//============================================================================
// softmax_unit_soft_reg.v -- SOFTERMAX-STYLE exp unit (registered, MODEs 0-3)
// LUT reduction: the 256x16 exp ROM is replaced by
//     e^x = 2^(x*log2e):  one constant multiply (x369>>8), then
//     2^y = frac_lut32[f] shifted by the integer part of y.
// ROM shrinks 256x16 -> 32x16 (frac_lut32.mem). Also uses right-sized widths.
// Cite: Softermax (Stevens et al., 2021) for the base-2 exponent idea.
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
    reg [15:0] frac_lut [0:31];
    initial $readmemh("frac_lut32.mem", frac_lut);
    localparam signed [15:0] GATE=16'sd768;

    reg [WIN*W-1:0] scores_r;  reg start_r;
    always @(posedge clk) begin
        if (!rst_n) begin scores_r<=0; start_r<=0; end
        else begin scores_r<=scores_flat; start_r<=start; end
    end

    integer k;
    reg signed [15:0] sc [0:WIN-1];
    reg signed [15:0] maxv, minv, thr;
    reg signed [16:0] sh;
    reg signed [25:0] y;        // sh*369 >> 8
    reg signed [17:0] Ipart;
    reg        [7:0]  f;
    reg        [15:0] t;
    reg        [15:0] ev [0:WIN-1];
    reg        [19:0] sumv;
    reg        [16:0] recip;
    reg        [32:0] prod;
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
                y  = (sh*369) >>> 8;          // x * log2(e) in Q8.8
                Ipart = y >>> 8;              // floor integer part
                f  = y[7:0];                  // fractional bits (mod-256)
                t  = frac_lut[f[7:3]];
                if (Ipart >= 0)
                    ev[k] = (Ipart > 7) ? 16'hFFFF :
                            ((t << Ipart) > 33'h0FFFF ? 16'hFFFF : (t << Ipart));
                else
                    ev[k] = (-Ipart < 16) ? (t >> (-Ipart)) : 16'd0;
            end else ev[k]=0;
            sumv = sumv + ev[k];
        end
        if (sumv==0) sumv=1;
        recip = (sumv > 20'd65536) ? 17'd0 : (17'd65536 / sumv);
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
