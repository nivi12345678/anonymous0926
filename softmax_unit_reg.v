//============================================================================
// softmax_unit_reg.v -- registered-I/O measurement version, MODES 0-3.
// 0=SWAT 1=STABLE 2=DDF 3=SG-DDF(gate=12.0).  Q8.8 in, Q8.8 weights out.
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
    reg signed [W-1:0] exp_lut [0:255];
    initial begin
        if (MODE==0) $readmemh("exp_lut_swat.mem",   exp_lut);
        else         $readmemh("exp_lut_stable.mem", exp_lut);
    end
    localparam signed [31:0] LO_SWAT=-1024, LO_STABLE=-2048;
    localparam integer SPAN_SHIFT=11;
    localparam signed [31:0] GATE=3072;   // 12.0 in Q8.8 (gate-sweep result)

    // ---- input registers ----
    reg [WIN*W-1:0] scores_r;  reg start_r;
    always @(posedge clk) begin
        if (!rst_n) begin scores_r<=0; start_r<=0; end
        else begin scores_r<=scores_flat; start_r<=start; end
    end

    // ---- combinational datapath on registered input ----
    integer k;
    reg signed [31:0] sc[0:WIN-1], maxv, minv, thr, sh, lo, idx32;
    reg signed [31:0] ev[0:WIN-1], sumv;
    reg [31:0] recip, prod;  reg keepf;
    reg [WIN*W-1:0] w_comb;
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
                lo=(MODE==0)?LO_SWAT:LO_STABLE;
                sh=(MODE==0)?sc[k]:(sc[k]-maxv);
                idx32=((sh-lo)*255)>>>SPAN_SHIFT;
                if (idx32<0) idx32=0;
                if (idx32>255) idx32=255;
                ev[k]=exp_lut[idx32[7:0]];
            end else ev[k]=0;
            sumv=sumv+ev[k];
        end
        if (sumv==0) sumv=1;
        recip = 65536/sumv;
        for (k=0;k<WIN;k=k+1) begin
            prod = ev[k]*recip;
            w_comb[k*W +: W] = prod[23:8];
        end
    end

    // ---- output registers ----
    always @(posedge clk) begin
        if (!rst_n) begin weights_flat<=0; done<=0; end
        else begin weights_flat<=w_comb; done<=start_r; end
    end
endmodule
