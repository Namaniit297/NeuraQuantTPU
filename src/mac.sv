module pe(
    input  wire clk,
    input  wire active,
    input  wire signed [7:0] datain,
    input  wire signed [7:0] win,
    input  wire signed [15:0] sumin,
    input  wire wwrite,

    output reg  signed [15:0] maccout,
    output reg  signed [7:0]  dataout,
    output reg  signed [7:0]  wout,
    output reg                wwriteout,
    output reg                activeout
);

    // Internal weight register
    reg signed [7:0] weight;

    // === Pipeline Stage Registers ===
    // Stage 1
    reg signed [7:0] data_s1, weight_s1;
    reg signed [15:0] sum_s1;
    // Stage 2
    reg signed [15:0] product_s2;
    reg signed [15:0] sum_s2;
    // Stage 3
    reg signed [15:0] acc_s3;

    // === Weight Logic ===
    always @(posedge clk) begin
        if (wwrite) begin
            weight <= win;
        end
        wout       <= weight;
        wwriteout  <= wwrite;
    end

    // === MAC Pipeline ===
    always @(posedge clk) begin
        if (active) begin
            // Stage 1: register inputs
            data_s1   <= datain;
            weight_s1 <= weight;
            sum_s1    <= sumin;

            // Stage 2: multiply
            product_s2 <= data_s1 * weight_s1;
            sum_s2     <= sum_s1;

            // Stage 3: accumulate
            acc_s3     <= product_s2 + sum_s2;

            // Output stage
            maccout    <= acc_s3;
            dataout    <= datain;
            activeout  <= active;
        end else begin
            // Pipeline stall: hold previous outputs
            maccout   <= maccout;
            dataout   <= dataout;
            activeout <= activeout;
        end
    end

endmodule
