struct VSInput {
    float2 pos : POSITION;
    float2 uv : TEXCOORD;
};

struct VSOutput {
    float4 pos : SV_POSITION;
    float2 uv : TEXCOORD;
};

VSOutput vs_main(VSInput input)
{
    VSOutput o;
    o.pos = float4(input.pos, 0.0, 1.0);
    o.uv = input.uv;
    return o;
}

Texture2D input_tex : register(t0);
SamplerState samp : register(s0);

float4 ps_main(VSOutput input) : SV_TARGET
{
    float4 color = input_tex.Sample(samp, input.uv);
    return float4(1.0 - color.rgb, color.a);
}
