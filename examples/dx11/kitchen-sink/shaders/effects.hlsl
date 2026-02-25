// Kitchen-sink HLSL shaders: grayscale (compute), tint (render), blend (compute).
//
// All four entry points are compiled independently from this single file.
// Register declarations at file scope are fine because only the ones
// referenced by each entry point are used during compilation.

// -----------------------------------------------------------------------
// Shared uniform struct
// -----------------------------------------------------------------------

cbuffer EffectParams : register(b0)
{
    float grayscale_amount;
    float tint_hue;
    float tint_saturation;
    float blend_amount;
};

// -----------------------------------------------------------------------
// HSV -> RGB helper
// -----------------------------------------------------------------------

float3 hsv_to_rgb(float h, float s, float v)
{
    float c = v * s;
    float hp = h * 6.0;
    float x = c * (1.0 - abs(fmod(hp, 2.0) - 1.0));
    float3 rgb;
    if      (hp < 1.0) rgb = float3(c, x, 0);
    else if (hp < 2.0) rgb = float3(x, c, 0);
    else if (hp < 3.0) rgb = float3(0, c, x);
    else if (hp < 4.0) rgb = float3(0, x, c);
    else if (hp < 5.0) rgb = float3(x, 0, c);
    else               rgb = float3(c, 0, x);
    float m = v - c;
    return rgb + m;
}

// -----------------------------------------------------------------------
// Pass 1: Grayscale (compute)
// -----------------------------------------------------------------------

Texture2D<float4>   gs_input  : register(t0);
RWTexture2D<float4> gs_output : register(u0);

[numthreads(16, 16, 1)]
void grayscale_cs(uint3 id : SV_DispatchThreadID)
{
    uint w, h;
    gs_input.GetDimensions(w, h);
    if (id.x >= w || id.y >= h) return;

    float4 color = gs_input[id.xy];
    float lum = dot(color.rgb, float3(0.2126, 0.7152, 0.0722));
    float3 gray = float3(lum, lum, lum);
    float3 result = lerp(color.rgb, gray, grayscale_amount);
    gs_output[id.xy] = float4(result, color.a);
}

// -----------------------------------------------------------------------
// Pass 2: Tint (render -- vertex + pixel shader)
// -----------------------------------------------------------------------

struct VSInput
{
    float2 pos : POSITION;
    float2 uv  : TEXCOORD;
};

struct VSOutput
{
    float4 pos : SV_POSITION;
    float2 uv  : TEXCOORD;
};

VSOutput tint_vs(VSInput input)
{
    VSOutput o;
    o.pos = float4(input.pos, 0.0, 1.0);
    o.uv  = input.uv;
    return o;
}

Texture2D    tint_input : register(t0);
SamplerState samp       : register(s0);

float4 tint_ps(VSOutput input) : SV_TARGET
{
    float4 color = tint_input.Sample(samp, input.uv);

    // Generate tint colour from hue + saturation
    float3 tint = hsv_to_rgb(tint_hue, tint_saturation, 1.0);

    // Multiply blend: overlay the tint onto the grayscaled image
    float3 tinted = color.rgb * tint;

    // Mix between untinted and tinted based on saturation strength
    float3 result = lerp(color.rgb, tinted, tint_saturation);
    return float4(result, color.a);
}

// -----------------------------------------------------------------------
// Pass 3: Blend original with processed (compute)
// -----------------------------------------------------------------------

Texture2D<float4>   blend_original  : register(t0);
Texture2D<float4>   blend_processed : register(t1);
RWTexture2D<float4> blend_output    : register(u0);

[numthreads(16, 16, 1)]
void blend_cs(uint3 id : SV_DispatchThreadID)
{
    uint w, h;
    blend_original.GetDimensions(w, h);
    if (id.x >= w || id.y >= h) return;

    float4 orig = blend_original[id.xy];
    float4 proc = blend_processed[id.xy];
    blend_output[id.xy] = lerp(orig, proc, blend_amount);
}
