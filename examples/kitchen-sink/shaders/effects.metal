#include <metal_stdlib>
using namespace metal;

// -----------------------------------------------------------------------
// Shared uniform struct
// -----------------------------------------------------------------------

struct EffectParams {
    float grayscale_amount; // 0..1
    float tint_hue;         // 0..1 (maps to 0..360 degrees)
    float tint_saturation;  // 0..1
    float blend;            // 0..1  mix(processed, original)
};

// -----------------------------------------------------------------------
// HSV <-> RGB helpers
// -----------------------------------------------------------------------

static float3 hsv_to_rgb(float h, float s, float v) {
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
// Pass 1: grayscale (compute)
// -----------------------------------------------------------------------

kernel void grayscale(
    texture2d<float, access::read> input [[texture(0)]],
    texture2d<float, access::write> output [[texture(1)]],
    constant EffectParams& params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint w = input.get_width();
    uint h = input.get_height();
    if (gid.x >= w || gid.y >= h) return;

    float4 color = input.read(gid);
    float lum = dot(color.rgb, float3(0.2126, 0.7152, 0.0722));
    float3 gray = float3(lum);
    float3 result = mix(color.rgb, gray, params.grayscale_amount);
    output.write(float4(result, color.a), gid);
}

// -----------------------------------------------------------------------
// Pass 2: tint (render pipeline — fullscreen quad)
// -----------------------------------------------------------------------

struct VertexOut {
    float4 position [[position]];
    float2 texcoord;
};

vertex VertexOut tint_vertex(
    const device float4* vertices [[buffer(0)]],
    uint vid [[vertex_id]])
{
    VertexOut out;
    out.position = float4(vertices[vid].xy, 0, 1);
    out.texcoord = vertices[vid].zw;
    return out;
}

fragment float4 tint_fragment(
    VertexOut in [[stage_in]],
    texture2d<float> input [[texture(0)]],
    constant EffectParams& params [[buffer(0)]])
{
    constexpr sampler s(mag_filter::linear, min_filter::linear);
    float4 color = input.sample(s, in.texcoord);

    // Generate tint color from hue + saturation
    float3 tint = hsv_to_rgb(params.tint_hue, params.tint_saturation, 1.0);

    // Multiply blend: overlay the tint onto the grayscaled image
    float3 tinted = color.rgb * tint;
    // Mix between untinted and tinted based on saturation strength
    float3 result = mix(color.rgb, tinted, params.tint_saturation);
    return float4(result, color.a);
}

// -----------------------------------------------------------------------
// Pass 3: blend original with processed (compute)
// -----------------------------------------------------------------------

kernel void blend(
    texture2d<float, access::read> original [[texture(0)]],
    texture2d<float, access::read> processed [[texture(1)]],
    texture2d<float, access::write> output [[texture(2)]],
    constant EffectParams& params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint w = original.get_width();
    uint h = original.get_height();
    if (gid.x >= w || gid.y >= h) return;

    float4 orig = original.read(gid);
    float4 proc = processed.read(gid);
    float4 result = mix(orig, proc, params.blend);
    output.write(result, gid);
}
