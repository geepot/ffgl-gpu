#include <metal_stdlib>
using namespace metal;

struct VertexOut {
    float4 position [[position]];
    float2 texcoord;
};

vertex VertexOut invert_vertex(
    const device float4* vertices [[buffer(0)]],
    uint vid [[vertex_id]])
{
    VertexOut out;
    out.position = float4(vertices[vid].xy, 0, 1);
    out.texcoord = vertices[vid].zw;
    return out;
}

fragment float4 invert_fragment(
    VertexOut in [[stage_in]],
    texture2d<float> input [[texture(0)]])
{
    constexpr sampler s(mag_filter::linear, min_filter::linear);
    float4 color = input.sample(s, in.texcoord);
    return float4(1.0 - color.rgb, color.a);
}
