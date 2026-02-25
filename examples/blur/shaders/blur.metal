#include <metal_stdlib>
using namespace metal;

/// Uniform buffer holding the blur radius in pixels.
struct BlurParams {
    int radius;
};

/// Horizontal box blur: reads from `input`, writes to `output`.
kernel void blur_horizontal(
    texture2d<float, access::read> input [[texture(0)]],
    texture2d<float, access::write> output [[texture(1)]],
    constant BlurParams& params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint w = input.get_width();
    uint h = input.get_height();
    if (gid.x >= w || gid.y >= h) return;

    int r = params.radius;
    if (r <= 0) {
        output.write(input.read(gid), gid);
        return;
    }

    float4 sum = float4(0.0);
    int count = 0;
    for (int dx = -r; dx <= r; dx++) {
        int sx = int(gid.x) + dx;
        if (sx >= 0 && sx < int(w)) {
            sum += input.read(uint2(sx, gid.y));
            count++;
        }
    }
    output.write(sum / float(max(count, 1)), gid);
}

/// Vertical box blur: reads from `input`, writes to `output`.
kernel void blur_vertical(
    texture2d<float, access::read> input [[texture(0)]],
    texture2d<float, access::write> output [[texture(1)]],
    constant BlurParams& params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint w = input.get_width();
    uint h = input.get_height();
    if (gid.x >= w || gid.y >= h) return;

    int r = params.radius;
    if (r <= 0) {
        output.write(input.read(gid), gid);
        return;
    }

    float4 sum = float4(0.0);
    int count = 0;
    for (int dy = -r; dy <= r; dy++) {
        int sy = int(gid.y) + dy;
        if (sy >= 0 && sy < int(h)) {
            sum += input.read(uint2(gid.x, sy));
            count++;
        }
    }
    output.write(sum / float(max(count, 1)), gid);
}
