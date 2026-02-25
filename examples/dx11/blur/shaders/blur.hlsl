// Separable box blur compute shaders for DX11.
//
// Two entry points: blur_horizontal and blur_vertical. Each samples a 1D
// neighborhood of the given radius and writes the average to the output.
// The radius is provided via a constant buffer (BlurParams).

Texture2D<float4> input : register(t0);
RWTexture2D<float4> output : register(u0);

cbuffer BlurParams : register(b0) {
    int radius;
};

[numthreads(16, 16, 1)]
void blur_horizontal(uint3 id : SV_DispatchThreadID)
{
    uint w, h;
    input.GetDimensions(w, h);
    if (id.x >= w || id.y >= h) return;

    int r = radius;
    if (r <= 0) {
        output[id.xy] = input[id.xy];
        return;
    }

    float4 sum = float4(0, 0, 0, 0);
    int count = 0;
    for (int dx = -r; dx <= r; dx++) {
        int sx = int(id.x) + dx;
        if (sx >= 0 && sx < int(w)) {
            sum += input[uint2(sx, id.y)];
            count++;
        }
    }
    output[id.xy] = sum / float(max(count, 1));
}

[numthreads(16, 16, 1)]
void blur_vertical(uint3 id : SV_DispatchThreadID)
{
    uint w, h;
    input.GetDimensions(w, h);
    if (id.x >= w || id.y >= h) return;

    int r = radius;
    if (r <= 0) {
        output[id.xy] = input[id.xy];
        return;
    }

    float4 sum = float4(0, 0, 0, 0);
    int count = 0;
    for (int dy = -r; dy <= r; dy++) {
        int sy = int(id.y) + dy;
        if (sy >= 0 && sy < int(h)) {
            sum += input[uint2(id.x, sy)];
            count++;
        }
    }
    output[id.xy] = sum / float(max(count, 1));
}
