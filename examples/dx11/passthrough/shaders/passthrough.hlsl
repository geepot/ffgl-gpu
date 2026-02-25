Texture2D<float4> input : register(t0);
RWTexture2D<float4> output : register(u0);

[numthreads(16, 16, 1)]
void main_cs(uint3 id : SV_DispatchThreadID)
{
    uint w, h;
    input.GetDimensions(w, h);
    if (id.x >= w || id.y >= h) return;
    output[id.xy] = input[id.xy];
}
