@group(0) @binding(0) var input: texture_2d<f32>;
@group(0) @binding(1) var output: texture_storage_2d<rgba8unorm, write>;

struct BlurParams {
    radius: i32,
}
@group(0) @binding(2) var<uniform> params: BlurParams;

@compute @workgroup_size(16, 16, 1)
fn blur(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(input);
    if gid.x >= dims.x || gid.y >= dims.y {
        return;
    }

    let r = params.radius;
    if r <= 0 {
        textureStore(output, gid.xy, textureLoad(input, gid.xy, 0));
        return;
    }

    var sum = vec4<f32>(0.0);
    var count = 0;
    for (var dy = -r; dy <= r; dy++) {
        for (var dx = -r; dx <= r; dx++) {
            let sx = i32(gid.x) + dx;
            let sy = i32(gid.y) + dy;
            if sx >= 0 && sx < i32(dims.x) && sy >= 0 && sy < i32(dims.y) {
                sum += textureLoad(input, vec2<u32>(u32(sx), u32(sy)), 0);
                count++;
            }
        }
    }
    textureStore(output, gid.xy, sum / f32(max(count, 1)));
}
