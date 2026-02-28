@group(0) @binding(0) var input: texture_2d<f32>;
@group(0) @binding(1) var output: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(16, 16, 1)
fn invert(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(input);
    if gid.x >= dims.x || gid.y >= dims.y {
        return;
    }
    let color = textureLoad(input, gid.xy, 0);
    textureStore(output, gid.xy, vec4<f32>(1.0 - color.r, 1.0 - color.g, 1.0 - color.b, color.a));
}
