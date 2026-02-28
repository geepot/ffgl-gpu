@group(0) @binding(0) var input: texture_2d<f32>;
@group(0) @binding(1) var output: texture_storage_2d<rgba8unorm, write>;

struct EffectParams {
    grayscale_amount: f32,
    tint_hue: f32,
    tint_saturation: f32,
    blend: f32,
}
@group(0) @binding(2) var<uniform> params: EffectParams;

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> vec3<f32> {
    let c = v * s;
    let x = c * (1.0 - abs(((h * 6.0) % 2.0) - 1.0));
    let m = v - c;
    var rgb: vec3<f32>;
    let hi = i32(h * 6.0) % 6;
    if hi == 0      { rgb = vec3<f32>(c, x, 0.0); }
    else if hi == 1 { rgb = vec3<f32>(x, c, 0.0); }
    else if hi == 2 { rgb = vec3<f32>(0.0, c, x); }
    else if hi == 3 { rgb = vec3<f32>(0.0, x, c); }
    else if hi == 4 { rgb = vec3<f32>(x, 0.0, c); }
    else            { rgb = vec3<f32>(c, 0.0, x); }
    return rgb + vec3<f32>(m);
}

@compute @workgroup_size(16, 16, 1)
fn effects(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(input);
    if gid.x >= dims.x || gid.y >= dims.y {
        return;
    }

    let color = textureLoad(input, gid.xy, 0);

    // Grayscale
    let lum = dot(color.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
    let gray = mix(color.rgb, vec3<f32>(lum), params.grayscale_amount);

    // Tint
    let tint = hsv_to_rgb(params.tint_hue, params.tint_saturation, 1.0);
    let tinted = gray * tint;
    let after_tint = mix(gray, tinted, params.tint_saturation);

    // Blend original with processed
    let result = mix(color.rgb, after_tint, params.blend);
    textureStore(output, gid.xy, vec4<f32>(result, color.a));
}
