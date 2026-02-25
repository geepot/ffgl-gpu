fn main() {
    #[cfg(target_os = "windows")]
    ffgl_gpu::build_support::compile_hlsl_shaders(
        std::path::Path::new("shaders"),
        &[
            ffgl_gpu::build_support::HlslEntry {
                file: "effects.hlsl",
                entry_point: "grayscale_cs",
                target: "cs_5_0",
            },
            ffgl_gpu::build_support::HlslEntry {
                file: "effects.hlsl",
                entry_point: "tint_vs",
                target: "vs_5_0",
            },
            ffgl_gpu::build_support::HlslEntry {
                file: "effects.hlsl",
                entry_point: "tint_ps",
                target: "ps_5_0",
            },
            ffgl_gpu::build_support::HlslEntry {
                file: "effects.hlsl",
                entry_point: "blend_cs",
                target: "cs_5_0",
            },
        ],
    );
}
