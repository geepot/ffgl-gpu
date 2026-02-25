fn main() {
    #[cfg(target_os = "windows")]
    ffgl_gpu::build_support::compile_hlsl_shaders(
        std::path::Path::new("shaders"),
        &[
            ffgl_gpu::build_support::HlslEntry {
                file: "invert.hlsl",
                entry_point: "vs_main",
                target: "vs_5_0",
            },
            ffgl_gpu::build_support::HlslEntry {
                file: "invert.hlsl",
                entry_point: "ps_main",
                target: "ps_5_0",
            },
        ],
    );
}
