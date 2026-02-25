fn main() {
    #[cfg(target_os = "windows")]
    ffgl_gpu::build_support::compile_hlsl_shaders(
        std::path::Path::new("shaders"),
        &[ffgl_gpu::build_support::HlslEntry {
            file: "passthrough.hlsl",
            entry_point: "main_cs",
            target: "cs_5_0",
        }],
    );
}
