fn main() {
    #[cfg(target_os = "windows")]
    ffgl_gpu::build_support::compile_hlsl_shaders(
        std::path::Path::new("shaders"),
        &[
            ffgl_gpu::build_support::HlslEntry {
                file: "blur.hlsl",
                entry_point: "blur_horizontal",
                target: "cs_5_0",
            },
            ffgl_gpu::build_support::HlslEntry {
                file: "blur.hlsl",
                entry_point: "blur_vertical",
                target: "cs_5_0",
            },
        ],
    );
}
