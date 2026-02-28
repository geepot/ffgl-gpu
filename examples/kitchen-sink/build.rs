fn main() {
    ffgl_gpu::build_support::compile_wgsl_shaders(
        std::path::Path::new("shaders"),
        &[ffgl_gpu::build_support::WgslEntry {
            file: "effects.wgsl",
            entry_point: "effects",
            stage: ffgl_gpu::build_support::ShaderStage::Compute,
        }],
    );
}
