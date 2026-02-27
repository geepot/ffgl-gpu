fn main() {
    ffgl_gpu::build_support::compile_wgsl_shaders(
        std::path::Path::new("shaders"),
        &[ffgl_gpu::build_support::WgslEntry {
            file: "passthrough.wgsl",
            entry_point: "passthrough",
            stage: ffgl_gpu::build_support::ShaderStage::Compute,
        }],
    );
}
