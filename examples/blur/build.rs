fn main() {
    ffgl_gpu::build_support::compile_wgsl_shaders(
        std::path::Path::new("shaders"),
        &[ffgl_gpu::build_support::WgslEntry {
            file: "blur.wgsl",
            entry_point: "blur",
            stage: ffgl_gpu::build_support::ShaderStage::Compute,
        }],
    );
}
