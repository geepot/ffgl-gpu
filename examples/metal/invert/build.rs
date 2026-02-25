fn main() {
    #[cfg(target_os = "macos")]
    ffgl_gpu::build_support::compile_metal_shaders(std::path::Path::new("shaders"));
}
