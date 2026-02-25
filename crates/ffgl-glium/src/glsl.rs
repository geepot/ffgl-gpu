//! GLSL version detection utilities.

use glium::CapabilitiesSource;

/// Supported GLSL transpilation targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlslVersion {
    Glsl120,
    Glsl140,
}

/// Try to get the best GLSL transpilation target for the given capabilities.
///
/// Returns `Glsl140` if supported, otherwise `Glsl120`, or `None` if neither
/// is available.
pub fn get_best_transpilation_target(ctx: &impl CapabilitiesSource) -> Option<GlslVersion> {
    let glsl_versions = &ctx.get_capabilities().supported_glsl_versions;

    if glsl_versions
        .iter()
        .any(|v| matches!(v, glium::Version(glium::Api::Gl, 1, 4)))
    {
        Some(GlslVersion::Glsl140)
    } else if glsl_versions
        .iter()
        .any(|v| matches!(v, glium::Version(glium::Api::Gl, 1, 2)))
    {
        Some(GlslVersion::Glsl120)
    } else {
        None
    }
}
