//! Validate and reset the state of the OpenGL context.

/// Describes an OpenGL texture target/binding pair.
pub struct TextureType {
    pub target: u32,
    pub binding: u32,
}

/// Common texture types to unbind during GL state reset.
pub const TEXTURE_TYPES: [TextureType; 2] = [
    TextureType {
        target: gl::TEXTURE_1D,
        binding: gl::TEXTURE_BINDING_1D,
    },
    TextureType {
        target: gl::TEXTURE_2D,
        binding: gl::TEXTURE_BINDING_2D,
    },
];

/// Reset OpenGL state back to the host's expected defaults.
///
/// This unbinds programs, textures, buffers, VAOs, and restores the host FBO.
///
/// # Safety
///
/// Must be called with a valid OpenGL context active and `frame_data` must
/// point to valid host data.
pub unsafe fn gl_reset(frame_data: &ffgl_core::ffi::ProcessOpenGLStruct) {
    gl::UseProgram(0);

    let mut num_samplers = 0;
    gl::GetIntegerv(gl::MAX_TEXTURE_IMAGE_UNITS, &mut num_samplers);

    for texture_type in TEXTURE_TYPES.iter() {
        for sampler in 0..num_samplers {
            gl::ActiveTexture(gl::TEXTURE0 + sampler as u32);
            gl::BindTexture(texture_type.target, 0);
        }
    }

    gl::ActiveTexture(gl::TEXTURE0);

    gl::BindBuffer(gl::ARRAY_BUFFER, 0);
    gl::BindBuffer(gl::VERTEX_BINDING_BUFFER, 0);
    gl::BindVertexArray(0);
    gl::Disable(gl::BLEND);

    gl::BlendFunc(gl::ONE, gl::ZERO);

    gl::BindFramebuffer(gl::FRAMEBUFFER, frame_data.HostFBO);
}

/// Assert that the viewport matches expected dimensions and scissor test is disabled.
///
/// # Safety
///
/// Must be called with a valid OpenGL context active.
pub unsafe fn validate_viewport(viewport: &[i32; 4]) {
    let scissor_enabled = gl::IsEnabled(gl::SCISSOR_TEST);
    assert_eq!(scissor_enabled, gl::FALSE, "SCISSOR_TEST is enabled");

    let mut dims: [i32; 4] = [0; 4];
    gl::GetIntegerv(gl::VIEWPORT, &mut dims[0]);
    assert_eq!(&dims, viewport, "VIEWPORT wrong value: {dims:?}");
}
