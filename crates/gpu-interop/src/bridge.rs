//! Common interface for GL-to-GPU texture bridging.

use anyhow::Result;
use gl::types::GLuint;

/// Common interface for GL-to-GPU texture bridging.
///
/// Implementations exist for Metal (macOS via IOSurface) and DX11 (Windows via
/// `WGL_NV_DX_interop2`).
///
/// The bridge manages a pair of shared textures (front/back) for double-buffered
/// rendering. Input textures receive data from the host's OpenGL FBO, and output
/// textures hold processed results to blit back.
pub trait GpuBridge {
    /// Downcast to a concrete type. Used by plugins to access platform-specific
    /// texture handles (e.g. `GlMetalBridge::input_metal_texture()`).
    fn as_any(&self) -> &dyn std::any::Any;

    /// Mutable downcast to a concrete type.
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
    /// Recreate shared textures if dimensions changed.
    fn ensure_dimensions(&mut self, width: u32, height: u32) -> Result<()>;

    /// Copy host OpenGL texture into the bridge's front input texture.
    ///
    /// Returns `false` if setup failed.
    fn blit_input_from_host_scaled(
        &mut self,
        host_texture: GLuint,
        src_w: u32,
        src_h: u32,
        dst_w: u32,
        dst_h: u32,
        bilinear: bool,
    ) -> bool;

    /// Copy the back output texture (previous frame result) to the host FBO.
    ///
    /// Returns `false` if setup failed.
    fn blit_back_output_to_target_scaled(
        &mut self,
        host_fbo: GLuint,
        src_w: u32,
        src_h: u32,
        dst_w: u32,
        dst_h: u32,
        bilinear: bool,
    ) -> bool;

    /// Copy the front output texture (current frame, sync path) to the host FBO.
    ///
    /// Returns `false` if setup failed.
    fn blit_output_to_target_scaled(
        &mut self,
        host_fbo: GLuint,
        src_w: u32,
        src_h: u32,
        dst_w: u32,
        dst_h: u32,
        bilinear: bool,
    ) -> bool;

    /// Check if a previous frame's result is ready for presentation.
    fn has_result_ready(&self, current_frame: u64) -> bool;

    /// Block until the previous frame's GPU work completes. Clears pending state.
    fn wait_for_previous(&mut self);

    /// Block until pending GPU work completes WITHOUT clearing pending state.
    fn wait_for_pending(&mut self);

    /// Swap front/back pairs for double-buffering.
    fn swap(&mut self);

    /// Store dispatch info for the current frame.
    fn mark_dispatch(&mut self, frame: u64);

    /// Clean up all GPU resources.
    fn cleanup(&mut self);

    /// Get current dimensions of the shared textures.
    fn dimensions(&self) -> (u32, u32);
}
