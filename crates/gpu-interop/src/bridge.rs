//! Common interface for GL-to-GPU texture bridging.

use anyhow::Result;
use gl::types::GLuint;
use std::time::Duration;

pub(crate) fn is_fresh_previous_frame(
    last_dispatch_frame: Option<u64>,
    current_frame: u64,
    age: Duration,
) -> bool {
    last_dispatch_frame.is_some_and(|last| current_frame == last.wrapping_add(1))
        && age < Duration::from_millis(250)
}

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
    #[allow(clippy::too_many_arguments)]
    fn blit_input_from_host_scaled(
        &mut self,
        host_texture: GLuint,
        src_w: u32,
        src_h: u32,
        dst_w: u32,
        dst_h: u32,
        bilinear: bool,
        uv_scale: (f32, f32),
    ) -> bool;

    /// Copy the back output texture (previous frame result) to the host FBO.
    ///
    /// Returns `false` if setup failed.
    #[allow(clippy::too_many_arguments)]
    fn blit_back_output_to_target_scaled(
        &mut self,
        host_fbo: GLuint,
        src_w: u32,
        src_h: u32,
        dst_x: i32,
        dst_y: i32,
        dst_w: u32,
        dst_h: u32,
        bilinear: bool,
    ) -> bool;

    /// Copy the front output texture (current frame, sync path) to the host FBO.
    ///
    /// Returns `false` if setup failed.
    #[allow(clippy::too_many_arguments)]
    fn blit_output_to_target_scaled(
        &mut self,
        host_fbo: GLuint,
        src_w: u32,
        src_h: u32,
        dst_x: i32,
        dst_y: i32,
        dst_w: u32,
        dst_h: u32,
        bilinear: bool,
    ) -> bool;

    /// Check if a previous frame's result is ready for presentation.
    fn has_result_ready(&self, current_frame: u64) -> bool;

    /// Block until the previous frame's GPU work completes. Clears pending state.
    /// Returns `false` when the GPU reported an error or timed out.
    fn wait_for_previous(&mut self) -> bool;

    /// Block until pending GPU work completes WITHOUT clearing pending state.
    /// Returns `false` when the GPU reported an error or timed out.
    fn wait_for_pending(&mut self) -> bool;

    /// Whether the plugin submitted GPU work for the current front buffer.
    /// Backends that cannot inspect submission state may keep the default.
    fn has_pending_work(&self) -> bool {
        true
    }

    /// Swap front/back pairs for double-buffering.
    fn swap(&mut self);

    /// Store dispatch info for the current frame.
    fn mark_dispatch(&mut self, frame: u64);

    /// Clean up all GPU resources.
    fn cleanup(&mut self);

    /// Get current dimensions of the shared textures.
    fn dimensions(&self) -> (u32, u32);
}

#[cfg(test)]
mod tests {
    use super::is_fresh_previous_frame;
    use std::time::Duration;

    #[test]
    fn accepts_heavy_but_continuous_playback() {
        assert!(is_fresh_previous_frame(
            Some(41),
            42,
            Duration::from_millis(180)
        ));
    }

    #[test]
    fn rejects_gaps_and_stale_results() {
        assert!(!is_fresh_previous_frame(
            Some(40),
            42,
            Duration::from_millis(16)
        ));
        assert!(!is_fresh_previous_frame(
            Some(41),
            42,
            Duration::from_millis(250)
        ));
    }
}
