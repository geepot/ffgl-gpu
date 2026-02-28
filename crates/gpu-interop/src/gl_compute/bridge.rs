//! [`GlComputeBridge`] — double-buffered output texture manager for GL compute.
//!
//! Unlike the macOS Metal bridge, no cross-API texture transfer is needed here
//! because the compute shader runs in the host's own GL context. The host's
//! input texture is bound directly. This bridge only manages:
//!
//! - Two scratch output textures (double-buffered)
//! - Blit from scratch output back to the host FBO
//! - GL fence synchronisation for double-buffer pipelining

use std::time::Instant;

use anyhow::Result;
use gl::types::{GLenum, GLint, GLsync, GLuint};

use crate::GpuBridge;

/// GL compute bridge — manages double-buffered output textures.
///
/// The host's input GL texture is bound directly by the draw loop (no input
/// copy). Output is written to a scratch texture, then blitted back to the
/// host FBO.
pub struct GlComputeBridge {
    /// Double-buffered output textures.
    output_textures: [GLuint; 2],
    /// Which slot (0 or 1) is the "current" output target.
    front: usize,
    width: u32,
    height: u32,
    /// Reusable FBO for blit operations.
    blit_fbo: GLuint,
    /// Fence from the most recent dispatch (current frame).
    pending_fence: Option<GLsync>,
    /// Fence from the previous frame (after swap).
    previous_fence: Option<GLsync>,
    /// Frame counter from the last dispatch.
    last_dispatch_frame: Option<u64>,
    /// Wall-clock time of the last dispatch (detects stale data).
    last_dispatch_time: Option<Instant>,
}

// SAFETY: GL sync objects are opaque handles. The bridge is used
// single-threaded from the FFGL host's render thread.
unsafe impl Send for GlComputeBridge {}
unsafe impl Sync for GlComputeBridge {}

impl GlComputeBridge {
    /// Create a new bridge. No GL resources are allocated until
    /// [`ensure_dimensions`](GpuBridge::ensure_dimensions) is called.
    pub fn new() -> Self {
        Self {
            output_textures: [0; 2],
            front: 0,
            width: 0,
            height: 0,
            blit_fbo: 0,
            pending_fence: None,
            previous_fence: None,
            last_dispatch_frame: None,
            last_dispatch_time: None,
        }
    }

    /// Returns the front output texture name for binding as an image unit.
    pub fn output_gl_texture(&self) -> GLuint {
        self.output_textures[self.front]
    }

    /// Check whether the bridge's FBO is still a valid GL object.
    pub fn is_valid(&self) -> bool {
        if self.blit_fbo == 0 && self.width == 0 && self.height == 0 {
            // Not yet initialised — valid (will be created on first use).
            return true;
        }
        unsafe { gl::IsFramebuffer(self.blit_fbo) != 0 }
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    fn delete_textures(&mut self) {
        unsafe {
            if self.output_textures[0] != 0 || self.output_textures[1] != 0 {
                gl::DeleteTextures(2, self.output_textures.as_ptr());
                self.output_textures = [0; 2];
            }
        }
    }

    fn delete_fbo(&mut self) {
        unsafe {
            if self.blit_fbo != 0 {
                gl::BindFramebuffer(gl::FRAMEBUFFER, 0);
                gl::DeleteFramebuffers(1, &self.blit_fbo);
                self.blit_fbo = 0;
            }
        }
    }

    fn delete_fence(fence: &mut Option<GLsync>) {
        if let Some(f) = fence.take() {
            unsafe {
                gl::DeleteSync(f);
            }
        }
    }

    fn wait_fence(fence: Option<GLsync>) {
        if let Some(f) = fence {
            unsafe {
                gl::ClientWaitSync(f, gl::SYNC_FLUSH_COMMANDS_BIT, u64::MAX);
            }
        }
    }

    /// Blit a texture to a target FBO using the bridge's blit FBO.
    fn blit_to_target(
        &self,
        src_texture: GLuint,
        host_fbo: GLuint,
        src_w: u32,
        src_h: u32,
        dst_w: u32,
        dst_h: u32,
        bilinear: bool,
    ) -> bool {
        if src_texture == 0 || self.blit_fbo == 0 {
            return false;
        }

        let filter: GLenum = if bilinear {
            gl::LINEAR
        } else {
            gl::NEAREST
        };

        unsafe {
            // Attach source texture to the blit FBO (read side).
            gl::BindFramebuffer(gl::READ_FRAMEBUFFER, self.blit_fbo);
            gl::FramebufferTexture2D(
                gl::READ_FRAMEBUFFER,
                gl::COLOR_ATTACHMENT0,
                gl::TEXTURE_2D,
                src_texture,
                0,
            );
            gl::ReadBuffer(gl::COLOR_ATTACHMENT0);

            // Draw to host FBO.
            gl::BindFramebuffer(gl::DRAW_FRAMEBUFFER, host_fbo);

            gl::BlitFramebuffer(
                0,
                0,
                src_w as GLint,
                src_h as GLint,
                0,
                0,
                dst_w as GLint,
                dst_h as GLint,
                gl::COLOR_BUFFER_BIT,
                filter,
            );

            gl::BindFramebuffer(gl::FRAMEBUFFER, 0);
        }

        true
    }
}

impl GpuBridge for GlComputeBridge {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn ensure_dimensions(&mut self, width: u32, height: u32) -> Result<()> {
        if self.width == width
            && self.height == height
            && self.output_textures[0] != 0
            && self.output_textures[1] != 0
        {
            return Ok(());
        }

        // Wait for in-flight work before recreating textures.
        self.wait_for_previous();
        Self::delete_fence(&mut self.pending_fence);

        // Clean up old resources.
        self.delete_textures();
        self.delete_fbo();

        // Create two output textures with immutable storage.
        unsafe {
            gl::GenTextures(2, self.output_textures.as_mut_ptr());
            for &tex in &self.output_textures {
                gl::BindTexture(gl::TEXTURE_2D, tex);
                gl::TexStorage2D(gl::TEXTURE_2D, 1, gl::RGBA8, width as i32, height as i32);
                gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::NEAREST as i32);
                gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::NEAREST as i32);
                gl::TexParameteri(
                    gl::TEXTURE_2D,
                    gl::TEXTURE_WRAP_S,
                    gl::CLAMP_TO_EDGE as i32,
                );
                gl::TexParameteri(
                    gl::TEXTURE_2D,
                    gl::TEXTURE_WRAP_T,
                    gl::CLAMP_TO_EDGE as i32,
                );
            }
            gl::BindTexture(gl::TEXTURE_2D, 0);

            // Create blit FBO.
            gl::GenFramebuffers(1, &mut self.blit_fbo);
        }

        self.width = width;
        self.height = height;
        self.front = 0;
        self.last_dispatch_frame = None;
        self.last_dispatch_time = None;

        Ok(())
    }

    fn blit_input_from_host_scaled(
        &mut self,
        _host_texture: GLuint,
        _src_w: u32,
        _src_h: u32,
        _dst_w: u32,
        _dst_h: u32,
        _bilinear: bool,
    ) -> bool {
        // No-op: host texture is bound directly on GL compute path.
        true
    }

    fn blit_output_to_target_scaled(
        &mut self,
        host_fbo: GLuint,
        src_w: u32,
        src_h: u32,
        dst_w: u32,
        dst_h: u32,
        bilinear: bool,
    ) -> bool {
        self.blit_to_target(
            self.output_textures[self.front],
            host_fbo,
            src_w,
            src_h,
            dst_w,
            dst_h,
            bilinear,
        )
    }

    fn blit_back_output_to_target_scaled(
        &mut self,
        host_fbo: GLuint,
        src_w: u32,
        src_h: u32,
        dst_w: u32,
        dst_h: u32,
        bilinear: bool,
    ) -> bool {
        let back = 1 - self.front;
        self.blit_to_target(
            self.output_textures[back],
            host_fbo,
            src_w,
            src_h,
            dst_w,
            dst_h,
            bilinear,
        )
    }

    fn has_result_ready(&self, current_frame: u64) -> bool {
        let frame_ok = match self.last_dispatch_frame {
            Some(f) => current_frame == f + 1,
            None => false,
        };
        let time_ok = match self.last_dispatch_time {
            Some(t) => t.elapsed().as_millis() < 100,
            None => false,
        };
        let fence_ok = self.pending_fence.is_some();

        frame_ok && time_ok && fence_ok
    }

    fn wait_for_previous(&mut self) {
        if let Some(f) = self.previous_fence.take() {
            unsafe {
                gl::ClientWaitSync(f, gl::SYNC_FLUSH_COMMANDS_BIT, u64::MAX);
                gl::DeleteSync(f);
            }
        }
    }

    fn wait_for_pending(&mut self) {
        // Wait but do NOT delete — the fence may still be needed.
        Self::wait_fence(self.pending_fence);
    }

    fn swap(&mut self) {
        self.front = 1 - self.front;
        // Rotate fences: pending → previous (previous was already waited on).
        Self::delete_fence(&mut self.previous_fence);
        self.previous_fence = self.pending_fence.take();
    }

    fn mark_dispatch(&mut self, frame: u64) {
        // Insert a GL fence for the work just dispatched.
        let fence = unsafe { gl::FenceSync(gl::SYNC_GPU_COMMANDS_COMPLETE, 0) };
        // If there's an old pending fence (shouldn't happen normally), clean it.
        Self::delete_fence(&mut self.pending_fence);
        self.pending_fence = Some(fence);
        self.last_dispatch_frame = Some(frame);
        self.last_dispatch_time = Some(Instant::now());
    }

    fn cleanup(&mut self) {
        // Wait for all in-flight work.
        self.wait_for_previous();
        Self::delete_fence(&mut self.pending_fence);

        self.delete_textures();
        self.delete_fbo();

        self.front = 0;
        self.width = 0;
        self.height = 0;
        self.last_dispatch_frame = None;
        self.last_dispatch_time = None;
    }

    fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }
}

impl Drop for GlComputeBridge {
    fn drop(&mut self) {
        self.cleanup();
    }
}
