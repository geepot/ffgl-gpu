//! Double-buffered GL-Metal bridge via IOSurface-backed shared textures.
//!
//! Two IOSurface pairs allow pipelining: Metal processes the "front" pair
//! while GL blits the previous result from the "back" pair to the host FBO.
//! This introduces one frame of latency but allows Metal compute to overlap
//! with host compositing between draw calls.

// The CGL / OpenGL API is deprecated by Apple but required for interop with
// FFGL hosts that provide an OpenGL context.
#![allow(deprecated)]

use std::time::Instant;

use anyhow::{bail, Result};
use gl::types::{GLenum, GLint, GLsizei, GLuint};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_core_foundation::{CFDictionary, CFNumber, CFRetained, CFString};
use objc2_io_surface::IOSurfaceRef;
use objc2_metal::{
    MTLCommandBuffer, MTLDevice, MTLPixelFormat, MTLStorageMode, MTLTexture,
    MTLTextureDescriptor, MTLTextureType, MTLTextureUsage,
};
use objc2_open_gl::{CGLError, CGLGetCurrentContext, CGLTexImageIOSurface2D};
use tracing::{error, warn};

use crate::GpuBridge;

/// Pixel format FourCC for BGRA8 ('BGRA' = 0x42475241).
const IOSURFACE_PIXEL_FORMAT_BGRA: u32 = 0x42475241;

/// `GL_TEXTURE_RECTANGLE` is not in the `gl` crate's default API.
const GL_TEXTURE_RECTANGLE: GLenum = 0x84F5;

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// A texture backed by an IOSurface, accessible from both OpenGL and Metal.
struct SharedTexture {
    /// Kept alive to prevent the IOSurface from being deallocated.
    _iosurface: CFRetained<IOSurfaceRef>,
    gl_texture: GLuint,
    metal_texture: Retained<ProtocolObject<dyn MTLTexture>>,
}

impl SharedTexture {
    fn new(device: &ProtocolObject<dyn MTLDevice>, width: u32, height: u32) -> Option<Self> {
        let iosurface = create_iosurface(width, height)?;
        let gl_texture = unsafe { create_gl_texture_from_iosurface(&iosurface, width, height)? };
        let metal_texture =
            create_metal_texture_from_iosurface(device, &iosurface, width, height)?;

        Some(Self {
            _iosurface: iosurface,
            gl_texture,
            metal_texture,
        })
    }
}

impl Drop for SharedTexture {
    fn drop(&mut self) {
        if self.gl_texture != 0 {
            unsafe {
                gl::DeleteTextures(1, &self.gl_texture);
            }
        }
        // Metal texture and IOSurface are reference-counted and drop automatically.
    }
}

/// A paired input/output IOSurface set for one frame slot.
struct IoSurfacePair {
    input: SharedTexture,
    output: SharedTexture,
}

impl IoSurfacePair {
    fn new(device: &ProtocolObject<dyn MTLDevice>, width: u32, height: u32) -> Option<Self> {
        Some(Self {
            input: SharedTexture::new(device, width, height)?,
            output: SharedTexture::new(device, width, height)?,
        })
    }
}

// ---------------------------------------------------------------------------
// IOSurface / texture creation
// ---------------------------------------------------------------------------

/// Create an IOSurface with BGRA8 pixel format via the CoreFoundation API.
fn create_iosurface(width: u32, height: u32) -> Option<CFRetained<IOSurfaceRef>> {
    unsafe {
        let k_width = objc2_io_surface::kIOSurfaceWidth;
        let k_height = objc2_io_surface::kIOSurfaceHeight;
        let k_bpe = objc2_io_surface::kIOSurfaceBytesPerElement;
        let k_pf = objc2_io_surface::kIOSurfacePixelFormat;

        let v_width = CFNumber::new_i32(width as i32);
        let v_height = CFNumber::new_i32(height as i32);
        let v_bpe = CFNumber::new_i32(4);
        let v_pf = CFNumber::new_i32(IOSURFACE_PIXEL_FORMAT_BGRA as i32);

        let keys: &[&CFString] = &[k_width, k_height, k_bpe, k_pf];
        let values: &[&CFNumber] = &[&v_width, &v_height, &v_bpe, &v_pf];

        let props = CFDictionary::from_slices(keys, values);
        let props_untyped: &CFDictionary = props.cast_unchecked();
        IOSurfaceRef::new(props_untyped)
    }
}

/// Create a GL `TEXTURE_RECTANGLE` backed by an IOSurface via
/// `CGLTexImageIOSurface2D`.
///
/// # Safety
/// A valid CGL context must be current.
unsafe fn create_gl_texture_from_iosurface(
    surface: &IOSurfaceRef,
    width: u32,
    height: u32,
) -> Option<GLuint> {
    let cgl_ctx = CGLGetCurrentContext();
    if cgl_ctx.is_null() {
        error!("No current CGL context for IOSurface texture creation");
        return None;
    }

    let mut tex: GLuint = 0;
    gl::GenTextures(1, &mut tex);
    gl::BindTexture(GL_TEXTURE_RECTANGLE, tex);

    gl::TexParameteri(
        GL_TEXTURE_RECTANGLE,
        gl::TEXTURE_MIN_FILTER,
        gl::LINEAR as GLint,
    );
    gl::TexParameteri(
        GL_TEXTURE_RECTANGLE,
        gl::TEXTURE_MAG_FILTER,
        gl::LINEAR as GLint,
    );

    let err = CGLTexImageIOSurface2D(
        cgl_ctx,
        GL_TEXTURE_RECTANGLE,
        gl::RGBA as GLenum,
        width as GLsizei,
        height as GLsizei,
        gl::BGRA,
        gl::UNSIGNED_INT_8_8_8_8_REV,
        surface,
        0, // plane
    );

    gl::BindTexture(GL_TEXTURE_RECTANGLE, 0);

    if err != CGLError::NoError {
        error!("CGLTexImageIOSurface2D failed with error: {err:?}");
        gl::DeleteTextures(1, &tex);
        return None;
    }

    Some(tex)
}

/// Create a Metal texture backed by an IOSurface.
fn create_metal_texture_from_iosurface(
    device: &ProtocolObject<dyn MTLDevice>,
    surface: &IOSurfaceRef,
    width: u32,
    height: u32,
) -> Option<Retained<ProtocolObject<dyn MTLTexture>>> {
    let desc = MTLTextureDescriptor::new();
    desc.setTextureType(MTLTextureType::Type2D);
    desc.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
    unsafe {
        desc.setWidth(width as usize);
        desc.setHeight(height as usize);
    }
    desc.setStorageMode(MTLStorageMode::Shared);
    desc.setUsage(MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite);

    device.newTextureWithDescriptor_iosurface_plane(&desc, surface, 0)
}

// ---------------------------------------------------------------------------
// GlMetalBridge
// ---------------------------------------------------------------------------

/// Double-buffered bridge between OpenGL and Metal via IOSurface-backed shared
/// textures.
///
/// Two IOSurface pairs allow pipelining: Metal processes the "front" pair
/// while GL blits the previous result from the "back" pair to the host FBO.
/// This introduces one frame of latency but allows Metal compute to overlap
/// with host compositing.
pub struct GlMetalBridge {
    /// The Metal device used to create shared textures.
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    /// Double-buffered IOSurface pairs.
    pairs: [Option<IoSurfacePair>; 2],
    /// Index of the pair currently being written by Metal compute.
    front: usize,
    /// The command buffer from the most recent Metal dispatch, if any.
    pending_command_buffer: Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
    /// Frame counter from the most recent draw call that dispatched Metal
    /// compute.  Used to detect gaps (deselection) -- if the current frame
    /// counter is not `last_frame + 1`, we had a gap and must not use stale
    /// back-buffer data.
    last_dispatch_frame: Option<u64>,
    /// Wall-clock time of the most recent dispatch.  Used to detect stale
    /// back-buffer data after deselection/reselection (where the frame counter
    /// is consecutive but real time has a gap).
    last_dispatch_time: Option<Instant>,
    read_fbo: GLuint,
    draw_fbo: GLuint,
    dimensions: (u32, u32),
    /// Cached GL texture target for the host's input texture
    /// (`TEXTURE_2D` or `TEXTURE_RECTANGLE`).  Zero means not yet probed --
    /// will be determined on first blit and cached.
    host_texture_type: GLenum,
}

impl GlMetalBridge {
    /// Create an uninitialised bridge.  Call [`GpuBridge::ensure_dimensions`]
    /// before use.
    pub fn new(device: Retained<ProtocolObject<dyn MTLDevice>>) -> Self {
        Self {
            device,
            pairs: [None, None],
            front: 0,
            pending_command_buffer: None,
            last_dispatch_frame: None,
            last_dispatch_time: None,
            read_fbo: 0,
            draw_fbo: 0,
            dimensions: (0, 0),
            host_texture_type: 0,
        }
    }

    /// Store the command buffer from a Metal dispatch for later waiting.
    ///
    /// This is Metal-specific (not part of [`GpuBridge`]).  It records both
    /// the command buffer reference **and** the frame counter / timestamp so
    /// that [`GpuBridge::has_result_ready`] can detect gaps.
    pub fn store_command_buffer(
        &mut self,
        command_buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        frame: u64,
    ) {
        self.pending_command_buffer = Some(command_buffer);
        self.last_dispatch_frame = Some(frame);
        self.last_dispatch_time = Some(Instant::now());
    }

    /// Get the Metal texture for the front input (read by compute shaders).
    pub fn input_metal_texture(&self) -> Option<&ProtocolObject<dyn MTLTexture>> {
        self.pairs[self.front]
            .as_ref()
            .map(|p| &*p.input.metal_texture)
    }

    /// Get the Metal texture for the front output (written by compute shaders).
    pub fn output_metal_texture(&self) -> Option<&ProtocolObject<dyn MTLTexture>> {
        self.pairs[self.front]
            .as_ref()
            .map(|p| &*p.output.metal_texture)
    }

    /// Get the Metal texture for the back output (previous frame's result).
    /// Useful for interleaved field modes that need to fill non-field rows.
    pub fn back_output_metal_texture(&self) -> Option<&ProtocolObject<dyn MTLTexture>> {
        let back = 1 - self.front;
        self.pairs[back].as_ref().map(|p| &*p.output.metal_texture)
    }

    /// Check whether the bridge FBO handles are still valid.
    pub fn is_valid(&self) -> bool {
        if self.read_fbo == 0 && self.draw_fbo == 0 {
            return self.dimensions == (0, 0); // not yet initialised is valid
        }
        unsafe { gl::IsFramebuffer(self.read_fbo) != 0 && gl::IsFramebuffer(self.draw_fbo) != 0 }
    }

    /// Borrow the stored Metal device.
    pub fn device(&self) -> &ProtocolObject<dyn MTLDevice> {
        &self.device
    }
}

// ---------------------------------------------------------------------------
// GpuBridge trait implementation
// ---------------------------------------------------------------------------

impl GpuBridge for GlMetalBridge {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn ensure_dimensions(&mut self, width: u32, height: u32) -> Result<()> {
        if self.dimensions == (width, height)
            && self.pairs[0].is_some()
            && self.pairs[1].is_some()
        {
            return Ok(());
        }

        // Dimension change: wait for any in-flight work before destroying textures.
        self.wait_for_previous();

        // Clean up old FBOs.
        unsafe {
            if self.read_fbo != 0 {
                gl::DeleteFramebuffers(1, &self.read_fbo);
            }
            if self.draw_fbo != 0 {
                gl::DeleteFramebuffers(1, &self.draw_fbo);
            }
        }

        self.pairs[0] = IoSurfacePair::new(&self.device, width, height);
        self.pairs[1] = IoSurfacePair::new(&self.device, width, height);

        if self.pairs[0].is_none() || self.pairs[1].is_none() {
            self.pairs = [None, None];
            self.read_fbo = 0;
            self.draw_fbo = 0;
            self.dimensions = (0, 0);
            bail!("Failed to create shared IOSurface texture pairs");
        }

        // Create separate FBOs for read and draw to avoid undefined behaviour.
        unsafe {
            gl::GenFramebuffers(1, &mut self.read_fbo);
            gl::GenFramebuffers(1, &mut self.draw_fbo);
        }

        self.dimensions = (width, height);
        self.front = 0;
        self.last_dispatch_frame = None;
        self.last_dispatch_time = None;
        self.host_texture_type = 0;
        Ok(())
    }

    fn blit_input_from_host_scaled(
        &mut self,
        host_texture: GLuint,
        src_w: u32,
        src_h: u32,
        dst_w: u32,
        dst_h: u32,
        bilinear: bool,
    ) -> bool {
        let input_gl = match &self.pairs[self.front] {
            Some(pair) => pair.input.gl_texture,
            None => return false,
        };

        unsafe {
            gl::BindFramebuffer(gl::READ_FRAMEBUFFER, self.read_fbo);

            // Probe / cache the host texture target on first call.
            if self.host_texture_type == 0 {
                gl::FramebufferTexture2D(
                    gl::READ_FRAMEBUFFER,
                    gl::COLOR_ATTACHMENT0,
                    gl::TEXTURE_2D,
                    host_texture,
                    0,
                );
                if gl::CheckFramebufferStatus(gl::READ_FRAMEBUFFER) == gl::FRAMEBUFFER_COMPLETE {
                    self.host_texture_type = gl::TEXTURE_2D;
                } else {
                    gl::FramebufferTexture2D(
                        gl::READ_FRAMEBUFFER,
                        gl::COLOR_ATTACHMENT0,
                        GL_TEXTURE_RECTANGLE,
                        host_texture,
                        0,
                    );
                    if gl::CheckFramebufferStatus(gl::READ_FRAMEBUFFER) == gl::FRAMEBUFFER_COMPLETE
                    {
                        self.host_texture_type = GL_TEXTURE_RECTANGLE;
                    } else {
                        warn!(
                            "READ_FRAMEBUFFER incomplete for host texture {host_texture}"
                        );
                        gl::BindFramebuffer(gl::FRAMEBUFFER, 0);
                        return false;
                    }
                }
            } else {
                gl::FramebufferTexture2D(
                    gl::READ_FRAMEBUFFER,
                    gl::COLOR_ATTACHMENT0,
                    self.host_texture_type,
                    host_texture,
                    0,
                );
            }
            gl::ReadBuffer(gl::COLOR_ATTACHMENT0);

            // DRAW side: attach IOSurface.
            gl::BindFramebuffer(gl::DRAW_FRAMEBUFFER, self.draw_fbo);
            gl::FramebufferTexture2D(
                gl::DRAW_FRAMEBUFFER,
                gl::COLOR_ATTACHMENT0,
                GL_TEXTURE_RECTANGLE,
                input_gl,
                0,
            );
            gl::DrawBuffer(gl::COLOR_ATTACHMENT0);

            let filter = if bilinear { gl::LINEAR } else { gl::NEAREST };

            gl::BlitFramebuffer(
                0,
                0,
                src_w as GLsizei,
                src_h as GLsizei,
                0,
                0,
                dst_w as GLsizei,
                dst_h as GLsizei,
                gl::COLOR_BUFFER_BIT,
                filter,
            );

            gl::BindFramebuffer(gl::FRAMEBUFFER, 0);
            gl::Flush();
        }
        true
    }

    fn blit_back_output_to_target_scaled(
        &mut self,
        host_fbo: GLuint,
        src_w: u32,
        src_h: u32,
        dst_w: u32,
        dst_h: u32,
        bilinear: bool,
    ) {
        let back = 1 - self.front;
        let output_gl = match &self.pairs[back] {
            Some(pair) => pair.output.gl_texture,
            None => return,
        };

        unsafe {
            gl::BindFramebuffer(gl::READ_FRAMEBUFFER, self.read_fbo);
            gl::FramebufferTexture2D(
                gl::READ_FRAMEBUFFER,
                gl::COLOR_ATTACHMENT0,
                GL_TEXTURE_RECTANGLE,
                output_gl,
                0,
            );
            gl::ReadBuffer(gl::COLOR_ATTACHMENT0);

            gl::BindFramebuffer(gl::DRAW_FRAMEBUFFER, host_fbo);

            let filter = if bilinear { gl::LINEAR } else { gl::NEAREST };

            gl::BlitFramebuffer(
                0,
                0,
                src_w as GLsizei,
                src_h as GLsizei,
                0,
                0,
                dst_w as GLsizei,
                dst_h as GLsizei,
                gl::COLOR_BUFFER_BIT,
                filter,
            );

            gl::BindFramebuffer(gl::FRAMEBUFFER, 0);
        }
    }

    fn blit_output_to_target_scaled(
        &mut self,
        host_fbo: GLuint,
        src_w: u32,
        src_h: u32,
        dst_w: u32,
        dst_h: u32,
        bilinear: bool,
    ) {
        let output_gl = match &self.pairs[self.front] {
            Some(pair) => pair.output.gl_texture,
            None => return,
        };

        unsafe {
            gl::BindFramebuffer(gl::READ_FRAMEBUFFER, self.read_fbo);
            gl::FramebufferTexture2D(
                gl::READ_FRAMEBUFFER,
                gl::COLOR_ATTACHMENT0,
                GL_TEXTURE_RECTANGLE,
                output_gl,
                0,
            );
            gl::ReadBuffer(gl::COLOR_ATTACHMENT0);

            gl::BindFramebuffer(gl::DRAW_FRAMEBUFFER, host_fbo);

            let filter = if bilinear { gl::LINEAR } else { gl::NEAREST };

            gl::BlitFramebuffer(
                0,
                0,
                src_w as GLsizei,
                src_h as GLsizei,
                0,
                0,
                dst_w as GLsizei,
                dst_h as GLsizei,
                gl::COLOR_BUFFER_BIT,
                filter,
            );

            gl::BindFramebuffer(gl::FRAMEBUFFER, 0);
        }
    }

    fn has_result_ready(&self, current_frame: u64) -> bool {
        self.pending_command_buffer.is_some()
            && self
                .last_dispatch_frame
                .is_some_and(|last| current_frame == last.wrapping_add(1))
            && self
                .last_dispatch_time
                .is_some_and(|t| t.elapsed().as_millis() < 100)
    }

    fn wait_for_previous(&mut self) {
        if let Some(cb) = self.pending_command_buffer.take() {
            cb.waitUntilCompleted();
        }
    }

    fn wait_for_pending(&mut self) {
        if let Some(cb) = &self.pending_command_buffer {
            cb.waitUntilCompleted();
        }
    }

    fn swap(&mut self) {
        self.front = 1 - self.front;
    }

    fn mark_dispatch(&mut self, frame: u64) {
        self.last_dispatch_frame = Some(frame);
        self.last_dispatch_time = Some(Instant::now());
    }

    fn cleanup(&mut self) {
        if let Some(cb) = self.pending_command_buffer.take() {
            cb.waitUntilCompleted();
        }
        self.pairs = [None, None];
        self.front = 0;
        self.last_dispatch_frame = None;
        self.last_dispatch_time = None;
        unsafe {
            if self.read_fbo != 0 {
                gl::DeleteFramebuffers(1, &self.read_fbo);
                self.read_fbo = 0;
            }
            if self.draw_fbo != 0 {
                gl::DeleteFramebuffers(1, &self.draw_fbo);
                self.draw_fbo = 0;
            }
        }
        self.dimensions = (0, 0);
        self.host_texture_type = 0;
    }

    fn dimensions(&self) -> (u32, u32) {
        self.dimensions
    }
}
