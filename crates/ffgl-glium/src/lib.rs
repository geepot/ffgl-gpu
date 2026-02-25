//! Utilities for creating FFGL plugins using the glium library.
//!
//! Use [`FFGLGlium`] in your plugin to render frames with a glium context.
//!
//! Just call [`FFGLGlium::draw`] inside your
//! [`ffgl_core::handler::FFGLInstance::draw`] method.
//!
//! ### Warning
//!
//! This module makes assumptions about the OpenGL context inside the host.
//! Bugs and crashes may occur. Testing infrastructure is required.

use std::{error::Error, fmt::Formatter, rc::Rc};

use ffgl_core::*;
use glium::{
    backend::Context,
    framebuffer::{RenderBuffer, SimpleFrameBuffer},
    CapabilitiesSource, Frame, Surface, Texture2d,
};
use std::fmt::Debug;
use tracing::trace;

mod gl_backend;
pub mod glsl;
pub mod texture;
pub mod validate_gl;

/// Cached render buffer to avoid per-frame GL allocations.
struct CachedRenderBuffer {
    rb: RenderBuffer,
    dims: (u32, u32),
}

/// Use this struct to render frames with a glium context, making assumptions
/// about the OpenGL context inside an FFGL host.
pub struct FFGLGlium {
    pub ctx: Rc<Context>,
    backend: Rc<gl_backend::RawGlBackend>,
    cached_rb: Option<CachedRenderBuffer>,
}

impl Debug for FFGLGlium {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FFGLGlium").finish()
    }
}

/// The default surface type used for rendering.
pub type DefaultSurface<'a> = SimpleFrameBuffer<'a>;

impl FFGLGlium {
    /// Create a new glium context from host-provided FFGL instance data.
    pub fn new(inst_data: &FFGLData) -> Self {
        let backend = Rc::new(gl_backend::RawGlBackend::new(inst_data.get_dimensions()));

        tracing::debug!("BACKEND: {backend:?}");

        let ctx = unsafe {
            glium::backend::Context::new(
                backend.clone(),
                false,
                glium::debug::DebugCallbackBehavior::Ignore,
            )
            .unwrap()
        };

        let valid_versions = &ctx.get_capabilities().supported_glsl_versions;

        tracing::debug!("VALID VERSIONS: {valid_versions:?}");

        tracing::debug!("OPENGL_VERSION {}", ctx.get_opengl_version_string());

        Self {
            ctx,
            backend,
            cached_rb: None,
        }
    }

    /// Main draw loop: create renderbuffer, import host textures, call user
    /// closure, blit result to host FBO.
    pub fn draw(
        &mut self,
        render_res: (u32, u32),
        out_res: (u32, u32),
        frame_data: GLInput<'_>,
        render_frame: &mut impl FnMut(
            &mut DefaultSurface,
            Vec<Texture2d>,
        ) -> Result<(), Box<dyn Error>>,
    ) {
        unsafe {
            self.ctx.rebuild(self.backend.clone()).unwrap();
        };

        // Cache the render buffer -- only recreate when dimensions change
        if self.cached_rb.as_ref().map(|c| c.dims) != Some(render_res) {
            let rb = RenderBuffer::new(
                &self.ctx,
                glium::texture::UncompressedFloatFormat::U8U8U8U8,
                render_res.0,
                render_res.1,
            )
            .expect("RenderBuffer could not be created");
            self.cached_rb = Some(CachedRenderBuffer {
                rb,
                dims: render_res,
            });
        }

        let rb = &self.cached_rb.as_ref().unwrap().rb;
        let mut fb = SimpleFrameBuffer::new(&self.ctx, rb)
            .expect("SimpleFrameBuffer could not be created");

        let textures: Vec<_> = frame_data
            .textures
            .iter()
            .map(|texture_info| unsafe {
                Texture2d::from_id(
                    &self.ctx,
                    glium::texture::UncompressedFloatFormat::U8U8U8U8,
                    texture_info.Handle,
                    false,
                    glium::texture::MipmapsOption::NoMipmap,
                    glium::texture::Dimensions::Texture2d {
                        width: texture_info.Width,
                        height: texture_info.Height,
                    },
                )
            })
            .collect();

        if let Err(err) = render_frame(&mut fb, textures) {
            tracing::error!("Render ERROR: {err:?}");
        }

        trace!(?out_res, ?render_res, "RENDERED");

        let frame = Frame::new(self.ctx.clone(), out_res);
        fb.fill(&frame, glium::uniforms::MagnifySamplerFilter::Nearest);

        unsafe {
            gl::BindFramebuffer(gl::DRAW_FRAMEBUFFER, frame_data.host);
            blit_fb(render_res, out_res);
            self.ctx.rebuild(self.backend.clone()).unwrap();
        };

        frame.finish().unwrap();
    }

    /// Swap buffers and rebind the host FBO as the draw framebuffer.
    pub fn set_default_db_to_ffgl_fb(&self, frame_data: &GLInput<'_>) {
        self.ctx.swap_buffers().expect("swap_buffers failed");
        unsafe {
            gl::BindFramebuffer(gl::DRAW_FRAMEBUFFER, frame_data.host);
        }
    }
}

/// Blit from the read framebuffer to the draw framebuffer.
///
/// # Safety
///
/// Caller must ensure valid GL context and correctly bound framebuffers.
unsafe fn blit_fb((read_w, read_h): (u32, u32), (write_w, write_h): (u32, u32)) {
    gl::BlitFramebuffer(
        0,
        0,
        read_w as gl::types::GLint,
        read_h as gl::types::GLint,
        0,
        0,
        write_w as gl::types::GLint,
        write_h as gl::types::GLint,
        gl::COLOR_BUFFER_BIT,
        gl::NEAREST,
    );
}
