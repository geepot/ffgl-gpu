//! Shader-based GL texture blit, used as a fallback when
//! `glBlitFramebuffer` can't read from the source.
//!
//! `glBlitFramebuffer` requires both the read and draw attachments to
//! be color-renderable, which excludes compressed texture formats
//! (BC1/DXT1, BC3/DXT5, etc.) and any non-renderable internal formats.
//! Resolume hands plugins clip-source textures *directly* — and those
//! are routinely BC1-compressed (video decoder output stored compressed
//! in VRAM, hardware-decompressed on sample). The FBO probe fails for
//! every frame on a clip-mounted effect.
//!
//! This fallback path samples the source texture (which only requires
//! the format to be sampleable, satisfied by every Resolume input) and
//! writes to a renderable destination via `DRAW_FRAMEBUFFER` + a
//! fullscreen quad. Sampling a compressed texture is exactly what the
//! GPU's sampler unit is for; hardware decompression on sample is
//! free. Two shader variants cover the two source target types we've
//! ever seen Resolume hand us (`TEXTURE_2D` and `TEXTURE_RECTANGLE`).

use gl::types::{GLchar, GLenum, GLint, GLsizei, GLsizeiptr, GLuint};
use std::ffi::CString;
use std::ptr;
use tracing::{error, warn};

/// `GL_TEXTURE_RECTANGLE` is not in the `gl` crate's default API on every
/// platform target combination — declare locally to match the macOS
/// interop module's convention.
#[cfg_attr(target_os = "windows", allow(dead_code))]
const GL_TEXTURE_RECTANGLE: GLenum = 0x84F5;

const VERT_SRC: &str = "#version 330 core\n\
layout(location = 0) in vec2 a_pos;\n\
out vec2 v_uv;\n\
void main() {\n\
    gl_Position = vec4(a_pos, 0.0, 1.0);\n\
    v_uv = a_pos * 0.5 + 0.5;\n\
}\n";

const FRAG_2D: &str = "#version 330 core\n\
in vec2 v_uv;\n\
out vec4 frag_color;\n\
uniform sampler2D u_src;\n\
uniform vec2 u_uv_scale;\n\
void main() {\n\
    frag_color = texture(u_src, v_uv * u_uv_scale);\n\
}\n";

const FRAG_RECT: &str = "#version 330 core\n\
in vec2 v_uv;\n\
out vec4 frag_color;\n\
uniform sampler2DRect u_src;\n\
uniform vec2 u_src_size;\n\
uniform vec2 u_uv_scale;\n\
void main() {\n\
    frag_color = texture(u_src, v_uv * u_uv_scale * u_src_size);\n\
}\n";

const QUAD: [f32; 8] = [
    -1.0, -1.0,
     1.0, -1.0,
    -1.0,  1.0,
     1.0,  1.0,
];

pub struct ShaderBlit {
    program_2d: GLuint,
    program_rect: GLuint,
    #[cfg_attr(target_os = "windows", allow(dead_code))]
    rect_size_loc: GLint,
    uv_scale_2d_loc: GLint,
    #[cfg_attr(target_os = "windows", allow(dead_code))]
    uv_scale_rect_loc: GLint,
    vao: GLuint,
    vbo: GLuint,
}

impl ShaderBlit {
    /// Compile both program variants and set up the fullscreen-quad
    /// VBO/VAO. Returns `None` if any compile/link/uniform-lookup step
    /// fails — caller falls back to "give up on this input texture."
    pub fn new() -> Option<Self> {
        unsafe {
            let program_2d = compile_program(VERT_SRC, FRAG_2D)?;
            let program_rect = compile_program(VERT_SRC, FRAG_RECT)?;

            // Bind the sampler uniform to texture unit 0 once per
            // program. Avoids re-setting on every blit.
            bind_sampler_zero(program_2d, "u_src");
            bind_sampler_zero(program_rect, "u_src");

            // Cache the rect-variant's size uniform location.
            let name = CString::new("u_src_size").ok()?;
            let rect_size_loc = gl::GetUniformLocation(program_rect, name.as_ptr());
            if rect_size_loc < 0 {
                error!("ShaderBlit: u_src_size uniform not found");
                gl::DeleteProgram(program_2d);
                gl::DeleteProgram(program_rect);
                return None;
            }
            let uv_name = CString::new("u_uv_scale").ok()?;
            let uv_scale_2d_loc = gl::GetUniformLocation(program_2d, uv_name.as_ptr());
            let uv_scale_rect_loc = gl::GetUniformLocation(program_rect, uv_name.as_ptr());
            if uv_scale_2d_loc < 0 || uv_scale_rect_loc < 0 {
                error!("ShaderBlit: u_uv_scale uniform not found");
                gl::DeleteProgram(program_2d);
                gl::DeleteProgram(program_rect);
                return None;
            }

            let mut vao: GLuint = 0;
            let mut vbo: GLuint = 0;
            gl::GenVertexArrays(1, &mut vao);
            gl::GenBuffers(1, &mut vbo);
            gl::BindVertexArray(vao);
            gl::BindBuffer(gl::ARRAY_BUFFER, vbo);
            gl::BufferData(
                gl::ARRAY_BUFFER,
                (QUAD.len() * std::mem::size_of::<f32>()) as GLsizeiptr,
                QUAD.as_ptr() as *const _,
                gl::STATIC_DRAW,
            );
            gl::EnableVertexAttribArray(0);
            gl::VertexAttribPointer(0, 2, gl::FLOAT, gl::FALSE, 0, ptr::null());
            gl::BindVertexArray(0);
            gl::BindBuffer(gl::ARRAY_BUFFER, 0);

            Some(Self {
                program_2d,
                program_rect,
                rect_size_loc,
                uv_scale_2d_loc,
                uv_scale_rect_loc,
                vao,
                vbo,
            })
        }
    }

    /// Sample-and-blit a `GL_TEXTURE_2D` source into the currently
    /// bound `DRAW_FRAMEBUFFER`. Caller is responsible for binding the
    /// draw FBO with the destination texture attached and for any
    /// upstream state (we save/restore the relevant pieces below).
    ///
    /// # Safety
    /// A current GL context must be active.
    pub unsafe fn blit_2d(
        &self,
        src_tex: GLuint,
        dst_w: u32,
        dst_h: u32,
        uv_scale: (f32, f32),
    ) {
        let uv_loc = self.uv_scale_2d_loc;
        self.blit_inner(
            self.program_2d,
            gl::TEXTURE_2D,
            src_tex,
            0,
            0,
            dst_w,
            dst_h,
            false,
            None,
            move |_| gl::Uniform2f(uv_loc, uv_scale.0, uv_scale.1),
        );
    }

    /// Sample-and-blit a `GL_TEXTURE_RECTANGLE` source. Rectangle
    /// samplers index in unnormalized `[0, w) × [0, h)` so the shader
    /// scales by the source dimensions.
    ///
    /// # Safety
    /// A current GL context must be active.
    #[cfg_attr(target_os = "windows", allow(dead_code))]
    pub unsafe fn blit_rect(
        &self,
        src_tex: GLuint,
        src_w: u32,
        src_h: u32,
        dst_w: u32,
        dst_h: u32,
        uv_scale: (f32, f32),
    ) {
        let size_loc = self.rect_size_loc;
        let uv_loc = self.uv_scale_rect_loc;
        self.blit_inner(
            self.program_rect,
            GL_TEXTURE_RECTANGLE,
            src_tex,
            0,
            0,
            dst_w,
            dst_h,
            false,
            None,
            move |_| {
                gl::Uniform2f(size_loc, src_w as f32, src_h as f32);
                gl::Uniform2f(uv_loc, uv_scale.0, uv_scale.1);
            },
        );
    }

    /// Raster-present a 2D texture into a viewport inside the currently bound
    /// host draw framebuffer. The caller's scissor state is honored.
    #[cfg_attr(target_os = "macos", allow(dead_code))]
    pub unsafe fn present_2d(
        &self,
        src_tex: GLuint,
        dst_x: i32,
        dst_y: i32,
        dst_w: u32,
        dst_h: u32,
        bilinear: bool,
    ) {
        self.blit_inner(
            self.program_2d,
            gl::TEXTURE_2D,
            src_tex,
            dst_x,
            dst_y,
            dst_w,
            dst_h,
            true,
            Some(if bilinear { gl::LINEAR } else { gl::NEAREST } as GLint),
            {
                let uv_loc = self.uv_scale_2d_loc;
                move |_| gl::Uniform2f(uv_loc, 1.0, 1.0)
            },
        );
    }

    /// Raster-present a rectangle texture into a viewport inside the
    /// currently bound host draw framebuffer. The caller's scissor is honored.
    #[allow(clippy::too_many_arguments)]
    #[cfg_attr(target_os = "windows", allow(dead_code))]
    pub unsafe fn present_rect(
        &self,
        src_tex: GLuint,
        src_w: u32,
        src_h: u32,
        dst_x: i32,
        dst_y: i32,
        dst_w: u32,
        dst_h: u32,
        bilinear: bool,
    ) {
        let size_loc = self.rect_size_loc;
        let uv_loc = self.uv_scale_rect_loc;
        self.blit_inner(
            self.program_rect,
            GL_TEXTURE_RECTANGLE,
            src_tex,
            dst_x,
            dst_y,
            dst_w,
            dst_h,
            true,
            Some(if bilinear { gl::LINEAR } else { gl::NEAREST } as GLint),
            move |_| {
                gl::Uniform2f(size_loc, src_w as f32, src_h as f32);
                gl::Uniform2f(uv_loc, 1.0, 1.0);
            },
        );
    }

    /// Whether this shader's GL names belong to the current context.
    pub unsafe fn is_valid(&self) -> bool {
        gl::IsProgram(self.program_2d) != 0 && gl::IsProgram(self.program_rect) != 0
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn blit_inner(
        &self,
        program: GLuint,
        src_target: GLenum,
        src_tex: GLuint,
        dst_x: i32,
        dst_y: i32,
        dst_w: u32,
        dst_h: u32,
        preserve_scissor: bool,
        texture_filter: Option<GLint>,
        set_extra_uniforms: impl FnOnce(GLuint),
    ) {
        // Snapshot the bits of GL state we touch so we don't surprise
        // the host's render flow on the way back out. The host may
        // re-bind everything on its next draw, but a defensive
        // restore here keeps the bridge composable with anything.
        let mut prev_program: GLint = 0;
        let mut prev_vao: GLint = 0;
        let mut prev_viewport: [GLint; 4] = [0; 4];
        let mut prev_active_tex: GLint = 0;
        let prev_blend = gl::IsEnabled(gl::BLEND);
        let prev_depth = gl::IsEnabled(gl::DEPTH_TEST);
        let prev_cull = gl::IsEnabled(gl::CULL_FACE);
        let prev_scissor = gl::IsEnabled(gl::SCISSOR_TEST);
        gl::GetIntegerv(gl::CURRENT_PROGRAM, &mut prev_program);
        gl::GetIntegerv(gl::VERTEX_ARRAY_BINDING, &mut prev_vao);
        gl::GetIntegerv(gl::VIEWPORT, prev_viewport.as_mut_ptr());
        gl::GetIntegerv(gl::ACTIVE_TEXTURE, &mut prev_active_tex);

        gl::UseProgram(program);
        set_extra_uniforms(program);
        gl::ActiveTexture(gl::TEXTURE0);
        gl::BindTexture(src_target, src_tex);
        let mut previous_filters = None;
        if let Some(filter) = texture_filter {
            let mut min_filter = 0;
            let mut mag_filter = 0;
            gl::GetTexParameteriv(src_target, gl::TEXTURE_MIN_FILTER, &mut min_filter);
            gl::GetTexParameteriv(src_target, gl::TEXTURE_MAG_FILTER, &mut mag_filter);
            gl::TexParameteri(src_target, gl::TEXTURE_MIN_FILTER, filter);
            gl::TexParameteri(src_target, gl::TEXTURE_MAG_FILTER, filter);
            previous_filters = Some((min_filter, mag_filter));
        }
        gl::Viewport(dst_x, dst_y, dst_w as GLsizei, dst_h as GLsizei);
        gl::Disable(gl::BLEND);
        gl::Disable(gl::DEPTH_TEST);
        gl::Disable(gl::CULL_FACE);
        if !preserve_scissor {
            gl::Disable(gl::SCISSOR_TEST);
        }
        gl::BindVertexArray(self.vao);
        gl::DrawArrays(gl::TRIANGLE_STRIP, 0, 4);

        if let Some((min_filter, mag_filter)) = previous_filters {
            gl::TexParameteri(src_target, gl::TEXTURE_MIN_FILTER, min_filter);
            gl::TexParameteri(src_target, gl::TEXTURE_MAG_FILTER, mag_filter);
        }

        // Restore.
        gl::BindVertexArray(prev_vao as GLuint);
        gl::BindTexture(src_target, 0);
        gl::ActiveTexture(prev_active_tex as GLenum);
        gl::Viewport(
            prev_viewport[0],
            prev_viewport[1],
            prev_viewport[2],
            prev_viewport[3],
        );
        if prev_blend == gl::TRUE {
            gl::Enable(gl::BLEND);
        }
        if prev_depth == gl::TRUE {
            gl::Enable(gl::DEPTH_TEST);
        }
        if prev_cull == gl::TRUE {
            gl::Enable(gl::CULL_FACE);
        }
        if prev_scissor == gl::TRUE {
            gl::Enable(gl::SCISSOR_TEST);
        }
        gl::UseProgram(prev_program as GLuint);
    }
}

impl Drop for ShaderBlit {
    fn drop(&mut self) {
        unsafe {
            gl::DeleteProgram(self.program_2d);
            gl::DeleteProgram(self.program_rect);
            gl::DeleteVertexArrays(1, &self.vao);
            gl::DeleteBuffers(1, &self.vbo);
        }
    }
}

unsafe fn bind_sampler_zero(program: GLuint, name: &str) {
    let cname = match CString::new(name) {
        Ok(c) => c,
        Err(_) => return,
    };
    gl::UseProgram(program);
    let loc = gl::GetUniformLocation(program, cname.as_ptr());
    if loc >= 0 {
        gl::Uniform1i(loc, 0);
    }
    gl::UseProgram(0);
}

unsafe fn compile_shader(stage: GLenum, src: &str) -> Option<GLuint> {
    let shader = gl::CreateShader(stage);
    if shader == 0 {
        error!("ShaderBlit: CreateShader returned 0");
        return None;
    }
    let csrc = CString::new(src).ok()?;
    let lengths: GLint = csrc.as_bytes().len() as GLint;
    gl::ShaderSource(
        shader,
        1,
        &csrc.as_ptr() as *const *const GLchar,
        &lengths,
    );
    gl::CompileShader(shader);

    let mut status: GLint = 0;
    gl::GetShaderiv(shader, gl::COMPILE_STATUS, &mut status);
    if status == 0 {
        let log = read_shader_info_log(shader);
        error!("ShaderBlit: shader compile failed ({stage:#x}): {log}");
        gl::DeleteShader(shader);
        return None;
    }

    Some(shader)
}

unsafe fn compile_program(vert: &str, frag: &str) -> Option<GLuint> {
    let vs = compile_shader(gl::VERTEX_SHADER, vert)?;
    let fs = compile_shader(gl::FRAGMENT_SHADER, frag)?;
    let program = gl::CreateProgram();
    if program == 0 {
        error!("ShaderBlit: CreateProgram returned 0");
        gl::DeleteShader(vs);
        gl::DeleteShader(fs);
        return None;
    }
    gl::AttachShader(program, vs);
    gl::AttachShader(program, fs);
    gl::LinkProgram(program);
    // Detach + delete component shaders unconditionally — once linked,
    // the program owns its own copy.
    gl::DetachShader(program, vs);
    gl::DetachShader(program, fs);
    gl::DeleteShader(vs);
    gl::DeleteShader(fs);

    let mut status: GLint = 0;
    gl::GetProgramiv(program, gl::LINK_STATUS, &mut status);
    if status == 0 {
        let log = read_program_info_log(program);
        warn!("ShaderBlit: program link failed: {log}");
        gl::DeleteProgram(program);
        return None;
    }

    Some(program)
}

unsafe fn read_shader_info_log(shader: GLuint) -> String {
    let mut len: GLint = 0;
    gl::GetShaderiv(shader, gl::INFO_LOG_LENGTH, &mut len);
    if len <= 0 {
        return String::new();
    }
    let mut buf = vec![0u8; len as usize];
    let mut written: GLsizei = 0;
    gl::GetShaderInfoLog(
        shader,
        len as GLsizei,
        &mut written,
        buf.as_mut_ptr() as *mut GLchar,
    );
    buf.truncate(written.max(0) as usize);
    String::from_utf8_lossy(&buf).into_owned()
}

unsafe fn read_program_info_log(program: GLuint) -> String {
    let mut len: GLint = 0;
    gl::GetProgramiv(program, gl::INFO_LOG_LENGTH, &mut len);
    if len <= 0 {
        return String::new();
    }
    let mut buf = vec![0u8; len as usize];
    let mut written: GLsizei = 0;
    gl::GetProgramInfoLog(
        program,
        len as GLsizei,
        &mut written,
        buf.as_mut_ptr() as *mut GLchar,
    );
    buf.truncate(written.max(0) as usize);
    String::from_utf8_lossy(&buf).into_owned()
}
