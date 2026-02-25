//! Raw OpenGL backend for glium, wrapping the host-provided GL context.

use std::sync::Once;

pub(crate) static GL_INIT_ONCE: Once = Once::new();

#[derive(Debug)]
pub(crate) struct RawGlBackend {
    pub(crate) size: (u32, u32),
}

impl RawGlBackend {
    /// Create a new backend wrapping the host GL context.
    ///
    /// GL function pointers are loaded exactly once via `gl_loader`.
    pub(crate) fn new(size: (u32, u32)) -> Self {
        GL_INIT_ONCE.call_once(|| {
            gl_loader::init_gl();
            gl::load_with(|s| gl_loader::get_proc_address(s).cast());
        });

        Self { size }
    }
}

/// # Safety
///
/// This implementation assumes it is only used inside FFGL host callbacks where
/// the host has already made the correct OpenGL context current. Using it outside
/// that context will cause undefined behavior.
unsafe impl glium::backend::Backend for RawGlBackend {
    fn swap_buffers(&self) -> Result<(), glium::SwapBuffersError> {
        Ok(())
    }

    unsafe fn get_proc_address(&self, symbol: &str) -> *const std::os::raw::c_void {
        gl_loader::get_proc_address(symbol).cast()
    }

    fn get_framebuffer_dimensions(&self) -> (u32, u32) {
        self.size
    }

    fn is_current(&self) -> bool {
        true
    }

    unsafe fn make_current(&self) {}

    fn resize(&self, _new_size: (u32, u32)) {}
}
