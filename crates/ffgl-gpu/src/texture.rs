//! Opaque GPU texture handle.

/// Platform-agnostic GPU texture handle.
///
/// On macOS wraps a `MTLTexture`. On Windows wraps a `GLuint` texture name
/// with associated format info for image unit binding.
pub struct GpuTexture {
    #[cfg(target_os = "macos")]
    pub(crate) metal: *const objc2::runtime::ProtocolObject<dyn objc2_metal::MTLTexture>,

    #[cfg(not(target_os = "macos"))]
    pub(crate) gl_name: gl::types::GLuint,
    #[cfg(not(target_os = "macos"))]
    pub(crate) gl_format: gl::types::GLenum,

    pub(crate) width: u32,
    pub(crate) height: u32,
}

#[cfg(target_os = "macos")]
impl GpuTexture {
    pub(crate) fn metal_ref(
        &self,
    ) -> &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLTexture> {
        unsafe { &*self.metal }
    }
}

// SAFETY: Texture handles are only accessed from the thread that created them
// (FFGL host guarantees single-threaded plugin calls).
unsafe impl Send for GpuTexture {}
unsafe impl Sync for GpuTexture {}
