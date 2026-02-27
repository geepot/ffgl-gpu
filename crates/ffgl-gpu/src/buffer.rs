//! GPU buffer type for structured data used in compute shaders.
//!
//! On macOS this wraps a `MTLBuffer`. On other platforms it wraps a GL
//! buffer object (SSBO) for compute shader access.

#[cfg(target_os = "macos")]
use objc2::rc::Retained;
#[cfg(target_os = "macos")]
use objc2::runtime::ProtocolObject;
#[cfg(target_os = "macos")]
use objc2_metal::MTLBuffer;

/// A GPU buffer for structured data used in compute shaders.
///
/// On macOS this is a `MTLBuffer` allocated with `StorageModePrivate`. On
/// other platforms it is an OpenGL buffer object bound as a Shader Storage
/// Buffer Object (SSBO) for read/write access in compute shaders.
pub struct GpuBuffer {
    /// Total size in bytes.
    pub(crate) size: usize,

    #[cfg(target_os = "macos")]
    pub(crate) metal: Retained<ProtocolObject<dyn MTLBuffer>>,

    #[cfg(not(target_os = "macos"))]
    pub(crate) gl_buffer: gl::types::GLuint,
}

impl GpuBuffer {
    /// Total size of this buffer in bytes.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Borrow the underlying Metal buffer (macOS).
    #[cfg(target_os = "macos")]
    pub fn metal_buffer(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.metal
    }
}
