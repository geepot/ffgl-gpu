//! GPU pipeline types for compute and render operations.
//!
//! These wrap platform-specific pipeline state objects created from shader
//! functions in the GPU context's shader library.

#[cfg(target_os = "macos")]
use objc2::rc::Retained;
#[cfg(target_os = "macos")]
use objc2::runtime::ProtocolObject;
#[cfg(target_os = "macos")]
use objc2_metal::{MTLBuffer, MTLComputePipelineState, MTLRenderPipelineState};

/// A compiled compute pipeline (kernel).
///
/// On macOS this wraps a `MTLComputePipelineState`. On other platforms it
/// wraps a GL program handle (linked compute shader).
pub struct ComputePipeline {
    #[cfg(target_os = "macos")]
    pub(crate) state: Retained<ProtocolObject<dyn MTLComputePipelineState>>,

    #[cfg(not(target_os = "macos"))]
    pub(crate) program: gl::types::GLuint,
}

/// A compiled render pipeline (vertex + fragment).
///
/// On macOS this wraps a `MTLRenderPipelineState` and a fullscreen quad vertex
/// buffer. On other platforms it wraps a GL program handle plus a fullscreen
/// quad VAO/VBO.
#[allow(dead_code)]
pub struct RenderPipeline {
    #[cfg(target_os = "macos")]
    pub(crate) state: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    /// Fullscreen quad vertex buffer (4 vertices: position + texcoord).
    #[cfg(target_os = "macos")]
    pub(crate) quad_vb: Retained<ProtocolObject<dyn MTLBuffer>>,

    #[cfg(not(target_os = "macos"))]
    pub(crate) program: gl::types::GLuint,
    #[cfg(not(target_os = "macos"))]
    pub(crate) quad_vao: gl::types::GLuint,
    #[cfg(not(target_os = "macos"))]
    pub(crate) quad_vb: gl::types::GLuint,
}
