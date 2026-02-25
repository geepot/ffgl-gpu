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
/// On macOS this wraps a `MTLComputePipelineState`. On Windows it wraps an
/// `ID3D11ComputeShader`.
pub struct ComputePipeline {
    #[cfg(target_os = "macos")]
    pub(crate) state: Retained<ProtocolObject<dyn MTLComputePipelineState>>,

    #[cfg(target_os = "windows")]
    pub(crate) shader: windows::Win32::Graphics::Direct3D11::ID3D11ComputeShader,
}

/// A compiled render pipeline (vertex + fragment).
///
/// On macOS this wraps a `MTLRenderPipelineState` and a fullscreen quad vertex
/// buffer. On Windows it wraps vertex and pixel shaders plus an input layout
/// and vertex buffer.
#[allow(dead_code)]
pub struct RenderPipeline {
    #[cfg(target_os = "macos")]
    pub(crate) state: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    /// Fullscreen quad vertex buffer (4 vertices: position + texcoord).
    #[cfg(target_os = "macos")]
    pub(crate) quad_vb: Retained<ProtocolObject<dyn MTLBuffer>>,

    #[cfg(target_os = "windows")]
    pub(crate) vs: windows::Win32::Graphics::Direct3D11::ID3D11VertexShader,
    #[cfg(target_os = "windows")]
    pub(crate) ps: windows::Win32::Graphics::Direct3D11::ID3D11PixelShader,
    #[cfg(target_os = "windows")]
    pub(crate) input_layout: windows::Win32::Graphics::Direct3D11::ID3D11InputLayout,
    #[cfg(target_os = "windows")]
    pub(crate) quad_vb: windows::Win32::Graphics::Direct3D11::ID3D11Buffer,
}
