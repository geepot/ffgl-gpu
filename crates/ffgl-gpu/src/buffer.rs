//! GPU buffer type for structured data used in compute shaders.
//!
//! On macOS this wraps a `MTLBuffer`. On Windows it wraps an `ID3D11Buffer`
//! with associated UAV and SRV views for compute shader access.

#[cfg(target_os = "macos")]
use objc2::rc::Retained;
#[cfg(target_os = "macos")]
use objc2::runtime::ProtocolObject;
#[cfg(target_os = "macos")]
use objc2_metal::MTLBuffer;

/// A GPU buffer for structured data used in compute shaders.
///
/// On macOS this is a `MTLBuffer` allocated with `StorageModePrivate`. On
/// Windows it is an `ID3D11Buffer` with both an `UnorderedAccessView` (UAV)
/// and a `ShaderResourceView` (SRV) so it can be bound for both read and write
/// access in compute shaders.
pub struct GpuBuffer {
    /// Total size in bytes.
    pub(crate) size: usize,

    #[cfg(target_os = "macos")]
    pub(crate) metal: Retained<ProtocolObject<dyn MTLBuffer>>,

    #[cfg(target_os = "windows")]
    pub(crate) dx11_buffer: windows::Win32::Graphics::Direct3D11::ID3D11Buffer,
    #[cfg(target_os = "windows")]
    pub(crate) dx11_uav: windows::Win32::Graphics::Direct3D11::ID3D11UnorderedAccessView,
    #[cfg(target_os = "windows")]
    pub(crate) dx11_srv: windows::Win32::Graphics::Direct3D11::ID3D11ShaderResourceView,
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

    /// Borrow the underlying DX11 buffer (Windows).
    #[cfg(target_os = "windows")]
    pub fn dx11_buffer(&self) -> &windows::Win32::Graphics::Direct3D11::ID3D11Buffer {
        &self.dx11_buffer
    }

    /// Borrow the DX11 unordered access view (Windows).
    #[cfg(target_os = "windows")]
    pub fn dx11_uav(&self) -> &windows::Win32::Graphics::Direct3D11::ID3D11UnorderedAccessView {
        &self.dx11_uav
    }

    /// Borrow the DX11 shader resource view (Windows).
    #[cfg(target_os = "windows")]
    pub fn dx11_srv(&self) -> &windows::Win32::Graphics::Direct3D11::ID3D11ShaderResourceView {
        &self.dx11_srv
    }
}
