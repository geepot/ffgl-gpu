//! GPU context wrapping platform-specific device + loaded shader library.
//!
//! Created lazily on first draw. On macOS this holds a [`MetalDevice`] and the
//! compiled Metal shader library. On Windows it holds a [`Dx11Device`] (shaders
//! are loaded individually per-pipeline from bytecode).

use anyhow::Result;

#[cfg(target_os = "macos")]
use objc2::rc::Retained;
#[cfg(target_os = "macos")]
use objc2::runtime::ProtocolObject;
#[cfg(target_os = "macos")]
use objc2_metal::MTLLibrary;

/// GPU context wrapping platform-specific device + loaded shader library.
///
/// On macOS this contains a `MetalDevice` and the compiled shader library
/// (`.metallib`). On Windows it contains a `Dx11Device`; shaders are loaded
/// individually per-pipeline from compiled bytecode (`.cso`).
pub struct GpuContext {
    #[cfg(target_os = "macos")]
    pub(crate) device: gpu_interop::metal::MetalDevice,
    #[cfg(target_os = "macos")]
    pub(crate) library: Retained<ProtocolObject<dyn MTLLibrary>>,

    #[cfg(target_os = "windows")]
    pub(crate) device: gpu_interop::dx11::Dx11Device,
}

impl GpuContext {
    /// Create from embedded Metal shader library bytes.
    ///
    /// The `metallib_bytes` should come from [`include_metallib!`] which embeds
    /// the compiled `.metallib` at build time.
    #[cfg(target_os = "macos")]
    pub fn new(metallib_bytes: &[u8]) -> Result<Self> {
        use dispatch2::DispatchData;
        use objc2_metal::MTLDevice;

        let device = gpu_interop::metal::MetalDevice::new()
            .ok_or_else(|| anyhow::anyhow!("Failed to create Metal device"))?;

        let data = DispatchData::from_bytes(metallib_bytes);
        let library = device
            .device()
            .newLibraryWithData_error(&data)
            .map_err(|e| anyhow::anyhow!("Failed to load Metal library: {e}"))?;

        Ok(Self { device, library })
    }

    /// Create a DX11 GPU context.
    ///
    /// On Windows, shaders are loaded individually per-pipeline from compiled
    /// bytecode, so no library bytes are needed at construction time.
    #[cfg(target_os = "windows")]
    pub fn new() -> Result<Self> {
        let device = gpu_interop::dx11::Dx11Device::new()
            .ok_or_else(|| anyhow::anyhow!("Failed to create D3D11 device"))?;
        Ok(Self { device })
    }

    /// Borrow the underlying Metal device (macOS).
    #[cfg(target_os = "macos")]
    pub fn metal_device(&self) -> &gpu_interop::metal::MetalDevice {
        &self.device
    }

    /// Borrow the Metal shader library (macOS).
    #[cfg(target_os = "macos")]
    pub fn metal_library(&self) -> &ProtocolObject<dyn MTLLibrary> {
        &self.library
    }

    /// Borrow the underlying DX11 device (Windows).
    #[cfg(target_os = "windows")]
    pub fn dx11_device(&self) -> &gpu_interop::dx11::Dx11Device {
        &self.device
    }
}
