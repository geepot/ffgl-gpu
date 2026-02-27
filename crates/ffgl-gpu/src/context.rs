//! GPU context wrapping platform-specific device + loaded shader library.
//!
//! Created lazily on first draw. On macOS this holds a [`MetalDevice`] and the
//! compiled Metal shader library. On other platforms it holds a map of GLSL
//! shader sources keyed by entry-point name (compiled at runtime by the GL
//! driver).

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
/// (`.metallib`). On other platforms it holds GLSL shader source strings
/// keyed by entry-point name; shaders are compiled at runtime by the GL
/// driver when pipelines are created.
pub struct GpuContext {
    #[cfg(target_os = "macos")]
    pub(crate) device: gpu_interop::metal::MetalDevice,
    #[cfg(target_os = "macos")]
    pub(crate) library: Retained<ProtocolObject<dyn MTLLibrary>>,

    #[cfg(not(target_os = "macos"))]
    pub(crate) shader_sources: std::collections::HashMap<String, String>,
}

impl GpuContext {
    /// Create from embedded Metal shader library bytes (macOS).
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

    /// Create from GLSL shader sources (non-macOS).
    ///
    /// The `sources` slice contains `(name, glsl_source)` pairs. Each name
    /// corresponds to a WGSL entry point transpiled to GLSL at build time.
    /// Shaders are compiled by the GL driver when pipelines are created.
    #[cfg(not(target_os = "macos"))]
    pub fn new(sources: &[(&str, &str)]) -> Result<Self> {
        let shader_sources = sources
            .iter()
            .map(|(name, src)| (name.to_string(), src.to_string()))
            .collect();
        Ok(Self { shader_sources })
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
}
