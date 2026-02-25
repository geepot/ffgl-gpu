//! Generic D3D11 device wrapper.
//!
//! Holds only the D3D11 device, immediate context, and a GPU event query for
//! synchronization -- no shader pipelines or application-specific constant
//! buffers.  Consumers (e.g. `ffgl-gpu`) are responsible for loading their own
//! shaders and building pipelines on top of this device.

use windows::Win32::Foundation::HMODULE;
use windows::Win32::Graphics::Direct3D::*;
use windows::Win32::Graphics::Direct3D11::*;

use tracing::{debug, error};

/// A generic D3D11 device with an immediate context and GPU sync query.
///
/// Created via [`Dx11Device::new()`] which tries hardware acceleration first
/// and falls back to WARP.  This struct intentionally contains *no* shader
/// pipelines or application-specific constant buffers -- those belong to
/// higher-level crates that know what shaders they need.
pub struct Dx11Device {
    device: ID3D11Device,
    context: ID3D11DeviceContext,
    /// GPU sync query (`D3D11_QUERY_EVENT`) for waiting on dispatch completion.
    gpu_query: ID3D11Query,
}

impl Dx11Device {
    /// Create a new D3D11 device using hardware acceleration, falling back to
    /// WARP if hardware is unavailable (e.g. CI/headless environments).
    ///
    /// Returns `None` if D3D11 is unavailable with any driver type.
    pub fn new() -> Option<Self> {
        let mut device = None;
        let mut context = None;

        // Try HARDWARE first, fall back to WARP for CI/headless
        let driver_types = [D3D_DRIVER_TYPE_HARDWARE, D3D_DRIVER_TYPE_WARP];
        let mut created = false;

        for &driver_type in &driver_types {
            let hr = unsafe {
                D3D11CreateDevice(
                    None,
                    driver_type,
                    HMODULE::default(),
                    D3D11_CREATE_DEVICE_SINGLETHREADED,
                    Some(&[D3D_FEATURE_LEVEL_11_0]),
                    D3D11_SDK_VERSION,
                    Some(&mut device as *mut _),
                    None,
                    Some(&mut context as *mut _),
                )
            };
            if hr.is_ok() {
                debug!("D3D11 device created with driver type {:?}", driver_type);
                created = true;
                break;
            }
        }

        if !created {
            error!("Failed to create D3D11 device with any driver type");
            return None;
        }

        let device = device?;
        let context = context?;
        let gpu_query = create_event_query(&device)?;

        Some(Self {
            device,
            context,
            gpu_query,
        })
    }

    /// Borrow the underlying `ID3D11Device`.
    pub fn device(&self) -> &ID3D11Device {
        &self.device
    }

    /// Borrow the immediate device context.
    pub fn context(&self) -> &ID3D11DeviceContext {
        &self.context
    }

    /// Borrow the GPU event query used for synchronization.
    pub fn query(&self) -> &ID3D11Query {
        &self.gpu_query
    }
}

/// Create a dynamic constant buffer of the given size (rounded up to 16-byte
/// alignment).
///
/// This is a general-purpose helper that consumers can use to create their own
/// constant buffers for shader parameters.
pub fn create_dynamic_cbuf(device: &ID3D11Device, size: usize) -> Option<ID3D11Buffer> {
    let aligned_size = (size + 15) & !15;
    let desc = D3D11_BUFFER_DESC {
        ByteWidth: aligned_size as u32,
        Usage: D3D11_USAGE_DYNAMIC,
        BindFlags: D3D11_BIND_CONSTANT_BUFFER.0 as u32,
        CPUAccessFlags: D3D11_CPU_ACCESS_WRITE.0 as u32,
        ..Default::default()
    };
    let mut buf = None;
    unsafe { device.CreateBuffer(&desc, None, Some(&mut buf as *mut _)) }.ok()?;
    buf
}

/// Create a `D3D11_QUERY_EVENT` for GPU synchronization.
pub fn create_event_query(device: &ID3D11Device) -> Option<ID3D11Query> {
    let desc = D3D11_QUERY_DESC {
        Query: D3D11_QUERY_EVENT,
        ..Default::default()
    };
    let mut query = None;
    unsafe { device.CreateQuery(&desc, Some(&mut query as *mut _)) }.ok()?;
    query
}
