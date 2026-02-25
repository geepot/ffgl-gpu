//! DX11 bridge implementation (Windows via WGL_NV_DX_interop2).

pub mod device;
pub mod interop;

pub use device::Dx11Device;
pub use interop::GlDx11Bridge;
