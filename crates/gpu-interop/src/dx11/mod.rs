//! DX11 bridge implementation (Windows via WGL_NV_DX_interop2).

pub mod device;
pub mod interop;

pub use device::{Dx11Device, create_dynamic_cbuf};
pub use interop::GlDx11Bridge;
