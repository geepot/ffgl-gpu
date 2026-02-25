//! Metal bridge implementation (macOS via IOSurface).

pub mod device;
pub mod interop;

pub use device::MetalDevice;
pub use interop::GlMetalBridge;
