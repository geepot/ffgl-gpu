//! GL compute bridge for non-macOS platforms.
//!
//! On these platforms the compute shader runs in the same GL context as the
//! host, so no cross-API texture bridging is needed. The bridge only manages
//! double-buffered output scratch textures and blit-back to the host FBO.

mod bridge;

pub use bridge::GlComputeBridge;
