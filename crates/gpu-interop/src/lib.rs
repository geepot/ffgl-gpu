//! GL to Metal and GL to DX11 texture bridging.
//!
//! This crate defines the [`GpuBridge`] trait, a common interface for
//! transferring OpenGL textures to platform-specific GPU APIs (Metal on macOS,
//! Direct3D 11 on Windows) and back.

pub mod bridge;
pub use bridge::GpuBridge;

// Platform-specific implementations.
// These modules will be populated in subsequent tasks.

#[cfg(target_os = "macos")]
pub mod metal;

#[cfg(target_os = "windows")]
pub mod dx11;
