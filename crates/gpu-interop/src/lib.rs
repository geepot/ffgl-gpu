//! GL to Metal and GL to DX11 texture bridging.
//!
//! This crate defines the [`GpuBridge`] trait, a common interface for
//! transferring OpenGL textures to platform-specific GPU APIs (Metal on macOS,
//! Direct3D 11 on Windows) and back.

pub mod bridge;
pub use bridge::GpuBridge;

// Shared GL fallback used by both platform bridges when the host's
// input texture isn't FBO-attachable (compressed video formats, etc.).
#[cfg(any(target_os = "macos", target_os = "windows"))]
pub mod shader_blit;

// Platform-specific implementations.
#[cfg(target_os = "macos")]
pub mod metal;

#[cfg(target_os = "windows")]
pub mod dx11;
