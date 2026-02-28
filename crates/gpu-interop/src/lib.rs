//! GL-to-GPU texture bridging.
//!
//! This crate defines the [`GpuBridge`] trait, a common interface for
//! double-buffered GL texture management. On macOS, textures are bridged to
//! Metal via IOSurface. On other platforms, GL compute runs in the host's
//! own context, so only output scratch textures and blit-back are needed.

pub mod bridge;
pub use bridge::GpuBridge;

// Platform-specific implementations.

#[cfg(target_os = "macos")]
pub mod metal;

#[cfg(target_os = "windows")]
pub mod dx11;

#[cfg(not(target_os = "macos"))]
pub mod gl_compute;
