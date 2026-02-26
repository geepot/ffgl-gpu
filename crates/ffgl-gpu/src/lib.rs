#![allow(clippy::too_many_arguments)]

//! High-level FFGL GPU plugin framework.
//!
//! This crate ties together [`ffgl_core`] (host protocol), [`ffgl_glium`]
//! (OpenGL context), and [`gpu_interop`] (GL-to-Metal/DX11 bridging) into a
//! single framework for writing GPU-accelerated FFGL plugins.
//!
//! # Overview
//!
//! - [`GpuContext`] wraps the platform GPU device and shader library.
//! - [`ComputePipeline`] / [`RenderPipeline`] are compiled pipeline states.
//! - [`GpuBuffer`] is a GPU buffer for structured compute data.
//! - [`GpuPlugin`] is the trait plugin authors implement.
//! - [`draw_gpu_effect`] is the main entry point that manages the
//!   double-buffered draw loop.
//! - [`build_support`] provides shader compilation helpers for `build.rs`.
//!
//! # Build-time shader compilation
//!
//! Use [`build_support::compile_metal_shaders`] and
//! [`build_support::compile_hlsl_shaders`] in your plugin's `build.rs`, then
//! load the compiled shaders with [`include_metallib!`] and
//! [`include_hlsl_shader!`].

pub mod buffer;
pub mod build_support;
pub mod bytes;
pub mod context;
pub mod dispatch;
pub mod drawing;
pub mod pipeline;
pub mod plugin;

// Re-export primary types at crate root for convenience.
pub use buffer::GpuBuffer;
pub use bytes::AsBytes;
pub use context::GpuContext;
pub use dispatch::{Binding, CommandBuffer, PendingWork};
pub use drawing::{draw_gpu_effect, ensure_instance_gl_resources, validate_gl_state_before_draw};
pub use pipeline::{ComputePipeline, RenderPipeline};
pub use plugin::{DrawInput, GpuPlugin};
