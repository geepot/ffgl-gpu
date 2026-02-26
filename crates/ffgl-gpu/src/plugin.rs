//! The [`GpuPlugin`] trait and [`DrawInput`] — the main interface for FFGL
//! plugins that use GPU compute/render.
//!
//! Plugin authors implement [`GpuPlugin`] on their effect struct. The framework
//! calls [`GpuPlugin::gpu_init`] once when the GPU context is first created,
//! then [`GpuPlugin::gpu_draw`] each frame with a [`DrawInput`] containing
//! pre-extracted platform textures.

use crate::context::GpuContext;
use ffgl_core::FFGLData;

// ---------------------------------------------------------------------------
// DrawInput — platform-specific pre-extracted textures
// ---------------------------------------------------------------------------

#[cfg(target_os = "macos")]
mod draw_input_impl {
    use gpu_interop::metal::GlMetalBridge;
    use objc2::runtime::ProtocolObject;
    use objc2_metal::MTLTexture;

    /// Pre-extracted GPU textures for the current frame.
    ///
    /// The framework populates this from the bridge before calling
    /// [`super::GpuPlugin::gpu_draw`]. Public fields give direct access to
    /// the input/output textures; the raw bridge is available via
    /// [`metal_bridge`](DrawInput::metal_bridge) for advanced operations
    /// like [`store_command_buffer`](GlMetalBridge::store_command_buffer).
    pub struct DrawInput<'a> {
        /// Input texture (the host's frame, already blitted).
        pub input: &'a ProtocolObject<dyn MTLTexture>,
        /// Output texture (write your result here).
        pub output: &'a ProtocolObject<dyn MTLTexture>,
        /// Processing width in pixels.
        pub width: u32,
        /// Processing height in pixels.
        pub height: u32,
        pub(crate) bridge: &'a mut GlMetalBridge,
    }

    impl<'a> DrawInput<'a> {
        /// Access the underlying Metal bridge for advanced operations
        /// (e.g. `store_command_buffer`, `back_output_metal_texture`).
        pub fn metal_bridge(&mut self) -> &mut GlMetalBridge {
            self.bridge
        }
    }
}

#[cfg(target_os = "windows")]
mod draw_input_impl {
    use gpu_interop::dx11::GlDx11Bridge;
    use windows::Win32::Graphics::Direct3D11::*;

    /// Pre-extracted GPU textures for the current frame.
    ///
    /// The framework populates this from the bridge before calling
    /// [`super::GpuPlugin::gpu_draw`]. Public fields give direct access to
    /// the input/output views; the raw bridge is available via
    /// [`dx11_bridge`](DrawInput::dx11_bridge) for advanced operations.
    pub struct DrawInput<'a> {
        /// Input SRV (read the host's frame from this).
        pub input_srv: ID3D11ShaderResourceView,
        /// Output UAV (write your result here).
        pub output_uav: ID3D11UnorderedAccessView,
        /// Output texture (use as render target for render pipelines).
        pub output_texture: ID3D11Texture2D,
        /// Processing width in pixels.
        pub width: u32,
        /// Processing height in pixels.
        pub height: u32,
        pub(crate) bridge: &'a mut GlDx11Bridge,
    }

    impl<'a> DrawInput<'a> {
        /// Access the underlying DX11 bridge for advanced operations
        /// (e.g. `device`, `context`, `back_output_srv`).
        pub fn dx11_bridge(&mut self) -> &mut GlDx11Bridge {
            self.bridge
        }
    }
}

#[cfg(not(any(target_os = "macos", target_os = "windows")))]
mod draw_input_impl {
    /// Stub for unsupported platforms.
    pub struct DrawInput<'a> {
        /// Processing width in pixels.
        pub width: u32,
        /// Processing height in pixels.
        pub height: u32,
        pub(crate) _lifetime: std::marker::PhantomData<&'a ()>,
    }
}

pub use draw_input_impl::DrawInput;

// ---------------------------------------------------------------------------
// GpuPlugin trait
// ---------------------------------------------------------------------------

/// Trait for FFGL plugins that use GPU compute/render.
///
/// Implementors should store their pipeline states, buffers, and other GPU
/// resources and create them in [`GpuPlugin::gpu_init`].
///
/// # Example
///
/// ```rust,ignore
/// struct MyEffect {
///     compute_pipeline: Option<ComputePipeline>,
/// }
///
/// impl GpuPlugin for MyEffect {
///     fn gpu_init(&mut self, ctx: &GpuContext) -> anyhow::Result<()> {
///         self.compute_pipeline = Some(ctx.create_compute_pipeline("my_kernel")?);
///         Ok(())
///     }
///
///     fn gpu_draw(
///         &mut self, ctx: &GpuContext, input: &mut DrawInput<'_>,
///         data: &FFGLData, frame: u64,
///     ) {
///         // input.input / input.output are ready to use — no downcasting needed
///     }
/// }
/// ```
pub trait GpuPlugin: Send + Sync + 'static {
    /// Called once when the GPU context is first available.
    ///
    /// Create pipelines, buffers, and other GPU resources here. The context
    /// provides access to the platform GPU device and shader library.
    fn gpu_init(&mut self, ctx: &GpuContext) -> anyhow::Result<()>;

    /// Called each frame to perform GPU rendering.
    ///
    /// The [`DrawInput`] provides pre-extracted input/output textures for the
    /// current platform. The plugin should:
    /// 1. Read from the input texture (the host's frame, already blitted)
    /// 2. Perform GPU compute/render work
    /// 3. Write results to the output texture
    ///
    /// For advanced operations (e.g. `store_command_buffer` on Metal), use
    /// `input.metal_bridge()` or `input.dx11_bridge()`.
    ///
    /// The `frame` counter is monotonically increasing and can be used for
    /// animation or temporal effects.
    fn gpu_draw(
        &mut self,
        ctx: &GpuContext,
        input: &mut DrawInput<'_>,
        data: &FFGLData,
        frame: u64,
    );
}
