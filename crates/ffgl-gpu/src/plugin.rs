//! The [`GpuPlugin`] trait — the main interface for FFGL plugins that use GPU
//! compute/render.
//!
//! Plugin authors implement this trait on their effect struct. The framework
//! calls [`GpuPlugin::gpu_init`] once when the GPU context is first created,
//! then [`GpuPlugin::gpu_draw`] each frame.

use crate::context::GpuContext;
use ffgl_core::inputs::GLInput;
use ffgl_core::FFGLData;
use gpu_interop::GpuBridge;

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
///         &mut self, ctx: &GpuContext, bridge: &mut dyn GpuBridge,
///         data: &FFGLData, input: GLInput, frame: u64,
///     ) {
///         // Dispatch compute work using ctx, read/write via bridge textures
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
    /// The `bridge` provides access to the shared input/output textures. The
    /// plugin should:
    /// 1. Read from the bridge's input texture (the host's frame, already
    ///    blitted)
    /// 2. Perform GPU compute/render work
    /// 3. Write results to the bridge's output texture
    ///
    /// The `frame` counter is monotonically increasing and can be used for
    /// animation or temporal effects.
    fn gpu_draw(
        &mut self,
        ctx: &GpuContext,
        bridge: &mut dyn GpuBridge,
        data: &FFGLData,
        input: &GLInput<'_>,
        frame: u64,
    );
}
