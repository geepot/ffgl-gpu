//! The [`GpuPlugin`] trait and [`DrawInput`] — the main interface for FFGL
//! plugins that use GPU compute/render.
//!
//! Plugin authors implement [`GpuPlugin`] on their effect struct. The framework
//! calls [`GpuPlugin::gpu_init`] once when the GPU context is first created,
//! then [`GpuPlugin::gpu_draw`] each frame with a [`DrawInput`] containing
//! pre-extracted platform textures.

use crate::context::GpuContext;
use crate::texture::GpuTexture;
use ffgl_core::FFGLData;

// ---------------------------------------------------------------------------
// DrawInput — unified pre-extracted textures
// ---------------------------------------------------------------------------

/// Pre-extracted GPU textures for the current frame.
///
/// The framework populates this from the bridge before calling
/// [`GpuPlugin::gpu_draw`]. Public fields give direct access to the
/// platform-agnostic input/output textures.
///
/// After performing GPU work via [`GpuContext`] dispatch methods, call
/// [`store_pending`](DrawInput::store_pending) with the returned
/// [`PendingWork`](crate::dispatch::PendingWork) so the framework can
/// synchronise the double-buffered pipeline.
pub struct DrawInput<'a> {
    /// Input texture (the host's frame, already blitted).
    pub input: &'a GpuTexture,
    /// Output texture (write your result here).
    pub output: &'a GpuTexture,
    /// Processing width in pixels.
    pub width: u32,
    /// Processing height in pixels.
    pub height: u32,
    /// Internal: bridge stores pending work for double-buffer sync.
    pub(crate) pending_work: Option<crate::dispatch::PendingWork>,
}

impl DrawInput<'_> {
    /// Store completed GPU work for the framework's double-buffer pipeline.
    ///
    /// Call this after dispatching compute or render work. The framework
    /// will use the stored [`PendingWork`](crate::dispatch::PendingWork) to
    /// synchronise output before presenting.
    pub fn store_pending(&mut self, work: crate::dispatch::PendingWork) {
        self.pending_work = Some(work);
    }
}

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
    /// After dispatching GPU work, call [`DrawInput::store_pending`] with
    /// the returned [`PendingWork`](crate::dispatch::PendingWork).
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
