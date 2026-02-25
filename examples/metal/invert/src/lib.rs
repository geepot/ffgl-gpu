//! Invert FFGL plugin example.
//!
//! Demonstrates a render pipeline (vertex + fragment shader) that inverts the
//! colors of the input image using a fullscreen quad pass.

use std::sync::atomic::{AtomicU64, Ordering};

use ffgl_core::handler::simplified::{SimpleFFGLHandler, SimpleFFGLInstance};
use ffgl_core::info::{PluginInfo, PluginType};
use ffgl_core::{FFGLData, GLInput};
use ffgl_glium::FFGLGlium;

static NEXT_INSTANCE_ID: AtomicU64 = AtomicU64::new(1);
use ffgl_gpu::pipeline::RenderPipeline;
use ffgl_gpu::plugin::GpuPlugin;
use ffgl_gpu::{GpuContext, draw_gpu_effect};

/// Compiled Metal shader library, embedded at build time.
#[cfg(target_os = "macos")]
const METALLIB_BYTES: &[u8] = ffgl_gpu::include_metallib!();

#[cfg(not(target_os = "macos"))]
const METALLIB_BYTES: &[u8] = &[];

/// Inner GPU state, separate from the glium context to avoid double-borrow.
struct GpuState {
    pipeline: Option<RenderPipeline>,
}

impl GpuPlugin for GpuState {
    fn gpu_init(&mut self, ctx: &GpuContext) -> anyhow::Result<()> {
        self.pipeline = Some(ctx.create_render_pipeline("invert_vertex", "invert_fragment")?);
        Ok(())
    }

    fn gpu_draw(
        &mut self,
        ctx: &GpuContext,
        bridge: &mut dyn gpu_interop::GpuBridge,
        _data: &FFGLData,
        _input: &GLInput<'_>,
        _frame: u64,
    ) {
        #[cfg(target_os = "macos")]
        {
            use gpu_interop::metal::GlMetalBridge;

            let pipeline = match &self.pipeline {
                Some(p) => p,
                None => return,
            };

            let metal_bridge = match bridge.as_any_mut().downcast_mut::<GlMetalBridge>() {
                Some(b) => b,
                None => return,
            };

            let input_tex = match metal_bridge.input_metal_texture() {
                Some(t) => t,
                None => return,
            };
            let output_tex = match metal_bridge.output_metal_texture() {
                Some(t) => t,
                None => return,
            };

            let pending = match ctx.dispatch_render(pipeline, output_tex, &[input_tex], &[]) {
                Ok(p) => p,
                Err(_) => return,
            };

            metal_bridge.store_command_buffer(pending.into_command_buffer());
        }

        #[cfg(not(target_os = "macos"))]
        {
            let _ = (ctx, bridge);
        }
    }
}

// SAFETY: FFGL plugins are called single-threaded from the host.
unsafe impl Send for GpuState {}
unsafe impl Sync for GpuState {}

pub struct Invert {
    glium: FFGLGlium,
    gpu: GpuState,
    frame_counter: u64,
    instance_id: u64,
}

// SAFETY: FFGL plugins are called single-threaded from the host.
unsafe impl Send for Invert {}
unsafe impl Sync for Invert {}

impl SimpleFFGLInstance for Invert {
    fn new(inst_data: &FFGLData) -> Self {
        Self {
            glium: FFGLGlium::new(inst_data),
            gpu: GpuState { pipeline: None },
            frame_counter: 0,
            instance_id: NEXT_INSTANCE_ID.fetch_add(1, Ordering::Relaxed),
        }
    }

    fn plugin_info() -> PluginInfo {
        PluginInfo {
            unique_id: *b"INVT",
            name: *b"Invert\0\0\0\0\0\0\0\0\0\0",
            ty: PluginType::Effect,
            about: "Color inversion via render pipeline".to_string(),
            description: "Inverts colors using a vertex/fragment shader pair".to_string(),
        }
    }

    fn draw(&mut self, data: &FFGLData, frame_data: GLInput) {
        self.frame_counter = self.frame_counter.wrapping_add(1);
        let id = self.instance_id;
        draw_gpu_effect(
            &mut self.gpu,
            id,
            &mut self.glium,
            data,
            frame_data,
            self.frame_counter,
            1.0,
            1.0,
            METALLIB_BYTES,
        );
    }
}

ffgl_core::plugin_main!(SimpleFFGLHandler<Invert>);
