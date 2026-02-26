//! Passthrough FFGL plugin example.
//!
//! Demonstrates the simplest possible GPU compute plugin: a single compute
//! kernel that copies the input texture to the output texture pixel-for-pixel.

use std::sync::atomic::{AtomicU64, Ordering};

use ffgl_core::handler::simplified::{SimpleFFGLHandler, SimpleFFGLInstance};
use ffgl_core::info::{PluginInfo, PluginType};
use ffgl_core::{FFGLData, GLInput};
use ffgl_glium::FFGLGlium;

static NEXT_INSTANCE_ID: AtomicU64 = AtomicU64::new(1);
use ffgl_gpu::pipeline::ComputePipeline;
use ffgl_gpu::plugin::GpuPlugin;
use ffgl_gpu::{DrawInput, GpuContext, draw_gpu_effect};

/// Compiled Metal shader library, embedded at build time.
#[cfg(target_os = "macos")]
const METALLIB_BYTES: &[u8] = ffgl_gpu::include_metallib!();

#[cfg(not(target_os = "macos"))]
const METALLIB_BYTES: &[u8] = &[];

/// Inner GPU state, separate from the glium context to avoid double-borrow
/// when calling [`draw_gpu_effect`].
struct GpuState {
    pipeline: Option<ComputePipeline>,
}

impl GpuPlugin for GpuState {
    fn gpu_init(&mut self, ctx: &GpuContext) -> anyhow::Result<()> {
        self.pipeline = Some(ctx.create_compute_pipeline("passthrough")?);
        Ok(())
    }

    fn gpu_draw(
        &mut self,
        ctx: &GpuContext,
        input: &mut DrawInput<'_>,
        _data: &FFGLData,
        _frame: u64,
    ) {
        #[cfg(target_os = "macos")]
        {
            let pipeline = match &self.pipeline {
                Some(p) => p,
                None => return,
            };

            let pending = match ctx.dispatch_compute(
                pipeline,
                &[input.input, input.output],
                &[],
                &[],
                (input.width as usize, input.height as usize),
                (16, 16),
            ) {
                Ok(p) => p,
                Err(_) => return,
            };
            input.metal_bridge().store_command_buffer(pending.into_command_buffer());
        }

        #[cfg(not(target_os = "macos"))]
        {
            let _ = (ctx, input);
        }
    }
}

// SAFETY: FFGL plugins are called single-threaded from the host.
unsafe impl Send for GpuState {}
unsafe impl Sync for GpuState {}

pub struct Passthrough {
    glium: FFGLGlium,
    gpu: GpuState,
    frame_counter: u64,
    instance_id: u64,
}

// SAFETY: FFGL plugins are called single-threaded from the host.
unsafe impl Send for Passthrough {}
unsafe impl Sync for Passthrough {}

impl SimpleFFGLInstance for Passthrough {
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
            unique_id: *b"PASS",
            name: *b"Passthrough\0\0\0\0\0",
            ty: PluginType::Effect,
            about: "Passthrough GPU compute example".to_string(),
            description: "Copies input to output via a Metal/DX11 compute shader".to_string(),
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

ffgl_core::plugin_main!(SimpleFFGLHandler<Passthrough>);
