//! Invert FFGL plugin example (WGSL transpiled).
//!
//! Inverts the colors of the input image via a single compute kernel.

use std::sync::atomic::{AtomicU64, Ordering};

use ffgl_core::handler::simplified::{SimpleFFGLHandler, SimpleFFGLInstance};
use ffgl_core::info::{PluginInfo, PluginType};
use ffgl_core::{FFGLData, GLInput};
use ffgl_glium::FFGLGlium;
use ffgl_gpu::pipeline::ComputePipeline;
use ffgl_gpu::plugin::GpuPlugin;
use ffgl_gpu::{DrawInput, GpuContext, draw_gpu_effect};

static NEXT_INSTANCE_ID: AtomicU64 = AtomicU64::new(1);

#[cfg(target_os = "macos")]
const METALLIB_BYTES: &[u8] = ffgl_gpu::include_metallib!();

#[cfg(not(target_os = "macos"))]
const METALLIB_BYTES: &[u8] = &[];

const GLSL_SOURCES: &[(&str, &str)] = &[
    ("invert", ffgl_gpu::include_glsl_shader!("invert")),
];

struct GpuState {
    pipeline: Option<ComputePipeline>,
}

impl GpuPlugin for GpuState {
    fn gpu_init(&mut self, ctx: &GpuContext) -> anyhow::Result<()> {
        self.pipeline = Some(ctx.create_compute_pipeline("invert")?);
        Ok(())
    }

    fn gpu_draw(
        &mut self,
        ctx: &GpuContext,
        input: &mut DrawInput<'_>,
        _data: &FFGLData,
        _frame: u64,
    ) {
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
        input.store_pending(pending);
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
            about: "Color inversion (WGSL transpiled)".to_string(),
            description: "Inverts colors via a WGSL compute shader".to_string(),
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
            GLSL_SOURCES,
        );
    }
}

ffgl_core::plugin_main!(SimpleFFGLHandler<Invert>);
