//! Passthrough FFGL plugin example.
//!
//! Demonstrates the simplest possible GPU compute plugin: a single compute
//! kernel that copies the input texture to the output texture pixel-for-pixel.

use ffgl_core::handler::simplified::{SimpleFFGLHandler, SimpleFFGLInstance};
use ffgl_core::info::{PluginInfo, PluginType};
use ffgl_core::{FFGLData, GLInput};
use ffgl_glium::FFGLGlium;
use ffgl_gpu::pipeline::ComputePipeline;
use ffgl_gpu::plugin::GpuPlugin;
use ffgl_gpu::{GpuContext, draw_gpu_effect};

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
        bridge: &mut dyn gpu_interop::GpuBridge,
        _data: &FFGLData,
        _input: &GLInput<'_>,
        frame: u64,
    ) {
        #[cfg(target_os = "macos")]
        {
            use gpu_interop::metal::GlMetalBridge;

            let pipeline = match &self.pipeline {
                Some(p) => p,
                None => return,
            };

            // Get dimensions before downcasting (avoids borrow conflict).
            let (w, h) = bridge.dimensions();

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

            let pass = match ctx.begin_compute_pass() {
                Ok(p) => p,
                Err(_) => return,
            };

            ctx.set_compute_pipeline(&pass, pipeline);
            ctx.bind_texture(&pass, input_tex, 0);
            ctx.bind_texture(&pass, output_tex, 1);
            ctx.dispatch_threads(&pass, (w as usize, h as usize), (16, 16));

            let pending = ctx.end_compute_pass(pass);
            metal_bridge.store_command_buffer(pending.into_command_buffer(), frame);
        }

        #[cfg(not(target_os = "macos"))]
        {
            let _ = (ctx, bridge, frame);
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
        let s = Self {
            glium: FFGLGlium::new(inst_data),
            gpu: GpuState { pipeline: None },
            frame_counter: 0,
            instance_id: 0,
        };
        // Use the struct address as a stable instance id.
        let id = &s as *const _ as u64;
        Self {
            instance_id: id,
            ..s
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
