#![allow(dead_code)]
//! DX11 Passthrough FFGL plugin example.
//!
//! Demonstrates the simplest possible GPU compute plugin on Windows: a single
//! DX11 compute shader that copies the input texture to the output texture
//! pixel-for-pixel via a `Texture2D` SRV and `RWTexture2D` UAV.

use std::sync::atomic::{AtomicU64, Ordering};

use ffgl_core::handler::simplified::{SimpleFFGLHandler, SimpleFFGLInstance};
use ffgl_core::info::{PluginInfo, PluginType};
use ffgl_core::{FFGLData, GLInput};
use ffgl_glium::FFGLGlium;
use ffgl_gpu::pipeline::ComputePipeline;
use ffgl_gpu::plugin::GpuPlugin;
use ffgl_gpu::{GpuContext, draw_gpu_effect};

static NEXT_INSTANCE_ID: AtomicU64 = AtomicU64::new(1);

#[cfg(target_os = "windows")]
use gpu_interop::dx11::GlDx11Bridge;

/// Compiled HLSL compute shader, embedded at build time.
#[cfg(target_os = "windows")]
const COMPUTE_SHADER: &[u8] = ffgl_gpu::include_hlsl_shader!("main_cs");

#[cfg(not(target_os = "windows"))]
const COMPUTE_SHADER: &[u8] = &[];

/// Metal shader library bytes. Empty for this DX11-only example.
const METALLIB_BYTES: &[u8] = &[];

/// Inner GPU state, separate from the glium context to avoid double-borrow
/// when calling [`draw_gpu_effect`].
struct GpuState {
    pipeline: Option<ComputePipeline>,
}

impl GpuPlugin for GpuState {
    fn gpu_init(&mut self, ctx: &GpuContext) -> anyhow::Result<()> {
        #[cfg(target_os = "windows")]
        {
            self.pipeline = Some(ctx.create_compute_pipeline_from_bytecode(COMPUTE_SHADER)?);
        }
        let _ = ctx;
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
        #[cfg(target_os = "windows")]
        {
            let pipeline = match &self.pipeline {
                Some(p) => p,
                None => return,
            };

            // Get dimensions before downcasting (avoids borrow conflict).
            let (w, h) = bridge.dimensions();

            let dx_bridge = match bridge.as_any_mut().downcast_mut::<GlDx11Bridge>() {
                Some(b) => b,
                None => return,
            };

            let input_srv = match dx_bridge.input_srv() {
                Some(srv) => srv,
                None => return,
            };
            let output_uav = match dx_bridge.output_uav() {
                Some(uav) => uav,
                None => return,
            };

            let thread_groups = ((w + 15) / 16, (h + 15) / 16, 1);

            ctx.dispatch_compute(
                pipeline,
                &[Some(output_uav)],
                &[Some(input_srv)],
                &[],
                thread_groups,
            );
        }

        #[cfg(not(target_os = "windows"))]
        {
            let _ = (ctx, bridge);
        }
    }
}

// SAFETY: GpuState contains DX11 COM pointers created with
// D3D11_CREATE_DEVICE_SINGLETHREADED, which omits internal locking. This is
// sound because the FFGL host guarantees single-threaded access per plugin
// instance — no concurrent &self or &mut self calls ever occur.
unsafe impl Send for GpuState {}
unsafe impl Sync for GpuState {}

pub struct Passthrough {
    glium: FFGLGlium,
    gpu: GpuState,
    frame_counter: u64,
    instance_id: u64,
}

// SAFETY: See GpuState safety comment above.
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
            unique_id: *b"DPas",
            name: *b"DX Passthrough\0\0",
            ty: PluginType::Effect,
            about: "DX11 Passthrough GPU compute example".to_string(),
            description: "Copies input to output via a DX11 compute shader".to_string(),
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
