#![allow(dead_code)]
//! DX11 Invert FFGL plugin example.
//!
//! Demonstrates a DX11 render pipeline (vertex + pixel shader) that inverts the
//! colors of the input image using a fullscreen quad pass.

use std::sync::atomic::{AtomicU64, Ordering};

use ffgl_core::handler::simplified::{SimpleFFGLHandler, SimpleFFGLInstance};
use ffgl_core::info::{PluginInfo, PluginType};
use ffgl_core::{FFGLData, GLInput};
use ffgl_glium::FFGLGlium;
use ffgl_gpu::pipeline::RenderPipeline;
use ffgl_gpu::plugin::GpuPlugin;
use ffgl_gpu::{DrawInput, GpuContext, draw_gpu_effect};

static NEXT_INSTANCE_ID: AtomicU64 = AtomicU64::new(1);

/// Compiled HLSL vertex shader bytecode, embedded at build time.
#[cfg(target_os = "windows")]
const VS_SHADER: &[u8] = ffgl_gpu::include_hlsl_shader!("vs_main");
#[cfg(target_os = "windows")]
const PS_SHADER: &[u8] = ffgl_gpu::include_hlsl_shader!("ps_main");

#[cfg(not(target_os = "windows"))]
const VS_SHADER: &[u8] = &[];
#[cfg(not(target_os = "windows"))]
const PS_SHADER: &[u8] = &[];

/// No Metal shaders for this DX11-only plugin.
const METALLIB_BYTES: &[u8] = &[];

/// Inner GPU state, separate from the glium context to avoid double-borrow.
struct GpuState {
    pipeline: Option<RenderPipeline>,
}

impl GpuPlugin for GpuState {
    fn gpu_init(&mut self, ctx: &GpuContext) -> anyhow::Result<()> {
        #[cfg(target_os = "windows")]
        {
            self.pipeline = Some(ctx.create_render_pipeline(VS_SHADER, PS_SHADER)?);
        }
        let _ = ctx;
        Ok(())
    }

    fn gpu_draw(
        &mut self,
        ctx: &GpuContext,
        input: &mut DrawInput<'_>,
        _data: &FFGLData,
        _frame: u64,
    ) {
        #[cfg(target_os = "windows")]
        {
            let pipeline = match &self.pipeline {
                Some(p) => p,
                None => return,
            };

            let _ = ctx.dispatch_render(
                pipeline,
                input.output_texture.clone(),
                &[Some(input.input_srv.clone())],
                &[],
            );
        }

        #[cfg(not(target_os = "windows"))]
        {
            let _ = (ctx, input);
        }
    }
}

// SAFETY: GpuState contains DX11 COM pointers created with
// D3D11_CREATE_DEVICE_SINGLETHREADED, which omits internal locking. This is
// sound because the FFGL host guarantees single-threaded access per plugin
// instance — no concurrent &self or &mut self calls ever occur.
unsafe impl Send for GpuState {}
unsafe impl Sync for GpuState {}

pub struct DxInvert {
    glium: FFGLGlium,
    gpu: GpuState,
    frame_counter: u64,
    instance_id: u64,
}

// SAFETY: See GpuState safety comment above.
unsafe impl Send for DxInvert {}
unsafe impl Sync for DxInvert {}

impl SimpleFFGLInstance for DxInvert {
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
            unique_id: *b"DInv",
            name: *b"DX Invert\0\0\0\0\0\0\0",
            ty: PluginType::Effect,
            about: "DX11 color inversion via render pipeline".to_string(),
            description: "Inverts colors using a DX11 vertex/pixel shader pair".to_string(),
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

ffgl_core::plugin_main!(SimpleFFGLHandler<DxInvert>);
