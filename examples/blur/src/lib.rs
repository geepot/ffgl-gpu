//! Blur FFGL plugin example (WGSL transpiled).
//!
//! Demonstrates a single-pass 2D box blur with a controllable radius parameter.

use std::ffi::CString;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::LazyLock;

use ffgl_core::handler::simplified::{SimpleFFGLHandler, SimpleFFGLInstance};
use ffgl_core::info::{PluginInfo, PluginType};
use ffgl_core::parameters::{ParamInfo, ParameterTypes, SimpleParamInfo};
use ffgl_core::{FFGLData, GLInput};
use ffgl_glium::FFGLGlium;
use ffgl_gpu::pipeline::ComputePipeline;
use ffgl_gpu::plugin::GpuPlugin;
use ffgl_gpu::{AsBytes, DrawInput, GpuContext, draw_gpu_effect};

static NEXT_INSTANCE_ID: AtomicU64 = AtomicU64::new(1);

#[cfg(target_os = "macos")]
const METALLIB_BYTES: &[u8] = ffgl_gpu::include_metallib!();

#[cfg(not(target_os = "macos"))]
const METALLIB_BYTES: &[u8] = &[];

const GLSL_SOURCES: &[(&str, &str)] = &[
    ("blur", ffgl_gpu::include_glsl_shader!("blur")),
];

#[repr(C)]
struct BlurParams {
    radius: i32,
    _pad: [i32; 3],
}

// SAFETY: BlurParams is #[repr(C)] with only plain numeric fields.
unsafe impl AsBytes for BlurParams {}

struct GpuState {
    pipeline: Option<ComputePipeline>,
    radius: f32,
}

impl GpuPlugin for GpuState {
    fn gpu_init(&mut self, ctx: &GpuContext) -> anyhow::Result<()> {
        self.pipeline = Some(ctx.create_compute_pipeline("blur")?);
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

        let params = BlurParams {
            radius: (self.radius * 20.0).round() as i32,
            _pad: [0; 3],
        };

        let pending = match ctx.dispatch_compute(
            pipeline,
            &[input.input, input.output],
            &[],
            &[(params.as_bytes(), 2)],
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

pub struct Blur {
    glium: FFGLGlium,
    gpu: GpuState,
    frame_counter: u64,
    instance_id: u64,
}

// SAFETY: FFGL plugins are called single-threaded from the host.
unsafe impl Send for Blur {}
unsafe impl Sync for Blur {}

static PARAM_RADIUS: LazyLock<SimpleParamInfo> = LazyLock::new(|| SimpleParamInfo {
    name: CString::new("Radius").unwrap(),
    param_type: ParameterTypes::Standard,
    default: Some(0.25),
    min: None,
    max: None,
    group: None,
    display_name: None,
    elements: None,
});

impl SimpleFFGLInstance for Blur {
    fn new(inst_data: &FFGLData) -> Self {
        Self {
            glium: FFGLGlium::new(inst_data),
            gpu: GpuState {
                pipeline: None,
                radius: 0.25,
            },
            frame_counter: 0,
            instance_id: NEXT_INSTANCE_ID.fetch_add(1, Ordering::Relaxed),
        }
    }

    fn num_params() -> usize {
        1
    }

    fn param_info(_index: usize) -> &'static dyn ParamInfo {
        &*PARAM_RADIUS
    }

    fn get_param(&self, _index: usize) -> f32 {
        self.gpu.radius
    }

    fn set_param(&mut self, _index: usize, value: f32) {
        self.gpu.radius = value;
    }

    fn plugin_info() -> PluginInfo {
        PluginInfo {
            unique_id: *b"WBLR",
            name: *b"WGSL Blur\0\0\0\0\0\0\0",
            ty: PluginType::Effect,
            about: "Box blur (WGSL transpiled)".to_string(),
            description: "2D box blur with controllable radius".to_string(),
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

ffgl_core::plugin_main!(SimpleFFGLHandler<Blur>);
