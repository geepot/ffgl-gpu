//! Kitchen-sink FFGL plugin example (WGSL transpiled).
//!
//! Demonstrates multiple effect parameters (grayscale, tint, blend) combined
//! in a single compute kernel with a uniform struct.

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
    ("effects", ffgl_gpu::include_glsl_shader!("effects")),
];

#[repr(C)]
struct EffectParams {
    grayscale_amount: f32,
    tint_hue: f32,
    tint_saturation: f32,
    blend: f32,
}

// SAFETY: EffectParams is #[repr(C)] with only f32 fields.
unsafe impl AsBytes for EffectParams {}

const NUM_PARAMS: usize = 4;
const DEFAULT_PARAMS: [f32; NUM_PARAMS] = [0.5, 0.0, 0.5, 1.0];

struct GpuState {
    pipeline: Option<ComputePipeline>,
    params: [f32; NUM_PARAMS],
}

impl GpuPlugin for GpuState {
    fn gpu_init(&mut self, ctx: &GpuContext) -> anyhow::Result<()> {
        self.pipeline = Some(ctx.create_compute_pipeline("effects")?);
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

        let params = EffectParams {
            grayscale_amount: self.params[0],
            tint_hue: self.params[1],
            tint_saturation: self.params[2],
            blend: self.params[3],
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

pub struct KitchenSink {
    glium: FFGLGlium,
    gpu: GpuState,
    frame_counter: u64,
    instance_id: u64,
}

// SAFETY: FFGL plugins are called single-threaded from the host.
unsafe impl Send for KitchenSink {}
unsafe impl Sync for KitchenSink {}

static PARAM_INFOS: LazyLock<[SimpleParamInfo; NUM_PARAMS]> = LazyLock::new(|| [
    SimpleParamInfo {
        name: CString::new("Grayscale Amount").unwrap(),
        param_type: ParameterTypes::Standard,
        default: Some(0.5),
        min: None,
        max: None,
        group: None,
        display_name: None,
        elements: None,
    },
    SimpleParamInfo {
        name: CString::new("Tint Hue").unwrap(),
        param_type: ParameterTypes::Hue,
        default: Some(0.0),
        min: None,
        max: None,
        group: None,
        display_name: None,
        elements: None,
    },
    SimpleParamInfo {
        name: CString::new("Tint Saturation").unwrap(),
        param_type: ParameterTypes::Saturation,
        default: Some(0.5),
        min: None,
        max: None,
        group: None,
        display_name: None,
        elements: None,
    },
    SimpleParamInfo {
        name: CString::new("Blend").unwrap(),
        param_type: ParameterTypes::Standard,
        default: Some(1.0),
        min: None,
        max: None,
        group: None,
        display_name: None,
        elements: None,
    },
]);

impl SimpleFFGLInstance for KitchenSink {
    fn new(inst_data: &FFGLData) -> Self {
        Self {
            glium: FFGLGlium::new(inst_data),
            gpu: GpuState {
                pipeline: None,
                params: DEFAULT_PARAMS,
            },
            frame_counter: 0,
            instance_id: NEXT_INSTANCE_ID.fetch_add(1, Ordering::Relaxed),
        }
    }

    fn num_params() -> usize {
        NUM_PARAMS
    }

    fn param_info(index: usize) -> &'static dyn ParamInfo {
        &PARAM_INFOS[index]
    }

    fn get_param(&self, index: usize) -> f32 {
        self.gpu.params[index]
    }

    fn set_param(&mut self, index: usize, value: f32) {
        self.gpu.params[index] = value;
    }

    fn plugin_info() -> PluginInfo {
        PluginInfo {
            unique_id: *b"WKSK",
            name: *b"WGSLKitchenSink\0",
            ty: PluginType::Effect,
            about: "Kitchen sink effects (WGSL transpiled)".to_string(),
            description: "Grayscale, tint, and blend via a WGSL compute shader".to_string(),
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

ffgl_core::plugin_main!(SimpleFFGLHandler<KitchenSink>);
