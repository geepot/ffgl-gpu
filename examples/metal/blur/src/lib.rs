//! Blur FFGL plugin example.
//!
//! Demonstrates multi-pass compute with an FFGL parameter. A separable box
//! blur is implemented as two compute dispatches (horizontal then vertical)
//! using an intermediate texture. The "Radius" parameter (0.0-1.0) maps to
//! 0-20 pixels of blur.

use std::ffi::CString;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use ffgl_core::handler::simplified::{SimpleFFGLHandler, SimpleFFGLInstance};
use ffgl_core::info::{PluginInfo, PluginType};
use ffgl_core::parameters::{ParamInfo, SimpleParamInfo};
use ffgl_core::{FFGLData, GLInput};
use ffgl_glium::FFGLGlium;
use ffgl_gpu::pipeline::ComputePipeline;
use ffgl_gpu::plugin::GpuPlugin;
use ffgl_gpu::{GpuContext, draw_gpu_effect};

static NEXT_INSTANCE_ID: AtomicU64 = AtomicU64::new(1);

/// Compiled Metal shader library, embedded at build time.
#[cfg(target_os = "macos")]
const METALLIB_BYTES: &[u8] = ffgl_gpu::include_metallib!();

#[cfg(not(target_os = "macos"))]
const METALLIB_BYTES: &[u8] = &[];

/// Maximum blur radius in pixels when the parameter is at 1.0.
const MAX_RADIUS: f32 = 20.0;

fn cached_params() -> &'static [SimpleParamInfo] {
    static PARAMS: OnceLock<Vec<SimpleParamInfo>> = OnceLock::new();
    PARAMS.get_or_init(|| {
        vec![SimpleParamInfo {
            name: CString::new("Radius").unwrap(),
            default: Some(0.25),
            ..Default::default()
        }]
    })
}

/// Uniform struct matching `BlurParams` in the Metal shader.
#[repr(C)]
struct BlurParams {
    radius: i32,
}

/// Inner GPU state, separate from glium to avoid double-borrow.
struct GpuState {
    radius_param: f32,
    h_pipeline: Option<ComputePipeline>,
    v_pipeline: Option<ComputePipeline>,
    #[cfg(target_os = "macos")]
    intermediate_texture:
        Option<objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLTexture>>>,
    #[cfg(target_os = "macos")]
    intermediate_dims: (u32, u32),
}

#[cfg(target_os = "macos")]
impl GpuState {
    fn ensure_intermediate_texture(&mut self, ctx: &GpuContext, width: u32, height: u32) {
        use objc2_metal::*;

        if self.intermediate_dims == (width, height) && self.intermediate_texture.is_some() {
            return;
        }

        let desc = MTLTextureDescriptor::new();
        desc.setTextureType(MTLTextureType::Type2D);
        desc.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
        unsafe {
            desc.setWidth(width as usize);
            desc.setHeight(height as usize);
        }
        desc.setStorageMode(MTLStorageMode::Private);
        desc.setUsage(MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite);

        self.intermediate_texture = ctx.metal_device().device().newTextureWithDescriptor(&desc);
        self.intermediate_dims = (width, height);
    }
}

impl GpuPlugin for GpuState {
    fn gpu_init(&mut self, ctx: &GpuContext) -> anyhow::Result<()> {
        self.h_pipeline = Some(ctx.create_compute_pipeline("blur_horizontal")?);
        self.v_pipeline = Some(ctx.create_compute_pipeline("blur_vertical")?);
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

            let (w, h) = bridge.dimensions();

            // Ensure intermediate texture before borrowing pipelines.
            self.ensure_intermediate_texture(ctx, w, h);

            let h_pipeline = match &self.h_pipeline {
                Some(p) => p,
                None => return,
            };
            let v_pipeline = match &self.v_pipeline {
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
            let intermediate_tex = match &self.intermediate_texture {
                Some(t) => t,
                None => return,
            };

            let pixel_radius = (self.radius_param * MAX_RADIUS).round() as i32;
            let params = BlurParams {
                radius: pixel_radius,
            };
            let params_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    &params as *const BlurParams as *const u8,
                    std::mem::size_of::<BlurParams>(),
                )
            };

            // Pass 1: horizontal blur (input -> intermediate)
            let pass = match ctx.begin_compute_pass() {
                Ok(p) => p,
                Err(_) => return,
            };
            ctx.set_compute_pipeline(&pass, h_pipeline);
            ctx.bind_texture(&pass, input_tex, 0);
            ctx.bind_texture(&pass, intermediate_tex, 1);
            ctx.bind_bytes(&pass, params_bytes, 0);
            ctx.dispatch_threads(&pass, (w as usize, h as usize), (16, 16));
            let pending = ctx.end_compute_pass(pass);
            pending.wait();

            // Pass 2: vertical blur (intermediate -> output)
            let pass = match ctx.begin_compute_pass() {
                Ok(p) => p,
                Err(_) => return,
            };
            ctx.set_compute_pipeline(&pass, v_pipeline);
            ctx.bind_texture(&pass, intermediate_tex, 0);
            ctx.bind_texture(&pass, output_tex, 1);
            ctx.bind_bytes(&pass, params_bytes, 0);
            ctx.dispatch_threads(&pass, (w as usize, h as usize), (16, 16));

            let pending = ctx.end_compute_pass(pass);
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

pub struct Blur {
    glium: FFGLGlium,
    gpu: GpuState,
    frame_counter: u64,
    instance_id: u64,
}

// SAFETY: FFGL plugins are called single-threaded from the host.
unsafe impl Send for Blur {}
unsafe impl Sync for Blur {}

impl SimpleFFGLInstance for Blur {
    fn new(inst_data: &FFGLData) -> Self {
        let default_radius = cached_params()[0].default_val();
        Self {
            glium: FFGLGlium::new(inst_data),
            gpu: GpuState {
                radius_param: default_radius,
                h_pipeline: None,
                v_pipeline: None,
                #[cfg(target_os = "macos")]
                intermediate_texture: None,
                #[cfg(target_os = "macos")]
                intermediate_dims: (0, 0),
            },
            frame_counter: 0,
            instance_id: NEXT_INSTANCE_ID.fetch_add(1, Ordering::Relaxed),
        }
    }

    fn num_params() -> usize {
        1
    }

    fn param_info(index: usize) -> &'static dyn ParamInfo {
        &cached_params()[index]
    }

    fn plugin_info() -> PluginInfo {
        PluginInfo {
            unique_id: *b"BLUR",
            name: *b"Blur\0\0\0\0\0\0\0\0\0\0\0\0",
            ty: PluginType::Effect,
            about: "Separable box blur via multi-pass compute".to_string(),
            description: "Two-pass GPU compute blur with adjustable radius parameter".to_string(),
        }
    }

    fn get_param(&self, _index: usize) -> f32 {
        self.gpu.radius_param
    }

    fn set_param(&mut self, _index: usize, value: f32) {
        self.gpu.radius_param = value;
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

ffgl_core::plugin_main!(SimpleFFGLHandler<Blur>);
