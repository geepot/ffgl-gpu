//! Kitchen-sink FFGL plugin example.
//!
//! Demonstrates a mixed compute + render pipeline with multiple FFGL
//! parameters. The effect chain is:
//!
//! 1. **Grayscale** (compute) -- desaturate based on the "Grayscale Amount"
//!    parameter.
//! 2. **Tint** (render / fragment shader) -- apply a colour tint controlled by
//!    "Tint Hue" and "Tint Saturation".
//! 3. **Blend** (compute) -- mix the fully-processed result with the original
//!    input via the "Blend" parameter.
//!
//! This shows how to:
//! - Use multiple pipelines (compute and render) in a single plugin.
//! - Pass uniform data to both compute and fragment shaders.
//! - Chain intermediate textures across passes.
//! - Expose multiple FFGL parameters.

use std::ffi::CString;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use ffgl_core::handler::simplified::{SimpleFFGLHandler, SimpleFFGLInstance};
use ffgl_core::info::{PluginInfo, PluginType};
use ffgl_core::parameters::{ParamInfo, ParameterTypes, SimpleParamInfo};
use ffgl_core::{FFGLData, GLInput};
use ffgl_glium::FFGLGlium;
use ffgl_gpu::pipeline::{ComputePipeline, RenderPipeline};
use ffgl_gpu::plugin::GpuPlugin;
use ffgl_gpu::{GpuContext, draw_gpu_effect};

static NEXT_INSTANCE_ID: AtomicU64 = AtomicU64::new(1);

/// Compiled Metal shader library, embedded at build time.
#[cfg(target_os = "macos")]
const METALLIB_BYTES: &[u8] = ffgl_gpu::include_metallib!();

#[cfg(not(target_os = "macos"))]
const METALLIB_BYTES: &[u8] = &[];

// ---------------------------------------------------------------------------
// Parameter indices
// ---------------------------------------------------------------------------

const PARAM_GRAYSCALE: usize = 0;
const PARAM_TINT_HUE: usize = 1;
const PARAM_TINT_SAT: usize = 2;
const PARAM_BLEND: usize = 3;
const PARAM_COUNT: usize = 4;

fn cached_params() -> &'static [SimpleParamInfo] {
    static PARAMS: OnceLock<Vec<SimpleParamInfo>> = OnceLock::new();
    PARAMS.get_or_init(|| {
        vec![
            SimpleParamInfo {
                name: CString::new("Grayscale Amount").unwrap(),
                default: Some(0.5),
                ..Default::default()
            },
            SimpleParamInfo {
                name: CString::new("Tint Hue").unwrap(),
                param_type: ParameterTypes::Hue,
                default: Some(0.0),
                ..Default::default()
            },
            SimpleParamInfo {
                name: CString::new("Tint Saturation").unwrap(),
                param_type: ParameterTypes::Saturation,
                default: Some(0.5),
                ..Default::default()
            },
            SimpleParamInfo {
                name: CString::new("Blend").unwrap(),
                default: Some(1.0),
                ..Default::default()
            },
        ]
    })
}

/// Uniform struct matching `EffectParams` in the Metal shaders.
#[repr(C)]
struct EffectParams {
    grayscale_amount: f32,
    tint_hue: f32,
    tint_saturation: f32,
    blend: f32,
}

/// Inner GPU state, separate from glium to avoid double-borrow.
struct GpuState {
    params: [f32; PARAM_COUNT],

    // Pipelines
    grayscale_pipeline: Option<ComputePipeline>,
    tint_pipeline: Option<RenderPipeline>,
    blend_pipeline: Option<ComputePipeline>,

    // Intermediate textures (macOS only)
    #[cfg(target_os = "macos")]
    tex_after_grayscale:
        Option<objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLTexture>>>,
    #[cfg(target_os = "macos")]
    tex_after_tint:
        Option<objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLTexture>>>,
    #[cfg(target_os = "macos")]
    intermediate_dims: (u32, u32),
}

// SAFETY: FFGL plugins are called single-threaded from the host.
unsafe impl Send for GpuState {}
unsafe impl Sync for GpuState {}

#[cfg(target_os = "macos")]
impl GpuState {
    fn ensure_intermediate_textures(&mut self, ctx: &GpuContext, width: u32, height: u32) {
        use objc2_metal::*;

        if self.intermediate_dims == (width, height)
            && self.tex_after_grayscale.is_some()
            && self.tex_after_tint.is_some()
        {
            return;
        }

        let make_texture = |usage: MTLTextureUsage| -> Option<
            objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn MTLTexture>>,
        > {
            let desc = MTLTextureDescriptor::new();
            desc.setTextureType(MTLTextureType::Type2D);
            desc.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
            unsafe {
                desc.setWidth(width as usize);
                desc.setHeight(height as usize);
            }
            desc.setStorageMode(MTLStorageMode::Private);
            desc.setUsage(usage);
            ctx.metal_device().device().newTextureWithDescriptor(&desc)
        };

        // After grayscale: read by tint fragment + written by compute
        self.tex_after_grayscale =
            make_texture(MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite);
        // After tint: read by blend compute, written by render pass
        self.tex_after_tint =
            make_texture(MTLTextureUsage::ShaderRead | MTLTextureUsage::RenderTarget);

        self.intermediate_dims = (width, height);
    }
}

impl GpuPlugin for GpuState {
    fn gpu_init(&mut self, ctx: &GpuContext) -> anyhow::Result<()> {
        self.grayscale_pipeline = Some(ctx.create_compute_pipeline("grayscale")?);
        self.tint_pipeline = Some(ctx.create_render_pipeline("tint_vertex", "tint_fragment")?);
        self.blend_pipeline = Some(ctx.create_compute_pipeline("blend")?);
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

            // Ensure intermediate textures before borrowing pipelines.
            self.ensure_intermediate_textures(ctx, w, h);

            let grayscale_pl = match &self.grayscale_pipeline {
                Some(p) => p,
                None => return,
            };
            let tint_pl = match &self.tint_pipeline {
                Some(p) => p,
                None => return,
            };
            let blend_pl = match &self.blend_pipeline {
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
            let after_gray = match &self.tex_after_grayscale {
                Some(t) => t,
                None => return,
            };
            let after_tint = match &self.tex_after_tint {
                Some(t) => t,
                None => return,
            };

            let uniforms = EffectParams {
                grayscale_amount: self.params[PARAM_GRAYSCALE],
                tint_hue: self.params[PARAM_TINT_HUE],
                tint_saturation: self.params[PARAM_TINT_SAT],
                blend: self.params[PARAM_BLEND],
            };
            let uniform_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    &uniforms as *const EffectParams as *const u8,
                    std::mem::size_of::<EffectParams>(),
                )
            };

            // --- Pass 1: grayscale compute (input -> after_gray) ---
            let pass = match ctx.begin_compute_pass() {
                Ok(p) => p,
                Err(_) => return,
            };
            ctx.set_compute_pipeline(&pass, grayscale_pl);
            ctx.bind_texture(&pass, input_tex, 0);
            ctx.bind_texture(&pass, after_gray, 1);
            ctx.bind_bytes(&pass, uniform_bytes, 0);
            ctx.dispatch_threads(&pass, (w as usize, h as usize), (16, 16));
            let pending = ctx.end_compute_pass(pass);
            pending.wait();

            // --- Pass 2: tint render (after_gray -> after_tint) ---
            let pending = match ctx.dispatch_render(
                tint_pl,
                after_tint,
                &[after_gray],
                &[(uniform_bytes, 0)],
            ) {
                Ok(p) => p,
                Err(_) => return,
            };
            pending.wait();

            // --- Pass 3: blend compute (original + after_tint -> output) ---
            let pass = match ctx.begin_compute_pass() {
                Ok(p) => p,
                Err(_) => return,
            };
            ctx.set_compute_pipeline(&pass, blend_pl);
            ctx.bind_texture(&pass, input_tex, 0);
            ctx.bind_texture(&pass, after_tint, 1);
            ctx.bind_texture(&pass, output_tex, 2);
            ctx.bind_bytes(&pass, uniform_bytes, 0);
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

pub struct KitchenSink {
    glium: FFGLGlium,
    gpu: GpuState,
    frame_counter: u64,
    instance_id: u64,
}

// SAFETY: FFGL plugins are called single-threaded from the host.
unsafe impl Send for KitchenSink {}
unsafe impl Sync for KitchenSink {}

impl SimpleFFGLInstance for KitchenSink {
    fn new(inst_data: &FFGLData) -> Self {
        let params_info = cached_params();
        let mut params = [0.0f32; PARAM_COUNT];
        for (i, p) in params.iter_mut().enumerate() {
            *p = params_info[i].default_val();
        }

        Self {
            glium: FFGLGlium::new(inst_data),
            gpu: GpuState {
                params,
                grayscale_pipeline: None,
                tint_pipeline: None,
                blend_pipeline: None,
                #[cfg(target_os = "macos")]
                tex_after_grayscale: None,
                #[cfg(target_os = "macos")]
                tex_after_tint: None,
                #[cfg(target_os = "macos")]
                intermediate_dims: (0, 0),
            },
            frame_counter: 0,
            instance_id: NEXT_INSTANCE_ID.fetch_add(1, Ordering::Relaxed),
        }
    }

    fn num_params() -> usize {
        PARAM_COUNT
    }

    fn param_info(index: usize) -> &'static dyn ParamInfo {
        &cached_params()[index]
    }

    fn plugin_info() -> PluginInfo {
        PluginInfo {
            unique_id: *b"KSNK",
            name: *b"Kitchen Sink\0\0\0\0",
            ty: PluginType::Effect,
            about: "Mixed compute + render pipeline demo".to_string(),
            description: "Grayscale (compute) -> Tint (render) -> Blend (compute)".to_string(),
        }
    }

    fn get_param(&self, index: usize) -> f32 {
        self.gpu.params.get(index).copied().unwrap_or(0.0)
    }

    fn set_param(&mut self, index: usize, value: f32) {
        if index < PARAM_COUNT {
            self.gpu.params[index] = value;
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

ffgl_core::plugin_main!(SimpleFFGLHandler<KitchenSink>);
