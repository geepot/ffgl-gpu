#![allow(dead_code)]
//! DX11 Kitchen-sink FFGL plugin example.
//!
//! Demonstrates a mixed compute + render pipeline with multiple FFGL
//! parameters on DX11. The effect chain is:
//!
//! 1. **Grayscale** (compute) -- desaturate based on the "Grayscale Amount"
//!    parameter.
//! 2. **Tint** (render / vertex + pixel shader) -- apply a colour tint
//!    controlled by "Tint Hue" and "Tint Saturation".
//! 3. **Blend** (compute) -- mix the fully-processed result with the original
//!    input via the "Blend" parameter.
//!
//! This shows how to:
//! - Use multiple pipelines (compute and render) in a single DX11 plugin.
//! - Pass uniform data via a dynamic constant buffer.
//! - Chain intermediate textures across passes.
//! - Expose multiple FFGL parameters.

use std::ffi::CString;
use std::sync::OnceLock;

use ffgl_core::handler::simplified::{SimpleFFGLHandler, SimpleFFGLInstance};
use ffgl_core::info::{PluginInfo, PluginType};
use ffgl_core::parameters::{ParamInfo, ParameterTypes, SimpleParamInfo};
use ffgl_core::{FFGLData, GLInput};
use ffgl_glium::FFGLGlium;
use ffgl_gpu::pipeline::{ComputePipeline, RenderPipeline};
use ffgl_gpu::plugin::GpuPlugin;
use ffgl_gpu::{GpuContext, draw_gpu_effect};

// ---------------------------------------------------------------------------
// Compiled HLSL shader bytecode, embedded at build time
// ---------------------------------------------------------------------------

#[cfg(target_os = "windows")]
const GRAYSCALE_CS: &[u8] = ffgl_gpu::include_hlsl_shader!("grayscale_cs");
#[cfg(target_os = "windows")]
const TINT_VS: &[u8] = ffgl_gpu::include_hlsl_shader!("tint_vs");
#[cfg(target_os = "windows")]
const TINT_PS: &[u8] = ffgl_gpu::include_hlsl_shader!("tint_ps");
#[cfg(target_os = "windows")]
const BLEND_CS: &[u8] = ffgl_gpu::include_hlsl_shader!("blend_cs");

#[cfg(not(target_os = "windows"))]
const GRAYSCALE_CS: &[u8] = &[];
#[cfg(not(target_os = "windows"))]
const TINT_VS: &[u8] = &[];
#[cfg(not(target_os = "windows"))]
const TINT_PS: &[u8] = &[];
#[cfg(not(target_os = "windows"))]
const BLEND_CS: &[u8] = &[];

/// No Metal shaders for this DX11-only plugin.
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

/// Uniform struct matching `EffectParams` in the HLSL shaders.
#[repr(C)]
struct EffectParams {
    grayscale_amount: f32,
    tint_hue: f32,
    tint_saturation: f32,
    blend: f32,
}

// ---------------------------------------------------------------------------
// GPU state
// ---------------------------------------------------------------------------

/// Inner GPU state, separate from glium to avoid double-borrow.
struct GpuState {
    params: [f32; PARAM_COUNT],

    // Pipelines
    grayscale_pipeline: Option<ComputePipeline>,
    tint_pipeline: Option<RenderPipeline>,
    blend_pipeline: Option<ComputePipeline>,

    // DX11-specific intermediate textures and views
    #[cfg(target_os = "windows")]
    tex_after_grayscale: Option<windows::Win32::Graphics::Direct3D11::ID3D11Texture2D>,
    #[cfg(target_os = "windows")]
    tex_after_grayscale_srv:
        Option<windows::Win32::Graphics::Direct3D11::ID3D11ShaderResourceView>,
    #[cfg(target_os = "windows")]
    tex_after_grayscale_uav:
        Option<windows::Win32::Graphics::Direct3D11::ID3D11UnorderedAccessView>,

    #[cfg(target_os = "windows")]
    tex_after_tint: Option<windows::Win32::Graphics::Direct3D11::ID3D11Texture2D>,
    #[cfg(target_os = "windows")]
    tex_after_tint_srv: Option<windows::Win32::Graphics::Direct3D11::ID3D11ShaderResourceView>,

    #[cfg(target_os = "windows")]
    intermediate_dims: (u32, u32),

    /// Dynamic constant buffer for `EffectParams`.
    #[cfg(target_os = "windows")]
    cbuf: Option<windows::Win32::Graphics::Direct3D11::ID3D11Buffer>,
}

// SAFETY: FFGL plugins are called single-threaded from the host.
unsafe impl Send for GpuState {}
unsafe impl Sync for GpuState {}

// ---------------------------------------------------------------------------
// DX11 intermediate texture management
// ---------------------------------------------------------------------------

#[cfg(target_os = "windows")]
impl GpuState {
    /// Ensure intermediate textures exist at the correct dimensions.
    ///
    /// - `tex_after_grayscale`: BGRA8, bind SRV + UAV (written by compute, read
    ///   by render).
    /// - `tex_after_tint`: BGRA8, bind SRV + RENDER_TARGET (written by render,
    ///   read by compute).
    fn ensure_intermediate_textures(&mut self, ctx: &GpuContext, width: u32, height: u32) {
        use windows::Win32::Graphics::Direct3D::D3D_SRV_DIMENSION_TEXTURE2D;
        use windows::Win32::Graphics::Direct3D11::*;
        use windows::Win32::Graphics::Dxgi::Common::*;

        if self.intermediate_dims == (width, height)
            && self.tex_after_grayscale.is_some()
            && self.tex_after_tint.is_some()
        {
            return;
        }

        let device = ctx.dx11_device().device();

        // --- tex_after_grayscale: SRV + UAV (compute write, render read) ---
        {
            let desc = D3D11_TEXTURE2D_DESC {
                Width: width,
                Height: height,
                MipLevels: 1,
                ArraySize: 1,
                Format: DXGI_FORMAT_B8G8R8A8_UNORM,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Usage: D3D11_USAGE_DEFAULT,
                BindFlags: (D3D11_BIND_SHADER_RESOURCE.0 | D3D11_BIND_UNORDERED_ACCESS.0) as u32,
                CPUAccessFlags: 0,
                MiscFlags: 0,
            };
            let mut tex = None;
            let _ = unsafe { device.CreateTexture2D(&desc, None, Some(&mut tex as *mut _)) };
            self.tex_after_grayscale = tex;

            if let Some(ref texture) = self.tex_after_grayscale {
                // SRV
                let srv_desc = D3D11_SHADER_RESOURCE_VIEW_DESC {
                    Format: DXGI_FORMAT_B8G8R8A8_UNORM,
                    ViewDimension: D3D_SRV_DIMENSION_TEXTURE2D,
                    Anonymous: D3D11_SHADER_RESOURCE_VIEW_DESC_0 {
                        Texture2D: D3D11_TEX2D_SRV {
                            MostDetailedMip: 0,
                            MipLevels: 1,
                        },
                    },
                };
                let mut srv = None;
                let _ = unsafe {
                    device.CreateShaderResourceView(
                        texture,
                        Some(&srv_desc),
                        Some(&mut srv as *mut _),
                    )
                };
                self.tex_after_grayscale_srv = srv;

                // UAV
                let uav_desc = D3D11_UNORDERED_ACCESS_VIEW_DESC {
                    Format: DXGI_FORMAT_B8G8R8A8_UNORM,
                    ViewDimension: D3D11_UAV_DIMENSION_TEXTURE2D,
                    Anonymous: D3D11_UNORDERED_ACCESS_VIEW_DESC_0 {
                        Texture2D: D3D11_TEX2D_UAV { MipSlice: 0 },
                    },
                };
                let mut uav = None;
                let _ = unsafe {
                    device.CreateUnorderedAccessView(
                        texture,
                        Some(&uav_desc),
                        Some(&mut uav as *mut _),
                    )
                };
                self.tex_after_grayscale_uav = uav;
            }
        }

        // --- tex_after_tint: SRV + RENDER_TARGET (render write, compute read) ---
        {
            let desc = D3D11_TEXTURE2D_DESC {
                Width: width,
                Height: height,
                MipLevels: 1,
                ArraySize: 1,
                Format: DXGI_FORMAT_B8G8R8A8_UNORM,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Usage: D3D11_USAGE_DEFAULT,
                BindFlags: (D3D11_BIND_SHADER_RESOURCE.0 | D3D11_BIND_RENDER_TARGET.0) as u32,
                CPUAccessFlags: 0,
                MiscFlags: 0,
            };
            let mut tex = None;
            let _ = unsafe { device.CreateTexture2D(&desc, None, Some(&mut tex as *mut _)) };
            self.tex_after_tint = tex;

            if let Some(ref texture) = self.tex_after_tint {
                let srv_desc = D3D11_SHADER_RESOURCE_VIEW_DESC {
                    Format: DXGI_FORMAT_B8G8R8A8_UNORM,
                    ViewDimension: D3D_SRV_DIMENSION_TEXTURE2D,
                    Anonymous: D3D11_SHADER_RESOURCE_VIEW_DESC_0 {
                        Texture2D: D3D11_TEX2D_SRV {
                            MostDetailedMip: 0,
                            MipLevels: 1,
                        },
                    },
                };
                let mut srv = None;
                let _ = unsafe {
                    device.CreateShaderResourceView(
                        texture,
                        Some(&srv_desc),
                        Some(&mut srv as *mut _),
                    )
                };
                self.tex_after_tint_srv = srv;
            }
        }

        self.intermediate_dims = (width, height);
    }
}

// ---------------------------------------------------------------------------
// GpuPlugin implementation
// ---------------------------------------------------------------------------

impl GpuPlugin for GpuState {
    fn gpu_init(&mut self, ctx: &GpuContext) -> anyhow::Result<()> {
        #[cfg(target_os = "windows")]
        {
            self.grayscale_pipeline =
                Some(ctx.create_compute_pipeline_from_bytecode(GRAYSCALE_CS)?);
            self.tint_pipeline =
                Some(ctx.create_render_pipeline_from_bytecode(TINT_VS, TINT_PS)?);
            self.blend_pipeline =
                Some(ctx.create_compute_pipeline_from_bytecode(BLEND_CS)?);
            self.cbuf = gpu_interop::dx11::create_dynamic_cbuf(
                ctx.dx11_device().device(),
                std::mem::size_of::<EffectParams>(),
            );
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
            use gpu_interop::dx11::GlDx11Bridge;
            use windows::Win32::Graphics::Direct3D11::*;

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

            let after_gray_srv = match &self.tex_after_grayscale_srv {
                Some(v) => v.clone(),
                None => return,
            };
            let after_gray_uav = match &self.tex_after_grayscale_uav {
                Some(v) => v.clone(),
                None => return,
            };
            let after_tint_texture = match &self.tex_after_tint {
                Some(t) => t.clone(),
                None => return,
            };
            let after_tint_srv = match &self.tex_after_tint_srv {
                Some(v) => v.clone(),
                None => return,
            };
            let cbuf = match &self.cbuf {
                Some(b) => b.clone(),
                None => return,
            };

            // Update constant buffer with current parameters.
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
            ctx.update_constant_buffer(&cbuf, uniform_bytes);

            let thread_groups = ((w + 15) / 16, (h + 15) / 16, 1);

            // --- Pass 1: grayscale compute (input_srv -> after_gray_uav) ---
            ctx.dispatch_compute(
                grayscale_pl,
                &[Some(after_gray_uav)],
                &[Some(input_srv.clone())],
                &[Some(cbuf.clone())],
                thread_groups,
            );

            // --- Pass 2: tint render (after_gray_srv -> after_tint texture) ---
            let _ = ctx.dispatch_render(
                tint_pl,
                &after_tint_texture,
                &[Some(after_gray_srv)],
                &[Some(cbuf.clone())],
            );

            // --- Pass 3: blend compute (input + after_tint -> output) ---
            ctx.dispatch_compute(
                blend_pl,
                &[Some(output_uav)],
                &[Some(input_srv), Some(after_tint_srv)],
                &[Some(cbuf)],
                thread_groups,
            );
        }

        #[cfg(not(target_os = "windows"))]
        {
            let _ = (ctx, bridge);
        }
    }
}

// ---------------------------------------------------------------------------
// FFGL plugin boilerplate
// ---------------------------------------------------------------------------

pub struct DxKitchenSink {
    glium: FFGLGlium,
    gpu: GpuState,
    frame_counter: u64,
    instance_id: u64,
}

// SAFETY: FFGL plugins are called single-threaded from the host.
unsafe impl Send for DxKitchenSink {}
unsafe impl Sync for DxKitchenSink {}

impl SimpleFFGLInstance for DxKitchenSink {
    fn new(inst_data: &FFGLData) -> Self {
        let params_info = cached_params();
        let mut params = [0.0f32; PARAM_COUNT];
        for (i, p) in params.iter_mut().enumerate() {
            *p = params_info[i].default_val();
        }

        let s = Self {
            glium: FFGLGlium::new(inst_data),
            gpu: GpuState {
                params,
                grayscale_pipeline: None,
                tint_pipeline: None,
                blend_pipeline: None,
                #[cfg(target_os = "windows")]
                tex_after_grayscale: None,
                #[cfg(target_os = "windows")]
                tex_after_grayscale_srv: None,
                #[cfg(target_os = "windows")]
                tex_after_grayscale_uav: None,
                #[cfg(target_os = "windows")]
                tex_after_tint: None,
                #[cfg(target_os = "windows")]
                tex_after_tint_srv: None,
                #[cfg(target_os = "windows")]
                intermediate_dims: (0, 0),
                #[cfg(target_os = "windows")]
                cbuf: None,
            },
            frame_counter: 0,
            instance_id: 0,
        };
        let id = &s as *const _ as u64;
        Self {
            instance_id: id,
            ..s
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
            unique_id: *b"DKSk",
            name: *b"DX Kitchen Sink\0",
            ty: PluginType::Effect,
            about: "DX11 mixed compute + render pipeline demo".to_string(),
            description: "Grayscale (compute) -> Tint (render) -> Blend (compute) on DX11"
                .to_string(),
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

ffgl_core::plugin_main!(SimpleFFGLHandler<DxKitchenSink>);
