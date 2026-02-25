#![allow(dead_code)]
//! DX11 Blur FFGL plugin example.
//!
//! Demonstrates multi-pass compute with an FFGL parameter on DX11. A separable
//! box blur is implemented as two compute dispatches (horizontal then vertical)
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

#[cfg(target_os = "windows")]
use gpu_interop::dx11::GlDx11Bridge;
#[cfg(target_os = "windows")]
use windows::Win32::Graphics::Direct3D::D3D_SRV_DIMENSION_TEXTURE2D;
#[cfg(target_os = "windows")]
use windows::Win32::Graphics::Direct3D11::*;
#[cfg(target_os = "windows")]
use windows::Win32::Graphics::Dxgi::Common::*;

/// Compiled HLSL horizontal blur shader, embedded at build time.
#[cfg(target_os = "windows")]
const H_SHADER: &[u8] = ffgl_gpu::include_hlsl_shader!("blur_horizontal");
#[cfg(not(target_os = "windows"))]
const H_SHADER: &[u8] = &[];

/// Compiled HLSL vertical blur shader, embedded at build time.
#[cfg(target_os = "windows")]
const V_SHADER: &[u8] = ffgl_gpu::include_hlsl_shader!("blur_vertical");
#[cfg(not(target_os = "windows"))]
const V_SHADER: &[u8] = &[];

/// No Metal shaders for this DX11-only example.
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

/// Uniform struct matching `BlurParams` in the HLSL shader.
#[repr(C)]
struct BlurParams {
    radius: i32,
}

/// Inner GPU state, separate from glium to avoid double-borrow.
struct GpuState {
    radius_param: f32,
    h_pipeline: Option<ComputePipeline>,
    v_pipeline: Option<ComputePipeline>,
    #[cfg(target_os = "windows")]
    intermediate_texture: Option<windows::Win32::Graphics::Direct3D11::ID3D11Texture2D>,
    #[cfg(target_os = "windows")]
    intermediate_srv: Option<windows::Win32::Graphics::Direct3D11::ID3D11ShaderResourceView>,
    #[cfg(target_os = "windows")]
    intermediate_uav: Option<windows::Win32::Graphics::Direct3D11::ID3D11UnorderedAccessView>,
    #[cfg(target_os = "windows")]
    intermediate_dims: (u32, u32),
    #[cfg(target_os = "windows")]
    cbuf: Option<windows::Win32::Graphics::Direct3D11::ID3D11Buffer>,
}

#[cfg(target_os = "windows")]
impl GpuState {
    /// Create or re-create the intermediate texture (and SRV/UAV views) when
    /// dimensions change.
    fn ensure_intermediate_texture(
        &mut self,
        device: &ID3D11Device,
        width: u32,
        height: u32,
    ) {
        if self.intermediate_dims == (width, height) && self.intermediate_texture.is_some() {
            return;
        }

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

        let mut texture = None;
        let hr = unsafe { device.CreateTexture2D(&desc, None, Some(&mut texture as *mut _)) };
        if hr.is_err() {
            return;
        }
        let texture = match texture {
            Some(t) => t,
            None => return,
        };

        // Create SRV for reading the intermediate texture
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
        let hr = unsafe {
            device.CreateShaderResourceView(
                &texture,
                Some(&srv_desc),
                Some(&mut srv as *mut _),
            )
        };
        if hr.is_err() {
            return;
        }

        // Create UAV for writing the intermediate texture
        let uav_desc = D3D11_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_B8G8R8A8_UNORM,
            ViewDimension: D3D11_UAV_DIMENSION_TEXTURE2D,
            Anonymous: D3D11_UNORDERED_ACCESS_VIEW_DESC_0 {
                Texture2D: D3D11_TEX2D_UAV { MipSlice: 0 },
            },
        };
        let mut uav = None;
        let hr = unsafe {
            device.CreateUnorderedAccessView(
                &texture,
                Some(&uav_desc),
                Some(&mut uav as *mut _),
            )
        };
        if hr.is_err() {
            return;
        }

        if srv.is_some() && uav.is_some() {
            self.intermediate_texture = Some(texture);
            self.intermediate_srv = srv;
            self.intermediate_uav = uav;
            self.intermediate_dims = (width, height);
        }
    }

    /// Map the dynamic constant buffer, write data, and unmap.
    fn update_cbuf(&self, context: &ID3D11DeviceContext, data: &[u8]) {
        let cbuf = match &self.cbuf {
            Some(b) => b,
            None => return,
        };
        unsafe {
            let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
            let hr = context.Map(
                cbuf,
                0,
                D3D11_MAP_WRITE_DISCARD,
                0,
                Some(&mut mapped),
            );
            if hr.is_err() {
                return;
            }
            std::ptr::copy_nonoverlapping(data.as_ptr(), mapped.pData as *mut u8, data.len());
            context.Unmap(cbuf, 0);
        }
    }
}

impl GpuPlugin for GpuState {
    fn gpu_init(&mut self, ctx: &GpuContext) -> anyhow::Result<()> {
        #[cfg(target_os = "windows")]
        {
            self.h_pipeline = Some(ctx.create_compute_pipeline_from_bytecode(H_SHADER)?);
            self.v_pipeline = Some(ctx.create_compute_pipeline_from_bytecode(V_SHADER)?);
            self.cbuf = gpu_interop::dx11::create_dynamic_cbuf(
                ctx.dx11_device().device(),
                std::mem::size_of::<BlurParams>(),
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
            let (w, h) = bridge.dimensions();

            let dx11_bridge = match bridge.as_any_mut().downcast_mut::<GlDx11Bridge>() {
                Some(b) => b,
                None => return,
            };

            let input_srv = match dx11_bridge.input_srv() {
                Some(s) => s,
                None => return,
            };
            let output_uav = match dx11_bridge.output_uav() {
                Some(u) => u,
                None => return,
            };

            // Get the device reference for ensure_intermediate_texture.
            // Clone the COM pointer so we don't hold a borrow on the bridge.
            let device = dx11_bridge.device().clone();
            let dx11_context = dx11_bridge.context().clone();

            // Ensure intermediate texture is allocated at the correct size.
            self.ensure_intermediate_texture(&device, w, h);

            let intermediate_srv = match &self.intermediate_srv {
                Some(s) => s.clone(),
                None => return,
            };
            let intermediate_uav = match &self.intermediate_uav {
                Some(u) => u.clone(),
                None => return,
            };

            // Compute pixel radius from the normalized parameter.
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

            // Update the constant buffer with the current blur radius.
            self.update_cbuf(&dx11_context, params_bytes);

            let cbuf_ref = match &self.cbuf {
                Some(b) => b.clone(),
                None => return,
            };

            let h_pipeline = match &self.h_pipeline {
                Some(p) => p,
                None => return,
            };
            let v_pipeline = match &self.v_pipeline {
                Some(p) => p,
                None => return,
            };

            // Thread groups: ceil(w/16) x ceil(h/16) x 1
            let groups_x = (w + 15) / 16;
            let groups_y = (h + 15) / 16;

            // Pass 1: horizontal blur (input -> intermediate)
            // dispatch_compute unbinds all CS resources after each dispatch,
            // so the intermediate UAV is safely unbound before pass 2 binds
            // it as an SRV.
            ctx.dispatch_compute(
                h_pipeline,
                &[Some(intermediate_uav)],
                &[Some(input_srv)],
                &[Some(cbuf_ref.clone())],
                (groups_x, groups_y, 1),
            );

            // Pass 2: vertical blur (intermediate -> output)
            ctx.dispatch_compute(
                v_pipeline,
                &[Some(output_uav)],
                &[Some(intermediate_srv)],
                &[Some(cbuf_ref)],
                (groups_x, groups_y, 1),
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

pub struct DxBlur {
    glium: FFGLGlium,
    gpu: GpuState,
    frame_counter: u64,
    instance_id: u64,
}

// SAFETY: See GpuState safety comment above.
unsafe impl Send for DxBlur {}
unsafe impl Sync for DxBlur {}

impl SimpleFFGLInstance for DxBlur {
    fn new(inst_data: &FFGLData) -> Self {
        let default_radius = cached_params()[0].default_val();
        Self {
            glium: FFGLGlium::new(inst_data),
            gpu: GpuState {
                radius_param: default_radius,
                h_pipeline: None,
                v_pipeline: None,
                #[cfg(target_os = "windows")]
                intermediate_texture: None,
                #[cfg(target_os = "windows")]
                intermediate_srv: None,
                #[cfg(target_os = "windows")]
                intermediate_uav: None,
                #[cfg(target_os = "windows")]
                intermediate_dims: (0, 0),
                #[cfg(target_os = "windows")]
                cbuf: None,
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
            unique_id: *b"DBlr",
            name: *b"DX Blur\0\0\0\0\0\0\0\0\0",
            ty: PluginType::Effect,
            about: "DX11 separable box blur via multi-pass compute".to_string(),
            description: "Two-pass DX11 GPU compute blur with adjustable radius parameter"
                .to_string(),
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

ffgl_core::plugin_main!(SimpleFFGLHandler<DxBlur>);
