//! Double-buffered GL-D3D11 bridge via WGL_NV_DX_interop2.
//!
//! Two SharedTexturePair slots allow pipelining: D3D11 processes the "front"
//! pair while GL blits the previous result from the "back" pair to the host
//! FBO. This introduces one frame of latency but allows D3D11 compute to
//! overlap with host compositing between draw calls.

use std::ffi::CStr;
use std::time::Instant;

use anyhow::{bail, Result};
use gl::types::{GLenum, GLint, GLsizei, GLuint, GLvoid};
use tracing::{debug, error, warn};
use windows::Win32::Graphics::Direct3D::D3D_SRV_DIMENSION_TEXTURE2D;
use windows::Win32::Graphics::Direct3D11::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Gdi::HDC;
use windows::Win32::Graphics::OpenGL::*;

use crate::GpuBridge;

/// WGL_NV_DX_interop2 constants.
const WGL_ACCESS_READ_WRITE_NV: GLenum = 0x0001;

// ---------------------------------------------------------------------------
// WGL function pointer types
// ---------------------------------------------------------------------------

type WglDxOpenDeviceNv = unsafe extern "system" fn(dx_device: *mut GLvoid) -> *mut GLvoid;
type WglDxCloseDeviceNv = unsafe extern "system" fn(h_device: *mut GLvoid) -> i32;
type WglDxRegisterObjectNv = unsafe extern "system" fn(
    h_device: *mut GLvoid,
    dx_object: *mut GLvoid,
    name: GLuint,
    obj_type: GLenum,
    access: GLenum,
) -> *mut GLvoid;
type WglDxUnregisterObjectNv =
    unsafe extern "system" fn(h_device: *mut GLvoid, h_object: *mut GLvoid) -> i32;
type WglDxLockObjectsNv = unsafe extern "system" fn(
    h_device: *mut GLvoid,
    count: GLint,
    h_objects: *mut *mut GLvoid,
) -> i32;
type WglDxUnlockObjectsNv = unsafe extern "system" fn(
    h_device: *mut GLvoid,
    count: GLint,
    h_objects: *mut *mut GLvoid,
) -> i32;
type WglGetExtensionsStringArb = unsafe extern "system" fn(hdc: HDC) -> *const i8;

// ---------------------------------------------------------------------------
// WglInteropFunctions
// ---------------------------------------------------------------------------

/// Loaded WGL_NV_DX_interop2 function pointers.
struct WglInteropFunctions {
    dx_open_device: WglDxOpenDeviceNv,
    dx_close_device: WglDxCloseDeviceNv,
    dx_register_object: WglDxRegisterObjectNv,
    dx_unregister_object: WglDxUnregisterObjectNv,
    dx_lock_objects: WglDxLockObjectsNv,
    dx_unlock_objects: WglDxUnlockObjectsNv,
}

impl WglInteropFunctions {
    /// Load all WGL_NV_DX_interop2 function pointers via wglGetProcAddress.
    fn load() -> Option<Self> {
        unsafe {
            let load = |name: &CStr| -> Option<*mut GLvoid> {
                let addr = wglGetProcAddress(windows::core::PCSTR(name.as_ptr() as *const u8));
                let addr = addr?;
                let ptr = addr as usize as *mut GLvoid;
                if ptr.is_null() {
                    None
                } else {
                    Some(ptr)
                }
            };

            Some(Self {
                dx_open_device: std::mem::transmute::<*mut GLvoid, WglDxOpenDeviceNv>(load(
                    c"wglDXOpenDeviceNV",
                )?),
                dx_close_device: std::mem::transmute::<*mut GLvoid, WglDxCloseDeviceNv>(load(
                    c"wglDXCloseDeviceNV",
                )?),
                dx_register_object: std::mem::transmute::<*mut GLvoid, WglDxRegisterObjectNv>(
                    load(c"wglDXRegisterObjectNV")?,
                ),
                dx_unregister_object: std::mem::transmute::<*mut GLvoid, WglDxUnregisterObjectNv>(
                    load(c"wglDXUnregisterObjectNV")?,
                ),
                dx_lock_objects: std::mem::transmute::<*mut GLvoid, WglDxLockObjectsNv>(load(
                    c"wglDXLockObjectsNV",
                )?),
                dx_unlock_objects: std::mem::transmute::<*mut GLvoid, WglDxUnlockObjectsNv>(load(
                    c"wglDXUnlockObjectsNV",
                )?),
            })
        }
    }
}

// ---------------------------------------------------------------------------
// SharedTexture / SharedTexturePair
// ---------------------------------------------------------------------------

/// A D3D11 texture shared with OpenGL via WGL_NV_DX_interop2.
struct SharedTexture {
    d3d_texture: ID3D11Texture2D,
    gl_texture: GLuint,
    /// WGL interop handle returned by wglDXRegisterObjectNV.
    interop_handle: *mut GLvoid,
}

impl SharedTexture {
    fn new(
        device: &ID3D11Device,
        wgl_fns: &WglInteropFunctions,
        interop_device: *mut GLvoid,
        width: u32,
        height: u32,
    ) -> Option<Self> {
        // Create D3D11 texture with SHARED flag for WGL interop
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
            MiscFlags: D3D11_RESOURCE_MISC_SHARED.0 as u32,
        };

        let mut d3d_texture = None;
        unsafe { device.CreateTexture2D(&desc, None, Some(&mut d3d_texture as *mut _)) }.ok()?;
        let d3d_texture = d3d_texture?;

        // Create GL texture name
        let mut gl_texture: GLuint = 0;
        unsafe { gl::GenTextures(1, &mut gl_texture) };
        if gl_texture == 0 {
            error!("Failed to generate GL texture name for interop");
            return None;
        }

        // Register the D3D11 texture with GL via WGL_NV_DX_interop2
        let interop_handle = unsafe {
            // Get the raw COM pointer for the D3D11 texture
            let d3d_raw: *mut GLvoid =
                windows::core::Interface::as_raw(&d3d_texture) as *mut GLvoid;
            (wgl_fns.dx_register_object)(
                interop_device,
                d3d_raw,
                gl_texture,
                gl::TEXTURE_2D,
                WGL_ACCESS_READ_WRITE_NV,
            )
        };

        if interop_handle.is_null() {
            error!(
                "wglDXRegisterObjectNV failed for texture {}x{}",
                width, height
            );
            unsafe { gl::DeleteTextures(1, &gl_texture) };
            return None;
        }

        Some(Self {
            d3d_texture,
            gl_texture,
            interop_handle,
        })
    }
}

/// A paired input/output shared texture set for one frame slot.
/// SRV and UAV views are cached here (created once per resize, not every frame).
struct SharedTexturePair {
    input: SharedTexture,
    output: SharedTexture,
    /// Cached SRV for reading the input texture in compute shaders.
    input_srv: ID3D11ShaderResourceView,
    /// Cached UAV for writing the output texture in compute shaders.
    output_uav: ID3D11UnorderedAccessView,
    /// Cached SRV for reading the output texture (used by interleaved field modes).
    output_srv: ID3D11ShaderResourceView,
}

impl SharedTexturePair {
    fn new(
        device: &ID3D11Device,
        wgl_fns: &WglInteropFunctions,
        interop_device: *mut GLvoid,
        width: u32,
        height: u32,
    ) -> Option<Self> {
        let input = SharedTexture::new(device, wgl_fns, interop_device, width, height)?;
        let output = SharedTexture::new(device, wgl_fns, interop_device, width, height)?;

        // Create and cache the SRV for the input texture
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
        let mut input_srv = None;
        unsafe {
            device.CreateShaderResourceView(
                &input.d3d_texture,
                Some(&srv_desc as *const _),
                Some(&mut input_srv as *mut _),
            )
        }
        .ok()?;

        // Create and cache the UAV for the output texture
        let uav_desc = D3D11_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_B8G8R8A8_UNORM,
            ViewDimension: D3D11_UAV_DIMENSION_TEXTURE2D,
            Anonymous: D3D11_UNORDERED_ACCESS_VIEW_DESC_0 {
                Texture2D: D3D11_TEX2D_UAV { MipSlice: 0 },
            },
        };
        let mut output_uav = None;
        unsafe {
            device.CreateUnorderedAccessView(
                &output.d3d_texture,
                Some(&uav_desc as *const _),
                Some(&mut output_uav as *mut _),
            )
        }
        .ok()?;

        // Create and cache the SRV for the output texture (for interleaved reading)
        let mut output_srv = None;
        unsafe {
            device.CreateShaderResourceView(
                &output.d3d_texture,
                Some(&srv_desc as *const _),
                Some(&mut output_srv as *mut _),
            )
        }
        .ok()?;

        Some(Self {
            input,
            output,
            input_srv: input_srv?,
            output_uav: output_uav?,
            output_srv: output_srv?,
        })
    }
}

// ---------------------------------------------------------------------------
// GlDx11Bridge
// ---------------------------------------------------------------------------

/// Double-buffered bridge between OpenGL and D3D11 via WGL_NV_DX_interop2.
///
/// Two SharedTexturePair slots allow pipelining: D3D11 processes the "front"
/// pair while GL blits the previous result from the "back" pair to the host
/// FBO. This introduces one frame of latency but allows D3D11 compute to
/// overlap with host compositing between draw calls.
pub struct GlDx11Bridge {
    /// Reference to the D3D11 device (for texture creation).
    device: ID3D11Device,
    /// Reference to the immediate context (for GPU sync queries).
    context: ID3D11DeviceContext,
    /// GPU event query for waiting on dispatch completion.
    gpu_query: ID3D11Query,
    /// Loaded WGL interop function pointers.
    wgl_fns: WglInteropFunctions,
    /// WGL interop device handle from wglDXOpenDeviceNV.
    interop_device: *mut GLvoid,
    /// Double-buffered shared texture pairs.
    pairs: [Option<SharedTexturePair>; 2],
    /// Index of the pair currently being written by D3D11 compute.
    front: usize,
    /// Whether the previous frame's D3D11 dispatch has been signaled as complete.
    pending_dispatch: bool,
    /// Frame counter from the most recent draw call that dispatched D3D11 compute.
    last_dispatch_frame: Option<u64>,
    /// Wall-clock time of the most recent dispatch. Used to detect stale
    /// back-buffer data after deselection/reselection (where the frame counter
    /// is consecutive but real time has a gap).
    last_dispatch_time: Option<Instant>,
    read_fbo: GLuint,
    draw_fbo: GLuint,
    dimensions: (u32, u32),
}

impl GlDx11Bridge {
    /// Create a new GL-D3D11 bridge. The D3D11 device, context, and query are
    /// borrowed (cloned COM references) from a [`Dx11Device`](super::Dx11Device).
    ///
    /// Returns `None` if WGL_NV_DX_interop2 is not available.
    pub fn new(
        device: &ID3D11Device,
        context: &ID3D11DeviceContext,
        query: &ID3D11Query,
    ) -> Option<Self> {
        let wgl_fns = WglInteropFunctions::load()?;

        // Open the D3D11 device for WGL interop
        let interop_device = unsafe {
            let d3d_raw: *mut GLvoid = windows::core::Interface::as_raw(device) as *mut GLvoid;
            (wgl_fns.dx_open_device)(d3d_raw)
        };

        if interop_device.is_null() {
            error!("wglDXOpenDeviceNV failed");
            return None;
        }

        debug!("GL-D3D11 interop bridge initialized via WGL_NV_DX_interop2");

        Some(Self {
            device: device.clone(),
            context: context.clone(),
            gpu_query: query.clone(),
            wgl_fns,
            interop_device,
            pairs: [None, None],
            front: 0,
            pending_dispatch: false,
            last_dispatch_frame: None,
            last_dispatch_time: None,
            read_fbo: 0,
            draw_fbo: 0,
            dimensions: (0, 0),
        })
    }

    /// Check if the WGL_NV_DX_interop2 extension is available in the current
    /// GL context.
    pub fn is_available() -> bool {
        unsafe {
            let get_ext: Option<WglGetExtensionsStringArb> = {
                let addr = wglGetProcAddress(windows::core::PCSTR(
                    c"wglGetExtensionsStringARB".as_ptr() as *const u8,
                ));
                addr.map(|a| {
                    std::mem::transmute::<
                        unsafe extern "system" fn() -> isize,
                        WglGetExtensionsStringArb,
                    >(a)
                })
            };

            let ext_fn = match get_ext {
                Some(f) => f,
                None => return false,
            };

            let hdc = wglGetCurrentDC();
            let ext_str = ext_fn(hdc);
            if ext_str.is_null() {
                return false;
            }

            let ext_cstr = CStr::from_ptr(ext_str);
            let ext_string = ext_cstr.to_string_lossy();
            ext_string.contains("WGL_NV_DX_interop2")
        }
    }

    // -- D3D11 view accessors (platform-specific, not part of GpuBridge) -----

    /// Get the D3D11 SRV for the front input texture (read by compute shaders).
    /// Returns a cloned COM reference (cheap AddRef, no device allocation).
    pub fn input_srv(&self) -> Option<ID3D11ShaderResourceView> {
        Some(self.pairs[self.front].as_ref()?.input_srv.clone())
    }

    /// Get the D3D11 UAV for the front output texture (written by compute shaders).
    /// Returns a cloned COM reference (cheap AddRef, no device allocation).
    pub fn output_uav(&self) -> Option<ID3D11UnorderedAccessView> {
        Some(self.pairs[self.front].as_ref()?.output_uav.clone())
    }

    /// Get the D3D11 SRV for the back output texture (previous frame's result).
    /// Used by interleaved field modes to fill non-field rows.
    pub fn back_output_srv(&self) -> Option<ID3D11ShaderResourceView> {
        let back = 1 - self.front;
        Some(self.pairs[back].as_ref()?.output_srv.clone())
    }

    /// Borrow the D3D11 device held by this bridge.
    pub fn device(&self) -> &ID3D11Device {
        &self.device
    }

    /// Borrow the D3D11 immediate context held by this bridge.
    pub fn context(&self) -> &ID3D11DeviceContext {
        &self.context
    }

    /// Borrow the GPU event query held by this bridge.
    pub fn query(&self) -> &ID3D11Query {
        &self.gpu_query
    }

    // -- Lock / unlock helpers ------------------------------------------------

    /// Lock the front pair's GL textures for GL access.
    /// Must be called before any GL operations on shared textures.
    unsafe fn lock_gl_textures_front(&self) -> bool {
        let pair = match &self.pairs[self.front] {
            Some(p) => p,
            None => return false,
        };

        let mut handles = [pair.input.interop_handle, pair.output.interop_handle];
        let result = (self.wgl_fns.dx_lock_objects)(
            self.interop_device,
            handles.len() as GLint,
            handles.as_mut_ptr(),
        );
        result != 0
    }

    /// Unlock the front pair's GL textures to release them back to D3D11.
    unsafe fn unlock_gl_textures_front(&self) -> bool {
        let pair = match &self.pairs[self.front] {
            Some(p) => p,
            None => return false,
        };

        let mut handles = [pair.input.interop_handle, pair.output.interop_handle];
        let result = (self.wgl_fns.dx_unlock_objects)(
            self.interop_device,
            handles.len() as GLint,
            handles.as_mut_ptr(),
        );
        result != 0
    }

    /// Lock the back pair's output GL texture for GL access (for reading the result).
    unsafe fn lock_gl_texture_back_output(&self) -> bool {
        let back = 1 - self.front;
        let pair = match &self.pairs[back] {
            Some(p) => p,
            None => return false,
        };

        let mut handles = [pair.output.interop_handle];
        let result = (self.wgl_fns.dx_lock_objects)(self.interop_device, 1, handles.as_mut_ptr());
        result != 0
    }

    /// Unlock the back pair's output GL texture.
    unsafe fn unlock_gl_texture_back_output(&self) -> bool {
        let back = 1 - self.front;
        let pair = match &self.pairs[back] {
            Some(p) => p,
            None => return false,
        };

        let mut handles = [pair.output.interop_handle];
        let result =
            (self.wgl_fns.dx_unlock_objects)(self.interop_device, 1, handles.as_mut_ptr());
        result != 0
    }

    /// Lock the front pair's output GL texture (for synchronous blit of current frame).
    unsafe fn lock_gl_texture_front_output(&self) -> bool {
        let pair = match &self.pairs[self.front] {
            Some(p) => p,
            None => return false,
        };

        let mut handles = [pair.output.interop_handle];
        let result = (self.wgl_fns.dx_lock_objects)(self.interop_device, 1, handles.as_mut_ptr());
        result != 0
    }

    /// Unlock the front pair's output GL texture.
    unsafe fn unlock_gl_texture_front_output(&self) -> bool {
        let pair = match &self.pairs[self.front] {
            Some(p) => p,
            None => return false,
        };

        let mut handles = [pair.output.interop_handle];
        let result =
            (self.wgl_fns.dx_unlock_objects)(self.interop_device, 1, handles.as_mut_ptr());
        result != 0
    }

    // -- GPU query polling ----------------------------------------------------

    /// Poll the GPU query until the dispatch completes (or timeout).
    fn poll_gpu_query(&self) {
        if !self.pending_dispatch {
            return;
        }
        let start = Instant::now();
        unsafe {
            // Poll until GPU signals completion or timeout.
            // For D3D11_QUERY_EVENT, GetData writes a BOOL: TRUE when the GPU is done.
            // GetData returns S_OK when data is ready and S_FALSE when not yet ready,
            // but the windows crate maps both to Ok(()) since S_FALSE is a success HRESULT.
            // Instead of checking the Result, we check the returned BOOL value directly.
            // On S_FALSE the output buffer is left unmodified, so our zero-init stays 0.
            loop {
                let mut done: u32 = 0;
                let _ = self.context.GetData(
                    &self.gpu_query,
                    Some(&mut done as *mut u32 as *mut GLvoid),
                    std::mem::size_of::<u32>() as u32,
                    0,
                );
                if done != 0 {
                    break;
                }
                if start.elapsed().as_millis() > 100 {
                    warn!("GPU query timed out after 100ms, proceeding anyway");
                    break;
                }
                std::thread::yield_now();
            }
        }
    }

    /// Unregister all shared textures and drop the pairs.
    fn destroy_pairs(&mut self) {
        for pair in &mut self.pairs {
            if let Some(p) = pair.take() {
                unsafe {
                    (self.wgl_fns.dx_unregister_object)(
                        self.interop_device,
                        p.input.interop_handle,
                    );
                    gl::DeleteTextures(1, &p.input.gl_texture);
                    (self.wgl_fns.dx_unregister_object)(
                        self.interop_device,
                        p.output.interop_handle,
                    );
                    gl::DeleteTextures(1, &p.output.gl_texture);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// GpuBridge trait implementation
// ---------------------------------------------------------------------------

impl GpuBridge for GlDx11Bridge {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn ensure_dimensions(&mut self, width: u32, height: u32) -> Result<()> {
        if self.dimensions == (width, height)
            && self.pairs[0].is_some()
            && self.pairs[1].is_some()
        {
            return Ok(());
        }

        // Dimension change: wait for any in-flight work before destroying textures
        self.wait_for_previous();

        // Clean up old pairs (unregister from interop first)
        self.destroy_pairs();

        // Clean up old FBOs (unbind first to avoid deleting a bound FBO)
        unsafe {
            gl::BindFramebuffer(gl::FRAMEBUFFER, 0);
            if self.read_fbo != 0 {
                gl::DeleteFramebuffers(1, &self.read_fbo);
            }
            if self.draw_fbo != 0 {
                gl::DeleteFramebuffers(1, &self.draw_fbo);
            }
        }

        self.pairs[0] = SharedTexturePair::new(
            &self.device,
            &self.wgl_fns,
            self.interop_device,
            width,
            height,
        );
        self.pairs[1] = SharedTexturePair::new(
            &self.device,
            &self.wgl_fns,
            self.interop_device,
            width,
            height,
        );

        if self.pairs[0].is_none() || self.pairs[1].is_none() {
            self.destroy_pairs();
            self.read_fbo = 0;
            self.draw_fbo = 0;
            self.dimensions = (0, 0);
            bail!("Failed to create shared D3D11-GL texture pairs");
        }

        // Create separate FBOs for read and draw
        unsafe {
            gl::GenFramebuffers(1, &mut self.read_fbo);
            gl::GenFramebuffers(1, &mut self.draw_fbo);
        }

        self.dimensions = (width, height);
        self.front = 0;
        self.last_dispatch_frame = None;
        self.last_dispatch_time = None;
        Ok(())
    }

    fn blit_input_from_host_scaled(
        &mut self,
        host_texture: GLuint,
        src_w: u32,
        src_h: u32,
        dst_w: u32,
        dst_h: u32,
        bilinear: bool,
    ) -> bool {
        let input_gl = match &self.pairs[self.front] {
            Some(pair) => pair.input.gl_texture,
            None => return false,
        };

        // Lock front input for GL access
        if unsafe { !self.lock_gl_textures_front() } {
            warn!("Failed to lock GL textures for input blit");
            return false;
        }

        unsafe {
            // READ side: attach the host texture (always TEXTURE_2D on Windows)
            gl::BindFramebuffer(gl::READ_FRAMEBUFFER, self.read_fbo);
            gl::FramebufferTexture2D(
                gl::READ_FRAMEBUFFER,
                gl::COLOR_ATTACHMENT0,
                gl::TEXTURE_2D,
                host_texture,
                0,
            );

            if gl::CheckFramebufferStatus(gl::READ_FRAMEBUFFER) != gl::FRAMEBUFFER_COMPLETE {
                warn!("READ_FRAMEBUFFER incomplete for host texture {host_texture}");
                gl::BindFramebuffer(gl::FRAMEBUFFER, 0);
                self.unlock_gl_textures_front();
                return false;
            }
            gl::ReadBuffer(gl::COLOR_ATTACHMENT0);

            // DRAW side: attach the shared TEXTURE_2D
            gl::BindFramebuffer(gl::DRAW_FRAMEBUFFER, self.draw_fbo);
            gl::FramebufferTexture2D(
                gl::DRAW_FRAMEBUFFER,
                gl::COLOR_ATTACHMENT0,
                gl::TEXTURE_2D,
                input_gl,
                0,
            );
            gl::DrawBuffer(gl::COLOR_ATTACHMENT0);

            let filter = if bilinear { gl::LINEAR } else { gl::NEAREST };

            gl::BlitFramebuffer(
                0,
                0,
                src_w as GLsizei,
                src_h as GLsizei,
                0,
                0,
                dst_w as GLsizei,
                dst_h as GLsizei,
                gl::COLOR_BUFFER_BIT,
                filter,
            );

            gl::BindFramebuffer(gl::FRAMEBUFFER, 0);
            gl::Flush();

            // Unlock so D3D11 can access the textures
            self.unlock_gl_textures_front();
        }
        true
    }

    fn blit_back_output_to_target_scaled(
        &mut self,
        host_fbo: GLuint,
        src_w: u32,
        src_h: u32,
        dst_w: u32,
        dst_h: u32,
        bilinear: bool,
    ) -> bool {
        let back = 1 - self.front;
        let output_gl = match &self.pairs[back] {
            Some(pair) => pair.output.gl_texture,
            None => return false,
        };

        // Lock back output for GL access
        if unsafe { !self.lock_gl_texture_back_output() } {
            warn!("Failed to lock back output GL texture for blit");
            return false;
        }

        unsafe {
            // Attach output as READ source
            gl::BindFramebuffer(gl::READ_FRAMEBUFFER, self.read_fbo);
            gl::FramebufferTexture2D(
                gl::READ_FRAMEBUFFER,
                gl::COLOR_ATTACHMENT0,
                gl::TEXTURE_2D,
                output_gl,
                0,
            );
            gl::ReadBuffer(gl::COLOR_ATTACHMENT0);

            // DRAW to target
            gl::BindFramebuffer(gl::DRAW_FRAMEBUFFER, host_fbo);

            let filter = if bilinear { gl::LINEAR } else { gl::NEAREST };

            gl::BlitFramebuffer(
                0,
                0,
                src_w as GLsizei,
                src_h as GLsizei,
                0,
                0,
                dst_w as GLsizei,
                dst_h as GLsizei,
                gl::COLOR_BUFFER_BIT,
                filter,
            );

            gl::BindFramebuffer(gl::FRAMEBUFFER, 0);

            self.unlock_gl_texture_back_output();
        }
        true
    }

    fn blit_output_to_target_scaled(
        &mut self,
        host_fbo: GLuint,
        src_w: u32,
        src_h: u32,
        dst_w: u32,
        dst_h: u32,
        bilinear: bool,
    ) -> bool {
        let output_gl = match &self.pairs[self.front] {
            Some(pair) => pair.output.gl_texture,
            None => return false,
        };

        // Lock front output for GL access
        if unsafe { !self.lock_gl_texture_front_output() } {
            warn!("Failed to lock front output GL texture for blit");
            return false;
        }

        unsafe {
            gl::BindFramebuffer(gl::READ_FRAMEBUFFER, self.read_fbo);
            gl::FramebufferTexture2D(
                gl::READ_FRAMEBUFFER,
                gl::COLOR_ATTACHMENT0,
                gl::TEXTURE_2D,
                output_gl,
                0,
            );
            gl::ReadBuffer(gl::COLOR_ATTACHMENT0);

            gl::BindFramebuffer(gl::DRAW_FRAMEBUFFER, host_fbo);

            let filter = if bilinear { gl::LINEAR } else { gl::NEAREST };

            gl::BlitFramebuffer(
                0,
                0,
                src_w as GLsizei,
                src_h as GLsizei,
                0,
                0,
                dst_w as GLsizei,
                dst_h as GLsizei,
                gl::COLOR_BUFFER_BIT,
                filter,
            );

            gl::BindFramebuffer(gl::FRAMEBUFFER, 0);

            self.unlock_gl_texture_front_output();
        }
        true
    }

    fn has_result_ready(&self, current_frame: u64) -> bool {
        self.pending_dispatch
            && self
                .last_dispatch_frame
                .is_some_and(|last| current_frame == last.wrapping_add(1))
            && self
                .last_dispatch_time
                .is_some_and(|t| t.elapsed().as_millis() < 100)
    }

    fn wait_for_previous(&mut self) {
        self.poll_gpu_query();
        self.pending_dispatch = false;
    }

    fn wait_for_pending(&mut self) {
        self.poll_gpu_query();
    }

    fn swap(&mut self) {
        self.front = 1 - self.front;
    }

    fn mark_dispatch(&mut self, frame: u64) {
        // Signal the GPU event query so poll_gpu_query can detect completion.
        // In the original ntsc-ffgl-plugin this was the caller's responsibility,
        // but since we now own the context + query it belongs here.
        unsafe {
            self.context.End(&self.gpu_query);
        }
        self.pending_dispatch = true;
        self.last_dispatch_frame = Some(frame);
        self.last_dispatch_time = Some(Instant::now());
    }

    fn cleanup(&mut self) {
        self.poll_gpu_query();
        self.pending_dispatch = false;
        self.destroy_pairs();
        self.front = 0;
        self.last_dispatch_frame = None;
        self.last_dispatch_time = None;
        unsafe {
            gl::BindFramebuffer(gl::FRAMEBUFFER, 0);
            if self.read_fbo != 0 {
                gl::DeleteFramebuffers(1, &self.read_fbo);
                self.read_fbo = 0;
            }
            if self.draw_fbo != 0 {
                gl::DeleteFramebuffers(1, &self.draw_fbo);
                self.draw_fbo = 0;
            }
        }
        self.dimensions = (0, 0);
    }

    fn dimensions(&self) -> (u32, u32) {
        self.dimensions
    }
}

impl Drop for GlDx11Bridge {
    fn drop(&mut self) {
        // Wait for any in-flight GPU work before destroying shared textures.
        self.poll_gpu_query();
        self.pending_dispatch = false;
        self.destroy_pairs();
        unsafe {
            if self.read_fbo != 0 {
                gl::DeleteFramebuffers(1, &self.read_fbo);
            }
            if self.draw_fbo != 0 {
                gl::DeleteFramebuffers(1, &self.draw_fbo);
            }
            if !self.interop_device.is_null() {
                (self.wgl_fns.dx_close_device)(self.interop_device);
            }
        }
    }
}
