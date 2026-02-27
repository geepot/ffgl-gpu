//! Thread-local draw infrastructure for GPU-accelerated FFGL plugins.
//!
//! This module provides [`draw_gpu_effect`], the main entry point that handles:
//! - Lazy GPU context initialization
//! - GL-to-GPU bridge management (Metal via IOSurface, DX11 via
//!   WGL_NV_DX_interop2)
//! - Double-buffered pipelining (one frame latency)
//! - GL state save/restore
//! - Instance tracking (resource release on instance switch)
//!
//! Thread-local state ensures multi-instance safety when an FFGL host calls
//! different plugin instances from the same thread.

use crate::context::GpuContext;
use crate::plugin::{DrawInput, GpuPlugin};
use ffgl_core::inputs::GLInput;
use ffgl_core::FFGLData;
use gl::types::{GLenum, GLint, GLuint};
use gpu_interop::GpuBridge as _;
use std::cell::RefCell;
use tracing::error;

// ---------------------------------------------------------------------------
// GL state save / restore
// ---------------------------------------------------------------------------

/// Saved GL state that we restore after our raw GL operations.
struct SavedGlState {
    pack_buffer: GLint,
    unpack_buffer: GLint,
    framebuffer: GLint,
    draw_framebuffer: GLint,
    read_framebuffer: GLint,
    texture_2d: GLint,
    active_texture: GLint,
    vao: GLint,
    viewport: [GLint; 4],
}

impl SavedGlState {
    unsafe fn save() -> Self {
        let mut s = Self {
            pack_buffer: 0,
            unpack_buffer: 0,
            framebuffer: 0,
            draw_framebuffer: 0,
            read_framebuffer: 0,
            texture_2d: 0,
            active_texture: 0,
            vao: 0,
            viewport: [0; 4],
        };
        gl::GetIntegerv(gl::PIXEL_PACK_BUFFER_BINDING, &mut s.pack_buffer);
        gl::GetIntegerv(gl::PIXEL_UNPACK_BUFFER_BINDING, &mut s.unpack_buffer);
        gl::GetIntegerv(gl::FRAMEBUFFER_BINDING, &mut s.framebuffer);
        gl::GetIntegerv(gl::DRAW_FRAMEBUFFER_BINDING, &mut s.draw_framebuffer);
        gl::GetIntegerv(gl::READ_FRAMEBUFFER_BINDING, &mut s.read_framebuffer);
        gl::GetIntegerv(gl::TEXTURE_BINDING_2D, &mut s.texture_2d);
        gl::GetIntegerv(gl::ACTIVE_TEXTURE, &mut s.active_texture);
        gl::GetIntegerv(gl::VERTEX_ARRAY_BINDING, &mut s.vao);
        gl::GetIntegerv(gl::VIEWPORT, s.viewport.as_mut_ptr());
        s
    }

    unsafe fn restore(&self) {
        gl::BindBuffer(gl::PIXEL_PACK_BUFFER, self.pack_buffer as GLuint);
        gl::BindBuffer(gl::PIXEL_UNPACK_BUFFER, self.unpack_buffer as GLuint);
        gl::BindFramebuffer(gl::FRAMEBUFFER, self.framebuffer as GLuint);
        gl::BindFramebuffer(gl::DRAW_FRAMEBUFFER, self.draw_framebuffer as GLuint);
        gl::BindFramebuffer(gl::READ_FRAMEBUFFER, self.read_framebuffer as GLuint);
        gl::ActiveTexture(self.active_texture as GLenum);
        gl::BindTexture(gl::TEXTURE_2D, self.texture_2d as GLuint);
        gl::BindVertexArray(self.vao as GLuint);
        gl::Viewport(
            self.viewport[0],
            self.viewport[1],
            self.viewport[2],
            self.viewport[3],
        );
    }
}

// ---------------------------------------------------------------------------
// Shared GL helpers
// ---------------------------------------------------------------------------

fn clear_gl_errors() {
    unsafe {
        while gl::GetError() != gl::NO_ERROR {}
    }
}

fn is_context_current() -> bool {
    unsafe { !gl::GetString(gl::VERSION).is_null() }
}

fn passthrough(glium_ctx: &mut ffgl_glium::FFGLGlium, data: &FFGLData, frame_data: GLInput<'_>) {
    use glium::Surface;
    let (width, height) = data.get_dimensions();
    glium_ctx.draw(
        (width, height),
        (width, height),
        frame_data,
        &mut |target, textures| {
            if let Some(input_texture) = textures.first() {
                input_texture
                    .as_surface()
                    .fill(target, glium::uniforms::MagnifySamplerFilter::Linear);
            }
            Ok(())
        },
    );
}

// ---------------------------------------------------------------------------
// macOS Metal draw path
// ---------------------------------------------------------------------------

#[cfg(target_os = "macos")]
mod metal_draw {
    use super::*;
    use gpu_interop::metal::GlMetalBridge;

    thread_local! {
        static GPU_CTX: RefCell<Option<GpuContext>> = const { RefCell::new(None) };
        static BRIDGE: RefCell<Option<GlMetalBridge>> = const { RefCell::new(None) };
        static LAST_INSTANCE_ID: RefCell<Option<u64>> = const { RefCell::new(None) };
        static GPU_INITIALIZED: RefCell<bool> = const { RefCell::new(false) };
    }

    fn release_resources() {
        BRIDGE.with(|cell| {
            if let Some(bridge) = cell.borrow_mut().as_mut() {
                bridge.cleanup();
            }
        });
        GPU_INITIALIZED.with(|cell| *cell.borrow_mut() = false);
    }

    pub fn ensure_instance_resources(instance_id: u64) {
        LAST_INSTANCE_ID.with(|cell| {
            let mut id = cell.borrow_mut();
            if *id != Some(instance_id) {
                release_resources();
                *id = Some(instance_id);
            }
        });
    }

    pub fn validate_gl_state() -> bool {
        clear_gl_errors();
        if !is_context_current() {
            return false;
        }
        let mut need_release = false;
        BRIDGE.with(|cell| {
            if let Some(bridge) = cell.borrow().as_ref() {
                if !bridge.is_valid() {
                    need_release = true;
                }
            }
        });
        if need_release {
            release_resources();
            LAST_INSTANCE_ID.with(|cell| *cell.borrow_mut() = None);
        }
        true
    }

    pub fn draw<P: GpuPlugin>(
        plugin: &mut P,
        instance_id: u64,
        glium: &mut ffgl_glium::FFGLGlium,
        data: &FFGLData,
        frame_data: GLInput<'_>,
        frame_counter: u64,
        internal_resolution: f32,
        filter_quality: f32,
        metallib_bytes: &[u8],
    ) {
        ensure_instance_resources(instance_id);
        if !validate_gl_state() {
            passthrough(glium, data, frame_data);
            return;
        }

        let (width, height) = data.get_dimensions();

        // Compute processing dimensions from internal_resolution scale factor.
        let res_scale = internal_resolution.clamp(0.125, 1.0);
        let proc_width = ((width as f32 * res_scale) as u32).max(2);
        let proc_height = ((height as f32 * res_scale) as u32).max(2);
        let use_bilinear = filter_quality >= 0.5;

        // Ensure GPU context is initialized
        let ctx_available = GPU_CTX.with(|cell| {
            let mut ctx = cell.borrow_mut();
            if ctx.is_none() {
                match GpuContext::new(metallib_bytes) {
                    Ok(c) => *ctx = Some(c),
                    Err(e) => {
                        error!("Failed to create GPU context: {e}");
                        return false;
                    }
                }
            }
            ctx.is_some()
        });

        if !ctx_available {
            passthrough(glium, data, frame_data);
            return;
        }

        // Get host FBO and texture
        let host_fbo = frame_data.host;
        let tex_id = match frame_data.textures.first() {
            Some(t) => t.Handle,
            None => {
                passthrough(glium, data, frame_data);
                return;
            }
        };

        let saved_state = unsafe { SavedGlState::save() };

        let success = objc2::rc::autoreleasepool(|_pool| {
            GPU_CTX.with(|ctx_cell| {
                let ctx_ref = ctx_cell.borrow();
                let ctx = ctx_ref.as_ref().unwrap();

                BRIDGE.with(|bridge_cell| {
                    // Initialize bridge if needed (separate borrow scope)
                    {
                        let mut bridge_opt = bridge_cell.borrow_mut();
                        if bridge_opt.is_none() {
                            // Retain-clone the Metal device to pass ownership
                            // to the bridge. The device is a protocol object
                            // so we use Retained::retain on its raw pointer.
                            let device_ref = ctx.device.device();
                            let device_ptr = device_ref
                                as *const objc2::runtime::ProtocolObject<
                                    dyn objc2_metal::MTLDevice,
                                >
                                as *mut objc2::runtime::ProtocolObject<
                                    dyn objc2_metal::MTLDevice,
                                >;
                            // SAFETY: device_ptr points to a valid, live ObjC
                            // object. Retained::retain increments its refcount.
                            let device_retained =
                                unsafe { objc2::rc::Retained::retain(device_ptr) }
                                    .expect("device pointer must be non-null");
                            *bridge_opt = Some(GlMetalBridge::new(device_retained));
                        }
                    }

                    // Call gpu_init on first use
                    let init_ok = GPU_INITIALIZED.with(|cell| {
                        let mut initialized = cell.borrow_mut();
                        if !*initialized {
                            match plugin.gpu_init(ctx) {
                                Ok(()) => {
                                    *initialized = true;
                                    true
                                }
                                Err(e) => {
                                    error!("GpuPlugin::gpu_init failed: {e}");
                                    false
                                }
                            }
                        } else {
                            true
                        }
                    });

                    if !init_ok {
                        return false;
                    }

                    // --- Double-buffered pipelined flow ---
                    // Single mutable borrow for all bridge operations.
                    let mut bridge_opt = bridge_cell.borrow_mut();
                    let bridge = bridge_opt.as_mut().unwrap();

                    if let Err(e) = bridge.ensure_dimensions(proc_width, proc_height) {
                        error!("Failed to ensure bridge dimensions: {e}");
                        return false;
                    }

                    let has_prev = bridge.has_result_ready(frame_counter);

                    bridge.wait_for_previous();

                    if has_prev {
                        bridge.swap();
                        bridge.blit_back_output_to_target_scaled(
                            host_fbo,
                            proc_width,
                            proc_height,
                            width,
                            height,
                            use_bilinear,
                        );
                    }

                    bridge.blit_input_from_host_scaled(
                        tex_id,
                        width,
                        height,
                        proc_width,
                        proc_height,
                        use_bilinear,
                    );

                    // Extract texture pointers from the bridge and wrap in
                    // GpuTexture. The raw pointers remain valid for the
                    // duration of gpu_draw because the bridge is held by this
                    // scope and no bridge methods that invalidate textures are
                    // called until after gpu_draw returns.
                    let input_tex = match bridge.input_metal_texture() {
                        Some(t) => crate::texture::GpuTexture {
                            metal: t as *const _,
                            width: proc_width,
                            height: proc_height,
                        },
                        None => return false,
                    };
                    let output_tex = match bridge.output_metal_texture() {
                        Some(t) => crate::texture::GpuTexture {
                            metal: t as *const _,
                            width: proc_width,
                            height: proc_height,
                        },
                        None => return false,
                    };

                    let mut draw_input = DrawInput {
                        input: &input_tex,
                        output: &output_tex,
                        width: proc_width,
                        height: proc_height,
                        pending_work: None,
                    };

                    plugin.gpu_draw(ctx, &mut draw_input, data, frame_counter);

                    // Store pending work command buffer in bridge for
                    // double-buffer sync.
                    if let Some(pending) = draw_input.pending_work.take() {
                        bridge.store_command_buffer(pending.command_buffer);
                    }

                    bridge.mark_dispatch(frame_counter);

                    if !has_prev {
                        bridge.wait_for_pending();
                        bridge.blit_output_to_target_scaled(
                            host_fbo,
                            proc_width,
                            proc_height,
                            width,
                            height,
                            use_bilinear,
                        );
                    }

                    true
                })
            })
        });

        unsafe {
            saved_state.restore();
        }

        if !success {
            passthrough(glium, data, frame_data);
        }
    }
}

// ---------------------------------------------------------------------------
// Windows DX11 draw path
// ---------------------------------------------------------------------------

#[cfg(target_os = "windows")]
mod dx11_draw {
    use super::*;
    use gpu_interop::dx11::GlDx11Bridge;

    thread_local! {
        static GPU_CTX: RefCell<Option<GpuContext>> = const { RefCell::new(None) };
        static BRIDGE: RefCell<Option<GlDx11Bridge>> = const { RefCell::new(None) };
        static LAST_INSTANCE_ID: RefCell<Option<u64>> = const { RefCell::new(None) };
        static GPU_INITIALIZED: RefCell<bool> = const { RefCell::new(false) };
    }

    fn release_resources() {
        BRIDGE.with(|cell| {
            if let Some(bridge) = cell.borrow_mut().as_mut() {
                bridge.cleanup();
            }
        });
        GPU_INITIALIZED.with(|cell| *cell.borrow_mut() = false);
    }

    pub fn ensure_instance_resources(instance_id: u64) {
        LAST_INSTANCE_ID.with(|cell| {
            let mut id = cell.borrow_mut();
            if *id != Some(instance_id) {
                release_resources();
                *id = Some(instance_id);
            }
        });
    }

    pub fn validate_gl_state() -> bool {
        clear_gl_errors();
        if !is_context_current() {
            return false;
        }
        let mut need_release = false;
        BRIDGE.with(|cell| {
            if let Some(bridge) = cell.borrow().as_ref() {
                if !bridge.is_valid() {
                    need_release = true;
                }
            }
        });
        if need_release {
            release_resources();
            LAST_INSTANCE_ID.with(|cell| *cell.borrow_mut() = None);
        }
        true
    }

    pub fn draw<P: GpuPlugin>(
        plugin: &mut P,
        instance_id: u64,
        glium: &mut ffgl_glium::FFGLGlium,
        data: &FFGLData,
        frame_data: GLInput<'_>,
        frame_counter: u64,
        internal_resolution: f32,
        filter_quality: f32,
        _metallib_bytes: &[u8],
    ) {
        ensure_instance_resources(instance_id);
        if !validate_gl_state() {
            passthrough(glium, data, frame_data);
            return;
        }

        let (width, height) = data.get_dimensions();

        let res_scale = internal_resolution.clamp(0.125, 1.0);
        let proc_width = ((width as f32 * res_scale) as u32).max(2);
        let proc_height = ((height as f32 * res_scale) as u32).max(2);
        let use_bilinear = filter_quality >= 0.5;

        // Ensure D3D11 context is initialized
        let ctx_available = GPU_CTX.with(|cell| {
            let mut ctx = cell.borrow_mut();
            if ctx.is_none() {
                match GpuContext::new() {
                    Ok(c) => *ctx = Some(c),
                    Err(e) => {
                        error!("Failed to create GPU context: {e}");
                        return false;
                    }
                }
            }
            ctx.is_some()
        });

        if !ctx_available {
            passthrough(glium, data, frame_data);
            return;
        }

        // Ensure GL-D3D11 interop bridge is initialized
        let bridge_available = GPU_CTX.with(|ctx_cell| {
            let ctx = ctx_cell.borrow();
            let ctx = ctx.as_ref().unwrap();
            BRIDGE.with(|bridge_cell| {
                let mut bridge = bridge_cell.borrow_mut();
                if bridge.is_none() {
                    *bridge = GlDx11Bridge::new(
                        ctx.device.device(),
                        ctx.device.context(),
                        ctx.device.query(),
                    );
                }
                bridge.is_some()
            })
        });

        if !bridge_available {
            passthrough(glium, data, frame_data);
            return;
        }

        let host_fbo = frame_data.host;
        let tex_id = match frame_data.textures.first() {
            Some(t) => t.Handle,
            None => {
                passthrough(glium, data, frame_data);
                return;
            }
        };

        let saved_state = unsafe { SavedGlState::save() };

        let success = GPU_CTX.with(|ctx_cell| {
            let ctx_ref = ctx_cell.borrow();
            let ctx = ctx_ref.as_ref().unwrap();

            BRIDGE.with(|bridge_cell| {
                // Call gpu_init on first use
                let init_ok = GPU_INITIALIZED.with(|cell| {
                    let mut initialized = cell.borrow_mut();
                    if !*initialized {
                        match plugin.gpu_init(ctx) {
                            Ok(()) => {
                                *initialized = true;
                                true
                            }
                            Err(e) => {
                                error!("GpuPlugin::gpu_init failed: {e}");
                                false
                            }
                        }
                    } else {
                        true
                    }
                });

                if !init_ok {
                    return false;
                }

                // --- Double-buffered pipelined flow ---
                // Single mutable borrow for all bridge operations.
                let mut bridge_opt = bridge_cell.borrow_mut();
                let bridge = bridge_opt.as_mut().unwrap();

                if let Err(e) = bridge.ensure_dimensions(proc_width, proc_height) {
                    error!("Failed to ensure bridge dimensions: {e}");
                    return false;
                }

                let has_prev = bridge.has_result_ready(frame_counter);

                bridge.wait_for_previous();

                if has_prev {
                    bridge.swap();
                    bridge.blit_back_output_to_target_scaled(
                        host_fbo,
                        proc_width,
                        proc_height,
                        width,
                        height,
                        use_bilinear,
                    );
                }

                bridge.blit_input_from_host_scaled(
                    tex_id,
                    width,
                    height,
                    proc_width,
                    proc_height,
                    use_bilinear,
                );

                // Extract owned COM refs from bridge (cheap AddRef).
                let input_srv = match bridge.input_srv() {
                    Some(s) => s,
                    None => return false,
                };
                let output_uav = match bridge.output_uav() {
                    Some(u) => u,
                    None => return false,
                };
                let output_texture = match bridge.output_texture() {
                    Some(t) => t,
                    None => return false,
                };

                let mut draw_input = DrawInput {
                    input_srv,
                    output_uav,
                    output_texture,
                    width: proc_width,
                    height: proc_height,
                    bridge,
                };

                plugin.gpu_draw(ctx, &mut draw_input, data, frame_counter);

                // Reclaim bridge from DrawInput for post-draw operations.
                let bridge = draw_input.bridge;

                bridge.mark_dispatch(frame_counter);

                if !has_prev {
                    bridge.wait_for_pending();
                    bridge.blit_output_to_target_scaled(
                        host_fbo,
                        proc_width,
                        proc_height,
                        width,
                        height,
                        use_bilinear,
                    );
                }

                true
            })
        });

        unsafe {
            saved_state.restore();
        }

        if !success {
            passthrough(glium, data, frame_data);
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// If a different plugin instance is now using this thread, release old
/// resources.
pub fn ensure_instance_gl_resources(instance_id: u64) {
    #[cfg(target_os = "macos")]
    metal_draw::ensure_instance_resources(instance_id);

    #[cfg(target_os = "windows")]
    dx11_draw::ensure_instance_resources(instance_id);

    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    let _ = instance_id;
}

/// Validate GL state before drawing. Returns `false` if the GL context is
/// invalid and drawing should be skipped.
pub fn validate_gl_state_before_draw() -> bool {
    #[cfg(target_os = "macos")]
    return metal_draw::validate_gl_state();

    #[cfg(target_os = "windows")]
    return dx11_draw::validate_gl_state();

    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    {
        clear_gl_errors();
        is_context_current()
    }
}

/// Main draw function called by FFGL plugins.
///
/// Handles lazy GPU context initialization, bridge management, and the
/// double-buffered blit pipeline. The plugin's [`GpuPlugin::gpu_draw`] is
/// called each frame to perform the actual GPU work.
///
/// # Arguments
///
/// * `plugin` - The plugin instance implementing [`GpuPlugin`].
/// * `instance_id` - Unique identifier for this plugin instance (for
///   thread-local resource tracking).
/// * `glium` - The glium context (used for passthrough fallback).
/// * `data` - Host-provided FFGL data (viewport dimensions, timing, etc).
/// * `frame_data` - Host input textures and FBO.
/// * `frame_counter` - Monotonically increasing frame counter.
/// * `internal_resolution` - Resolution scale factor `[0.125, 1.0]`.
/// * `filter_quality` - Filter quality `[0.0, 1.0]`. Values >= 0.5 use
///   bilinear filtering.
/// * `metallib_bytes` - Compiled Metal shader library bytes (from
///   [`include_metallib!`]). Ignored on Windows.
pub fn draw_gpu_effect<P: GpuPlugin>(
    plugin: &mut P,
    instance_id: u64,
    glium: &mut ffgl_glium::FFGLGlium,
    data: &FFGLData,
    frame_data: GLInput<'_>,
    frame_counter: u64,
    internal_resolution: f32,
    filter_quality: f32,
    metallib_bytes: &[u8],
) {
    #[cfg(target_os = "macos")]
    metal_draw::draw(
        plugin,
        instance_id,
        glium,
        data,
        frame_data,
        frame_counter,
        internal_resolution,
        filter_quality,
        metallib_bytes,
    );

    #[cfg(target_os = "windows")]
    dx11_draw::draw(
        plugin,
        instance_id,
        glium,
        data,
        frame_data,
        frame_counter,
        internal_resolution,
        filter_quality,
        metallib_bytes,
    );

    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    {
        let _ = (
            plugin,
            instance_id,
            frame_counter,
            internal_resolution,
            filter_quality,
            metallib_bytes,
        );
        passthrough(glium, data, frame_data);
    }
}
