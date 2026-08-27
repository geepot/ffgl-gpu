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

// `GpuPlugin` is referenced at module level by `draw_gpu_effect<P: GpuPlugin>`
// so it stays ungated. Everything else is only used inside the cfg-gated
// `metal_draw` / `dx11_draw` submodules below; gate those imports so they
// don't warn-as-unused on a non-mac, non-windows host build (e.g. the
// Linux Docker container compiling ffgl-gpu as a build-dep).
#[cfg(any(target_os = "macos", target_os = "windows"))]
use crate::context::GpuContext;
#[cfg(any(target_os = "macos", target_os = "windows"))]
use crate::plugin::DrawInput;
use crate::plugin::GpuPlugin;
use ffgl_core::inputs::GLInput;
use ffgl_core::FFGLData;
#[cfg(any(target_os = "macos", target_os = "windows"))]
use gl::types::{GLenum, GLint, GLuint};
#[cfg(any(target_os = "macos", target_os = "windows"))]
use gpu_interop::GpuBridge as _;
#[cfg(any(target_os = "macos", target_os = "windows"))]
use std::cell::RefCell;
#[cfg(any(target_os = "macos", target_os = "windows"))]
use std::collections::{HashMap, HashSet};
#[cfg(any(target_os = "macos", target_os = "windows"))]
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(any(target_os = "macos", target_os = "windows"))]
use std::sync::{LazyLock, Mutex, Once};
#[cfg(any(target_os = "macos", target_os = "windows"))]
use std::time::{Duration, Instant};
#[cfg(any(target_os = "macos", target_os = "windows"))]
use tracing::error;

#[cfg(any(target_os = "macos", target_os = "windows"))]
static PENDING_RELEASES: LazyLock<Mutex<HashMap<u64, Instant>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

#[cfg(any(target_os = "macos", target_os = "windows"))]
const RELEASE_TTL: Duration = Duration::from_secs(60);

#[cfg(any(target_os = "macos", target_os = "windows"))]
static GL_INIT: Once = Once::new();

#[cfg(any(target_os = "macos", target_os = "windows"))]
static PIPELINE_FAILURE_LOGGED: AtomicBool = AtomicBool::new(false);

#[cfg(any(target_os = "macos", target_os = "windows"))]
fn ensure_gl_loaded() {
    GL_INIT.call_once(|| {
        gl_loader::init_gl();
        gl::load_with(|symbol| gl_loader::get_proc_address(symbol).cast());
    });
}

#[cfg(any(target_os = "macos", target_os = "windows"))]
fn pending_release_snapshot() -> Vec<u64> {
    let mut pending = PENDING_RELEASES
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    pending.retain(|_, queued| queued.elapsed() < RELEASE_TTL);
    pending.keys().copied().collect()
}

#[cfg(any(target_os = "macos", target_os = "windows"))]
fn queue_release(instance_id: u64) {
    PENDING_RELEASES
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .insert(instance_id, Instant::now());
}

#[cfg(any(target_os = "macos", target_os = "windows"))]
fn log_pipeline_fallback_once() {
    if !PIPELINE_FAILURE_LOGGED.swap(true, Ordering::Relaxed) {
        error!(
            "FFGL GPU pipeline failed; presenting the unprocessed input instead. \
             Report this if the host shows unprocessed, black, or flickering frames"
        );
    }
}

// ---------------------------------------------------------------------------
// GL state save / restore
// ---------------------------------------------------------------------------

/// Saved GL state that we restore after our raw GL operations. Only used
/// from the cfg-gated `metal_draw` / `dx11_draw` submodules below.
#[cfg(any(target_os = "macos", target_os = "windows"))]
struct SavedGlState {
    pack_buffer: GLint,
    unpack_buffer: GLint,
    draw_framebuffer: GLint,
    read_framebuffer: GLint,
    texture_2d_active: GLint,
    texture_rectangle_active: GLint,
    texture_2d_unit0: GLint,
    texture_rectangle_unit0: GLint,
    sampler_unit0: GLint,
    sampler_objects_supported: bool,
    active_texture: GLint,
    vao: GLint,
    current_program: GLint,
    viewport: [GLint; 4],
    scissor_box: [GLint; 4],
    polygon_mode: [GLint; 2],
    scissor_test: bool,
    blend: bool,
    depth_test: bool,
    stencil_test: bool,
    cull_face: bool,
    framebuffer_srgb: bool,
    rasterizer_discard: bool,
    color_logic_op: bool,
    dither: bool,
    color_mask: [gl::types::GLboolean; 4],
    blend_src_rgb: GLint,
    blend_dst_rgb: GLint,
    blend_src_alpha: GLint,
    blend_dst_alpha: GLint,
    blend_equation_rgb: GLint,
    blend_equation_alpha: GLint,
}

#[cfg(any(target_os = "macos", target_os = "windows"))]
impl SavedGlState {
    unsafe fn save() -> Self {
        let mut s = Self {
            pack_buffer: 0,
            unpack_buffer: 0,
            draw_framebuffer: 0,
            read_framebuffer: 0,
            texture_2d_active: 0,
            texture_rectangle_active: 0,
            texture_2d_unit0: 0,
            texture_rectangle_unit0: 0,
            sampler_unit0: 0,
            sampler_objects_supported: false,
            active_texture: 0,
            vao: 0,
            current_program: 0,
            viewport: [0; 4],
            scissor_box: [0; 4],
            polygon_mode: [gl::FILL as GLint; 2],
            scissor_test: false,
            blend: false,
            depth_test: false,
            stencil_test: false,
            cull_face: false,
            framebuffer_srgb: false,
            rasterizer_discard: false,
            color_logic_op: false,
            dither: false,
            color_mask: [gl::TRUE; 4],
            blend_src_rgb: gl::ONE as GLint,
            blend_dst_rgb: gl::ZERO as GLint,
            blend_src_alpha: gl::ONE as GLint,
            blend_dst_alpha: gl::ZERO as GLint,
            blend_equation_rgb: gl::FUNC_ADD as GLint,
            blend_equation_alpha: gl::FUNC_ADD as GLint,
        };
        gl::GetIntegerv(gl::PIXEL_PACK_BUFFER_BINDING, &mut s.pack_buffer);
        gl::GetIntegerv(gl::PIXEL_UNPACK_BUFFER_BINDING, &mut s.unpack_buffer);
        gl::GetIntegerv(gl::DRAW_FRAMEBUFFER_BINDING, &mut s.draw_framebuffer);
        gl::GetIntegerv(gl::READ_FRAMEBUFFER_BINDING, &mut s.read_framebuffer);
        gl::GetIntegerv(gl::ACTIVE_TEXTURE, &mut s.active_texture);
        gl::GetIntegerv(gl::TEXTURE_BINDING_2D, &mut s.texture_2d_active);
        gl::GetIntegerv(
            gl::TEXTURE_BINDING_RECTANGLE,
            &mut s.texture_rectangle_active,
        );
        if s.active_texture != gl::TEXTURE0 as GLint {
            gl::ActiveTexture(gl::TEXTURE0);
            gl::GetIntegerv(gl::TEXTURE_BINDING_2D, &mut s.texture_2d_unit0);
            gl::GetIntegerv(
                gl::TEXTURE_BINDING_RECTANGLE,
                &mut s.texture_rectangle_unit0,
            );
            gl::ActiveTexture(s.active_texture as GLenum);
        } else {
            s.texture_2d_unit0 = s.texture_2d_active;
            s.texture_rectangle_unit0 = s.texture_rectangle_active;
        }
        s.sampler_objects_supported =
            gl::GetIntegeri_v::is_loaded() && gl::BindSampler::is_loaded();
        if s.sampler_objects_supported {
            gl::GetIntegeri_v(gl::SAMPLER_BINDING, 0, &mut s.sampler_unit0);
        }
        gl::GetIntegerv(gl::VERTEX_ARRAY_BINDING, &mut s.vao);
        gl::GetIntegerv(gl::CURRENT_PROGRAM, &mut s.current_program);
        gl::GetIntegerv(gl::VIEWPORT, s.viewport.as_mut_ptr());
        gl::GetIntegerv(gl::SCISSOR_BOX, s.scissor_box.as_mut_ptr());
        gl::GetIntegerv(gl::POLYGON_MODE, s.polygon_mode.as_mut_ptr());
        s.scissor_test = gl::IsEnabled(gl::SCISSOR_TEST) != 0;
        s.blend = gl::IsEnabled(gl::BLEND) != 0;
        s.depth_test = gl::IsEnabled(gl::DEPTH_TEST) != 0;
        s.stencil_test = gl::IsEnabled(gl::STENCIL_TEST) != 0;
        s.cull_face = gl::IsEnabled(gl::CULL_FACE) != 0;
        s.framebuffer_srgb = gl::IsEnabled(gl::FRAMEBUFFER_SRGB) != 0;
        s.rasterizer_discard = gl::IsEnabled(gl::RASTERIZER_DISCARD) != 0;
        s.color_logic_op = gl::IsEnabled(gl::COLOR_LOGIC_OP) != 0;
        s.dither = gl::IsEnabled(gl::DITHER) != 0;
        gl::GetBooleanv(gl::COLOR_WRITEMASK, s.color_mask.as_mut_ptr());
        gl::GetIntegerv(gl::BLEND_SRC_RGB, &mut s.blend_src_rgb);
        gl::GetIntegerv(gl::BLEND_DST_RGB, &mut s.blend_dst_rgb);
        gl::GetIntegerv(gl::BLEND_SRC_ALPHA, &mut s.blend_src_alpha);
        gl::GetIntegerv(gl::BLEND_DST_ALPHA, &mut s.blend_dst_alpha);
        gl::GetIntegerv(gl::BLEND_EQUATION_RGB, &mut s.blend_equation_rgb);
        gl::GetIntegerv(gl::BLEND_EQUATION_ALPHA, &mut s.blend_equation_alpha);

        s.apply_plugin_state();
        s
    }

    unsafe fn with_host_scissor<R>(&self, operation: impl FnOnce() -> R) -> R {
        gl::Scissor(
            self.scissor_box[0],
            self.scissor_box[1],
            self.scissor_box[2],
            self.scissor_box[3],
        );
        set_enabled(gl::SCISSOR_TEST, self.scissor_test);
        set_enabled(gl::FRAMEBUFFER_SRGB, self.framebuffer_srgb);
        let result = operation();
        gl::Disable(gl::SCISSOR_TEST);
        gl::Disable(gl::FRAMEBUFFER_SRGB);
        result
    }

    unsafe fn apply_plugin_state(&self) {
        gl::BindBuffer(gl::PIXEL_PACK_BUFFER, 0);
        gl::BindBuffer(gl::PIXEL_UNPACK_BUFFER, 0);
        gl::ActiveTexture(gl::TEXTURE0);
        gl::BindTexture(gl::TEXTURE_2D, 0);
        gl::BindTexture(gl::TEXTURE_RECTANGLE, 0);
        if self.sampler_objects_supported {
            gl::BindSampler(0, 0);
        }
        gl::Disable(gl::SCISSOR_TEST);
        gl::Disable(gl::BLEND);
        gl::Disable(gl::DEPTH_TEST);
        gl::Disable(gl::STENCIL_TEST);
        gl::Disable(gl::CULL_FACE);
        gl::Disable(gl::FRAMEBUFFER_SRGB);
        gl::Disable(gl::RASTERIZER_DISCARD);
        gl::Disable(gl::COLOR_LOGIC_OP);
        gl::Disable(gl::DITHER);
        gl::ColorMask(gl::TRUE, gl::TRUE, gl::TRUE, gl::TRUE);
        gl::PolygonMode(gl::FRONT_AND_BACK, gl::FILL);
    }

    unsafe fn restore(&self) {
        gl::BindBuffer(gl::PIXEL_PACK_BUFFER, self.pack_buffer as GLuint);
        gl::BindBuffer(gl::PIXEL_UNPACK_BUFFER, self.unpack_buffer as GLuint);
        gl::BindFramebuffer(gl::DRAW_FRAMEBUFFER, self.draw_framebuffer as GLuint);
        gl::BindFramebuffer(gl::READ_FRAMEBUFFER, self.read_framebuffer as GLuint);

        gl::ActiveTexture(gl::TEXTURE0);
        gl::BindTexture(gl::TEXTURE_2D, self.texture_2d_unit0 as GLuint);
        gl::BindTexture(
            gl::TEXTURE_RECTANGLE,
            self.texture_rectangle_unit0 as GLuint,
        );
        if self.sampler_objects_supported {
            gl::BindSampler(0, self.sampler_unit0 as GLuint);
        }
        gl::ActiveTexture(self.active_texture as GLenum);
        gl::BindTexture(gl::TEXTURE_2D, self.texture_2d_active as GLuint);
        gl::BindTexture(
            gl::TEXTURE_RECTANGLE,
            self.texture_rectangle_active as GLuint,
        );

        gl::UseProgram(self.current_program as GLuint);
        gl::BindVertexArray(self.vao as GLuint);
        gl::Viewport(
            self.viewport[0],
            self.viewport[1],
            self.viewport[2],
            self.viewport[3],
        );
        gl::Scissor(
            self.scissor_box[0],
            self.scissor_box[1],
            self.scissor_box[2],
            self.scissor_box[3],
        );
        gl::PolygonMode(gl::FRONT, self.polygon_mode[0] as GLenum);
        gl::PolygonMode(gl::BACK, self.polygon_mode[1] as GLenum);
        gl::ColorMask(
            self.color_mask[0],
            self.color_mask[1],
            self.color_mask[2],
            self.color_mask[3],
        );
        gl::BlendFuncSeparate(
            self.blend_src_rgb as GLenum,
            self.blend_dst_rgb as GLenum,
            self.blend_src_alpha as GLenum,
            self.blend_dst_alpha as GLenum,
        );
        gl::BlendEquationSeparate(
            self.blend_equation_rgb as GLenum,
            self.blend_equation_alpha as GLenum,
        );
        set_enabled(gl::SCISSOR_TEST, self.scissor_test);
        set_enabled(gl::BLEND, self.blend);
        set_enabled(gl::DEPTH_TEST, self.depth_test);
        set_enabled(gl::STENCIL_TEST, self.stencil_test);
        set_enabled(gl::CULL_FACE, self.cull_face);
        set_enabled(gl::FRAMEBUFFER_SRGB, self.framebuffer_srgb);
        set_enabled(gl::RASTERIZER_DISCARD, self.rasterizer_discard);
        set_enabled(gl::COLOR_LOGIC_OP, self.color_logic_op);
        set_enabled(gl::DITHER, self.dither);
    }
}

#[cfg(any(target_os = "macos", target_os = "windows"))]
unsafe fn set_enabled(capability: GLenum, enabled: bool) {
    if enabled {
        gl::Enable(capability);
    } else {
        gl::Disable(capability);
    }
}

// ---------------------------------------------------------------------------
// Shared GL helpers
// ---------------------------------------------------------------------------

fn clear_gl_errors() {
    unsafe { while gl::GetError() != gl::NO_ERROR {} }
}

fn is_context_current() -> bool {
    unsafe { !gl::GetString(gl::VERSION).is_null() }
}

fn resolve_output_viewport(live: [i32; 4], cached: [i32; 4]) -> [i32; 4] {
    if live[2] > 0 && live[3] > 0 {
        live
    } else {
        cached
    }
}

fn content_uv_scale(
    width: u32,
    height: u32,
    hardware_width: u32,
    hardware_height: u32,
) -> (f32, f32) {
    (
        if hardware_width == 0 {
            1.0
        } else {
            width as f32 / hardware_width as f32
        },
        if hardware_height == 0 {
            1.0
        } else {
            height as f32 / hardware_height as f32
        },
    )
}

#[cfg(any(target_os = "macos", target_os = "windows"))]
fn passthrough(data: &FFGLData, frame_data: GLInput<'_>) {
    use gpu_interop::shader_blit::ShaderBlit;

    thread_local! {
        static PASSTHROUGH_SHADER: RefCell<Option<ShaderBlit>> = const { RefCell::new(None) };
    }

    let Some(input) = frame_data.textures.first() else {
        return;
    };
    let (width, height) = data.get_dimensions();
    let uv_scale = content_uv_scale(
        input.Width,
        input.Height,
        input.HardwareWidth,
        input.HardwareHeight,
    );

    unsafe {
        let saved = SavedGlState::save();
        let output_fbo = saved.draw_framebuffer.max(0) as GLuint;
        let viewport = resolve_output_viewport(
            saved.viewport,
            [
                data.viewport.x as i32,
                data.viewport.y as i32,
                width as i32,
                height as i32,
            ],
        );

        let mut read_fbo = 0;
        gl::GenFramebuffers(1, &mut read_fbo);
        gl::BindFramebuffer(gl::READ_FRAMEBUFFER, read_fbo);

        let mut attachable_target = None;
        for target in [gl::TEXTURE_2D, gl::TEXTURE_RECTANGLE] {
            gl::FramebufferTexture2D(
                gl::READ_FRAMEBUFFER,
                gl::COLOR_ATTACHMENT0,
                target,
                input.Handle,
                0,
            );
            if gl::CheckFramebufferStatus(gl::READ_FRAMEBUFFER) == gl::FRAMEBUFFER_COMPLETE {
                attachable_target = Some(target);
                break;
            }
        }

        if attachable_target.is_some() {
            gl::ReadBuffer(gl::COLOR_ATTACHMENT0);
            gl::BindFramebuffer(gl::DRAW_FRAMEBUFFER, output_fbo);
            saved.with_host_scissor(|| {
                gl::BlitFramebuffer(
                    0,
                    0,
                    input.Width as GLint,
                    input.Height as GLint,
                    viewport[0],
                    viewport[1],
                    viewport[0] + viewport[2],
                    viewport[1] + viewport[3],
                    gl::COLOR_BUFFER_BIT,
                    gl::NEAREST,
                );
            });
        } else {
            let source_target =
                [gl::TEXTURE_2D, gl::TEXTURE_RECTANGLE]
                    .into_iter()
                    .find(|target| {
                        clear_gl_errors();
                        gl::BindTexture(*target, input.Handle);
                        gl::GetError() == gl::NO_ERROR
                    });

            if let Some(source_target) = source_target {
                PASSTHROUGH_SHADER.with(|cell| {
                    let mut shader = cell.borrow_mut();
                    if shader
                        .as_ref()
                        .is_some_and(|candidate| !candidate.is_valid())
                    {
                        *shader = None;
                    }
                    if shader.is_none() {
                        *shader = ShaderBlit::new();
                    }
                    if let Some(shader) = shader.as_ref() {
                        gl::BindFramebuffer(gl::DRAW_FRAMEBUFFER, output_fbo);
                        saved.with_host_scissor(|| match source_target {
                            gl::TEXTURE_2D => shader.present_2d_scaled(
                                input.Handle,
                                viewport[0],
                                viewport[1],
                                viewport[2] as u32,
                                viewport[3] as u32,
                                true,
                                uv_scale,
                            ),
                            _ => shader.present_rect_scaled(
                                input.Handle,
                                input.HardwareWidth.max(input.Width),
                                input.HardwareHeight.max(input.Height),
                                viewport[0],
                                viewport[1],
                                viewport[2] as u32,
                                viewport[3] as u32,
                                true,
                                uv_scale,
                            ),
                        });
                    }
                });
            }
        }

        gl::DeleteFramebuffers(1, &read_fbo);
        clear_gl_errors();
        saved.restore();
    }
}

#[cfg(not(any(target_os = "macos", target_os = "windows")))]
fn passthrough(_data: &FFGLData, _frame_data: GLInput<'_>) {}

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
        static CACHED_BRIDGES: RefCell<HashMap<(u64, usize), GlMetalBridge>> = RefCell::new(HashMap::new());
        static LAST_RESOURCE_KEY: RefCell<Option<(u64, usize)>> = const { RefCell::new(None) };
        static GPU_INITIALIZED: RefCell<HashSet<u64>> = RefCell::new(HashSet::new());
    }

    fn release_active_resources() {
        BRIDGE.with(|cell| {
            if let Some(bridge) = cell.borrow_mut().as_mut() {
                bridge.cleanup();
            }
            *cell.borrow_mut() = None;
        });
    }

    pub fn ensure_instance_resources(instance_id: u64) {
        for released_id in pending_release_snapshot() {
            release_instance_resources(released_id);
        }
        let resource_key = (instance_id, GlMetalBridge::current_context_key());
        LAST_RESOURCE_KEY.with(|cell| {
            let mut key = cell.borrow_mut();
            if *key != Some(resource_key) {
                if let Some(previous_key) = *key {
                    if let Some(bridge) = BRIDGE.with(|bridge| bridge.borrow_mut().take()) {
                        CACHED_BRIDGES.with(|cache| {
                            cache.borrow_mut().insert(previous_key, bridge);
                        });
                    }
                }
                let bridge = CACHED_BRIDGES.with(|cache| cache.borrow_mut().remove(&resource_key));
                BRIDGE.with(|slot| *slot.borrow_mut() = bridge);
                *key = Some(resource_key);
            }
        });
    }

    pub fn release_instance_resources(instance_id: u64) {
        let resource_key = (instance_id, GlMetalBridge::current_context_key());
        let is_active = LAST_RESOURCE_KEY.with(|last| *last.borrow() == Some(resource_key));
        if is_active {
            release_active_resources();
            LAST_RESOURCE_KEY.with(|last| *last.borrow_mut() = None);
        }
        CACHED_BRIDGES.with(|cache| {
            cache.borrow_mut().remove(&resource_key);
        });
        GPU_INITIALIZED.with(|ids| {
            ids.borrow_mut().remove(&instance_id);
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
            release_active_resources();
            LAST_RESOURCE_KEY.with(|last| *last.borrow_mut() = None);
        }
        true
    }

    #[allow(clippy::too_many_arguments)]
    pub fn draw<P: GpuPlugin>(
        plugin: &mut P,
        instance_id: u64,
        data: &FFGLData,
        frame_data: GLInput<'_>,
        frame_counter: u64,
        internal_resolution: f32,
        filter_quality: f32,
        metallib_bytes: &[u8],
    ) {
        ensure_instance_resources(instance_id);
        if !validate_gl_state() {
            passthrough(data, frame_data);
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
            passthrough(data, frame_data);
            return;
        }

        // Get host texture. The actual output target is captured from live GL
        // state below; hosts may pass an ABI FBO that differs from the FBO
        // bound for this render callback.
        let tex_id = match frame_data.textures.first() {
            Some(t) => t.Handle,
            None => {
                passthrough(data, frame_data);
                return;
            }
        };
        let input_uv_scale = frame_data.textures.first().map_or((1.0, 1.0), |texture| {
            content_uv_scale(
                texture.Width,
                texture.Height,
                texture.HardwareWidth,
                texture.HardwareHeight,
            )
        });

        let saved_state = unsafe { SavedGlState::save() };
        let output_fbo = saved_state.draw_framebuffer.max(0) as u32;
        let output_viewport = resolve_output_viewport(
            saved_state.viewport,
            [
                data.viewport.x as i32,
                data.viewport.y as i32,
                width as i32,
                height as i32,
            ],
        );

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
                                as *const objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>
                                as *mut objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>;
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
                        if !initialized.contains(&instance_id) {
                            match plugin.gpu_init(ctx) {
                                Ok(()) => {
                                    initialized.insert(instance_id);
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

                    if !bridge.wait_for_previous() {
                        return false;
                    }

                    let mut output_blitted = false;

                    if has_prev {
                        bridge.swap();
                        output_blitted = unsafe {
                            saved_state.with_host_scissor(|| {
                                bridge.blit_back_output_to_target_scaled(
                                    output_fbo,
                                    proc_width,
                                    proc_height,
                                    output_viewport[0],
                                    output_viewport[1],
                                    output_viewport[2] as u32,
                                    output_viewport[3] as u32,
                                    use_bilinear,
                                )
                            })
                        };
                    }

                    if !bridge.blit_input_from_host_scaled(
                        tex_id,
                        width,
                        height,
                        proc_width,
                        proc_height,
                        use_bilinear,
                        input_uv_scale,
                    ) {
                        return output_blitted;
                    }

                    // Extract texture references via raw pointers to avoid
                    // conflicting borrows (shared refs to textures + mutable
                    // ref to bridge in DrawInput).
                    let input_ptr = match bridge.input_metal_texture() {
                        Some(t) => t as *const _,
                        None => return false,
                    };
                    let output_ptr = match bridge.output_metal_texture() {
                        Some(t) => t as *const _,
                        None => return false,
                    };

                    // SAFETY: The texture pointers point into bridge's internal
                    // IOSurface-backed texture pairs. They remain valid for the
                    // duration of gpu_draw because the bridge is held by this
                    // scope and no bridge methods that invalidate textures are
                    // called until after gpu_draw returns.
                    let mut draw_input = DrawInput {
                        input: unsafe { &*input_ptr },
                        output: unsafe { &*output_ptr },
                        width: proc_width,
                        height: proc_height,
                        bridge,
                    };

                    plugin.gpu_draw(ctx, &mut draw_input, data, frame_counter);

                    // Reclaim bridge from DrawInput for post-draw operations.
                    let bridge = draw_input.bridge;

                    if !bridge.has_pending_work() {
                        error!("GpuPlugin::gpu_draw did not submit Metal work");
                        return output_blitted;
                    }
                    bridge.mark_dispatch(frame_counter);

                    if !has_prev {
                        if !bridge.wait_for_pending() {
                            return false;
                        }
                        output_blitted = unsafe {
                            saved_state.with_host_scissor(|| {
                                bridge.blit_output_to_target_scaled(
                                    output_fbo,
                                    proc_width,
                                    proc_height,
                                    output_viewport[0],
                                    output_viewport[1],
                                    output_viewport[2] as u32,
                                    output_viewport[3] as u32,
                                    use_bilinear,
                                )
                            })
                        };
                    }

                    output_blitted
                })
            })
        });

        unsafe {
            saved_state.restore();
        }

        if !success {
            log_pipeline_fallback_once();
            passthrough(data, frame_data);
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
        static CACHED_BRIDGES: RefCell<HashMap<(u64, usize), GlDx11Bridge>> = RefCell::new(HashMap::new());
        static LAST_RESOURCE_KEY: RefCell<Option<(u64, usize)>> = const { RefCell::new(None) };
        static GPU_INITIALIZED: RefCell<HashSet<u64>> = RefCell::new(HashSet::new());
    }

    fn release_active_resources() {
        BRIDGE.with(|cell| {
            if let Some(bridge) = cell.borrow_mut().as_mut() {
                bridge.cleanup();
            }
            *cell.borrow_mut() = None;
        });
    }

    pub fn ensure_instance_resources(instance_id: u64) {
        for released_id in pending_release_snapshot() {
            release_instance_resources(released_id);
        }
        let resource_key = (instance_id, GlDx11Bridge::current_context_key());
        LAST_RESOURCE_KEY.with(|cell| {
            let mut key = cell.borrow_mut();
            if *key != Some(resource_key) {
                if let Some(previous_key) = *key {
                    if let Some(bridge) = BRIDGE.with(|bridge| bridge.borrow_mut().take()) {
                        CACHED_BRIDGES.with(|cache| {
                            cache.borrow_mut().insert(previous_key, bridge);
                        });
                    }
                }
                let bridge = CACHED_BRIDGES.with(|cache| cache.borrow_mut().remove(&resource_key));
                BRIDGE.with(|slot| *slot.borrow_mut() = bridge);
                *key = Some(resource_key);
            }
        });
    }

    pub fn release_instance_resources(instance_id: u64) {
        let resource_key = (instance_id, GlDx11Bridge::current_context_key());
        let is_active = LAST_RESOURCE_KEY.with(|last| *last.borrow() == Some(resource_key));
        if is_active {
            release_active_resources();
            LAST_RESOURCE_KEY.with(|last| *last.borrow_mut() = None);
        }
        CACHED_BRIDGES.with(|cache| {
            cache.borrow_mut().remove(&resource_key);
        });
        GPU_INITIALIZED.with(|ids| {
            ids.borrow_mut().remove(&instance_id);
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
            release_active_resources();
            LAST_RESOURCE_KEY.with(|last| *last.borrow_mut() = None);
        }
        true
    }

    #[allow(clippy::too_many_arguments)]
    pub fn draw<P: GpuPlugin>(
        plugin: &mut P,
        instance_id: u64,
        data: &FFGLData,
        frame_data: GLInput<'_>,
        frame_counter: u64,
        internal_resolution: f32,
        filter_quality: f32,
        _metallib_bytes: &[u8],
    ) {
        ensure_instance_resources(instance_id);
        if !validate_gl_state() {
            passthrough(data, frame_data);
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
            passthrough(data, frame_data);
            return;
        }

        // Ensure GL-D3D11 interop bridge is initialized
        let bridge_available = GPU_CTX.with(|ctx_cell| {
            let ctx = ctx_cell.borrow();
            let ctx = ctx.as_ref().unwrap();
            BRIDGE.with(|bridge_cell| {
                let mut bridge = bridge_cell.borrow_mut();
                if bridge.is_none() {
                    *bridge = GlDx11Bridge::new(ctx.device.device(), ctx.device.context());
                }
                bridge.is_some()
            })
        });

        if !bridge_available {
            passthrough(data, frame_data);
            return;
        }

        let tex_id = match frame_data.textures.first() {
            Some(t) => t.Handle,
            None => {
                passthrough(data, frame_data);
                return;
            }
        };
        let input_uv_scale = frame_data.textures.first().map_or((1.0, 1.0), |texture| {
            content_uv_scale(
                texture.Width,
                texture.Height,
                texture.HardwareWidth,
                texture.HardwareHeight,
            )
        });

        let saved_state = unsafe { SavedGlState::save() };
        let output_fbo = saved_state.draw_framebuffer.max(0) as u32;
        let output_viewport = resolve_output_viewport(
            saved_state.viewport,
            [
                data.viewport.x as i32,
                data.viewport.y as i32,
                width as i32,
                height as i32,
            ],
        );

        let success = GPU_CTX.with(|ctx_cell| {
            let ctx_ref = ctx_cell.borrow();
            let ctx = ctx_ref.as_ref().unwrap();

            BRIDGE.with(|bridge_cell| {
                // Call gpu_init on first use
                let init_ok = GPU_INITIALIZED.with(|cell| {
                    let mut initialized = cell.borrow_mut();
                    if !initialized.contains(&instance_id) {
                        match plugin.gpu_init(ctx) {
                            Ok(()) => {
                                initialized.insert(instance_id);
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

                if !bridge.wait_for_previous() {
                    return false;
                }

                let mut output_blitted = false;

                if has_prev {
                    bridge.swap();
                    output_blitted = unsafe {
                        saved_state.with_host_scissor(|| {
                            bridge.blit_back_output_to_target_scaled(
                                output_fbo,
                                proc_width,
                                proc_height,
                                output_viewport[0],
                                output_viewport[1],
                                output_viewport[2] as u32,
                                output_viewport[3] as u32,
                                use_bilinear,
                            )
                        })
                    };
                }

                if !bridge.blit_input_from_host_scaled(
                    tex_id,
                    width,
                    height,
                    proc_width,
                    proc_height,
                    use_bilinear,
                    input_uv_scale,
                ) {
                    return output_blitted;
                }

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
                    if !bridge.wait_for_pending() {
                        return false;
                    }
                    output_blitted = unsafe {
                        saved_state.with_host_scissor(|| {
                            bridge.blit_output_to_target_scaled(
                                output_fbo,
                                proc_width,
                                proc_height,
                                output_viewport[0],
                                output_viewport[1],
                                output_viewport[2] as u32,
                                output_viewport[3] as u32,
                                use_bilinear,
                            )
                        })
                    };
                }

                output_blitted
            })
        });

        unsafe {
            saved_state.restore();
        }

        if !success {
            log_pipeline_fallback_once();
            passthrough(data, frame_data);
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

/// Release cached bridge resources for one plugin instance.
///
/// Call this from the instance's `Drop` implementation. When destruction runs
/// on the render thread, resources are reclaimed immediately; context-identity
/// guards prevent foreign-context cleanup from deleting unrelated host GL
/// objects.
pub fn release_instance_gl_resources(instance_id: u64) {
    #[cfg(target_os = "macos")]
    metal_draw::release_instance_resources(instance_id);

    #[cfg(target_os = "windows")]
    dx11_draw::release_instance_resources(instance_id);

    #[cfg(any(target_os = "macos", target_os = "windows"))]
    queue_release(instance_id);

    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    let _ = instance_id;
}

/// Validate GL state before drawing. Returns `false` if the GL context is
/// invalid and drawing should be skipped.
pub fn validate_gl_state_before_draw() -> bool {
    #[cfg(any(target_os = "macos", target_os = "windows"))]
    ensure_gl_loaded();

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
/// * `data` - Host-provided FFGL data (viewport dimensions, timing, etc).
/// * `frame_data` - Host input textures and FBO.
/// * `frame_counter` - Monotonically increasing frame counter.
/// * `internal_resolution` - Resolution scale factor `[0.125, 1.0]`.
/// * `filter_quality` - Filter quality `[0.0, 1.0]`. Values >= 0.5 use
///   bilinear filtering.
/// * `metallib_bytes` - Compiled Metal shader library bytes (from
///   [`include_metallib!`]). Ignored on Windows.
#[allow(clippy::too_many_arguments)]
pub fn draw_gpu_effect<P: GpuPlugin>(
    plugin: &mut P,
    instance_id: u64,
    data: &FFGLData,
    frame_data: GLInput<'_>,
    frame_counter: u64,
    internal_resolution: f32,
    filter_quality: f32,
    metallib_bytes: &[u8],
) {
    #[cfg(any(target_os = "macos", target_os = "windows"))]
    ensure_gl_loaded();

    #[cfg(target_os = "macos")]
    metal_draw::draw(
        plugin,
        instance_id,
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
        passthrough(data, frame_data);
    }
}

#[cfg(test)]
mod tests {
    use super::{content_uv_scale, resolve_output_viewport};

    #[test]
    fn live_host_viewport_wins_over_cached_abi_viewport() {
        assert_eq!(
            resolve_output_viewport([19, 13, 640, 360], [0, 0, 1920, 1080]),
            [19, 13, 640, 360]
        );
        assert_eq!(
            resolve_output_viewport([0, 0, 0, 0], [7, 5, 320, 240]),
            [7, 5, 320, 240]
        );
    }

    #[test]
    fn padded_texture_uvs_exclude_hardware_padding() {
        assert_eq!(
            content_uv_scale(1919, 1079, 1920, 1080),
            (1919.0 / 1920.0, 1079.0 / 1080.0)
        );
        assert_eq!(content_uv_scale(640, 480, 0, 0), (1.0, 1.0));
    }

    #[cfg(any(target_os = "macos", target_os = "windows"))]
    #[test]
    fn release_requests_are_visible_across_threads() {
        use super::{pending_release_snapshot, queue_release};

        let instance_id = u64::MAX - 17;
        std::thread::spawn(move || queue_release(instance_id))
            .join()
            .unwrap();
        assert!(pending_release_snapshot().contains(&instance_id));
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[allow(deprecated)]
    fn real_gl_passthrough_restores_host_state_and_honors_scissor() {
        use super::{ensure_gl_loaded, passthrough};
        use ffgl_core::ffi::{FFGLTextureStruct, FFGLViewportStruct};
        use ffgl_core::inputs::{FFGLData, GLInput};
        use gl::types::{GLint, GLuint};
        use objc2_open_gl::{
            CGLChoosePixelFormat, CGLContextObj, CGLCreateContext, CGLDestroyContext,
            CGLDestroyPixelFormat, CGLError, CGLOpenGLProfile, CGLPixelFormatAttribute,
            CGLSetCurrentContext,
        };
        use std::ptr::{self, NonNull};

        struct ContextGuard(CGLContextObj);
        impl Drop for ContextGuard {
            fn drop(&mut self) {
                unsafe {
                    CGLSetCurrentContext(ptr::null_mut());
                    CGLDestroyContext(self.0);
                }
            }
        }

        let attributes = [
            CGLPixelFormatAttribute::CGLPFAOpenGLProfile,
            CGLPixelFormatAttribute(CGLOpenGLProfile::CGLOGLPVersion_GL4_Core.0),
            CGLPixelFormatAttribute::CGLPFAAllowOfflineRenderers,
            CGLPixelFormatAttribute(0),
        ];
        let mut pixel_format = ptr::null_mut();
        let mut pixel_format_count = 0;
        let choose_error = unsafe {
            CGLChoosePixelFormat(
                NonNull::new(attributes.as_ptr() as *mut _).unwrap(),
                NonNull::new(&mut pixel_format).unwrap(),
                NonNull::new(&mut pixel_format_count).unwrap(),
            )
        };
        assert_eq!(choose_error, CGLError::NoError);

        let mut context = ptr::null_mut();
        let create_error = unsafe {
            CGLCreateContext(
                pixel_format,
                ptr::null_mut(),
                NonNull::new(&mut context).unwrap(),
            )
        };
        unsafe { CGLDestroyPixelFormat(pixel_format) };
        assert_eq!(create_error, CGLError::NoError);
        unsafe { CGLSetCurrentContext(context) };
        let _context = ContextGuard(context);
        ensure_gl_loaded();

        unsafe {
            let source_pixels = [[255u8, 0, 0, 255]; 16];
            let mut source_texture: GLuint = 0;
            gl::GenTextures(1, &mut source_texture);
            gl::BindTexture(gl::TEXTURE_2D, source_texture);
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::NEAREST as GLint);
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::NEAREST as GLint);
            gl::TexImage2D(
                gl::TEXTURE_2D,
                0,
                gl::RGBA8 as GLint,
                4,
                4,
                0,
                gl::RGBA,
                gl::UNSIGNED_BYTE,
                source_pixels.as_ptr().cast(),
            );

            let mut output_texture: GLuint = 0;
            let mut output_fbo: GLuint = 0;
            gl::GenTextures(1, &mut output_texture);
            gl::BindTexture(gl::TEXTURE_2D, output_texture);
            gl::TexImage2D(
                gl::TEXTURE_2D,
                0,
                gl::RGBA8 as GLint,
                8,
                8,
                0,
                gl::RGBA,
                gl::UNSIGNED_BYTE,
                ptr::null(),
            );
            gl::GenFramebuffers(1, &mut output_fbo);
            gl::BindFramebuffer(gl::FRAMEBUFFER, output_fbo);
            gl::FramebufferTexture2D(
                gl::FRAMEBUFFER,
                gl::COLOR_ATTACHMENT0,
                gl::TEXTURE_2D,
                output_texture,
                0,
            );
            assert_eq!(
                gl::CheckFramebufferStatus(gl::FRAMEBUFFER),
                gl::FRAMEBUFFER_COMPLETE
            );
            gl::Disable(gl::SCISSOR_TEST);
            gl::ColorMask(gl::TRUE, gl::TRUE, gl::TRUE, gl::TRUE);
            gl::ClearColor(1.0, 0.0, 1.0, 1.0);
            gl::Clear(gl::COLOR_BUFFER_BIT);

            gl::BindFramebuffer(gl::DRAW_FRAMEBUFFER, output_fbo);
            gl::BindFramebuffer(gl::READ_FRAMEBUFFER, output_fbo);
            gl::Viewport(2, 2, 4, 4);
            gl::Scissor(3, 3, 2, 2);
            gl::Enable(gl::SCISSOR_TEST);
            gl::Enable(gl::BLEND);
            gl::ColorMask(gl::TRUE, gl::FALSE, gl::TRUE, gl::FALSE);

            let data = FFGLData::new(&FFGLViewportStruct {
                x: 0,
                y: 0,
                width: 4,
                height: 4,
            });
            let texture = FFGLTextureStruct {
                Width: 4,
                Height: 4,
                HardwareWidth: 4,
                HardwareHeight: 4,
                Handle: source_texture,
            };
            passthrough(
                &data,
                GLInput {
                    textures: &[texture],
                    host: output_fbo,
                },
            );

            let mut draw_fbo = 0;
            let mut read_fbo = 0;
            let mut viewport = [0; 4];
            let mut scissor = [0; 4];
            let mut color_mask = [gl::FALSE; 4];
            gl::GetIntegerv(gl::DRAW_FRAMEBUFFER_BINDING, &mut draw_fbo);
            gl::GetIntegerv(gl::READ_FRAMEBUFFER_BINDING, &mut read_fbo);
            gl::GetIntegerv(gl::VIEWPORT, viewport.as_mut_ptr());
            gl::GetIntegerv(gl::SCISSOR_BOX, scissor.as_mut_ptr());
            gl::GetBooleanv(gl::COLOR_WRITEMASK, color_mask.as_mut_ptr());
            assert_eq!(draw_fbo as GLuint, output_fbo);
            assert_eq!(read_fbo as GLuint, output_fbo);
            assert_eq!(viewport, [2, 2, 4, 4]);
            assert_eq!(scissor, [3, 3, 2, 2]);
            assert_ne!(gl::IsEnabled(gl::SCISSOR_TEST), 0);
            assert_ne!(gl::IsEnabled(gl::BLEND), 0);
            assert_eq!(color_mask, [gl::TRUE, gl::FALSE, gl::TRUE, gl::FALSE]);

            gl::Disable(gl::SCISSOR_TEST);
            gl::ColorMask(gl::TRUE, gl::TRUE, gl::TRUE, gl::TRUE);
            let mut output_pixels = [[0u8; 4]; 64];
            gl::ReadPixels(
                0,
                0,
                8,
                8,
                gl::RGBA,
                gl::UNSIGNED_BYTE,
                output_pixels.as_mut_ptr().cast(),
            );
            for y in 0..8 {
                for x in 0..8 {
                    let expected = if (3..5).contains(&x) && (3..5).contains(&y) {
                        [255, 0, 0, 255]
                    } else {
                        [255, 0, 255, 255]
                    };
                    assert_eq!(output_pixels[y * 8 + x], expected, "pixel ({x}, {y})");
                }
            }

            gl::DeleteFramebuffers(1, &output_fbo);
            gl::DeleteTextures(1, &output_texture);
            gl::DeleteTextures(1, &source_texture);
        }
    }
}
