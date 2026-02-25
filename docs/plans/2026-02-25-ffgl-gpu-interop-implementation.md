# FFGL GPU Interop Library Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a generic Rust library for FFGL plugins that bridges OpenGL to Metal/DX11 for GPU compute and rendering, ported from ntsc-ffgl-plugin.

**Architecture:** Layered workspace with 4 crates (ffgl-core, ffgl-glium, gpu-interop, ffgl-gpu) plus examples. Each crate builds on the previous. Platform backends selected via `#[cfg(target_os)]`.

**Tech Stack:** Rust, OpenGL (gl/glium), Metal (objc2-metal, IOSurface), DirectX 11 (windows crate, WGL_NV_DX_interop2), bindgen for FFGL C headers.

**Reference Code:** All source to port is in `ntsc-ffgl-plugin/` (unchanged).

---

## Task 1: Initialize Workspace and Directory Structure

**Files:**
- Create: `Cargo.toml` (workspace root)
- Create: `crates/ffgl-core/Cargo.toml`
- Create: `crates/ffgl-core/src/lib.rs` (empty)
- Create: `crates/ffgl-glium/Cargo.toml`
- Create: `crates/ffgl-glium/src/lib.rs` (empty)
- Create: `crates/gpu-interop/Cargo.toml`
- Create: `crates/gpu-interop/src/lib.rs` (empty)
- Create: `crates/ffgl-gpu/Cargo.toml`
- Create: `crates/ffgl-gpu/src/lib.rs` (empty)
- Create: `.gitignore`

**Step 1: Create directory structure**

```bash
mkdir -p crates/{ffgl-core,ffgl-glium,gpu-interop,ffgl-gpu}/src
mkdir -p examples/{passthrough,invert,blur,kitchen-sink}/src
mkdir -p examples/{passthrough,invert,blur,kitchen-sink}/shaders
```

**Step 2: Write workspace Cargo.toml**

```toml
[workspace]
members = [
    "crates/ffgl-core",
    "crates/ffgl-glium",
    "crates/gpu-interop",
    "crates/ffgl-gpu",
]
resolver = "2"

[workspace.package]
edition = "2021"
license = "MIT"

[workspace.dependencies]
# Internal crates
ffgl-core = { path = "crates/ffgl-core" }
ffgl-glium = { path = "crates/ffgl-glium" }
gpu-interop = { path = "crates/gpu-interop" }
ffgl-gpu = { path = "crates/ffgl-gpu" }

# OpenGL
gl = "0.14.0"
glium = "0.36.0"
gl_loader = "0.1.2"

# Utilities
tracing = "0.1"
anyhow = "1"
once_cell = "1"
num = "0.4"
num-derive = "0.4"
num-traits = "0.2"

# macOS Metal
objc2 = "0.6"
objc2-foundation = "0.3"
objc2-metal = { version = "0.3", features = ["MTLDevice", "MTLCommandQueue", "MTLCommandBuffer", "MTLComputeCommandEncoder", "MTLComputePipeline", "MTLLibrary", "MTLTexture", "MTLBuffer", "MTLResource", "MTLRenderPipeline", "MTLRenderCommandEncoder", "MTLRenderPass"] }
objc2-io-surface = "0.3"
objc2-open-gl = "0.3"
objc2-core-foundation = "0.3"
dispatch2 = "0.3"

# Windows DX11
[workspace.dependencies.windows]
version = "0.62"
features = [
    "Win32_Graphics_Direct3D",
    "Win32_Graphics_Direct3D11",
    "Win32_Graphics_Dxgi",
    "Win32_Graphics_Dxgi_Common",
    "Win32_Graphics_Gdi",
    "Win32_Graphics_OpenGL",
]
```

**Step 3: Write individual crate Cargo.toml files**

`crates/ffgl-core/Cargo.toml`:
```toml
[package]
name = "ffgl-core"
version = "0.1.0"
edition.workspace = true

[dependencies]
gl = { workspace = true }
num = { workspace = true }
num-derive = { workspace = true }
num-traits = { workspace = true }
once_cell = { workspace = true }
tracing = { workspace = true }
anyhow = { workspace = true }

[build-dependencies]
bindgen = "0.71"
cfg-if = "1"
```

`crates/ffgl-glium/Cargo.toml`:
```toml
[package]
name = "ffgl-glium"
version = "0.1.0"
edition.workspace = true

[dependencies]
ffgl-core = { workspace = true }
glium = { workspace = true }
gl = { workspace = true }
gl_loader = { workspace = true }
```

`crates/gpu-interop/Cargo.toml`:
```toml
[package]
name = "gpu-interop"
version = "0.1.0"
edition.workspace = true

[dependencies]
gl = { workspace = true }
tracing = { workspace = true }
anyhow = { workspace = true }

[target.'cfg(target_os = "macos")'.dependencies]
objc2 = { workspace = true }
objc2-foundation = { workspace = true }
objc2-metal = { workspace = true }
objc2-io-surface = { workspace = true }
objc2-open-gl = { workspace = true }
objc2-core-foundation = { workspace = true }
dispatch2 = { workspace = true }

[target.'cfg(target_os = "windows")'.dependencies]
windows = { workspace = true }
```

`crates/ffgl-gpu/Cargo.toml`:
```toml
[package]
name = "ffgl-gpu"
version = "0.1.0"
edition.workspace = true

[dependencies]
ffgl-core = { workspace = true }
ffgl-glium = { workspace = true }
gpu-interop = { workspace = true }
gl = { workspace = true }
tracing = { workspace = true }
anyhow = { workspace = true }
```

**Step 4: Write stub lib.rs files and .gitignore**

Each `lib.rs` starts as `//! Crate description` placeholder.

`.gitignore`:
```
/target
Cargo.lock
*.dylib
*.dll
*.so
```

**Step 5: Verify workspace compiles**

Run: `cargo check`
Expected: All 4 crates compile (stub only, no real code yet). ffgl-core build.rs may need attention for bindgen -- if FFGL headers aren't available yet, skip bindgen for now with a cfg flag.

**Step 6: Commit**

```bash
git add -A
git commit -m "feat: initialize workspace with 4 crate stubs"
```

---

## Task 2: Port ffgl-core — FFGL Types and Enums

Port the FFGL type definitions, enums, and conversion types from `ntsc-ffgl-plugin/ffgl-rs/ffgl-core/src/`.

**Files:**
- Create: `crates/ffgl-core/src/conversions.rs`
- Create: `crates/ffgl-core/src/info.rs`
- Create: `crates/ffgl-core/src/inputs.rs`
- Modify: `crates/ffgl-core/src/lib.rs`

**Step 1: Port conversions.rs**

Port from `ntsc-ffgl-plugin/ffgl-rs/ffgl-core/src/conversions.rs`:
- `Op` enum (all 40+ FFGL operation codes)
- `FFGLVal` union (u32/pointer union for C ABI)
- `SuccessVal`, `SupportVal`, `BoolVal` enums
- All `From` impls for FFGLVal
- Hardcode the FFGL constants (FF_SUCCESS, FF_FAIL, etc.) rather than depending on bindgen for now

**Step 2: Port info.rs**

Port from `ntsc-ffgl-plugin/ffgl-rs/ffgl-core/src/info.rs`:
- `PluginInfo` struct (unique_id, name, type, about, description)
- `PluginType` enum (Effect, Source, Mixer)
- `PluginExtendedInfoStruct` and layout

**Step 3: Port inputs.rs**

Port from `ntsc-ffgl-plugin/ffgl-rs/ffgl-core/src/inputs.rs`:
- `FFGLData` struct (viewport, host_time, beat info, dimensions)
- `GLInput` struct (textures, host FBO)
- `FFGLTextureStruct` (handle, width, height)
- `FFGLViewportStruct`

**Step 4: Wire up lib.rs**

```rust
pub mod conversions;
pub mod info;
pub mod inputs;
```

**Step 5: Verify compilation**

Run: `cargo check -p ffgl-core`
Expected: PASS

**Step 6: Commit**

```bash
git add crates/ffgl-core/
git commit -m "feat(ffgl-core): port FFGL types, enums, and conversions"
```

---

## Task 3: Port ffgl-core — Parameter System

**Files:**
- Create: `crates/ffgl-core/src/parameters/mod.rs`
- Create: `crates/ffgl-core/src/parameters/info.rs`
- Create: `crates/ffgl-core/src/parameters/handler.rs`

**Step 1: Port parameters/info.rs**

Port from `ntsc-ffgl-plugin/ffgl-rs/ffgl-core/src/parameters/info.rs`:
- `ParameterTypes` enum (Boolean, Event, Standard, Option, etc.)
- `ParameterUsages` enum
- `ParamInfo` trait (name, display_name, param_type, min, max, default_val, group, elements)
- `SimpleParamInfo` struct with builder pattern

**Step 2: Port parameters/handler.rs**

Port from `ntsc-ffgl-plugin/ffgl-rs/ffgl-core/src/parameters/handler.rs`:
- `ParamInfoHandler` trait
- `ParamValueHandler` trait

**Step 3: Wire up parameters module**

```rust
// parameters/mod.rs
pub mod info;
pub mod handler;
pub use info::*;
pub use handler::*;
```

**Step 4: Verify compilation**

Run: `cargo check -p ffgl-core`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/ffgl-core/
git commit -m "feat(ffgl-core): port parameter system traits and types"
```

---

## Task 4: Port ffgl-core — Handler Traits and Entry Point

**Files:**
- Create: `crates/ffgl-core/src/handler/mod.rs`
- Create: `crates/ffgl-core/src/handler/simplified.rs`
- Create: `crates/ffgl-core/src/entry.rs`
- Create: `crates/ffgl-core/src/plugin_main.rs`
- Create: `crates/ffgl-core/src/log.rs`
- Create: `crates/ffgl-core/src/ffi.rs`

**Step 1: Port handler traits**

Port from `ntsc-ffgl-plugin/ffgl-rs/ffgl-core/src/handler/`:
- `FFGLHandler` trait (init, num_params, param_info, plugin_info, new_instance)
- `FFGLInstance` trait (get_param, set_param, draw)
- `Instance<T>` struct (data + renderer wrapper)
- `SimpleFFGLInstance` trait (simplified API)
- `SimpleFFGLHandler<T>` struct

**Step 2: Port FFI constants**

Create `ffi.rs` with hardcoded FFGL constants instead of bindgen-generated code. This avoids the FFGL SDK header dependency for now:
- FF_SUCCESS, FF_FAIL, FF_TRUE, FF_FALSE
- FF_CAP_* constants (capabilities)
- FF_EFFECT, FF_SOURCE, FF_MIXER
- C struct layouts for FFGLViewportStruct, ProcessOpenGLStruct, FFGLTextureStruct, etc.

**Step 3: Port log.rs**

Port from `ntsc-ffgl-plugin/ffgl-rs/ffgl-core/src/log.rs`:
- `FFGLLogger` type (C function pointer)
- `FFGLWriter` (io::Write impl routing to host logger)
- `init_logger()` function
- `try_init_default_subscriber()` (tracing integration)

**Step 4: Port entry.rs**

Port from `ntsc-ffgl-plugin/ffgl-rs/ffgl-core/src/entry.rs`:
- `default_ffgl_entry()` function — main FFGL operation dispatcher
- Maps Op enum to handler/instance calls
- Handles InstantiateGL, DeinstantiateGL, ProcessOpenGL, parameter queries, etc.
- OnceLock-based static handler initialization

**Step 5: Port plugin_main.rs**

Port from `ntsc-ffgl-plugin/ffgl-rs/ffgl-core/src/plugin_main.rs`:
- `plugin_main!` macro generating `plugMain` and `SetLogCallback` extern C functions
- `handle_plugin_main()` function (panic-catching wrapper)

**Step 6: Update lib.rs with all modules**

```rust
pub mod conversions;
pub mod entry;
pub mod ffi;
pub mod handler;
pub mod info;
pub mod inputs;
pub mod log;
pub mod parameters;
pub mod plugin_main;

pub use handler::*;
pub use info::*;
pub use inputs::*;
pub use parameters::*;
```

**Step 7: Verify compilation**

Run: `cargo check -p ffgl-core`
Expected: PASS

**Step 8: Commit**

```bash
git add crates/ffgl-core/
git commit -m "feat(ffgl-core): port handler traits, entry point, and plugin_main macro"
```

---

## Task 5: Port ffgl-glium — OpenGL Context Wrapper

**Files:**
- Create: `crates/ffgl-glium/src/gl_backend.rs`
- Create: `crates/ffgl-glium/src/texture.rs`
- Create: `crates/ffgl-glium/src/validate_gl.rs`
- Modify: `crates/ffgl-glium/src/lib.rs`

**Step 1: Port gl_backend.rs**

Port from `ntsc-ffgl-plugin/ffgl-rs/ffgl-glium/src/gl_backend.rs`:
- `RawGlBackend` struct (wraps host GL context for glium)
- `Backend` trait impl (swap_buffers, get_proc_address, get_framebuffer_dimensions, is_current, make_current)
- GL_INIT_ONCE (one-time GL function pointer loading)

**Step 2: Port texture.rs and validate_gl.rs**

Port from `ntsc-ffgl-plugin/ffgl-rs/ffgl-glium/src/`:
- Texture creation helpers (from_id, BGRA format handling)
- GL state validation utilities

**Step 3: Port FFGLGlium**

Port from `ntsc-ffgl-plugin/ffgl-rs/ffgl-glium/src/lib.rs`:
- `FFGLGlium` struct (ctx, backend, cached_rb)
- `new()` — Initialize glium context from host OpenGL
- `draw()` — The main draw loop (create renderbuffer, import host textures, call user closure, blit to host FBO)

**Step 4: Verify compilation**

Run: `cargo check -p ffgl-glium`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/ffgl-glium/
git commit -m "feat(ffgl-glium): port OpenGL context wrapper and draw loop"
```

---

## Task 6: Create gpu-interop — Common Bridge Interface

**Files:**
- Modify: `crates/gpu-interop/src/lib.rs`
- Create: `crates/gpu-interop/src/bridge.rs`

**Step 1: Define the GpuBridge trait**

```rust
// bridge.rs
use anyhow::Result;
use gl::types::GLuint;

/// Common interface for GL↔GPU texture bridging.
/// Implementations exist for Metal (macOS) and DX11 (Windows).
pub trait GpuBridge {
    /// Recreate shared textures if dimensions changed.
    fn ensure_dimensions(&mut self, width: u32, height: u32) -> Result<()>;

    /// Copy host OpenGL texture into the bridge's front input texture.
    /// Returns false if dimensions don't match.
    fn blit_input_from_host_scaled(
        &mut self,
        host_texture: GLuint,
        src_w: u32, src_h: u32,
        dst_w: u32, dst_h: u32,
        bilinear: bool,
    ) -> bool;

    /// Copy the back output texture (previous frame) to the host FBO.
    fn blit_back_output_to_target_scaled(
        &mut self,
        host_fbo: GLuint,
        src_w: u32, src_h: u32,
        dst_w: u32, dst_h: u32,
        bilinear: bool,
    );

    /// Copy the front output texture (current frame, sync path) to the host FBO.
    fn blit_output_to_target_scaled(
        &mut self,
        host_fbo: GLuint,
        src_w: u32, src_h: u32,
        dst_w: u32, dst_h: u32,
        bilinear: bool,
    );

    /// Check if a previous frame's result is ready for presentation.
    fn has_result_ready(&self, current_frame: u64) -> bool;

    /// Block until the previous frame's GPU work completes.
    fn wait_for_previous(&mut self);

    /// Swap front/back pairs for double-buffering.
    fn swap(&mut self);

    /// Clean up all GPU resources.
    fn cleanup(&mut self);

    /// Get current dimensions.
    fn dimensions(&self) -> (u32, u32);
}
```

**Step 2: Wire up lib.rs**

```rust
pub mod bridge;
pub use bridge::GpuBridge;

#[cfg(target_os = "macos")]
pub mod metal;

#[cfg(target_os = "windows")]
pub mod dx11;
```

**Step 3: Verify compilation**

Run: `cargo check -p gpu-interop`
Expected: PASS

**Step 4: Commit**

```bash
git add crates/gpu-interop/
git commit -m "feat(gpu-interop): define GpuBridge trait interface"
```

---

## Task 7: Port gpu-interop — Metal Backend (macOS)

**Files:**
- Create: `crates/gpu-interop/src/metal/mod.rs`
- Create: `crates/gpu-interop/src/metal/interop.rs`
- Create: `crates/gpu-interop/src/metal/device.rs`

**Step 1: Port Metal device context**

Port from `ntsc-ffgl-plugin/shared/src/metal/device.rs`, generalized:
- `MetalDevice` struct (device, command_queue) — no shader library (that's ffgl-gpu's job)
- `MetalDevice::new()` — create default device + queue
- Remove NTSC-specific pipeline states

**Step 2: Port Metal interop bridge**

Port from `ntsc-ffgl-plugin/shared/src/metal/interop.rs`:
- `SharedTexture` (IOSurface + GL texture + Metal texture)
- `IoSurfacePair` (input + output)
- `GlMetalBridge` struct (double-buffered pairs, FBOs, pending CB, frame tracking)
- Helper functions: `create_iosurface()`, `create_gl_texture_from_iosurface()`, `create_metal_texture_from_iosurface()`
- All blit methods, swap, cleanup
- Implement `GpuBridge` trait for `GlMetalBridge`
- Add public accessors for Metal textures: `input_metal_texture()`, `output_metal_texture()`

**Step 3: Wire up metal module**

```rust
// metal/mod.rs
pub mod device;
pub mod interop;
pub use device::MetalDevice;
pub use interop::GlMetalBridge;
```

**Step 4: Verify compilation (macOS only)**

Run: `cargo check -p gpu-interop`
Expected: PASS on macOS

**Step 5: Commit**

```bash
git add crates/gpu-interop/
git commit -m "feat(gpu-interop): port Metal GL↔IOSurface bridge"
```

---

## Task 8: Port gpu-interop — DX11 Backend (Windows)

**Files:**
- Create: `crates/gpu-interop/src/dx11/mod.rs`
- Create: `crates/gpu-interop/src/dx11/interop.rs`
- Create: `crates/gpu-interop/src/dx11/device.rs`

**Step 1: Port DX11 device context**

Port from `ntsc-ffgl-plugin/shared/src/dx11/device.rs`, generalized:
- `Dx11Device` struct (device, context) — no NTSC-specific constant buffers or pipelines
- `Dx11Device::new()` — create D3D11 device (HARDWARE → WARP fallback)
- Keep GPU query creation for synchronization

**Step 2: Port DX11 interop bridge**

Port from `ntsc-ffgl-plugin/shared/src/dx11/interop.rs`:
- `WglInteropFunctions` (dynamic WGL function loading)
- `SharedTexture` (D3D11 texture + GL texture + interop handle)
- `SharedTexturePair` (input + output with SRV/UAV views)
- `GlDx11Bridge` struct (double-buffered pairs, WGL device, FBOs, frame tracking)
- Lock/unlock methods
- All blit methods, swap, cleanup
- Implement `GpuBridge` trait for `GlDx11Bridge`
- Add public accessors for D3D11 views: `input_srv()`, `output_uav()`

**Step 3: Wire up dx11 module**

```rust
// dx11/mod.rs
pub mod device;
pub mod interop;
pub use device::Dx11Device;
pub use interop::GlDx11Bridge;
```

**Step 4: Verify compilation (Windows only)**

Run: `cargo check -p gpu-interop`
Expected: PASS on Windows (will get cfg errors on macOS, which is fine)

**Step 5: Commit**

```bash
git add crates/gpu-interop/
git commit -m "feat(gpu-interop): port DX11 GL↔WGL_NV_DX_interop2 bridge"
```

---

## Task 9: Create ffgl-gpu — GPU Context and Pipeline Types

**Files:**
- Modify: `crates/ffgl-gpu/src/lib.rs`
- Create: `crates/ffgl-gpu/src/context.rs`
- Create: `crates/ffgl-gpu/src/pipeline.rs`
- Create: `crates/ffgl-gpu/src/buffer.rs`
- Create: `crates/ffgl-gpu/src/dispatch.rs`

**Step 1: Define GPU context**

```rust
// context.rs — wraps platform GPU device + shader library
pub struct GpuContext {
    #[cfg(target_os = "macos")]
    pub(crate) metal: MetalGpuContext,
    #[cfg(target_os = "windows")]
    pub(crate) dx11: Dx11GpuContext,
}

#[cfg(target_os = "macos")]
pub(crate) struct MetalGpuContext {
    pub device: gpu_interop::metal::MetalDevice,
    pub library: Retained<ProtocolObject<dyn MTLLibrary>>,
}

#[cfg(target_os = "windows")]
pub(crate) struct Dx11GpuContext {
    pub device: gpu_interop::dx11::Dx11Device,
    // Shader bytecodes loaded separately
}
```

**Step 2: Define pipeline types**

```rust
// pipeline.rs
pub struct ComputePipeline {
    #[cfg(target_os = "macos")]
    pub(crate) metal: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    #[cfg(target_os = "windows")]
    pub(crate) dx11: ID3D11ComputeShader,
}

pub struct RenderPipeline {
    #[cfg(target_os = "macos")]
    pub(crate) metal: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    #[cfg(target_os = "windows")]
    pub(crate) dx11: Dx11RenderPipeline, // VS + PS + input layout
}
```

**Step 3: Define buffer types**

```rust
// buffer.rs
pub struct GpuBuffer {
    #[cfg(target_os = "macos")]
    pub(crate) metal: Retained<ProtocolObject<dyn MTLBuffer>>,
    #[cfg(target_os = "windows")]
    pub(crate) dx11: Dx11Buffer, // buffer + UAV + SRV
}
```

**Step 4: Define dispatch/binding helpers**

```rust
// dispatch.rs
pub enum Binding<'a> {
    Buffer(&'a GpuBuffer),
    Texture(&'a GpuTexture),
    UniformData(&'a [u8]),
}

impl GpuContext {
    pub fn dispatch_compute(&self, pipeline: &ComputePipeline,
                            bindings: &[Binding], threads: [u32; 3],
                            group_size: [u32; 3]);
    pub fn dispatch_render(&self, pipeline: &RenderPipeline,
                           bindings: &[Binding], output: &GpuTexture,
                           width: u32, height: u32);
}
```

**Step 5: Wire up lib.rs**

```rust
pub mod buffer;
pub mod context;
pub mod dispatch;
pub mod pipeline;

pub use buffer::GpuBuffer;
pub use context::GpuContext;
pub use dispatch::Binding;
pub use pipeline::{ComputePipeline, RenderPipeline};

// Re-export interop for convenience
pub use gpu_interop::GpuBridge;
pub use ffgl_core;
pub use ffgl_glium;
```

**Step 6: Verify compilation**

Run: `cargo check -p ffgl-gpu`
Expected: PASS

**Step 7: Commit**

```bash
git add crates/ffgl-gpu/
git commit -m "feat(ffgl-gpu): define GPU context, pipeline, buffer, and dispatch types"
```

---

## Task 10: Create ffgl-gpu — Shader Build System

**Files:**
- Create: `crates/ffgl-gpu/src/build_support.rs` (library code for consumer build.rs)

**Step 1: Port Metal shader compilation logic**

Port from `ntsc-ffgl-plugin/shared/build.rs` (macOS path), generalized:
- `compile_metal_shaders(shader_dir: &Path, out_dir: &Path)` function
- Scans `shader_dir` for `.metal` files
- Compiles each to `.air` via `xcrun -sdk macosx metal`
- Links all `.air` into single `.metallib` via `xcrun metallib`
- Emits cargo:rerun-if-changed for each shader file

**Step 2: Port HLSL shader compilation logic**

Port from `ntsc-ffgl-plugin/shared/build.rs` (Windows path), generalized:
- `compile_hlsl_shaders(shader_dir: &Path, out_dir: &Path, entries: &[HlslEntry])` function
- `HlslEntry { file: &str, entry_point: &str, target: &str }` — CS/VS/PS targets
- Finds fxc.exe (PATH then Windows SDK scan)
- Compiles each entry to `.cso`
- Emits cargo:rerun-if-changed

**Step 3: Provide macros for loading compiled shaders**

```rust
/// Macro for loading embedded Metal shader library in consumer code
#[macro_export]
macro_rules! load_metal_library {
    () => {
        include_bytes!(concat!(env!("OUT_DIR"), "/shaders.metallib"))
    };
}

/// Macro for loading an embedded HLSL compute shader
#[macro_export]
macro_rules! load_hlsl_shader {
    ($name:literal) => {
        include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".cso"))
    };
}
```

**Step 4: Verify compilation**

Run: `cargo check -p ffgl-gpu`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/ffgl-gpu/
git commit -m "feat(ffgl-gpu): add shader build system for Metal and HLSL compilation"
```

---

## Task 11: Create ffgl-gpu — GpuPlugin Trait and Drawing Infrastructure

**Files:**
- Create: `crates/ffgl-gpu/src/plugin.rs`
- Create: `crates/ffgl-gpu/src/drawing.rs`

**Step 1: Define GpuPlugin trait**

```rust
// plugin.rs
use ffgl_core::{FFGLData, GLInput};
use crate::{GpuContext, GpuBridge};

/// Trait for FFGL plugins that use GPU compute/render.
/// Implement this to build a GPU-accelerated FFGL plugin.
pub trait GpuPlugin: Send + Sync + 'static {
    /// Called once when the GPU context is first available.
    /// Create pipelines, buffers, and other GPU resources here.
    fn gpu_init(&mut self, ctx: &GpuContext) -> anyhow::Result<()>;

    /// Called each frame to perform GPU rendering.
    fn gpu_draw(
        &mut self,
        ctx: &GpuContext,
        bridge: &mut dyn GpuBridge,
        data: &FFGLData,
        input: GLInput,
        frame: u64,
    );
}
```

**Step 2: Port drawing infrastructure**

Port the thread-local state pattern and draw loop from `ntsc-ffgl-plugin/shared/src/parameters/common.rs`:
- Thread-local `GpuContext`, bridge, and frame tracking
- `draw_gpu_effect()` function that handles:
  - Lazy GPU context initialization
  - Bridge dimension management
  - Double-buffered blit pipeline (previous output → host, input → bridge)
  - Calls `GpuPlugin::gpu_draw()`
  - Handles first-frame sync path
  - GL state save/restore

**Step 3: Verify compilation**

Run: `cargo check -p ffgl-gpu`
Expected: PASS

**Step 4: Commit**

```bash
git add crates/ffgl-gpu/
git commit -m "feat(ffgl-gpu): add GpuPlugin trait and drawing infrastructure"
```

---

## Task 12: Metal Pipeline Implementation — Compute Dispatch

**Files:**
- Modify: `crates/ffgl-gpu/src/dispatch.rs`
- Modify: `crates/ffgl-gpu/src/context.rs`

**Step 1: Implement Metal compute pipeline creation**

```rust
impl GpuContext {
    #[cfg(target_os = "macos")]
    pub fn create_compute_pipeline(&self, name: &str) -> Result<ComputePipeline> {
        let ns_name = NSString::from_str(name);
        let function = self.metal.library.newFunctionWithName(&ns_name)
            .ok_or_else(|| anyhow!("Shader function '{}' not found", name))?;
        let pipeline = self.metal.device.device
            .newComputePipelineStateWithFunction_error(&function)?;
        Ok(ComputePipeline { metal: pipeline })
    }
}
```

**Step 2: Implement Metal compute dispatch**

Port dispatch encoding pattern from `ntsc-ffgl-plugin/shared/src/metal/dispatch.rs`:
- Create command buffer and compute encoder
- Set pipeline state
- Bind buffers and textures at indices
- Set uniform bytes inline
- `dispatchThreads:threadsPerThreadgroup:`
- End encoding and commit

**Step 3: Implement Metal render pipeline creation and dispatch**

- Create `MTLRenderPipelineDescriptor` with vertex + fragment functions
- Set pixel format to BGRA8Unorm
- Create fullscreen quad vertex buffer (4 vertices: -1,-1 to 1,1 with UVs)
- Dispatch: create render pass descriptor targeting output texture, draw fullscreen quad

**Step 4: Verify compilation (macOS)**

Run: `cargo check -p ffgl-gpu`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/ffgl-gpu/
git commit -m "feat(ffgl-gpu): implement Metal compute and render pipeline dispatch"
```

---

## Task 13: DX11 Pipeline Implementation — Compute Dispatch

**Files:**
- Modify: `crates/ffgl-gpu/src/dispatch.rs`
- Modify: `crates/ffgl-gpu/src/context.rs`
- Modify: `crates/ffgl-gpu/src/buffer.rs`

**Step 1: Implement DX11 compute pipeline creation**

```rust
impl GpuContext {
    #[cfg(target_os = "windows")]
    pub fn create_compute_pipeline(&self, bytecode: &[u8]) -> Result<ComputePipeline> {
        let mut shader = None;
        unsafe { self.dx11.device.device.CreateComputeShader(bytecode, None, Some(&mut shader)) }?;
        Ok(ComputePipeline { dx11: shader.unwrap() })
    }
}
```

**Step 2: Implement DX11 compute dispatch**

Port dispatch pattern from `ntsc-ffgl-plugin/shared/src/dx11/dispatch.rs`:
- `CSSetShader`
- `CSSetUnorderedAccessViews` for buffer/texture UAVs
- `CSSetShaderResources` for SRVs
- `CSSetConstantBuffers` for uniform data
- `Dispatch(groups_x, groups_y, groups_z)`
- Unbind UAVs after dispatch

**Step 3: Implement DX11 constant buffer management**

Port from `ntsc-ffgl-plugin/shared/src/dx11/device.rs`:
- Dynamic constant buffer creation
- Map/Unmap pattern for uploading uniforms

**Step 4: Implement DX11 render pipeline creation and dispatch**

- Create vertex shader, pixel shader, input layout
- Create fullscreen quad vertex buffer
- Dispatch: `IASetVertexBuffers`, `IASetInputLayout`, `IASetPrimitiveTopology`, `VSSetShader`, `PSSetShader`, `OMSetRenderTargets`, `Draw`

**Step 5: Verify compilation (Windows)**

Run: `cargo check -p ffgl-gpu`
Expected: PASS on Windows

**Step 6: Commit**

```bash
git add crates/ffgl-gpu/
git commit -m "feat(ffgl-gpu): implement DX11 compute and render pipeline dispatch"
```

---

## Task 14: Create Passthrough Example

**Files:**
- Create: `examples/passthrough/Cargo.toml`
- Create: `examples/passthrough/src/lib.rs`
- Create: `examples/passthrough/shaders/passthrough.metal`
- Create: `examples/passthrough/shaders/passthrough.hlsl`
- Create: `examples/passthrough/build.rs`

**Step 1: Write compute shaders**

Metal (`passthrough.metal`):
```metal
#include <metal_stdlib>
kernel void passthrough(
    texture2d<float, access::read> input [[texture(0)]],
    texture2d<float, access::write> output [[texture(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= input.get_width() || gid.y >= input.get_height()) return;
    output.write(input.read(gid), gid);
}
```

HLSL (`passthrough.hlsl`):
```hlsl
Texture2D<float4> input : register(t0);
RWTexture2D<float4> output : register(u0);

[numthreads(16, 16, 1)]
void passthrough(uint3 id : SV_DispatchThreadID) {
    uint w, h;
    input.GetDimensions(w, h);
    if (id.x >= w || id.y >= h) return;
    output[id.xy] = input[id.xy];
}
```

**Step 2: Write build.rs**

```rust
fn main() {
    ffgl_gpu::build_support::compile_shaders("shaders");
}
```

**Step 3: Write plugin implementation**

```rust
// src/lib.rs
use ffgl_core::*;
use ffgl_gpu::*;

struct PassthroughPlugin {
    pipeline: Option<ComputePipeline>,
}

impl GpuPlugin for PassthroughPlugin {
    fn gpu_init(&mut self, ctx: &GpuContext) -> anyhow::Result<()> {
        self.pipeline = Some(ctx.create_compute_pipeline("passthrough")?);
        Ok(())
    }

    fn gpu_draw(&mut self, ctx: &GpuContext, bridge: &mut dyn GpuBridge,
                _data: &FFGLData, _input: GLInput, _frame: u64) {
        let (w, h) = bridge.dimensions();
        ctx.dispatch_compute(
            self.pipeline.as_ref().unwrap(),
            &[Binding::Texture(bridge.input_texture()),
              Binding::Texture(bridge.output_texture())],
            [w, h, 1],
            [16, 16, 1],
        );
    }
}

impl SimpleFFGLInstance for PassthroughPlugin { /* ... */ }
plugin_main!(SimpleFFGLHandler<PassthroughPlugin>);
```

**Step 4: Add to workspace and verify compilation**

Add `"examples/passthrough"` to workspace members.

Run: `cargo check -p ffgl-passthrough`
Expected: PASS

**Step 5: Commit**

```bash
git add examples/passthrough/ Cargo.toml
git commit -m "feat(examples): add passthrough compute shader example"
```

---

## Task 15: Create Invert Example (Fragment Shader)

**Files:**
- Create: `examples/invert/Cargo.toml`
- Create: `examples/invert/src/lib.rs`
- Create: `examples/invert/shaders/invert.metal` (vertex + fragment functions)
- Create: `examples/invert/shaders/invert.hlsl` (VS + PS entry points)
- Create: `examples/invert/build.rs`

**Step 1: Write vertex/fragment shaders**

Metal: fullscreen quad vertex function + fragment function that samples input and outputs `1.0 - color`.
HLSL: VS that generates fullscreen quad from vertex ID + PS that inverts.

**Step 2: Write plugin using RenderPipeline**

Demonstrates `create_render_pipeline()` and `dispatch_render()` instead of compute.

**Step 3: Add to workspace and verify**

Run: `cargo check -p ffgl-invert`
Expected: PASS

**Step 4: Commit**

```bash
git add examples/invert/ Cargo.toml
git commit -m "feat(examples): add invert fragment shader example"
```

---

## Task 16: Create Blur Example (Multi-Pass Compute)

**Files:**
- Create: `examples/blur/Cargo.toml`
- Create: `examples/blur/src/lib.rs`
- Create: `examples/blur/shaders/blur.metal`
- Create: `examples/blur/shaders/blur.hlsl`
- Create: `examples/blur/build.rs`

**Step 1: Write two-pass blur shaders**

Horizontal and vertical Gaussian blur kernels with radius uniform.

**Step 2: Write plugin with FFGL parameter**

- Expose "Radius" parameter via FFGL
- Create intermediate GpuBuffer for horizontal pass output
- Two dispatches per frame: horizontal → vertical

**Step 3: Add to workspace and verify**

Run: `cargo check -p ffgl-blur`
Expected: PASS

**Step 4: Commit**

```bash
git add examples/blur/ Cargo.toml
git commit -m "feat(examples): add blur multi-pass compute example with FFGL parameter"
```

---

## Task 17: Create Kitchen-Sink Example (Mixed Pipelines)

**Files:**
- Create: `examples/kitchen-sink/Cargo.toml`
- Create: `examples/kitchen-sink/src/lib.rs`
- Create: `examples/kitchen-sink/shaders/edge_detect.metal`
- Create: `examples/kitchen-sink/shaders/color_grade.metal`
- Create: `examples/kitchen-sink/shaders/composite.metal`
- Create: `examples/kitchen-sink/shaders/edge_detect.hlsl`
- Create: `examples/kitchen-sink/shaders/color_grade.hlsl`
- Create: `examples/kitchen-sink/shaders/composite.hlsl`
- Create: `examples/kitchen-sink/build.rs`

**Step 1: Write shaders**

- `edge_detect` — Compute shader (Sobel operator)
- `color_grade` — Fragment shader (hue/saturation/brightness)
- `composite` — Compute shader (blend edge + graded)

**Step 2: Write plugin combining compute + render**

- Multiple FFGL parameters (edge strength, hue shift, saturation, brightness, blend amount)
- Pipeline: compute edge detect → render color grade → compute composite
- Demonstrates full framework capabilities

**Step 3: Add to workspace and verify**

Run: `cargo check -p ffgl-kitchen-sink`
Expected: PASS

**Step 4: Commit**

```bash
git add examples/kitchen-sink/ Cargo.toml
git commit -m "feat(examples): add kitchen-sink mixed compute+render example"
```

---

## Task 18: Final Integration and Build Verification

**Step 1: Verify full workspace compiles**

Run: `cargo check --workspace`
Expected: All crates and examples compile successfully.

**Step 2: Run clippy**

Run: `cargo clippy --workspace -- -D warnings`
Fix any warnings.

**Step 3: Write workspace-level README or CLAUDE.md**

Brief description of the project, crate purposes, and how to build examples.

**Step 4: Final commit**

```bash
git add -A
git commit -m "feat: complete FFGL GPU interop library with all examples"
```
