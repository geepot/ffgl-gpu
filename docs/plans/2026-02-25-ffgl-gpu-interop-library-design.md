# FFGL GPU Interop Library Design

**Date:** 2026-02-25
**Status:** Approved

## Goal

Create a generic, reusable Rust library for building FFGL plugins that use Metal (macOS) or DirectX 11 (Windows) for GPU compute and rendering. The library bridges FFGL's OpenGL context to native GPU APIs, handling texture sharing, double-buffering, shader compilation, and pipeline dispatch.

Ported from the interop architecture in [ntsc-ffgl-plugin](./ntsc-ffgl-plugin/).

## Architecture: Layered Crate Workspace

```
ffgl-dx11-metal/
├── Cargo.toml                        # Workspace root
├── crates/
│   ├── ffgl-core/                    # FFGL 2.2 host protocol
│   ├── ffgl-glium/                   # OpenGL context wrapper for FFGL
│   ├── gpu-interop/                  # GL↔Metal + GL↔DX11 texture bridges
│   └── ffgl-gpu/                     # High-level FFGL+GPU plugin framework
├── examples/
│   ├── passthrough/                  # Identity compute shader
│   ├── invert/                       # Color invert via fragment shader
│   ├── blur/                         # Multi-pass compute with FFGL params
│   └── kitchen-sink/                 # Mixed compute + render pipelines
└── ntsc-ffgl-plugin/                 # Original reference (unchanged)
```

## Crate Details

### 1. `ffgl-core` — FFGL Host Protocol

Ported from `ntsc-ffgl-plugin/ffgl-rs/ffgl-core`.

- FFI entry points (`plugMain`, `plugMainGlobal`)
- FFGL type definitions (`PluginInfo`, `ParamInfo`, capability flags)
- `FFGLHandler` trait that plugin authors implement
- `plugin_main!` macro for cdylib entry point generation
- No GPU dependencies — purely FFGL protocol

### 2. `ffgl-glium` — OpenGL Context Wrapper

Ported from `ntsc-ffgl-plugin/ffgl-rs/ffgl-glium`.

- `FFGLGlium` struct wrapping the host-provided OpenGL context
- Provides safe access to GL state within the FFGL lifecycle
- Depends on `glium` and `gl` crates

### 3. `gpu-interop` — GL↔GPU Texture Bridges

Core of the library. Platform-conditional compilation.

#### Common Interface

```rust
pub trait GpuBridge {
    type GpuTexture;

    fn ensure_dimensions(&mut self, width: u32, height: u32) -> Result<()>;
    fn blit_input(&mut self, host_texture: GLuint, src_w: u32, src_h: u32,
                  dst_w: u32, dst_h: u32, bilinear: bool);
    fn blit_output(&mut self, host_fbo: GLuint, src_w: u32, src_h: u32,
                   dst_w: u32, dst_h: u32, bilinear: bool);
    fn input_texture(&self) -> &Self::GpuTexture;
    fn output_texture(&self) -> &Self::GpuTexture;
    fn has_result_ready(&self, frame: u64) -> bool;
    fn wait_for_previous(&mut self);
    fn swap(&mut self);
}
```

#### Metal Backend (macOS)

- `GlMetalBridge` — IOSurface-backed shared textures
- IOSurface allows same GPU memory to be accessed by both OpenGL (via `CGLTexImageIOSurface2D`) and Metal (via `newTextureWithDescriptor:iosurface:`)
- Double-buffered pairs: front pair receives new input while back pair's output is blitted
- One-frame latency for pipelining
- Stores pending `MTLCommandBuffer` for synchronization

#### DX11 Backend (Windows)

- `GlDx11Bridge` — WGL_NV_DX_interop2 shared textures
- Dynamically loads `wglDXOpenDeviceNV`, `wglDXRegisterObjectNV`, etc.
- D3D11 textures created with `D3D11_RESOURCE_MISC_SHARED` flag
- Lock/unlock semantics for GL↔D3D11 synchronization
- Double-buffered pairs with GPU event query polling
- D3D11 textures have both SRV (shader read) and UAV (compute write) views

#### Dependencies

- macOS: `objc2`, `objc2-metal`, `objc2-io-surface`, `objc2-open-gl`, `objc2-core-foundation`
- Windows: `windows` crate with `Win32_Graphics_Direct3D11`, `Win32_Graphics_OpenGL`, etc.
- Both: `gl` crate for OpenGL bindings

### 4. `ffgl-gpu` — High-Level Plugin Framework

User-facing crate that combines everything.

#### GPU Context

```rust
pub struct GpuContext {
    // Wraps MetalContext (device, queue, library) or Dx11Context (device, context)
}

impl GpuContext {
    pub fn create_compute_pipeline(&self, name: &str) -> Result<ComputePipeline>;
    pub fn create_render_pipeline(&self, vertex: &str, fragment: &str) -> Result<RenderPipeline>;
    pub fn create_buffer(&self, size: usize) -> Result<GpuBuffer>;
    pub fn dispatch_compute(&self, pipeline: &ComputePipeline,
                            bindings: &[Binding], threads: [u32; 3]);
    pub fn dispatch_render(&self, pipeline: &RenderPipeline,
                           bindings: &[Binding], output: &GpuTexture);
}
```

#### Pipeline Types

- **ComputePipeline** — wraps `MTLComputePipelineState` or `ID3D11ComputeShader`
- **RenderPipeline** — wraps `MTLRenderPipelineState` or `ID3D11VertexShader` + `ID3D11PixelShader` + fullscreen quad geometry
- **GpuBuffer** — wraps `MTLBuffer` or `ID3D11Buffer` with SRV/UAV views

#### Shader Build System (`build.rs`)

- Scans a `shaders/` directory in the consumer's crate
- **Metal:** Compiles `.metal` files → `.metallib` via `xcrun metal` + `xcrun metallib`
- **DX11:** Compiles `.hlsl` files → `.cso` via `fxc.exe` or `dxc.exe`, with separate targets for CS (compute), VS (vertex), PS (pixel)
- Compiled bytecode is embedded via `include_bytes!` at build time
- Consumer crates call a helper function or macro to invoke the build

#### Plugin Trait

```rust
pub trait GpuPlugin: FFGLHandler {
    fn gpu_init(&mut self, ctx: &GpuContext) -> Result<()>;
    fn gpu_draw(&mut self, ctx: &GpuContext, bridge: &mut dyn GpuBridge,
                input: GLInput, frame: u64);
}
```

## Examples

### `passthrough/`
Identity transform via compute shader. Copies input→output pixel-for-pixel. Validates the GL↔GPU bridge works end-to-end. No FFGL parameters.

### `invert/`
Color inversion via fragment shader. Draws a fullscreen quad through a render pipeline that samples the input and outputs `1.0 - color`. Validates the render pipeline path.

### `blur/`
Gaussian blur via two-pass compute shader (horizontal + vertical). Exposes a "Radius" FFGL parameter. Demonstrates multi-pass compute, intermediate GPU buffers, and parameter mapping.

### `kitchen-sink/`
Combined effect using both compute and render passes. Example pipeline: compute edge detection → fragment color grading → compute compositing. Multiple FFGL parameters. Demonstrates the full capability of the framework.

Each example is a `cdylib` crate with:
- `Cargo.toml` (depends on `ffgl-gpu`)
- `src/lib.rs` (plugin implementation)
- `shaders/` directory with `.metal` and `.hlsl` source files

## Key Design Decisions

1. **Double-buffering with one-frame latency** — preserved from original for performance
2. **Thread-local GPU state** — avoids global statics, safe for multi-instance plugins
3. **Platform conditional compilation** — `#[cfg(target_os)]` cleanly splits Metal/DX11
4. **Embedded shaders** — no external file dependencies at runtime
5. **Trait-based abstraction** — `GpuBridge` trait with platform-specific impls
6. **Lazy initialization** — GPU contexts created on first draw, cached thereafter

## Dependencies

### Workspace-level
- `gl = "0.14"` — OpenGL bindings
- `glium = "0.36"` — OpenGL wrapper
- `tracing = "0.1"` — Logging
- `anyhow = "1"` — Error handling

### macOS-only
- `objc2 = "0.6"`, `objc2-metal = "0.3"`, `objc2-io-surface = "0.3"`, `objc2-open-gl = "0.3"`, `objc2-core-foundation = "0.3"`, `objc2-foundation = "0.3"`, `dispatch2 = "0.3"`

### Windows-only
- `windows = "0.62"` with Direct3D11, DXGI, OpenGL features
