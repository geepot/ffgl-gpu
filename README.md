# ffgl-gpu

A Rust framework for building GPU-accelerated [FFGL 2.2](https://github.com/resolume/ffgl) plugins. Write Metal compute/render shaders on macOS and HLSL compute shaders on Windows, and the framework handles the OpenGL interop, double-buffered pipelining, and FFGL host protocol for you.

Plugins built with this framework work in any FFGL 2.2 host: Resolume Arena/Avenue, VDMX, MadMapper, and others.

## Features

- **Cross-platform GPU compute and render pipelines** via a unified `GpuPlugin` trait
- **Metal on macOS** (IOSurface-backed GL-Metal texture sharing)
- **Direct3D 11 on Windows** (WGL\_NV\_DX\_interop2 GL-DX11 texture sharing)
- **Double-buffered pipelining** for one-frame-latency GPU overlap
- **Build-time shader compilation** (`.metal` to `.metallib`, `.hlsl` to `.cso`)
- **GL state save/restore** to keep the host compositor stable
- **FFGL parameter support** for exposing controls to the host
- **Four example plugins** demonstrating compute, render, multi-pass, and parameter patterns

## Project Structure

| Crate | Description |
|---|---|
| `crates/ffgl-core` | FFGL 2.2 host protocol: FFI bindings, plugin traits, parameter system |
| `crates/ffgl-glium` | OpenGL context wrapper for rendering inside an FFGL host |
| `crates/gpu-interop` | Platform-specific GL-GPU texture bridges (`GpuBridge` trait) |
| `crates/ffgl-gpu` | GPU context, pipelines, shader build support, drawing loop |

```
ffgl-core          (no GPU deps — pure FFGL protocol)
  ^
ffgl-glium         (OpenGL abstraction via glium)
  ^
gpu-interop        (Metal/DX11 ↔ GL texture sharing)
  ^
ffgl-gpu           (ties it all together: context, pipelines, draw loop)
  ^
examples/*         (your plugins go here)
```

## Prerequisites

**All platforms:**
- [Rust](https://rustup.rs/) (stable toolchain)

**macOS:**
- Xcode or Command Line Tools (for Metal compiler and `libclang`)

**Windows:**
- Visual Studio with C++ workload (for MSVC toolchain and `libclang`)
- Windows SDK (for Direct3D 11 headers)
- An NVIDIA GPU with WGL\_NV\_DX\_interop2 support (most modern NVIDIA GPUs)

## Building

### macOS

```bash
# Build all plugins for current architecture (release)
./build.sh

# Universal binary (arm64 + x86_64)
./build.sh --platform macos --arch universal

# Single plugin, debug
./build.sh --plugin blur --profile debug
```

### Windows

```powershell
# Build all plugins (release, MSVC)
.\build.ps1

# Single plugin
.\build.ps1 -Plugin blur

# MinGW toolchain
.\build.ps1 -Toolchain gnu
```

### Cross-compilation

```bash
# Build Windows DLLs from macOS (requires cross-compilation toolchain)
./build.sh --platform windows

# Build everything for every platform
./build.sh --platform all --arch universal
```

Built artifacts land in `dist/<platform>/<arch>/`. On macOS, `.bundle` directories are created automatically for Resolume compatibility.

## Deploying

```bash
# Deploy all plugins to detected VJ software directories
./deploy.sh

# Preview without copying
./deploy.sh --dry-run

# Deploy a specific plugin
./deploy.sh --plugin blur
```

The deploy script detects installed hosts (Resolume Arena, VDMX, MadMapper on macOS; Resolume Arena/Avenue on Windows) and copies the appropriate format (`.bundle` for Resolume, `.dylib` for others on macOS, `.dll` on Windows).

## Writing a Plugin

A minimal plugin needs three things: a GPU state struct, shaders, and a build script.

### 1. Implement `GpuPlugin`

```rust
use ffgl_gpu::{GpuContext, GpuPlugin};
use ffgl_gpu::pipeline::ComputePipeline;
use gpu_interop::GpuBridge;

struct MyEffect {
    pipeline: Option<ComputePipeline>,
}

impl GpuPlugin for MyEffect {
    fn gpu_init(&mut self, ctx: &GpuContext) -> anyhow::Result<()> {
        self.pipeline = Some(ctx.create_compute_pipeline("my_kernel")?);
        Ok(())
    }

    fn gpu_draw(
        &mut self,
        ctx: &GpuContext,
        bridge: &mut dyn GpuBridge,
        data: &FFGLData,
        input: &GLInput<'_>,
        frame: u64,
    ) {
        // Downcast bridge to platform type, get textures, dispatch work
    }
}
```

### 2. Write shaders

Place `.metal` files in a `shaders/` directory:

```metal
#include <metal_stdlib>
using namespace metal;

kernel void my_kernel(
    texture2d<float, access::read> input [[texture(0)]],
    texture2d<float, access::write> output [[texture(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= input.get_width() || gid.y >= input.get_height()) return;
    float4 color = input.read(gid);
    // ... your effect here ...
    output.write(color, gid);
}
```

### 3. Add a build script

```rust
// build.rs
fn main() {
    #[cfg(target_os = "macos")]
    ffgl_gpu::build_support::compile_metal_shaders(std::path::Path::new("shaders"));
}
```

### 4. Wire up the FFGL entry point

Wrap your GPU state in a `SimpleFFGLInstance` and use the `plugin_main!` macro:

```rust
impl SimpleFFGLInstance for MyPlugin {
    fn plugin_info() -> PluginInfo {
        PluginInfo {
            unique_id: *b"MYFX",
            name: *b"My Effect\0\0\0\0\0\0\0",
            ty: PluginType::Effect,
            ..
        }
    }

    fn draw(&mut self, data: &FFGLData, frame_data: GLInput) {
        draw_gpu_effect(&mut self.gpu, self.instance_id, &mut self.glium,
            data, frame_data, self.frame_counter, 1.0, 1.0, METALLIB_BYTES);
    }
}

ffgl_core::plugin_main!(SimpleFFGLHandler<MyPlugin>);
```

See the `examples/` directory for complete working implementations.

## Examples

| Example | What it demonstrates |
|---|---|
| `passthrough` | Simplest possible compute plugin: copies input to output pixel-for-pixel |
| `invert` | Render pipeline (vertex + fragment shader) instead of compute |
| `blur` | Multi-pass compute (separable box blur) with an FFGL parameter |
| `kitchen-sink` | Chained compute + render pipelines with multiple parameters |

## License

MIT
