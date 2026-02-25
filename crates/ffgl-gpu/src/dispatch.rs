//! GPU dispatch: pipeline creation, buffer allocation, and compute/render
//! command encoding.
//!
//! All pipeline creation and dispatch methods live on [`GpuContext`].

use anyhow::Result;

use crate::buffer::GpuBuffer;
use crate::context::GpuContext;
use crate::pipeline::{ComputePipeline, RenderPipeline};

// ---------------------------------------------------------------------------
// Binding enum — platform-agnostic resource binding descriptor
// ---------------------------------------------------------------------------

/// A resource to bind at a numbered slot when dispatching a compute or render
/// pipeline.
pub enum Binding<'a> {
    /// A GPU buffer.
    Buffer(&'a GpuBuffer),
    /// A platform-specific texture. On macOS this is a `&ProtocolObject<dyn
    /// MTLTexture>`; on Windows an `&ID3D11ShaderResourceView` or
    /// `&ID3D11UnorderedAccessView`. The caller must cast via `Any`.
    Texture(&'a dyn std::any::Any),
    /// Inline uniform / constant data (copied into the command encoder).
    UniformData(&'a [u8]),
}

// ---------------------------------------------------------------------------
// Compute pass — in-progress compute encoding
// ---------------------------------------------------------------------------

/// An in-progress compute pass.
///
/// On macOS this holds the command buffer and compute command encoder. On
/// Windows this is a zero-sized marker (D3D11 immediate context is stateful).
pub struct ComputePass {
    #[cfg(target_os = "macos")]
    pub(crate) command_buffer:
        objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandBuffer>>,
    #[cfg(target_os = "macos")]
    pub(crate) encoder: objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>,
    >,
}

/// A token representing GPU work that has been submitted but may not yet be
/// complete.
pub struct PendingWork {
    #[cfg(target_os = "macos")]
    pub(crate) command_buffer:
        objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandBuffer>>,
}

#[cfg(target_os = "macos")]
impl PendingWork {
    /// Block until the GPU work completes.
    pub fn wait(&self) {
        use objc2_metal::MTLCommandBuffer;
        self.command_buffer.waitUntilCompleted();
    }

    /// Consume this token and return the underlying Metal command buffer.
    ///
    /// Useful for storing in a [`GlMetalBridge`](gpu_interop::metal::GlMetalBridge)
    /// for pipelined synchronization.
    pub fn into_command_buffer(
        self,
    ) -> objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandBuffer>>
    {
        self.command_buffer
    }
}

// ---------------------------------------------------------------------------
// macOS Metal implementation
// ---------------------------------------------------------------------------

#[cfg(target_os = "macos")]
mod metal_impl {
    use super::*;
    use objc2::runtime::ProtocolObject;
    use objc2_foundation::NSString;
    use objc2_metal::*;

    /// Fullscreen quad vertex data: 4 vertices, each with (x, y, u, v).
    /// Triangle-strip order: bottom-left, bottom-right, top-left, top-right.
    const FULLSCREEN_QUAD: [[f32; 4]; 4] = [
        [-1.0, -1.0, 0.0, 1.0], // bottom-left  (pos.xy, uv)
        [1.0, -1.0, 1.0, 1.0],  // bottom-right
        [-1.0, 1.0, 0.0, 0.0],  // top-left
        [1.0, 1.0, 1.0, 0.0],   // top-right
    ];

    impl GpuContext {
        /// Create a compute pipeline from a named kernel function in the loaded
        /// Metal shader library.
        pub fn create_compute_pipeline(&self, name: &str) -> Result<ComputePipeline> {
            let func_name = NSString::from_str(name);
            let function = self
                .library
                .newFunctionWithName(&func_name)
                .ok_or_else(|| anyhow::anyhow!("Metal function '{name}' not found in library"))?;

            let state = self
                .device
                .device()
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|e| {
                    anyhow::anyhow!("Failed to create compute pipeline for '{name}': {e}")
                })?;

            Ok(ComputePipeline { state })
        }

        /// Create a render pipeline from vertex and fragment function names.
        ///
        /// The pipeline is configured for BGRA8Unorm output and alpha blending
        /// disabled, suitable for fullscreen quad rendering.
        pub fn create_render_pipeline(
            &self,
            vertex_name: &str,
            fragment_name: &str,
        ) -> Result<RenderPipeline> {
            let vs_name = NSString::from_str(vertex_name);
            let fs_name = NSString::from_str(fragment_name);

            let vs_func = self
                .library
                .newFunctionWithName(&vs_name)
                .ok_or_else(|| {
                    anyhow::anyhow!("Metal vertex function '{vertex_name}' not found")
                })?;
            let fs_func = self
                .library
                .newFunctionWithName(&fs_name)
                .ok_or_else(|| {
                    anyhow::anyhow!("Metal fragment function '{fragment_name}' not found")
                })?;

            let desc = MTLRenderPipelineDescriptor::new();
            desc.setVertexFunction(Some(&vs_func));
            desc.setFragmentFunction(Some(&fs_func));

            {
                let attachment = unsafe {
                    desc.colorAttachments().objectAtIndexedSubscript(0)
                };
                attachment.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
                attachment.setBlendingEnabled(false);
            }

            let state = self
                .device
                .device()
                .newRenderPipelineStateWithDescriptor_error(&desc)
                .map_err(|e| {
                    anyhow::anyhow!(
                        "Failed to create render pipeline '{vertex_name}'/'{fragment_name}': {e}"
                    )
                })?;

            // Create fullscreen quad vertex buffer
            let quad_data = FULLSCREEN_QUAD;
            let quad_bytes = quad_data.as_ptr() as *const std::ffi::c_void;
            let quad_len = std::mem::size_of_val(&quad_data);
            let quad_vb = unsafe {
                self.device
                    .device()
                    .newBufferWithBytes_length_options(
                        std::ptr::NonNull::new_unchecked(quad_bytes as *mut _),
                        quad_len,
                        MTLResourceOptions::StorageModeShared,
                    )
            }
            .ok_or_else(|| anyhow::anyhow!("Failed to create fullscreen quad vertex buffer"))?;

            Ok(RenderPipeline {
                state,
                quad_vb,
            })
        }

        /// Create a GPU buffer with the given number of elements and element
        /// size.
        ///
        /// The buffer is allocated with `StorageModePrivate` (GPU-only memory)
        /// for optimal compute shader performance.
        pub fn create_buffer(
            &self,
            num_elements: usize,
            element_size: usize,
        ) -> Result<GpuBuffer> {
            let size = num_elements * element_size;
            let buffer = self
                .device
                .device()
                .newBufferWithLength_options(size, MTLResourceOptions::StorageModePrivate)
                .ok_or_else(|| {
                    anyhow::anyhow!("Failed to allocate Metal buffer of {size} bytes")
                })?;

            Ok(GpuBuffer {
                size,
                metal: buffer,
            })
        }

        /// Create a GPU buffer with `StorageModeShared` (CPU + GPU accessible).
        ///
        /// Useful for constant/uniform buffers that need CPU writes.
        pub fn create_shared_buffer(
            &self,
            num_elements: usize,
            element_size: usize,
        ) -> Result<GpuBuffer> {
            let size = num_elements * element_size;
            let buffer = self
                .device
                .device()
                .newBufferWithLength_options(size, MTLResourceOptions::StorageModeShared)
                .ok_or_else(|| {
                    anyhow::anyhow!("Failed to allocate shared Metal buffer of {size} bytes")
                })?;

            Ok(GpuBuffer {
                size,
                metal: buffer,
            })
        }

        /// Begin a compute pass: create a command buffer and compute command
        /// encoder.
        pub fn begin_compute_pass(&self) -> Result<ComputePass> {
            let command_buffer = self
                .device
                .command_queue()
                .commandBuffer()
                .ok_or_else(|| anyhow::anyhow!("Failed to create Metal command buffer"))?;

            let encoder = command_buffer
                .computeCommandEncoder()
                .ok_or_else(|| anyhow::anyhow!("Failed to create Metal compute encoder"))?;

            Ok(ComputePass {
                command_buffer,
                encoder,
            })
        }

        /// Set a compute pipeline on the given pass.
        pub fn set_compute_pipeline(&self, pass: &ComputePass, pipeline: &ComputePipeline) {
            pass.encoder.setComputePipelineState(&pipeline.state);
        }

        /// Bind a buffer at the given index on a compute pass.
        pub fn bind_buffer(&self, pass: &ComputePass, buffer: &GpuBuffer, index: usize) {
            unsafe {
                pass.encoder
                    .setBuffer_offset_atIndex(Some(&buffer.metal), 0, index);
            }
        }

        /// Bind a Metal texture at the given index on a compute pass.
        ///
        /// The `texture` must be a `&ProtocolObject<dyn MTLTexture>`.
        pub fn bind_texture(
            &self,
            pass: &ComputePass,
            texture: &ProtocolObject<dyn MTLTexture>,
            index: usize,
        ) {
            unsafe {
                pass.encoder.setTexture_atIndex(Some(texture), index);
            }
        }

        /// Bind inline bytes as a constant buffer at the given index.
        pub fn bind_bytes(&self, pass: &ComputePass, data: &[u8], index: usize) {
            unsafe {
                pass.encoder.setBytes_length_atIndex(
                    std::ptr::NonNull::new_unchecked(data.as_ptr() as *mut _),
                    data.len(),
                    index,
                );
            }
        }

        /// Dispatch a 2D compute grid.
        ///
        /// `grid` is the total number of threads (width, height), and
        /// `threadgroup` is the threadgroup size. The encoder will compute the
        /// correct number of threadgroups.
        pub fn dispatch_threads(
            &self,
            pass: &ComputePass,
            grid: (usize, usize),
            threadgroup: (usize, usize),
        ) {
            let grid_size = MTLSize {
                width: grid.0,
                height: grid.1,
                depth: 1,
            };
            let tg_size = MTLSize {
                width: threadgroup.0,
                height: threadgroup.1,
                depth: 1,
            };
            pass.encoder
                .dispatchThreads_threadsPerThreadgroup(grid_size, tg_size);
        }

        /// Dispatch a 1D compute grid.
        pub fn dispatch_threads_1d(
            &self,
            pass: &ComputePass,
            thread_count: usize,
            threadgroup_size: usize,
        ) {
            let grid_size = MTLSize {
                width: thread_count,
                height: 1,
                depth: 1,
            };
            let tg_size = MTLSize {
                width: threadgroup_size,
                height: 1,
                depth: 1,
            };
            pass.encoder
                .dispatchThreads_threadsPerThreadgroup(grid_size, tg_size);
        }

        /// Dispatch threadgroups (indirect grid size, explicit threadgroup
        /// count).
        pub fn dispatch_threadgroups(
            &self,
            pass: &ComputePass,
            threadgroups: (usize, usize),
            threadgroup_size: (usize, usize),
        ) {
            let tg_count = MTLSize {
                width: threadgroups.0,
                height: threadgroups.1,
                depth: 1,
            };
            let tg_size = MTLSize {
                width: threadgroup_size.0,
                height: threadgroup_size.1,
                depth: 1,
            };
            pass.encoder
                .dispatchThreadgroups_threadsPerThreadgroup(tg_count, tg_size);
        }

        /// End the compute pass, commit the command buffer, and return a
        /// [`PendingWork`] token for synchronization.
        pub fn end_compute_pass(&self, pass: ComputePass) -> PendingWork {
            pass.encoder.endEncoding();
            pass.command_buffer.commit();
            PendingWork {
                command_buffer: pass.command_buffer,
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Windows DX11 stub implementation
// ---------------------------------------------------------------------------

#[cfg(target_os = "windows")]
mod dx11_impl {
    use super::*;
    use windows::Win32::Graphics::Direct3D11::*;

    impl GpuContext {
        /// Create a compute pipeline from pre-compiled HLSL bytecode (`.cso`).
        pub fn create_compute_pipeline_from_bytecode(
            &self,
            bytecode: &[u8],
        ) -> Result<ComputePipeline> {
            let mut shader = None;
            unsafe {
                self.device
                    .device()
                    .CreateComputeShader(bytecode, None, Some(&mut shader as *mut _))
            }
            .map_err(|e| anyhow::anyhow!("Failed to create D3D11 compute shader: {e}"))?;

            let shader =
                shader.ok_or_else(|| anyhow::anyhow!("D3D11 CreateComputeShader returned null"))?;

            Ok(ComputePipeline { shader })
        }

        /// Create a GPU buffer as a structured buffer with UAV + SRV views.
        pub fn create_buffer(
            &self,
            num_elements: usize,
            element_size: usize,
        ) -> Result<GpuBuffer> {
            let size = num_elements * element_size;

            let buffer_desc = D3D11_BUFFER_DESC {
                ByteWidth: size as u32,
                Usage: D3D11_USAGE_DEFAULT,
                BindFlags: (D3D11_BIND_UNORDERED_ACCESS.0 | D3D11_BIND_SHADER_RESOURCE.0) as u32,
                MiscFlags: D3D11_RESOURCE_MISC_BUFFER_STRUCTURED.0 as u32,
                StructureByteStride: element_size as u32,
                ..Default::default()
            };

            let mut buffer = None;
            unsafe {
                self.device.device().CreateBuffer(
                    &buffer_desc,
                    None,
                    Some(&mut buffer as *mut _),
                )
            }
            .map_err(|e| anyhow::anyhow!("Failed to create D3D11 structured buffer: {e}"))?;
            let buffer =
                buffer.ok_or_else(|| anyhow::anyhow!("D3D11 CreateBuffer returned null"))?;

            // Create UAV
            let uav_desc = D3D11_UNORDERED_ACCESS_VIEW_DESC {
                Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_UNKNOWN,
                ViewDimension: D3D11_UAV_DIMENSION_BUFFER,
                Anonymous: D3D11_UNORDERED_ACCESS_VIEW_DESC_0 {
                    Buffer: D3D11_BUFFER_UAV {
                        FirstElement: 0,
                        NumElements: num_elements as u32,
                        Flags: 0,
                    },
                },
            };

            let mut uav = None;
            unsafe {
                self.device.device().CreateUnorderedAccessView(
                    &buffer,
                    Some(&uav_desc),
                    Some(&mut uav as *mut _),
                )
            }
            .map_err(|e| anyhow::anyhow!("Failed to create D3D11 UAV: {e}"))?;
            let uav = uav.ok_or_else(|| anyhow::anyhow!("D3D11 CreateUAV returned null"))?;

            // Create SRV
            let srv_desc = D3D11_SHADER_RESOURCE_VIEW_DESC {
                Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_UNKNOWN,
                ViewDimension: D3D_SRV_DIMENSION_BUFFER,
                Anonymous: D3D11_SHADER_RESOURCE_VIEW_DESC_0 {
                    Buffer: D3D11_BUFFER_SRV {
                        Anonymous1: D3D11_BUFFER_SRV_0 {
                            FirstElement: 0,
                        },
                        Anonymous2: D3D11_BUFFER_SRV_1 {
                            NumElements: num_elements as u32,
                        },
                    },
                },
            };

            let mut srv = None;
            unsafe {
                self.device.device().CreateShaderResourceView(
                    &buffer,
                    Some(&srv_desc),
                    Some(&mut srv as *mut _),
                )
            }
            .map_err(|e| anyhow::anyhow!("Failed to create D3D11 SRV: {e}"))?;
            let srv = srv.ok_or_else(|| anyhow::anyhow!("D3D11 CreateSRV returned null"))?;

            Ok(GpuBuffer {
                size,
                dx11_buffer: buffer,
                dx11_uav: uav,
                dx11_srv: srv,
            })
        }

        /// Dispatch a compute shader on the immediate context.
        ///
        /// Binds the compute shader, UAVs, SRVs, and constant buffers, then
        /// dispatches the given number of thread groups.
        pub fn dispatch_compute(
            &self,
            pipeline: &ComputePipeline,
            uavs: &[Option<ID3D11UnorderedAccessView>],
            srvs: &[Option<ID3D11ShaderResourceView>],
            cbufs: &[Option<ID3D11Buffer>],
            thread_groups: (u32, u32, u32),
        ) {
            let ctx = self.device.context();
            unsafe {
                ctx.CSSetShader(&pipeline.shader, None);
                if !uavs.is_empty() {
                    ctx.CSSetUnorderedAccessViews(0, uavs, None);
                }
                if !srvs.is_empty() {
                    ctx.CSSetShaderResources(0, srvs);
                }
                if !cbufs.is_empty() {
                    ctx.CSSetConstantBuffers(0, cbufs);
                }
                ctx.Dispatch(thread_groups.0, thread_groups.1, thread_groups.2);
            }
        }

        /// Signal the GPU event query for synchronization.
        pub fn signal_gpu_query(&self) {
            unsafe {
                self.device.context().End(self.device.query());
            }
        }
    }
}
