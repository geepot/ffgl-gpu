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

/// A command buffer for encoding multiple passes.
///
/// On macOS this wraps a `MTLCommandBuffer`. Create one with
/// [`GpuContext::create_command_buffer`], encode compute and render passes
/// on it, then call [`GpuContext::commit`] to submit all work at once.
///
/// Metal automatically serialises encoders on the same command buffer, so
/// there is no need for mid-frame `PendingWork::wait()` calls between passes.
pub struct CommandBuffer {
    #[cfg(target_os = "macos")]
    pub(crate) inner:
        objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandBuffer>>,
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

    /// Encode a compute dispatch onto `encoder`: set pipeline, bind resources,
    /// dispatch threads, and end the encoder.
    fn encode_compute_inner(
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        pipeline: &ComputePipeline,
        textures: &[&ProtocolObject<dyn MTLTexture>],
        buffers: &[(&GpuBuffer, usize)],
        bytes: &[(&[u8], usize)],
        grid: (usize, usize),
        threadgroup: (usize, usize),
    ) {
        encoder.setComputePipelineState(&pipeline.state);

        for (i, tex) in textures.iter().enumerate() {
            unsafe {
                encoder.setTexture_atIndex(Some(*tex), i);
            }
        }

        for (buf, idx) in buffers {
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&buf.metal), 0, *idx);
            }
        }

        for (data, idx) in bytes {
            unsafe {
                encoder.setBytes_length_atIndex(
                    std::ptr::NonNull::new_unchecked(data.as_ptr() as *mut _),
                    data.len(),
                    *idx,
                );
            }
        }

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
        encoder.dispatchThreads_threadsPerThreadgroup(grid_size, tg_size);
        encoder.endEncoding();
    }

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

        /// Dispatch a single compute pass: create a command buffer, encode
        /// the pipeline with all bindings, dispatch, commit, and return a
        /// [`PendingWork`] token.
        ///
        /// Textures are bound sequentially starting at index 0. Buffers and
        /// bytes are bound at their specified slot indices.
        pub fn dispatch_compute(
            &self,
            pipeline: &ComputePipeline,
            textures: &[&ProtocolObject<dyn MTLTexture>],
            buffers: &[(&GpuBuffer, usize)],
            bytes: &[(&[u8], usize)],
            grid: (usize, usize),
            threadgroup: (usize, usize),
        ) -> Result<PendingWork> {
            let command_buffer = self
                .device
                .command_queue()
                .commandBuffer()
                .ok_or_else(|| anyhow::anyhow!("Failed to create Metal command buffer"))?;

            let encoder = command_buffer
                .computeCommandEncoder()
                .ok_or_else(|| anyhow::anyhow!("Failed to create Metal compute encoder"))?;

            encode_compute_inner(
                &encoder, pipeline, textures, buffers, bytes, grid, threadgroup,
            );

            command_buffer.commit();
            Ok(PendingWork { command_buffer })
        }

        /// Dispatch a fullscreen render pass: renders a quad using the given
        /// render pipeline with the output texture as the render target and
        /// input textures bound to fragment shader slots.
        ///
        /// Returns a [`PendingWork`] token for synchronization.
        pub fn dispatch_render(
            &self,
            pipeline: &RenderPipeline,
            output_texture: &ProtocolObject<dyn MTLTexture>,
            fragment_textures: &[&ProtocolObject<dyn MTLTexture>],
            fragment_bytes: &[(&[u8], usize)],
        ) -> Result<PendingWork> {
            let command_buffer = self
                .device
                .command_queue()
                .commandBuffer()
                .ok_or_else(|| anyhow::anyhow!("Failed to create command buffer for render"))?;

            let render_desc = MTLRenderPassDescriptor::new();
            {
                let attachment = unsafe {
                    render_desc
                        .colorAttachments()
                        .objectAtIndexedSubscript(0)
                };
                attachment.setTexture(Some(output_texture));
                attachment.setLoadAction(MTLLoadAction::DontCare);
                attachment.setStoreAction(MTLStoreAction::Store);
            }

            let encoder = command_buffer
                .renderCommandEncoderWithDescriptor(&render_desc)
                .ok_or_else(|| anyhow::anyhow!("Failed to create render encoder"))?;

            encoder.setRenderPipelineState(&pipeline.state);

            // Bind fullscreen quad vertex buffer at index 0
            unsafe {
                encoder.setVertexBuffer_offset_atIndex(Some(&pipeline.quad_vb), 0, 0);
            }

            // Bind fragment textures
            for (i, tex) in fragment_textures.iter().enumerate() {
                unsafe {
                    encoder.setFragmentTexture_atIndex(Some(*tex), i);
                }
            }

            // Bind fragment constant data
            for (data, index) in fragment_bytes {
                unsafe {
                    encoder.setFragmentBytes_length_atIndex(
                        std::ptr::NonNull::new_unchecked(data.as_ptr() as *mut _),
                        data.len(),
                        *index,
                    );
                }
            }

            // Draw fullscreen quad as triangle strip (4 vertices)
            unsafe {
                encoder
                    .drawPrimitives_vertexStart_vertexCount(MTLPrimitiveType::TriangleStrip, 0, 4);
            }

            encoder.endEncoding();
            command_buffer.commit();

            Ok(PendingWork {
                command_buffer,
            })
        }

        // =================================================================
        // Multi-pass command buffer API
        // =================================================================

        /// Create a command buffer for encoding multiple passes.
        ///
        /// Use [`begin_compute_encoder`](Self::begin_compute_encoder),
        /// [`encode_render_pass`](Self::encode_render_pass), and
        /// [`commit`](Self::commit) to build and submit a multi-pass
        /// pipeline in a single GPU submission with zero mid-frame stalls.
        pub fn create_command_buffer(&self) -> Result<CommandBuffer> {
            let inner = self
                .device
                .command_queue()
                .commandBuffer()
                .ok_or_else(|| anyhow::anyhow!("Failed to create Metal command buffer"))?;
            Ok(CommandBuffer { inner })
        }

        /// Encode a compute pass on an existing command buffer.
        ///
        /// Same encoding as [`dispatch_compute`](Self::dispatch_compute) but
        /// targets the given [`CommandBuffer`] instead of creating a new one.
        /// Call [`commit`](Self::commit) after encoding all passes.
        ///
        /// Textures are bound sequentially starting at index 0. Buffers and
        /// bytes are bound at their specified slot indices.
        pub fn encode_compute_pass(
            &self,
            cb: &CommandBuffer,
            pipeline: &ComputePipeline,
            textures: &[&ProtocolObject<dyn MTLTexture>],
            buffers: &[(&GpuBuffer, usize)],
            bytes: &[(&[u8], usize)],
            grid: (usize, usize),
            threadgroup: (usize, usize),
        ) -> Result<()> {
            let encoder = cb
                .inner
                .computeCommandEncoder()
                .ok_or_else(|| anyhow::anyhow!("Failed to create Metal compute encoder"))?;

            encode_compute_inner(
                &encoder, pipeline, textures, buffers, bytes, grid, threadgroup,
            );

            Ok(())
        }

        /// Encode a fullscreen render pass on an existing command buffer.
        ///
        /// Same rendering as [`dispatch_render`](Self::dispatch_render) but
        /// encodes onto the given [`CommandBuffer`] instead of creating a
        /// new one, avoiding an extra GPU submission boundary.
        pub fn encode_render_pass(
            &self,
            cb: &CommandBuffer,
            pipeline: &RenderPipeline,
            output_texture: &ProtocolObject<dyn MTLTexture>,
            fragment_textures: &[&ProtocolObject<dyn MTLTexture>],
            fragment_bytes: &[(&[u8], usize)],
        ) -> Result<()> {
            let render_desc = MTLRenderPassDescriptor::new();
            {
                let attachment = unsafe {
                    render_desc
                        .colorAttachments()
                        .objectAtIndexedSubscript(0)
                };
                attachment.setTexture(Some(output_texture));
                attachment.setLoadAction(MTLLoadAction::DontCare);
                attachment.setStoreAction(MTLStoreAction::Store);
            }

            let encoder = cb
                .inner
                .renderCommandEncoderWithDescriptor(&render_desc)
                .ok_or_else(|| anyhow::anyhow!("Failed to create render encoder"))?;

            encoder.setRenderPipelineState(&pipeline.state);

            // Bind fullscreen quad vertex buffer at index 0
            unsafe {
                encoder.setVertexBuffer_offset_atIndex(Some(&pipeline.quad_vb), 0, 0);
            }

            // Bind fragment textures
            for (i, tex) in fragment_textures.iter().enumerate() {
                unsafe {
                    encoder.setFragmentTexture_atIndex(Some(*tex), i);
                }
            }

            // Bind fragment constant data
            for (data, index) in fragment_bytes {
                unsafe {
                    encoder.setFragmentBytes_length_atIndex(
                        std::ptr::NonNull::new_unchecked(data.as_ptr() as *mut _),
                        data.len(),
                        *index,
                    );
                }
            }

            // Draw fullscreen quad as triangle strip (4 vertices)
            unsafe {
                encoder.drawPrimitives_vertexStart_vertexCount(
                    MTLPrimitiveType::TriangleStrip,
                    0,
                    4,
                );
            }

            encoder.endEncoding();
            Ok(())
        }

        /// Commit a command buffer and return a [`PendingWork`] token.
        ///
        /// Call this after encoding all passes. The returned token can be
        /// stored via
        /// [`GlMetalBridge::store_command_buffer`](gpu_interop::metal::GlMetalBridge::store_command_buffer)
        /// for pipelined double-buffered synchronisation.
        pub fn commit(&self, cb: CommandBuffer) -> PendingWork {
            cb.inner.commit();
            PendingWork {
                command_buffer: cb.inner,
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
    use windows::core::PCSTR;
    use windows::Win32::Graphics::Direct3D::D3D_SRV_DIMENSION_BUFFER;
    use windows::Win32::Graphics::Direct3D11::*;
    use windows::Win32::Graphics::Dxgi::Common::*;

    /// Fullscreen quad vertex data: 4 vertices, each with (x, y, u, v).
    /// Triangle-strip order: bottom-left, bottom-right, top-left, top-right.
    const FULLSCREEN_QUAD: [[f32; 4]; 4] = [
        [-1.0, -1.0, 0.0, 1.0], // bottom-left  (pos.xy, uv)
        [1.0, -1.0, 1.0, 1.0],  // bottom-right
        [-1.0, 1.0, 0.0, 0.0],  // top-left
        [1.0, 1.0, 1.0, 0.0],   // top-right
    ];

    impl GpuContext {
        /// Create a compute pipeline from pre-compiled HLSL bytecode (`.cso`).
        pub fn create_compute_pipeline(
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

        /// Create a render pipeline from pre-compiled HLSL vertex and pixel
        /// shader bytecode (`.cso`).
        ///
        /// Sets up a fullscreen quad vertex buffer with `POSITION float2 +
        /// TEXCOORD float2` layout and a linear/clamp sampler for fragment
        /// shader texture sampling.
        pub fn create_render_pipeline(
            &self,
            vs_bytecode: &[u8],
            ps_bytecode: &[u8],
        ) -> Result<RenderPipeline> {
            let device = self.device.device();

            // Create vertex shader
            let mut vs = None;
            unsafe { device.CreateVertexShader(vs_bytecode, None, Some(&mut vs as *mut _)) }
                .map_err(|e| anyhow::anyhow!("Failed to create D3D11 vertex shader: {e}"))?;
            let vs = vs.ok_or_else(|| anyhow::anyhow!("D3D11 CreateVertexShader returned null"))?;

            // Create pixel shader
            let mut ps = None;
            unsafe { device.CreatePixelShader(ps_bytecode, None, Some(&mut ps as *mut _)) }
                .map_err(|e| anyhow::anyhow!("Failed to create D3D11 pixel shader: {e}"))?;
            let ps = ps.ok_or_else(|| anyhow::anyhow!("D3D11 CreatePixelShader returned null"))?;

            // Create input layout: POSITION float2 + TEXCOORD float2
            let input_elements = [
                D3D11_INPUT_ELEMENT_DESC {
                    SemanticName: PCSTR(b"POSITION\0".as_ptr()),
                    SemanticIndex: 0,
                    Format: DXGI_FORMAT_R32G32_FLOAT,
                    InputSlot: 0,
                    AlignedByteOffset: 0,
                    InputSlotClass: D3D11_INPUT_PER_VERTEX_DATA,
                    InstanceDataStepRate: 0,
                },
                D3D11_INPUT_ELEMENT_DESC {
                    SemanticName: PCSTR(b"TEXCOORD\0".as_ptr()),
                    SemanticIndex: 0,
                    Format: DXGI_FORMAT_R32G32_FLOAT,
                    InputSlot: 0,
                    AlignedByteOffset: 8,
                    InputSlotClass: D3D11_INPUT_PER_VERTEX_DATA,
                    InstanceDataStepRate: 0,
                },
            ];

            let mut input_layout = None;
            unsafe {
                device.CreateInputLayout(
                    &input_elements,
                    vs_bytecode,
                    Some(&mut input_layout as *mut _),
                )
            }
            .map_err(|e| anyhow::anyhow!("Failed to create D3D11 input layout: {e}"))?;
            let input_layout = input_layout
                .ok_or_else(|| anyhow::anyhow!("D3D11 CreateInputLayout returned null"))?;

            // Create fullscreen quad vertex buffer
            let quad_data = FULLSCREEN_QUAD;
            let vb_desc = D3D11_BUFFER_DESC {
                ByteWidth: std::mem::size_of_val(&quad_data) as u32,
                Usage: D3D11_USAGE_IMMUTABLE,
                BindFlags: D3D11_BIND_VERTEX_BUFFER.0 as u32,
                ..Default::default()
            };
            let vb_init = D3D11_SUBRESOURCE_DATA {
                pSysMem: quad_data.as_ptr() as *const _,
                ..Default::default()
            };
            let mut quad_vb = None;
            unsafe {
                device.CreateBuffer(&vb_desc, Some(&vb_init), Some(&mut quad_vb as *mut _))
            }
            .map_err(|e| anyhow::anyhow!("Failed to create fullscreen quad VB: {e}"))?;
            let quad_vb =
                quad_vb.ok_or_else(|| anyhow::anyhow!("D3D11 CreateBuffer(VB) returned null"))?;

            // Create linear/clamp sampler
            let sampler_desc = D3D11_SAMPLER_DESC {
                Filter: D3D11_FILTER_MIN_MAG_MIP_LINEAR,
                AddressU: D3D11_TEXTURE_ADDRESS_CLAMP,
                AddressV: D3D11_TEXTURE_ADDRESS_CLAMP,
                AddressW: D3D11_TEXTURE_ADDRESS_CLAMP,
                MaxAnisotropy: 1,
                ComparisonFunc: D3D11_COMPARISON_NEVER,
                MinLOD: 0.0,
                MaxLOD: f32::MAX,
                ..Default::default()
            };
            let mut sampler = None;
            unsafe {
                device.CreateSamplerState(&sampler_desc, Some(&mut sampler as *mut _))
            }
            .map_err(|e| anyhow::anyhow!("Failed to create D3D11 sampler: {e}"))?;
            let sampler =
                sampler.ok_or_else(|| anyhow::anyhow!("D3D11 CreateSamplerState returned null"))?;

            Ok(RenderPipeline {
                vs,
                ps,
                input_layout,
                quad_vb,
                sampler,
            })
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
                Format: DXGI_FORMAT_UNKNOWN,
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
                Format: DXGI_FORMAT_UNKNOWN,
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
        /// dispatches enough thread groups to cover `grid` total threads with
        /// the given `threadgroup` size. Unbinds all CS resources after dispatch
        /// to prevent resource hazards in multi-pass scenarios.
        pub fn dispatch_compute(
            &self,
            pipeline: &ComputePipeline,
            uavs: &[Option<ID3D11UnorderedAccessView>],
            srvs: &[Option<ID3D11ShaderResourceView>],
            cbufs: &[Option<ID3D11Buffer>],
            grid: (usize, usize),
            threadgroup: (usize, usize),
        ) {
            let groups_x = ((grid.0 + threadgroup.0 - 1) / threadgroup.0) as u32;
            let groups_y = ((grid.1 + threadgroup.1 - 1) / threadgroup.1) as u32;

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
                ctx.Dispatch(groups_x, groups_y, 1);

                // Unbind all CS resources to prevent hazards when the same
                // texture is used as SRV in a subsequent pass.
                let null_uavs: [Option<ID3D11UnorderedAccessView>; 8] = Default::default();
                let null_srvs: [Option<ID3D11ShaderResourceView>; 8] = Default::default();
                let null_cbufs: [Option<ID3D11Buffer>; 1] = Default::default();
                ctx.CSSetUnorderedAccessViews(0, &null_uavs, None);
                ctx.CSSetShaderResources(0, &null_srvs);
                ctx.CSSetConstantBuffers(0, &null_cbufs);
            }
        }

        /// Dispatch a fullscreen render pass using the given render pipeline.
        ///
        /// Creates a temporary render target view from `output_texture`, sets
        /// up the viewport, draws a fullscreen quad, and unbinds all resources
        /// afterward to prevent hazards.
        pub fn dispatch_render(
            &self,
            pipeline: &RenderPipeline,
            output_texture: &ID3D11Texture2D,
            fragment_srvs: &[Option<ID3D11ShaderResourceView>],
            fragment_cbufs: &[Option<ID3D11Buffer>],
        ) -> Result<()> {
            let device = self.device.device();
            let ctx = self.device.context();

            // Query texture dimensions for viewport
            let mut desc = D3D11_TEXTURE2D_DESC::default();
            unsafe { output_texture.GetDesc(&mut desc) };

            // Create temporary RTV
            let mut rtv = None;
            unsafe {
                device.CreateRenderTargetView(output_texture, None, Some(&mut rtv as *mut _))
            }
            .map_err(|e| anyhow::anyhow!("Failed to create RTV for render dispatch: {e}"))?;
            let rtv = rtv.ok_or_else(|| anyhow::anyhow!("D3D11 CreateRTV returned null"))?;

            unsafe {
                // Set viewport
                let viewport = D3D11_VIEWPORT {
                    TopLeftX: 0.0,
                    TopLeftY: 0.0,
                    Width: desc.Width as f32,
                    Height: desc.Height as f32,
                    MinDepth: 0.0,
                    MaxDepth: 1.0,
                };
                ctx.RSSetViewports(Some(&[viewport]));

                // Input assembler
                ctx.IASetInputLayout(&pipeline.input_layout);
                let stride = std::mem::size_of::<[f32; 4]>() as u32;
                let offset = 0u32;
                ctx.IASetVertexBuffers(
                    0,
                    1,
                    Some(&Some(pipeline.quad_vb.clone())),
                    Some(&stride),
                    Some(&offset),
                );
                ctx.IASetPrimitiveTopology(
                    windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP,
                );

                // Vertex shader
                ctx.VSSetShader(&pipeline.vs, None);

                // Fragment (pixel) shader
                ctx.PSSetShader(&pipeline.ps, None);
                if !fragment_srvs.is_empty() {
                    ctx.PSSetShaderResources(0, fragment_srvs);
                }
                if !fragment_cbufs.is_empty() {
                    ctx.PSSetConstantBuffers(0, fragment_cbufs);
                }
                ctx.PSSetSamplers(0, Some(&[Some(pipeline.sampler.clone())]));

                // Output merger
                ctx.OMSetRenderTargets(Some(&[Some(rtv)]), None);

                // Draw fullscreen quad
                ctx.Draw(4, 0);

                // Unbind render target and PS SRVs to prevent resource hazards
                let null_rtvs: [Option<ID3D11RenderTargetView>; 1] = Default::default();
                ctx.OMSetRenderTargets(Some(&null_rtvs), None);
                let null_srvs: [Option<ID3D11ShaderResourceView>; 8] = Default::default();
                ctx.PSSetShaderResources(0, &null_srvs);
                let null_cbufs: [Option<ID3D11Buffer>; 1] = Default::default();
                ctx.PSSetConstantBuffers(0, &null_cbufs);
            }

            Ok(())
        }

        /// Map a dynamic constant buffer, copy data into it, and unmap.
        ///
        /// The buffer must have been created with `D3D11_USAGE_DYNAMIC` and
        /// `D3D11_CPU_ACCESS_WRITE` (e.g. via
        /// [`create_dynamic_cbuf`](gpu_interop::dx11::create_dynamic_cbuf)).
        pub fn update_constant_buffer(&self, buffer: &ID3D11Buffer, data: &[u8]) {
            let ctx = self.device.context();
            unsafe {
                let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
                let hr = ctx.Map(buffer, 0, D3D11_MAP_WRITE_DISCARD, 0, Some(&mut mapped));
                if hr.is_ok() {
                    debug_assert!(
                        data.len() <= mapped.RowPitch as usize,
                        "update_constant_buffer: data ({} bytes) exceeds mapped region ({} bytes)",
                        data.len(),
                        mapped.RowPitch,
                    );
                    std::ptr::copy_nonoverlapping(data.as_ptr(), mapped.pData as *mut u8, data.len());
                    ctx.Unmap(buffer, 0);
                }
            }
        }
    }
}
