//! GPU dispatch: pipeline creation, buffer allocation, and compute/render
//! command encoding.
//!
//! All pipeline creation and dispatch methods live on [`GpuContext`].

use crate::buffer::GpuBuffer;

// Used only by the cfg-gated `metal_impl` / `dx11_impl` modules below;
// gate to match so they don't warn-as-unused on a non-mac, non-windows
// host build.
#[cfg(any(target_os = "macos", target_os = "windows"))]
use anyhow::Result;
#[cfg(any(target_os = "macos", target_os = "windows"))]
use crate::bind_set::BindSet;
#[cfg(any(target_os = "macos", target_os = "windows"))]
use crate::context::GpuContext;
#[cfg(any(target_os = "macos", target_os = "windows"))]
use crate::pipeline::{ComputePipeline, RenderPipeline};

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
    //
    // SAFETY: All `encoder.set*` calls below are `unsafe` because objc2 marks
    // every method on its protocol-object wrappers as unsafe (the FFI
    // boundary). The Rust-side invariants — that the texture/sampler/buffer
    // is non-null and lives at least as long as the encoder — are upheld
    // because `&'a T` borrows guarantee non-null + outlives-this-call. The
    // one genuinely interesting cast is the `setBytes` block: see the
    // SAFETY comment above that loop.
    fn encode_compute_inner(
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        pipeline: &ComputePipeline,
        textures: &[&ProtocolObject<dyn MTLTexture>],
        samplers: &[&ProtocolObject<dyn MTLSamplerState>],
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

        for (i, samp) in samplers.iter().enumerate() {
            unsafe {
                encoder.setSamplerState_atIndex(Some(*samp), i);
            }
        }

        for (buf, idx) in buffers {
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&buf.metal), 0, *idx);
            }
        }

        // SAFETY: `setBytes_length_atIndex` takes a `*mut c_void` purely
        // because the Objective-C selector is non-const-qualified on the
        // pointer parameter; semantically the API is read-only — Metal
        // copies `length` bytes out of the pointer before returning, and
        // the encoder never retains the pointer past this call. So the
        // immutable-to-mutable cast (`as *mut _`) does not enable any
        // actual mutation. `NonNull::new_unchecked` is sound because
        // `data.as_ptr()` for a `&[u8]` is always non-null (slices
        // always reference at least one well-aligned byte, even when
        // empty).
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

    /// Encode a BindSet onto `encoder` then dispatch and end. Slot-aware:
    /// every binding lands at the slot the caller specified, including
    /// sparse slots (e.g. binding 7 with nothing at 0..6).
    //
    // SAFETY: same overall invariants as `encode_compute_inner`. Each
    // `encoder.set*_atIndex` call is FFI-unsafe but receives a non-null
    // borrowed reference whose lifetime outlives the encoder (enforced by
    // BindSet's lifetime parameter). The `setBytes` loop reuses the
    // immutable-to-mutable cast pattern justified there.
    fn encode_bind_set(
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        pipeline: &ComputePipeline,
        bindings: &BindSet<'_>,
        grid: (usize, usize),
        threadgroup: (usize, usize),
    ) {
        encoder.setComputePipelineState(&pipeline.state);

        // Metal does not have a separate UAV register class — both
        // `texture_read` and `texture_rw` map to the same texture-slot
        // space. Bind both tables; if the WGSL author used the same
        // slot for both, the rw view wins (last-set).
        for (slot, tex) in bindings.textures_ro.iter().enumerate() {
            if let Some(t) = tex {
                unsafe { encoder.setTexture_atIndex(Some(*t), slot); }
            }
        }
        for (slot, tex) in bindings.textures_rw.iter().enumerate() {
            if let Some(t) = tex {
                unsafe { encoder.setTexture_atIndex(Some(*t), slot); }
            }
        }

        for (slot, samp) in bindings.samplers.iter().enumerate() {
            if let Some(s) = samp {
                unsafe { encoder.setSamplerState_atIndex(Some(*s), slot); }
            }
        }

        // Metal unifies storage buffers and uniform buffers into a
        // single `[[buffer(N)]]` slot space — buffers_ro, buffers_rw,
        // and uniforms all bind via setBuffer/setBytes at their slot.
        for (slot, buf) in bindings.buffers_ro.iter().enumerate() {
            if let Some(b) = buf {
                unsafe { encoder.setBuffer_offset_atIndex(Some(&b.metal), 0, slot); }
            }
        }
        for (slot, buf) in bindings.buffers_rw.iter().enumerate() {
            if let Some(b) = buf {
                unsafe { encoder.setBuffer_offset_atIndex(Some(&b.metal), 0, slot); }
            }
        }

        // SAFETY: see `encode_compute_inner` — `setBytes_length_atIndex`
        // copies before returning, immutable cast is sound.
        for (slot, data) in bindings.uniforms.iter().enumerate() {
            if let Some(d) = data {
                unsafe {
                    encoder.setBytes_length_atIndex(
                        std::ptr::NonNull::new_unchecked(d.as_ptr() as *mut _),
                        d.len(),
                        slot,
                    );
                }
            }
        }

        let grid_size = MTLSize { width: grid.0, height: grid.1, depth: 1 };
        let tg_size = MTLSize { width: threadgroup.0, height: threadgroup.1, depth: 1 };
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
                // SAFETY: `colorAttachments()` returns a non-null
                // `MTLRenderPipelineColorAttachmentDescriptorArray` per
                // MTLRenderPipelineDescriptor invariants, and slot 0 is
                // always valid (the array is allocated to its declared
                // capacity). `objectAtIndexedSubscript` is unsafe purely
                // because objc2 marks it so for FFI hygiene.
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

            // Create fullscreen quad vertex buffer.
            //
            // SAFETY: `newBufferWithBytes_length_options` copies `quad_len`
            // bytes from the source pointer into a fresh GPU-resident
            // buffer before returning, so the immutable-to-mutable cast
            // does not enable any actual mutation. `NonNull::new_unchecked`
            // is sound because `quad_data.as_ptr()` references a non-null
            // stack-allocated `[[f32; 4]; 4]`.
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
        /// Textures and samplers are bound sequentially starting at index 0.
        /// Buffers and bytes are bound at their specified slot indices.
        #[deprecated(
            note = "use dispatch_compute_with(&BindSet) — slot-aware bindings, \
                    HLSL register-class collision detection, no parallel-array ceremony"
        )]
        #[allow(clippy::too_many_arguments)]
        pub fn dispatch_compute(
            &self,
            pipeline: &ComputePipeline,
            textures: &[&ProtocolObject<dyn MTLTexture>],
            samplers: &[&ProtocolObject<dyn MTLSamplerState>],
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
                &encoder, pipeline, textures, samplers, buffers, bytes, grid, threadgroup,
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

            // SAFETY: each `encoder.set*` / `colorAttachments` /
            // `drawPrimitives` call is FFI-unsafe per objc2 conventions,
            // not memory-unsafe. Borrowed references guarantee non-null +
            // outlive-this-call. The `setFragmentBytes` cast follows the
            // same justification as `encode_compute_inner`'s setBytes
            // loop: Metal copies before returning, immutable-to-mutable
            // cast is sound.
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
        /// Textures and samplers are bound sequentially starting at index 0.
        /// Buffers and bytes are bound at their specified slot indices.
        #[deprecated(
            note = "use encode_compute_pass_with(cb, pipeline, &BindSet, ...) — \
                    slot-aware bindings, HLSL register-class collision detection"
        )]
        #[allow(clippy::too_many_arguments)]
        pub fn encode_compute_pass(
            &self,
            cb: &CommandBuffer,
            pipeline: &ComputePipeline,
            textures: &[&ProtocolObject<dyn MTLTexture>],
            samplers: &[&ProtocolObject<dyn MTLSamplerState>],
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
                &encoder, pipeline, textures, samplers, buffers, bytes, grid, threadgroup,
            );

            Ok(())
        }

        /// Dispatch a compute pass using a [`BindSet`] for resource
        /// binding. Slot-aware on both texture and buffer indices, and
        /// hides the platform asymmetry between Metal's sequential
        /// bindings and DX11's parallel slot arrays.
        ///
        /// Same submission semantics as [`dispatch_compute`](Self::dispatch_compute).
        pub fn dispatch_compute_with(
            &self,
            pipeline: &ComputePipeline,
            bindings: &BindSet<'_>,
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

            encode_bind_set(&encoder, pipeline, bindings, grid, threadgroup);

            command_buffer.commit();
            Ok(PendingWork { command_buffer })
        }

        /// Encode a compute pass on an existing command buffer using a
        /// [`BindSet`] for resource binding. Same as
        /// [`encode_compute_pass`](Self::encode_compute_pass) but with
        /// the unified bind-set API.
        pub fn encode_compute_pass_with(
            &self,
            cb: &CommandBuffer,
            pipeline: &ComputePipeline,
            bindings: &BindSet<'_>,
            grid: (usize, usize),
            threadgroup: (usize, usize),
        ) -> Result<()> {
            let encoder = cb
                .inner
                .computeCommandEncoder()
                .ok_or_else(|| anyhow::anyhow!("Failed to create Metal compute encoder"))?;

            encode_bind_set(&encoder, pipeline, bindings, grid, threadgroup);

            Ok(())
        }

        /// Create a Metal sampler state with linear filtering and clamp-to-edge
        /// addressing — the standard "bilinear sampler" used by image-processing
        /// compute kernels. Returns a `Retained<...>` that owns one ref count;
        /// keep it alive (e.g. on the plugin instance) for as long as you want
        /// to dispatch with it bound.
        pub fn create_linear_clamp_sampler(
            &self,
        ) -> Result<objc2::rc::Retained<ProtocolObject<dyn MTLSamplerState>>> {
            let desc = MTLSamplerDescriptor::new();
            desc.setMinFilter(MTLSamplerMinMagFilter::Linear);
            desc.setMagFilter(MTLSamplerMinMagFilter::Linear);
            desc.setMipFilter(MTLSamplerMipFilter::NotMipmapped);
            desc.setSAddressMode(MTLSamplerAddressMode::ClampToEdge);
            desc.setTAddressMode(MTLSamplerAddressMode::ClampToEdge);
            desc.setRAddressMode(MTLSamplerAddressMode::ClampToEdge);
            self.device
                .device()
                .newSamplerStateWithDescriptor(&desc)
                .ok_or_else(|| anyhow::anyhow!("Failed to create Metal sampler state"))
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
            // SAFETY: every objc2 call below is FFI-unsafe rather than
            // memory-unsafe; same invariants as `dispatch_render` apply.
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
        /// TEXCOORD float2` layout and a linear/clamp sampler for pixel shader
        /// texture sampling.
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
        /// Binds the compute shader, UAVs, SRVs, samplers, and constant
        /// buffers, then dispatches enough thread groups to cover `grid`
        /// total threads with the given `threadgroup` size. Unbinds all CS
        /// resources after dispatch to prevent resource hazards in
        /// multi-pass scenarios.
        #[deprecated(
            note = "use dispatch_compute_with(&BindSet) — slot-aware bindings, \
                    HLSL register-class collision detection, no manual UAV/SRV padding"
        )]
        #[allow(clippy::too_many_arguments)]
        pub fn dispatch_compute(
            &self,
            pipeline: &ComputePipeline,
            uavs: &[Option<ID3D11UnorderedAccessView>],
            srvs: &[Option<ID3D11ShaderResourceView>],
            samplers: &[Option<ID3D11SamplerState>],
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
                    ctx.CSSetUnorderedAccessViews(
                        0,
                        uavs.len() as u32,
                        Some(uavs.as_ptr()),
                        None,
                    );
                }
                if !srvs.is_empty() {
                    ctx.CSSetShaderResources(0, Some(srvs));
                }
                if !samplers.is_empty() {
                    ctx.CSSetSamplers(0, Some(samplers));
                }
                if !cbufs.is_empty() {
                    ctx.CSSetConstantBuffers(0, Some(cbufs));
                }
                ctx.Dispatch(groups_x, groups_y, 1);

                // Unbind all CS resources to prevent hazards when the same
                // texture is used as SRV in a subsequent pass.
                let null_uavs: [Option<ID3D11UnorderedAccessView>; 8] = Default::default();
                let null_srvs: [Option<ID3D11ShaderResourceView>; 8] = Default::default();
                let null_samplers: [Option<ID3D11SamplerState>; 4] = Default::default();
                let null_cbufs: [Option<ID3D11Buffer>; 1] = Default::default();
                ctx.CSSetUnorderedAccessViews(
                    0,
                    null_uavs.len() as u32,
                    Some(null_uavs.as_ptr()),
                    None,
                );
                ctx.CSSetShaderResources(0, Some(&null_srvs));
                ctx.CSSetSamplers(0, Some(&null_samplers));
                ctx.CSSetConstantBuffers(0, Some(&null_cbufs));
            }
        }

        /// Dispatch a compute shader using a [`BindSet`] for resource
        /// binding. Same submission semantics as
        /// [`dispatch_compute`](Self::dispatch_compute) but the
        /// platform-specific UAV/SRV slot packing happens internally.
        pub fn dispatch_compute_with(
            &self,
            pipeline: &ComputePipeline,
            bindings: &BindSet<'_>,
            grid: (usize, usize),
            threadgroup: (usize, usize),
        ) {
            // Pack BindSet's sparse tables into the dense `Vec<Option<View>>`
            // form CSSet*(0, ...) expects. Slot indexes are preserved one-to-one;
            // unset slots become `None`.
            let uav_len = bindings.textures_rw.len().max(bindings.buffers_rw.len());
            let mut uavs: Vec<Option<ID3D11UnorderedAccessView>> = vec![None; uav_len];
            for (slot, tex) in bindings.textures_rw.iter().enumerate() {
                if let Some(t) = tex { uavs[slot] = Some((*t).clone()); }
            }
            for (slot, buf) in bindings.buffers_rw.iter().enumerate() {
                if let Some(b) = buf { uavs[slot] = Some(b.dx11_uav.clone()); }
            }

            let srv_len = bindings.textures_ro.len().max(bindings.buffers_ro.len());
            let mut srvs: Vec<Option<ID3D11ShaderResourceView>> = vec![None; srv_len];
            for (slot, tex) in bindings.textures_ro.iter().enumerate() {
                if let Some(t) = tex { srvs[slot] = Some((*t).clone()); }
            }
            for (slot, buf) in bindings.buffers_ro.iter().enumerate() {
                if let Some(b) = buf { srvs[slot] = Some(b.dx11_srv.clone()); }
            }

            let mut samplers: Vec<Option<ID3D11SamplerState>> = vec![None; bindings.samplers.len()];
            for (slot, samp) in bindings.samplers.iter().enumerate() {
                if let Some(s) = samp { samplers[slot] = Some((*s).clone()); }
            }

            let mut cbufs: Vec<Option<ID3D11Buffer>> = vec![None; bindings.uniforms.len()];
            for (slot, cb) in bindings.uniforms.iter().enumerate() {
                if let Some(b) = cb { cbufs[slot] = Some((*b).clone()); }
            }

            // Internal use of the deprecated entry-point — `BindSet` flattens
            // to its parallel-array shape and we just hand them off.
            #[allow(deprecated)]
            self.dispatch_compute(pipeline, &uavs, &srvs, &samplers, &cbufs, grid, threadgroup);
        }

        /// Create a D3D11 sampler with linear filtering and clamp addressing.
        pub fn create_linear_clamp_sampler(&self) -> Result<ID3D11SamplerState> {
            let desc = D3D11_SAMPLER_DESC {
                Filter: D3D11_FILTER_MIN_MAG_MIP_LINEAR,
                AddressU: D3D11_TEXTURE_ADDRESS_CLAMP,
                AddressV: D3D11_TEXTURE_ADDRESS_CLAMP,
                AddressW: D3D11_TEXTURE_ADDRESS_CLAMP,
                MipLODBias: 0.0,
                MaxAnisotropy: 1,
                ComparisonFunc: D3D11_COMPARISON_NEVER,
                BorderColor: [0.0; 4],
                MinLOD: 0.0,
                MaxLOD: f32::MAX,
            };
            let mut sampler = None;
            unsafe {
                self.device
                    .device()
                    .CreateSamplerState(&desc, Some(&mut sampler))
            }
            .map_err(|e| anyhow::anyhow!("Failed to create D3D11 sampler: {e}"))?;
            sampler.ok_or_else(|| anyhow::anyhow!("CreateSamplerState returned null"))
        }

        /// Dispatch a fullscreen render pass using the given render pipeline.
        ///
        /// Creates a temporary render target view from `output_texture`, sets
        /// up the viewport, draws a fullscreen quad, and unbinds all resources
        /// afterward to prevent hazards.
        #[allow(clippy::too_many_arguments)]
        pub fn dispatch_render(
            &self,
            pipeline: &RenderPipeline,
            output_texture: &ID3D11Texture2D,
            pixel_srvs: &[Option<ID3D11ShaderResourceView>],
            pixel_cbufs: &[Option<ID3D11Buffer>],
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

                // Pixel shader
                ctx.PSSetShader(&pipeline.ps, None);
                if !pixel_srvs.is_empty() {
                    ctx.PSSetShaderResources(0, Some(pixel_srvs));
                }
                if !pixel_cbufs.is_empty() {
                    ctx.PSSetConstantBuffers(0, Some(pixel_cbufs));
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
                ctx.PSSetShaderResources(0, Some(&null_srvs));
                let null_cbufs: [Option<ID3D11Buffer>; 1] = Default::default();
                ctx.PSSetConstantBuffers(0, Some(&null_cbufs));
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
