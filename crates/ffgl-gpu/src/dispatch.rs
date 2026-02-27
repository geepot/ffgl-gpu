//! GPU dispatch: pipeline creation, buffer allocation, and compute/render
//! command encoding.
//!
//! All pipeline creation and dispatch methods live on [`GpuContext`].

use anyhow::Result;

use crate::buffer::GpuBuffer;
use crate::context::GpuContext;
use crate::pipeline::{ComputePipeline, RenderPipeline};
use crate::texture::GpuTexture;

// ---------------------------------------------------------------------------
// Command buffer + pending work types
// ---------------------------------------------------------------------------

/// A command buffer for encoding multiple passes.
///
/// On macOS this wraps a `MTLCommandBuffer`. On other platforms it is a
/// thin marker — dispatches execute immediately and [`GpuContext::commit`]
/// inserts a GL fence.
///
/// Create one with [`GpuContext::create_command_buffer`], encode compute and
/// render passes on it, then call [`GpuContext::commit`] to submit all work
/// at once.
pub struct CommandBuffer {
    #[cfg(target_os = "macos")]
    pub(crate) inner:
        objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandBuffer>>,

    #[cfg(not(target_os = "macos"))]
    pub(crate) _marker: (),
}

/// A token representing GPU work that has been submitted but may not yet be
/// complete.
pub struct PendingWork {
    #[cfg(target_os = "macos")]
    pub(crate) command_buffer:
        objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandBuffer>>,

    #[cfg(not(target_os = "macos"))]
    pub(crate) fence: gl::types::GLsync,
}

// SAFETY: GL sync objects are thread-safe handles.
#[cfg(not(target_os = "macos"))]
unsafe impl Send for PendingWork {}
#[cfg(not(target_os = "macos"))]
unsafe impl Sync for PendingWork {}

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

#[cfg(not(target_os = "macos"))]
impl PendingWork {
    /// Block until the GPU work completes.
    pub fn wait(&self) {
        unsafe {
            gl::ClientWaitSync(self.fence, gl::SYNC_FLUSH_COMMANDS_BIT, u64::MAX);
        }
    }
}

#[cfg(not(target_os = "macos"))]
impl Drop for PendingWork {
    fn drop(&mut self) {
        unsafe {
            gl::DeleteSync(self.fence);
        }
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
        textures: &[&GpuTexture],
        buffers: &[(&GpuBuffer, usize)],
        bytes: &[(&[u8], usize)],
        grid: (usize, usize),
        threadgroup: (usize, usize),
    ) {
        encoder.setComputePipelineState(&pipeline.state);

        for (i, tex) in textures.iter().enumerate() {
            unsafe {
                encoder.setTexture_atIndex(Some(tex.metal_ref()), i);
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

    /// Encode a fullscreen render pass onto `encoder`.
    fn encode_render_inner(
        cb: &ProtocolObject<dyn MTLCommandBuffer>,
        pipeline: &RenderPipeline,
        output: &GpuTexture,
        fragment_textures: &[&GpuTexture],
        fragment_bytes: &[(&[u8], usize)],
    ) -> Result<()> {
        let render_desc = MTLRenderPassDescriptor::new();
        {
            let attachment = unsafe {
                render_desc
                    .colorAttachments()
                    .objectAtIndexedSubscript(0)
            };
            attachment.setTexture(Some(output.metal_ref()));
            attachment.setLoadAction(MTLLoadAction::DontCare);
            attachment.setStoreAction(MTLStoreAction::Store);
        }

        let encoder = cb
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
                encoder.setFragmentTexture_atIndex(Some(tex.metal_ref()), i);
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
        Ok(())
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

        /// Dispatch a single compute pass and return a [`PendingWork`] token.
        ///
        /// Textures are bound sequentially starting at index 0. Buffers and
        /// bytes are bound at their specified slot indices.
        pub fn dispatch_compute(
            &self,
            pipeline: &ComputePipeline,
            textures: &[&GpuTexture],
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

        /// Dispatch a fullscreen render pass and return a [`PendingWork`] token.
        pub fn dispatch_render(
            &self,
            pipeline: &RenderPipeline,
            output: &GpuTexture,
            fragment_textures: &[&GpuTexture],
            fragment_bytes: &[(&[u8], usize)],
        ) -> Result<PendingWork> {
            let command_buffer = self
                .device
                .command_queue()
                .commandBuffer()
                .ok_or_else(|| anyhow::anyhow!("Failed to create command buffer for render"))?;

            encode_render_inner(
                &command_buffer,
                pipeline,
                output,
                fragment_textures,
                fragment_bytes,
            )?;

            command_buffer.commit();
            Ok(PendingWork { command_buffer })
        }

        // =================================================================
        // Multi-pass command buffer API
        // =================================================================

        /// Create a command buffer for encoding multiple passes.
        pub fn create_command_buffer(&self) -> Result<CommandBuffer> {
            let inner = self
                .device
                .command_queue()
                .commandBuffer()
                .ok_or_else(|| anyhow::anyhow!("Failed to create Metal command buffer"))?;
            Ok(CommandBuffer { inner })
        }

        /// Encode a compute pass on an existing command buffer.
        pub fn encode_compute_pass(
            &self,
            cb: &CommandBuffer,
            pipeline: &ComputePipeline,
            textures: &[&GpuTexture],
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
        pub fn encode_render_pass(
            &self,
            cb: &CommandBuffer,
            pipeline: &RenderPipeline,
            output: &GpuTexture,
            fragment_textures: &[&GpuTexture],
            fragment_bytes: &[(&[u8], usize)],
        ) -> Result<()> {
            encode_render_inner(
                &cb.inner,
                pipeline,
                output,
                fragment_textures,
                fragment_bytes,
            )
        }

        /// Commit a command buffer and return a [`PendingWork`] token.
        pub fn commit(&self, cb: CommandBuffer) -> PendingWork {
            cb.inner.commit();
            PendingWork {
                command_buffer: cb.inner,
            }
        }
    }
}

// ---------------------------------------------------------------------------
// OpenGL 4.6 compute implementation (non-macOS)
// ---------------------------------------------------------------------------

#[cfg(not(target_os = "macos"))]
mod gl_impl {
    use super::*;
    use std::ffi::CString;

    /// Fullscreen quad vertex data: 4 vertices, each with (x, y, u, v).
    /// Triangle-strip order: bottom-left, bottom-right, top-left, top-right.
    const FULLSCREEN_QUAD: [[f32; 4]; 4] = [
        [-1.0, -1.0, 0.0, 1.0], // bottom-left  (pos.xy, uv)
        [1.0, -1.0, 1.0, 1.0],  // bottom-right
        [-1.0, 1.0, 0.0, 0.0],  // top-left
        [1.0, 1.0, 1.0, 0.0],   // top-right
    ];

    /// Compile a GLSL shader source and return a linked program.
    fn compile_shader_program(
        shader_type: gl::types::GLenum,
        source: &str,
    ) -> Result<gl::types::GLuint> {
        unsafe {
            let shader = gl::CreateShader(shader_type);
            if shader == 0 {
                anyhow::bail!("glCreateShader returned 0");
            }

            let c_source = CString::new(source)
                .map_err(|_| anyhow::anyhow!("Shader source contains null byte"))?;
            let sources = [c_source.as_ptr()];
            gl::ShaderSource(shader, 1, sources.as_ptr(), std::ptr::null());
            gl::CompileShader(shader);

            let mut success = 0i32;
            gl::GetShaderiv(shader, gl::COMPILE_STATUS, &mut success);
            if success == 0 {
                let mut len = 0i32;
                gl::GetShaderiv(shader, gl::INFO_LOG_LENGTH, &mut len);
                let mut log = vec![0u8; len as usize];
                gl::GetShaderInfoLog(shader, len, std::ptr::null_mut(), log.as_mut_ptr() as *mut _);
                let msg = String::from_utf8_lossy(&log);
                gl::DeleteShader(shader);
                anyhow::bail!("Shader compilation failed: {msg}");
            }

            let program = gl::CreateProgram();
            gl::AttachShader(program, shader);
            gl::LinkProgram(program);

            let mut link_success = 0i32;
            gl::GetProgramiv(program, gl::LINK_STATUS, &mut link_success);
            if link_success == 0 {
                let mut len = 0i32;
                gl::GetProgramiv(program, gl::INFO_LOG_LENGTH, &mut len);
                let mut log = vec![0u8; len as usize];
                gl::GetProgramInfoLog(
                    program,
                    len,
                    std::ptr::null_mut(),
                    log.as_mut_ptr() as *mut _,
                );
                let msg = String::from_utf8_lossy(&log);
                gl::DeleteShader(shader);
                gl::DeleteProgram(program);
                anyhow::bail!("Program link failed: {msg}");
            }

            gl::DeleteShader(shader);
            Ok(program)
        }
    }

    /// Compile a vertex + fragment shader pair and return a linked program.
    fn compile_render_program(
        vs_source: &str,
        fs_source: &str,
    ) -> Result<gl::types::GLuint> {
        unsafe {
            let vs = gl::CreateShader(gl::VERTEX_SHADER);
            let fs = gl::CreateShader(gl::FRAGMENT_SHADER);
            if vs == 0 || fs == 0 {
                anyhow::bail!("glCreateShader returned 0");
            }

            // Compile vertex shader
            let c_vs = CString::new(vs_source)
                .map_err(|_| anyhow::anyhow!("VS source contains null byte"))?;
            let vs_sources = [c_vs.as_ptr()];
            gl::ShaderSource(vs, 1, vs_sources.as_ptr(), std::ptr::null());
            gl::CompileShader(vs);

            let mut success = 0i32;
            gl::GetShaderiv(vs, gl::COMPILE_STATUS, &mut success);
            if success == 0 {
                let mut len = 0i32;
                gl::GetShaderiv(vs, gl::INFO_LOG_LENGTH, &mut len);
                let mut log = vec![0u8; len as usize];
                gl::GetShaderInfoLog(vs, len, std::ptr::null_mut(), log.as_mut_ptr() as *mut _);
                let msg = String::from_utf8_lossy(&log);
                gl::DeleteShader(vs);
                gl::DeleteShader(fs);
                anyhow::bail!("Vertex shader compilation failed: {msg}");
            }

            // Compile fragment shader
            let c_fs = CString::new(fs_source)
                .map_err(|_| anyhow::anyhow!("FS source contains null byte"))?;
            let fs_sources = [c_fs.as_ptr()];
            gl::ShaderSource(fs, 1, fs_sources.as_ptr(), std::ptr::null());
            gl::CompileShader(fs);

            gl::GetShaderiv(fs, gl::COMPILE_STATUS, &mut success);
            if success == 0 {
                let mut len = 0i32;
                gl::GetShaderiv(fs, gl::INFO_LOG_LENGTH, &mut len);
                let mut log = vec![0u8; len as usize];
                gl::GetShaderInfoLog(fs, len, std::ptr::null_mut(), log.as_mut_ptr() as *mut _);
                let msg = String::from_utf8_lossy(&log);
                gl::DeleteShader(vs);
                gl::DeleteShader(fs);
                anyhow::bail!("Fragment shader compilation failed: {msg}");
            }

            // Link program
            let program = gl::CreateProgram();
            gl::AttachShader(program, vs);
            gl::AttachShader(program, fs);
            gl::LinkProgram(program);

            let mut link_success = 0i32;
            gl::GetProgramiv(program, gl::LINK_STATUS, &mut link_success);
            if link_success == 0 {
                let mut len = 0i32;
                gl::GetProgramiv(program, gl::INFO_LOG_LENGTH, &mut len);
                let mut log = vec![0u8; len as usize];
                gl::GetProgramInfoLog(
                    program,
                    len,
                    std::ptr::null_mut(),
                    log.as_mut_ptr() as *mut _,
                );
                let msg = String::from_utf8_lossy(&log);
                gl::DeleteShader(vs);
                gl::DeleteShader(fs);
                gl::DeleteProgram(program);
                anyhow::bail!("Program link failed: {msg}");
            }

            gl::DeleteShader(vs);
            gl::DeleteShader(fs);
            Ok(program)
        }
    }

    /// Bind textures for a compute dispatch.
    ///
    /// Each texture is bound to BOTH a GL texture unit (for `sampler2D`
    /// uniforms) and a GL image unit (for `image2D` uniforms) at the same
    /// index. Naga generates `layout(binding = N)` qualifiers that route
    /// to the correct binding type based on the WGSL declaration.
    unsafe fn bind_compute_textures(textures: &[&GpuTexture]) {
        for (i, tex) in textures.iter().enumerate() {
            let unit = i as u32;

            // Bind as texture unit (for sampler2D / texelFetch)
            gl::ActiveTexture(gl::TEXTURE0 + unit);
            gl::BindTexture(gl::TEXTURE_2D, tex.gl_name);

            // Also bind as image unit (for image2D / imageStore)
            // READ_WRITE covers all access patterns; the shader's layout
            // qualifier determines actual access.
            gl::BindImageTexture(
                unit,
                tex.gl_name,
                0,
                gl::FALSE,
                0,
                gl::READ_WRITE,
                tex.gl_format,
            );
        }
    }

    impl GpuContext {
        /// Create a compute pipeline from a named entry point.
        ///
        /// Looks up the GLSL source by name from the shader sources provided
        /// at construction time, compiles it with the GL driver, and returns
        /// a linked compute program.
        pub fn create_compute_pipeline(&self, name: &str) -> Result<ComputePipeline> {
            let source = self
                .shader_sources
                .get(name)
                .ok_or_else(|| anyhow::anyhow!("GLSL shader source '{name}' not found"))?;

            let program = compile_shader_program(gl::COMPUTE_SHADER, source)?;
            Ok(ComputePipeline { program })
        }

        /// Create a render pipeline from vertex and fragment entry points.
        ///
        /// Looks up the GLSL sources by name and compiles a linked VS+FS
        /// program. Also creates a fullscreen quad VAO/VBO.
        pub fn create_render_pipeline(
            &self,
            vertex_name: &str,
            fragment_name: &str,
        ) -> Result<RenderPipeline> {
            let vs_source = self
                .shader_sources
                .get(vertex_name)
                .ok_or_else(|| anyhow::anyhow!("GLSL vertex shader '{vertex_name}' not found"))?;
            let fs_source = self.shader_sources.get(fragment_name).ok_or_else(|| {
                anyhow::anyhow!("GLSL fragment shader '{fragment_name}' not found")
            })?;

            let program = compile_render_program(vs_source, fs_source)?;

            // Create fullscreen quad VAO + VBO
            let (mut vao, mut vbo) = (0u32, 0u32);
            unsafe {
                gl::GenVertexArrays(1, &mut vao);
                gl::GenBuffers(1, &mut vbo);

                gl::BindVertexArray(vao);
                gl::BindBuffer(gl::ARRAY_BUFFER, vbo);
                gl::BufferData(
                    gl::ARRAY_BUFFER,
                    std::mem::size_of_val(&FULLSCREEN_QUAD) as isize,
                    FULLSCREEN_QUAD.as_ptr() as *const _,
                    gl::STATIC_DRAW,
                );

                // position: location 0, vec2 at offset 0
                gl::EnableVertexAttribArray(0);
                gl::VertexAttribPointer(
                    0,
                    2,
                    gl::FLOAT,
                    gl::FALSE,
                    (4 * std::mem::size_of::<f32>()) as i32,
                    std::ptr::null(),
                );

                // texcoord: location 1, vec2 at offset 8
                gl::EnableVertexAttribArray(1);
                gl::VertexAttribPointer(
                    1,
                    2,
                    gl::FLOAT,
                    gl::FALSE,
                    (4 * std::mem::size_of::<f32>()) as i32,
                    (2 * std::mem::size_of::<f32>()) as *const _,
                );

                gl::BindVertexArray(0);
                gl::BindBuffer(gl::ARRAY_BUFFER, 0);
            }

            Ok(RenderPipeline {
                program,
                quad_vao: vao,
                quad_vb: vbo,
            })
        }

        /// Create a GPU buffer (SSBO) with the given number of elements and
        /// element size.
        pub fn create_buffer(
            &self,
            num_elements: usize,
            element_size: usize,
        ) -> Result<GpuBuffer> {
            let size = num_elements * element_size;
            let mut buf = 0u32;
            unsafe {
                gl::GenBuffers(1, &mut buf);
                gl::BindBuffer(gl::SHADER_STORAGE_BUFFER, buf);
                gl::BufferData(
                    gl::SHADER_STORAGE_BUFFER,
                    size as isize,
                    std::ptr::null(),
                    gl::DYNAMIC_DRAW,
                );
                gl::BindBuffer(gl::SHADER_STORAGE_BUFFER, 0);
            }
            Ok(GpuBuffer {
                size,
                gl_buffer: buf,
            })
        }

        /// Dispatch a single compute pass and return a [`PendingWork`] token.
        ///
        /// Textures are bound at their WGSL `@binding(N)` index to BOTH the
        /// corresponding GL texture unit (for `sampler2D`) AND image unit
        /// (for `image2D`). Naga's GLSL output uses `layout(binding = N)` to
        /// route to the correct one. Buffers are bound as SSBOs. Bytes are
        /// set via uniform buffer objects.
        pub fn dispatch_compute(
            &self,
            pipeline: &ComputePipeline,
            textures: &[&GpuTexture],
            buffers: &[(&GpuBuffer, usize)],
            bytes: &[(&[u8], usize)],
            grid: (usize, usize),
            threadgroup: (usize, usize),
        ) -> Result<PendingWork> {
            unsafe {
                gl::UseProgram(pipeline.program);

                bind_compute_textures(textures);

                // Bind SSBOs
                for (buf, idx) in buffers {
                    gl::BindBufferBase(gl::SHADER_STORAGE_BUFFER, *idx as u32, buf.gl_buffer);
                }

                // Bind inline uniform data via temporary UBOs
                let mut temp_ubos = Vec::new();
                for (data, idx) in bytes {
                    let mut ubo = 0u32;
                    gl::GenBuffers(1, &mut ubo);
                    gl::BindBuffer(gl::UNIFORM_BUFFER, ubo);
                    gl::BufferData(
                        gl::UNIFORM_BUFFER,
                        data.len() as isize,
                        data.as_ptr() as *const _,
                        gl::STREAM_DRAW,
                    );
                    gl::BindBufferBase(gl::UNIFORM_BUFFER, *idx as u32, ubo);
                    temp_ubos.push(ubo);
                }

                // Dispatch
                let gx = ((grid.0 + threadgroup.0 - 1) / threadgroup.0) as u32;
                let gy = ((grid.1 + threadgroup.1 - 1) / threadgroup.1) as u32;
                gl::DispatchCompute(gx, gy, 1);

                // Memory barrier
                gl::MemoryBarrier(
                    gl::SHADER_IMAGE_ACCESS_BARRIER_BIT
                        | gl::SHADER_STORAGE_BARRIER_BIT
                        | gl::TEXTURE_FETCH_BARRIER_BIT,
                );

                // Cleanup temp UBOs
                if !temp_ubos.is_empty() {
                    gl::DeleteBuffers(temp_ubos.len() as i32, temp_ubos.as_ptr());
                }

                gl::UseProgram(0);

                // Insert fence for synchronisation
                let fence = gl::FenceSync(gl::SYNC_GPU_COMMANDS_COMPLETE, 0);
                Ok(PendingWork { fence })
            }
        }

        /// Dispatch a fullscreen render pass and return a [`PendingWork`] token.
        ///
        /// Creates a temporary FBO, attaches the output texture, and draws
        /// a fullscreen quad with the given render pipeline.
        pub fn dispatch_render(
            &self,
            pipeline: &RenderPipeline,
            output: &GpuTexture,
            fragment_textures: &[&GpuTexture],
            fragment_bytes: &[(&[u8], usize)],
        ) -> Result<PendingWork> {
            unsafe {
                // Create and bind temporary FBO
                let mut fbo = 0u32;
                gl::GenFramebuffers(1, &mut fbo);
                gl::BindFramebuffer(gl::FRAMEBUFFER, fbo);
                gl::FramebufferTexture2D(
                    gl::FRAMEBUFFER,
                    gl::COLOR_ATTACHMENT0,
                    gl::TEXTURE_2D,
                    output.gl_name,
                    0,
                );

                gl::Viewport(0, 0, output.width as i32, output.height as i32);
                gl::UseProgram(pipeline.program);

                // Bind fragment textures to texture units.
                // Naga generates `layout(binding = N)` qualifiers, so
                // binding to texture unit N is automatic.
                for (i, tex) in fragment_textures.iter().enumerate() {
                    gl::ActiveTexture(gl::TEXTURE0 + i as u32);
                    gl::BindTexture(gl::TEXTURE_2D, tex.gl_name);
                }

                // Bind uniform data
                let mut temp_ubos = Vec::new();
                for (data, idx) in fragment_bytes {
                    let mut ubo = 0u32;
                    gl::GenBuffers(1, &mut ubo);
                    gl::BindBuffer(gl::UNIFORM_BUFFER, ubo);
                    gl::BufferData(
                        gl::UNIFORM_BUFFER,
                        data.len() as isize,
                        data.as_ptr() as *const _,
                        gl::STREAM_DRAW,
                    );
                    gl::BindBufferBase(gl::UNIFORM_BUFFER, *idx as u32, ubo);
                    temp_ubos.push(ubo);
                }

                // Draw fullscreen quad
                gl::BindVertexArray(pipeline.quad_vao);
                gl::DrawArrays(gl::TRIANGLE_STRIP, 0, 4);
                gl::BindVertexArray(0);

                // Cleanup
                gl::UseProgram(0);
                gl::BindFramebuffer(gl::FRAMEBUFFER, 0);
                gl::DeleteFramebuffers(1, &fbo);
                if !temp_ubos.is_empty() {
                    gl::DeleteBuffers(temp_ubos.len() as i32, temp_ubos.as_ptr());
                }

                // Insert fence
                let fence = gl::FenceSync(gl::SYNC_GPU_COMMANDS_COMPLETE, 0);
                Ok(PendingWork { fence })
            }
        }

        // =================================================================
        // Multi-pass command buffer API
        // =================================================================

        /// Create a command buffer (thin marker on GL).
        pub fn create_command_buffer(&self) -> Result<CommandBuffer> {
            Ok(CommandBuffer { _marker: () })
        }

        /// Encode a compute pass (dispatches immediately on GL).
        pub fn encode_compute_pass(
            &self,
            _cb: &CommandBuffer,
            pipeline: &ComputePipeline,
            textures: &[&GpuTexture],
            buffers: &[(&GpuBuffer, usize)],
            bytes: &[(&[u8], usize)],
            grid: (usize, usize),
            threadgroup: (usize, usize),
        ) -> Result<()> {
            unsafe {
                gl::UseProgram(pipeline.program);

                bind_compute_textures(textures);

                for (buf, idx) in buffers {
                    gl::BindBufferBase(gl::SHADER_STORAGE_BUFFER, *idx as u32, buf.gl_buffer);
                }

                let mut temp_ubos = Vec::new();
                for (data, idx) in bytes {
                    let mut ubo = 0u32;
                    gl::GenBuffers(1, &mut ubo);
                    gl::BindBuffer(gl::UNIFORM_BUFFER, ubo);
                    gl::BufferData(
                        gl::UNIFORM_BUFFER,
                        data.len() as isize,
                        data.as_ptr() as *const _,
                        gl::STREAM_DRAW,
                    );
                    gl::BindBufferBase(gl::UNIFORM_BUFFER, *idx as u32, ubo);
                    temp_ubos.push(ubo);
                }

                let gx = ((grid.0 + threadgroup.0 - 1) / threadgroup.0) as u32;
                let gy = ((grid.1 + threadgroup.1 - 1) / threadgroup.1) as u32;
                gl::DispatchCompute(gx, gy, 1);
                gl::MemoryBarrier(
                    gl::SHADER_IMAGE_ACCESS_BARRIER_BIT | gl::SHADER_STORAGE_BARRIER_BIT,
                );

                if !temp_ubos.is_empty() {
                    gl::DeleteBuffers(temp_ubos.len() as i32, temp_ubos.as_ptr());
                }

                gl::UseProgram(0);
            }
            Ok(())
        }

        /// Encode a fullscreen render pass (draws immediately on GL).
        pub fn encode_render_pass(
            &self,
            _cb: &CommandBuffer,
            pipeline: &RenderPipeline,
            output: &GpuTexture,
            fragment_textures: &[&GpuTexture],
            fragment_bytes: &[(&[u8], usize)],
        ) -> Result<()> {
            unsafe {
                let mut fbo = 0u32;
                gl::GenFramebuffers(1, &mut fbo);
                gl::BindFramebuffer(gl::FRAMEBUFFER, fbo);
                gl::FramebufferTexture2D(
                    gl::FRAMEBUFFER,
                    gl::COLOR_ATTACHMENT0,
                    gl::TEXTURE_2D,
                    output.gl_name,
                    0,
                );

                gl::Viewport(0, 0, output.width as i32, output.height as i32);
                gl::UseProgram(pipeline.program);

                for (i, tex) in fragment_textures.iter().enumerate() {
                    gl::ActiveTexture(gl::TEXTURE0 + i as u32);
                    gl::BindTexture(gl::TEXTURE_2D, tex.gl_name);
                    let loc = gl::GetUniformLocation(
                        pipeline.program,
                        format!("_group_0_binding_{i}_fs\0").as_ptr() as *const _,
                    );
                    if loc >= 0 {
                        gl::Uniform1i(loc, i as i32);
                    }
                }

                let mut temp_ubos = Vec::new();
                for (data, idx) in fragment_bytes {
                    let mut ubo = 0u32;
                    gl::GenBuffers(1, &mut ubo);
                    gl::BindBuffer(gl::UNIFORM_BUFFER, ubo);
                    gl::BufferData(
                        gl::UNIFORM_BUFFER,
                        data.len() as isize,
                        data.as_ptr() as *const _,
                        gl::STREAM_DRAW,
                    );
                    gl::BindBufferBase(gl::UNIFORM_BUFFER, *idx as u32, ubo);
                    temp_ubos.push(ubo);
                }

                gl::BindVertexArray(pipeline.quad_vao);
                gl::DrawArrays(gl::TRIANGLE_STRIP, 0, 4);
                gl::BindVertexArray(0);

                gl::UseProgram(0);
                gl::BindFramebuffer(gl::FRAMEBUFFER, 0);
                gl::DeleteFramebuffers(1, &fbo);
                if !temp_ubos.is_empty() {
                    gl::DeleteBuffers(temp_ubos.len() as i32, temp_ubos.as_ptr());
                }
            }
            Ok(())
        }

        /// Commit a command buffer — inserts a GL fence and returns a
        /// [`PendingWork`] token.
        pub fn commit(&self, _cb: CommandBuffer) -> PendingWork {
            let fence = unsafe { gl::FenceSync(gl::SYNC_GPU_COMMANDS_COMPLETE, 0) };
            PendingWork { fence }
        }
    }
}
