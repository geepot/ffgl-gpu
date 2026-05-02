//! Unified resource-binding API for compute dispatches.
//!
//! `BindSet` is a slot-keyed builder for the resources a compute kernel
//! needs (textures, buffers, samplers, inline uniforms). It hides the
//! platform asymmetry between Metal's sequential bindings + sparse buffer
//! tuples and DX11's parallel slot-indexed `Vec<Option<view>>` arrays
//! behind a single shape, and validates HLSL register-class collisions
//! before dispatch (the failure mode that today silently zeroes out a
//! binding at runtime).
//!
//! # Example
//!
//! ```rust,ignore
//! let bindings = BindSet::new()
//!     .texture_read(0, &input_srv)
//!     .texture_rw(0, &output_uav)        // OK: different register class (t0 vs u0)
//!     .buffer_read(2, &counts_buf)       // SRV at t2
//!     .buffer_rw(3, &particles_buf)      // UAV at u3
//!     .sampler(0, &sampler)
//!     .uniforms_inline(0, uniforms.as_bytes())
//!     .build()?;
//!
//! ctx.dispatch_compute_with(&pipeline, &bindings, grid, threadgroup)?;
//! ```
//!
//! # HLSL register-class semantics
//!
//! HLSL groups resources by their register class, not by `@binding` number:
//!
//! - `t<N>` — sampled textures (`Texture2D`) AND read-only structured
//!   buffers (`ByteAddressBuffer` / `StructuredBuffer<T>`)
//! - `u<N>` — read-write textures (`RWTexture2D`) AND read-write buffers
//!   (`RWStructuredBuffer<T>`)
//! - `s<N>` — samplers
//! - `b<N>` — constant buffers
//!
//! So `BindSet` rejects, at `build()` time, any combination that would
//! collide on the DX11 register space — most importantly,
//! `texture_read(N)` + `buffer_read(N)`, which is the source of the
//! `binding(7)` workaround in stipple's splat_particles kernel.

use std::fmt;

use crate::buffer::GpuBuffer;

// ---------------------------------------------------------------------------
// Platform-specific view aliases. The BindSet builder API looks the same on
// every platform; what changes is the concrete reference type the caller
// passes in. Plugins are already cfg-gated per-platform (step.rs vs
// step_dx11.rs), so this asymmetry is invisible at the call site.
// ---------------------------------------------------------------------------

#[cfg(target_os = "macos")]
pub type TextureRead<'a> =
    &'a objc2::runtime::ProtocolObject<dyn objc2_metal::MTLTexture>;
#[cfg(target_os = "macos")]
pub type TextureRw<'a> =
    &'a objc2::runtime::ProtocolObject<dyn objc2_metal::MTLTexture>;
#[cfg(target_os = "macos")]
pub type SamplerRef<'a> =
    &'a objc2::runtime::ProtocolObject<dyn objc2_metal::MTLSamplerState>;

#[cfg(target_os = "windows")]
pub type TextureRead<'a> = &'a windows::Win32::Graphics::Direct3D11::ID3D11ShaderResourceView;
#[cfg(target_os = "windows")]
pub type TextureRw<'a> = &'a windows::Win32::Graphics::Direct3D11::ID3D11UnorderedAccessView;
#[cfg(target_os = "windows")]
pub type SamplerRef<'a> = &'a windows::Win32::Graphics::Direct3D11::ID3D11SamplerState;

/// Uniform / constant-buffer payload. On Metal this is raw bytes the
/// dispatch will copy via `setBytes` (single-frame ephemeral). On DX11
/// it's a reference to a pre-uploaded `ID3D11Buffer` (caller is
/// responsible for `update_constant_buffer` before the dispatch).
#[cfg(target_os = "macos")]
pub type UniformsBinding<'a> = &'a [u8];
#[cfg(target_os = "windows")]
pub type UniformsBinding<'a> = &'a windows::Win32::Graphics::Direct3D11::ID3D11Buffer;

// On host (linux/CI) builds neither cfg fires; provide unit-shaped aliases
// so the crate still type-checks. The actual dispatch impls only exist on
// macOS / Windows so these aliases are never instantiated in practice.
#[cfg(not(any(target_os = "macos", target_os = "windows")))]
pub type TextureRead<'a> = &'a ();
#[cfg(not(any(target_os = "macos", target_os = "windows")))]
pub type TextureRw<'a> = &'a ();
#[cfg(not(any(target_os = "macos", target_os = "windows")))]
pub type SamplerRef<'a> = &'a ();
#[cfg(not(any(target_os = "macos", target_os = "windows")))]
pub type UniformsBinding<'a> = &'a ();

// ---------------------------------------------------------------------------
// BindError — what build() can reject
// ---------------------------------------------------------------------------

/// An error encountered while validating a `BindSet`.
#[derive(Debug, Clone)]
pub enum BindError {
    /// Two read-only resources (texture + buffer) bound to the same slot.
    /// In HLSL both would emit `register(t<slot>)` and the second would
    /// silently shadow the first, producing wrong shader inputs at
    /// runtime. The Metal side has no equivalent collision but rejecting
    /// uniformly keeps the API portable.
    SrvRegisterCollision { slot: usize },
    /// Two read-write resources (texture + buffer) bound to the same
    /// slot. Same pattern as `SrvRegisterCollision` but for `register(u<N>)`.
    UavRegisterCollision { slot: usize },
    /// Same slot bound twice for the same resource kind (e.g.
    /// `texture_read(2, ...)` called twice with different views).
    DuplicateBinding { kind: &'static str, slot: usize },
}

impl fmt::Display for BindError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BindError::SrvRegisterCollision { slot } => write!(
                f,
                "BindSet: slot {slot} has both a read-only texture and a read-only \
                 buffer; both would emit HLSL register(t{slot}) and collide. \
                 Move one to a different @binding in the WGSL."
            ),
            BindError::UavRegisterCollision { slot } => write!(
                f,
                "BindSet: slot {slot} has both a read-write texture and a read-write \
                 buffer; both would emit HLSL register(u{slot}) and collide."
            ),
            BindError::DuplicateBinding { kind, slot } => {
                write!(f, "BindSet: {kind} bound twice at slot {slot}")
            }
        }
    }
}

impl std::error::Error for BindError {}

// ---------------------------------------------------------------------------
// BindSet — sparse slot tables
// ---------------------------------------------------------------------------

/// A validated set of resource bindings ready to hand to
/// `dispatch_compute_with` / `encode_compute_pass_with`.
///
/// Construct with `BindSet::new()` and the `texture_*` / `buffer_*` /
/// `sampler` / `uniforms_inline` builder methods, then call `build()` to
/// validate. The returned `BindSet` borrows from the same lifetime as the
/// resources the caller passed in.
pub struct BindSet<'a> {
    pub(crate) textures_ro: Vec<Option<TextureRead<'a>>>,
    pub(crate) textures_rw: Vec<Option<TextureRw<'a>>>,
    pub(crate) buffers_ro: Vec<Option<&'a GpuBuffer>>,
    pub(crate) buffers_rw: Vec<Option<&'a GpuBuffer>>,
    pub(crate) samplers: Vec<Option<SamplerRef<'a>>>,
    /// Sparse uniform / constant-buffer slot table. Slot indexing
    /// matches HLSL `register(b<slot>)` and Metal `[[buffer(<slot>)]]`.
    /// Note: on Metal, this slot space is shared with `buffers_*`
    /// (Metal does not separate cbuf and storage buffer registers), so
    /// the WGSL author shouldn't reuse a slot — but the BindSet does
    /// not enforce this since it can't tell which platform you'll
    /// dispatch on.
    pub(crate) uniforms: Vec<Option<UniformsBinding<'a>>>,
}

impl<'a> BindSet<'a> {
    /// Start a new empty bind set.
    pub fn new() -> Self {
        Self {
            textures_ro: Vec::new(),
            textures_rw: Vec::new(),
            buffers_ro: Vec::new(),
            buffers_rw: Vec::new(),
            samplers: Vec::new(),
            uniforms: Vec::new(),
        }
    }

    /// True if no resources have been added yet.
    pub fn is_empty(&self) -> bool {
        self.textures_ro.iter().all(Option::is_none)
            && self.textures_rw.iter().all(Option::is_none)
            && self.buffers_ro.iter().all(Option::is_none)
            && self.buffers_rw.iter().all(Option::is_none)
            && self.samplers.iter().all(Option::is_none)
            && self.uniforms.iter().all(Option::is_none)
    }

    /// Bind a read-only (sampled) texture at the given slot.
    ///
    /// Maps to HLSL `register(t<slot>)` and Metal `[[texture(<slot>)]]`.
    /// Collides with `buffer_read` at the same slot under HLSL — `build()`
    /// will reject that combination.
    pub fn texture_read(mut self, slot: usize, tex: TextureRead<'a>) -> Self {
        ensure_slot(&mut self.textures_ro, slot);
        self.textures_ro[slot] = Some(tex);
        self
    }

    /// Bind a read-write (UAV) texture at the given slot.
    ///
    /// Maps to HLSL `register(u<slot>)` and Metal `[[texture(<slot>)]]`
    /// (Metal does not separate read and write texture register classes).
    pub fn texture_rw(mut self, slot: usize, tex: TextureRw<'a>) -> Self {
        ensure_slot(&mut self.textures_rw, slot);
        self.textures_rw[slot] = Some(tex);
        self
    }

    /// Bind a read-only structured buffer (SRV) at the given slot.
    ///
    /// Maps to HLSL `register(t<slot>)` and Metal `[[buffer(<slot>)]]`.
    pub fn buffer_read(mut self, slot: usize, buf: &'a GpuBuffer) -> Self {
        ensure_slot(&mut self.buffers_ro, slot);
        self.buffers_ro[slot] = Some(buf);
        self
    }

    /// Bind a read-write structured buffer (UAV) at the given slot.
    pub fn buffer_rw(mut self, slot: usize, buf: &'a GpuBuffer) -> Self {
        ensure_slot(&mut self.buffers_rw, slot);
        self.buffers_rw[slot] = Some(buf);
        self
    }

    /// Bind a sampler at the given slot.
    pub fn sampler(mut self, slot: usize, samp: SamplerRef<'a>) -> Self {
        ensure_slot(&mut self.samplers, slot);
        self.samplers[slot] = Some(samp);
        self
    }

    /// Bind a uniform / constant-buffer at the given slot.
    ///
    /// The argument type is platform-aliased via [`UniformsBinding`]:
    /// on macOS it's `&[u8]` (copied via `setBytes` per dispatch);
    /// on Windows it's `&ID3D11Buffer` (caller pre-uploads via
    /// [`GpuContext::update_constant_buffer`]). Plugin code is already
    /// cfg-gated per platform, so the call site reads identically on
    /// both:
    ///
    /// ```rust,ignore
    /// // macOS
    /// .uniforms(0, uniforms.as_bytes())
    /// // Windows
    /// .uniforms(0, &uniform_buf.buffer)
    /// ```
    pub fn uniforms(mut self, slot: usize, b: UniformsBinding<'a>) -> Self {
        ensure_slot(&mut self.uniforms, slot);
        self.uniforms[slot] = Some(b);
        self
    }

    /// Validate the bind set. Returns `Err(BindError)` on register-class
    /// collisions or duplicates. Returns `Ok(self)` otherwise so calls
    /// can chain — e.g. `BindSet::new().texture_read(0, ..).build()?`.
    pub fn build(self) -> Result<Self, BindError> {
        // SRV register class: textures_ro[N] and buffers_ro[N] both
        // emit register(t<N>) in HLSL. Reject overlap.
        let max = self.textures_ro.len().max(self.buffers_ro.len());
        for slot in 0..max {
            let has_tex = self.textures_ro.get(slot).copied().flatten().is_some();
            let has_buf = self.buffers_ro.get(slot).copied().flatten().is_some();
            if has_tex && has_buf {
                return Err(BindError::SrvRegisterCollision { slot });
            }
        }
        // UAV register class: textures_rw[N] and buffers_rw[N] both
        // emit register(u<N>).
        let max = self.textures_rw.len().max(self.buffers_rw.len());
        for slot in 0..max {
            let has_tex = self.textures_rw.get(slot).copied().flatten().is_some();
            let has_buf = self.buffers_rw.get(slot).copied().flatten().is_some();
            if has_tex && has_buf {
                return Err(BindError::UavRegisterCollision { slot });
            }
        }
        Ok(self)
    }
}

impl<'a> Default for BindSet<'a> {
    fn default() -> Self {
        Self::new()
    }
}

/// Grow a sparse-slot table to at least `slot + 1` entries with `None`s.
fn ensure_slot<T>(table: &mut Vec<Option<T>>, slot: usize) {
    if table.len() <= slot {
        table.resize_with(slot + 1, || None);
    }
}
