//! Safe byte-slice conversion for GPU uniform structs.

/// Convert a `#[repr(C)]` struct to a byte slice for GPU uniform upload.
///
/// # Safety
///
/// Only implement on `#[repr(C)]` structs whose fields are all plain numeric
/// types (f32, i32, u32, etc.) with no pointers, references, or
/// padding-dependent invariants.
///
/// # Example
///
/// ```rust,ignore
/// #[repr(C)]
/// struct MyParams {
///     brightness: f32,
///     contrast: f32,
/// }
///
/// unsafe impl AsBytes for MyParams {}
///
/// // Then in gpu_draw:
/// ctx.update_constant_buffer(&cbuf, params.as_bytes());
/// ```
pub unsafe trait AsBytes: Sized {
    /// View `self` as a byte slice. The returned slice has length
    /// `std::mem::size_of::<Self>()`.
    fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(self as *const Self as *const u8, std::mem::size_of::<Self>())
        }
    }
}
