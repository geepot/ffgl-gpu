//! Hardcoded FFGL 2.2 constants and C-repr structs.
//!
//! These replace the bindgen-generated bindings from the original ffgl-rs crate.
//! Constants are sourced from the FFGL SDK headers (FreeFrame.h, FFGL.h).

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]

use std::ffi::c_void;

// =====================================================================
// Function codes (Op codes)
// =====================================================================

// FFGL 1.x function codes (FreeFrame.h)
pub const FF_GET_INFO: u32 = 0;
pub const FF_INITIALISE: u32 = 1;
pub const FF_DEINITIALISE: u32 = 2;
pub const FF_PROCESSFRAME: u32 = 3;
pub const FF_GETNUMPARAMETERS: u32 = 4;
pub const FF_GETPARAMETERNAME: u32 = 5;
pub const FF_GETPARAMETERDEFAULT: u32 = 6;
pub const FF_GETPARAMETERDISPLAY: u32 = 7;
pub const FF_SETPARAMETER: u32 = 8;
pub const FF_GETPARAMETER: u32 = 9;
pub const FF_GETPLUGINCAPS: u32 = 10;
pub const FF_INSTANTIATE: u32 = 11;
pub const FF_DEINSTANTIATE: u32 = 12;
pub const FF_GETEXTENDEDINFO: u32 = 13;
pub const FF_PROCESSFRAMECOPY: u32 = 14;
pub const FF_GETPARAMETERTYPE: u32 = 15;
pub const FF_GETINPUTSTATUS: u32 = 16;

// FFGL 1.x GL function codes (FFGL.h)
pub const FF_PROCESSOPENGL: u32 = 17;
pub const FF_INSTANTIATEGL: u32 = 18;
pub const FF_DEINSTANTIATEGL: u32 = 19;
pub const FF_SETTIME: u32 = 20;
pub const FF_CONNECT: u32 = 21;
pub const FF_DISCONNECT: u32 = 22;
pub const FF_RESIZE: u32 = 23;

// FFGL 2.x function codes (FFGL.h / resolume)
pub const FF_INITIALISE_V2: u32 = 34;
pub const FF_GET_NUM_PARAMETER_ELEMENTS: u32 = 31;
pub const FF_GET_PARAMETER_ELEMENT_NAME: u32 = 35;
pub const FF_GET_PARAMETER_ELEMENT_VALUE: u32 = 36;
pub const FF_SET_PARAMETER_ELEMENT_VALUE: u32 = 37;
pub const FF_GET_PARAMETER_USAGE: u32 = 32;
pub const FF_GET_PLUGIN_SHORT_NAME: u32 = 33;
pub const FF_SET_BEATINFO: u32 = 38;
pub const FF_SET_HOSTINFO: u32 = 39;
pub const FF_SET_SAMPLERATE: u32 = 40;
pub const FF_GET_RANGE: u32 = 41;
pub const FF_GET_THUMBNAIL: u32 = 42;
pub const FF_GET_NUM_FILE_PARAMETER_EXTENSIONS: u32 = 43;
pub const FF_GET_FILE_PARAMETER_EXTENSION: u32 = 44;
pub const FF_GET_PRAMETER_VISIBILITY: u32 = 45;
pub const FF_GET_PARAMETER_EVENTS: u32 = 46;
pub const FF_GET_NUM_ELEMENT_SEPARATORS: u32 = 47;
pub const FF_GET_SEPARATOR_ELEMENT_INDEX: u32 = 48;
pub const FF_ENABLE_PLUGIN_CAP: u32 = 49;
pub const FF_GET_PARAM_GROUP: u32 = 50;
pub const FF_GET_PARAM_DISPLAY_NAME: u32 = 51;

// =====================================================================
// Result codes
// =====================================================================
pub const FF_SUCCESS: u32 = 0;
pub const FF_FAIL: u32 = 0xFFFFFFFF;
pub const FF_TRUE: u32 = 1;
pub const FF_FALSE: u32 = 0;
pub const FF_SUPPORTED: u32 = 1;
pub const FF_UNSUPPORTED: u32 = 0;

// =====================================================================
// Plugin types
// =====================================================================
pub const FF_EFFECT: u32 = 0;
pub const FF_SOURCE: u32 = 1;
pub const FF_MIXER: u32 = 2;

// =====================================================================
// Plugin capabilities
// =====================================================================

// FFGL 1.x capabilities (FreeFrame.h)
pub const FF_CAP_16BITVIDEO: u32 = 0;
pub const FF_CAP_24BITVIDEO: u32 = 1;
pub const FF_CAP_32BITVIDEO: u32 = 2;
pub const FF_CAP_PROCESSFRAMECOPY: u32 = 3;
pub const FF_CAP_PROCESSOPENGL: u32 = 4;

// FFGL 2.x capabilities (FFGL.h / resolume)
pub const FF_CAP_SET_TIME: u32 = 5;
pub const FF_CAP_MINIMUM_INPUT_FRAMES: u32 = 10;
pub const FF_CAP_MAXIMUM_INPUT_FRAMES: u32 = 11;
pub const FF_CAP_COPYORINPLACE: u32 = 15;
pub const FF_CAP_TOP_LEFT_TEXTURE_ORIENTATION: u32 = 16;

// =====================================================================
// Parameter types
// =====================================================================
pub const FF_TYPE_BOOLEAN: u32 = 0;
pub const FF_TYPE_EVENT: u32 = 1;
pub const FF_TYPE_RED: u32 = 2;
pub const FF_TYPE_GREEN: u32 = 3;
pub const FF_TYPE_BLUE: u32 = 4;
pub const FF_TYPE_XPOS: u32 = 5;
pub const FF_TYPE_YPOS: u32 = 6;
pub const FF_TYPE_STANDARD: u32 = 10;
pub const FF_TYPE_OPTION: u32 = 11;
pub const FF_TYPE_BUFFER: u32 = 12;
pub const FF_TYPE_INTEGER: u32 = 13;
pub const FF_TYPE_FILE: u32 = 14;
pub const FF_TYPE_TEXT: u32 = 100;
pub const FF_TYPE_HUE: u32 = 200;
pub const FF_TYPE_SATURATION: u32 = 201;
pub const FF_TYPE_BRIGHTNESS: u32 = 202;
pub const FF_TYPE_ALPHA: u32 = 203;

// =====================================================================
// Input status
// =====================================================================
pub const FF_INPUT_NOTINUSE: u32 = 0;
pub const FF_INPUT_INUSE: u32 = 1;

// =====================================================================
// Parameter usages
// =====================================================================
pub const FF_USAGE_STANDARD: u32 = 0;
pub const FF_USAGE_FFT: u32 = 1;

// =====================================================================
// Parameter event flags
// =====================================================================
pub const FF_EVENT_FLAG_VISIBILITY: u64 = 0x01;
pub const FF_EVENT_FLAG_DISPLAY_NAME: u64 = 0x02;
pub const FF_EVENT_FLAG_VALUE: u64 = 0x04;
pub const FF_EVENT_FLAG_ELEMENTS: u64 = 0x08;

// =====================================================================
// C-repr structs matching the FFGL SDK
// =====================================================================

/// Union type used for parameter values in the FFGL ABI.
#[repr(C)]
#[derive(Copy, Clone)]
pub union FFMixed {
    pub UIntValue: u32,
    pub PointerValue: *mut c_void,
}

/// Plugin info struct returned by FF_GET_INFO.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct PluginInfoStruct {
    pub APIMajorVersion: u32,
    pub APIMinorVersion: u32,
    pub PluginUniqueID: [i8; 4],
    pub PluginName: [i8; 16],
    pub PluginType: u32,
}

/// Extended plugin info returned by FF_GET_EXTENDED_INFO.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct PluginExtendedInfoStruct {
    pub PluginMajorVersion: u32,
    pub PluginMinorVersion: u32,
    pub Description: *mut c_void,
    pub About: *mut c_void,
    pub FreeFrameExtendedDataSize: u32,
    pub FreeFrameExtendedDataBlock: *mut c_void,
}

/// Struct passed to the plugin when setting a parameter.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct SetParameterStruct {
    pub ParameterNumber: u32,
    pub NewParameterValue: FFMixed,
}

/// Beat information provided by the host.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct SetBeatinfoStruct {
    pub bpm: f32,
    pub barPhase: f32,
}

/// Range information for a parameter.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct RangeStruct {
    pub min: f32,
    pub max: f32,
}

/// Struct for getting a parameter's range.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct GetRangeStruct {
    pub parameterNumber: u32,
    pub range: RangeStruct,
}

/// String buffer struct used for host-provided string buffers.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct StringBufferStruct {
    pub address: *mut i8,
    pub maxToWrite: u32,
}

/// Struct for getting a string from the plugin (group name, display name, etc.).
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct GetStringStruct {
    pub parameterNumber: u32,
    pub stringBuffer: StringBufferStruct,
}

/// Viewport struct for InstantiateGL.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct FFGLViewportStruct {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

/// Texture struct for ProcessOpenGL.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct FFGLTextureStruct {
    pub Width: u32,
    pub Height: u32,
    pub HardwareWidth: u32,
    pub HardwareHeight: u32,
    pub Handle: u32,
}

/// Struct passed to ProcessOpenGL.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ProcessOpenGLStruct {
    pub numInputTextures: u32,
    pub inputTextures: *mut *mut FFGLTextureStruct,
    pub HostFBO: u32,
}

/// Struct for getting a parameter element name.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct GetParameterElementNameStruct {
    pub ParameterNumber: u32,
    pub ElementNumber: u32,
}

/// Struct for getting a parameter element value.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct GetParameterElementValueStruct {
    pub ParameterNumber: u32,
    pub ElementNumber: u32,
}

/// Struct for setting a parameter element value.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct SetParameterElementValueStruct {
    pub ParameterNumber: u32,
    pub ElementNumber: u32,
    pub NewParameterValue: FFMixed,
}

// =====================================================================
// Utility
// =====================================================================

/// Copy a Rust string into a host-provided buffer, null-terminating it.
///
/// # Safety
///
/// `address` must be a valid pointer to a buffer of at least `max_to_write` bytes.
pub unsafe fn copy_str_to_host_buffer(address: *mut u8, max_to_write: usize, string: &str) {
    use std::ffi::CString;

    if max_to_write == 0 {
        return;
    }

    let cstr = CString::new(string).unwrap().into_bytes_with_nul();
    let to_copy = cstr.len().min(max_to_write);
    let dest = unsafe { std::slice::from_raw_parts_mut(address, to_copy) };

    dest.copy_from_slice(&cstr[..to_copy]);

    // If we truncated, ensure the buffer is still null-terminated
    if to_copy < cstr.len() {
        dest[to_copy - 1] = 0;
    }
}
