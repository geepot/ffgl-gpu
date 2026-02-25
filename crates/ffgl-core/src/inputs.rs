//! Inputs from the host to your plugin

use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::ffi::*;

/// Texture inputs
#[derive(Debug)]
pub struct GLInput<'a> {
    pub textures: &'a [FFGLTextureStruct],
    pub host: u32,
}

impl<'a> From<&'a ProcessOpenGLStruct> for GLInput<'a> {
    fn from(val: &'a ProcessOpenGLStruct) -> GLInput<'a> {
        GLInput {
            textures: unsafe {
                std::slice::from_raw_parts(
                    *val.inputTextures as *const _,
                    val.numInputTextures as usize,
                )
            },
            host: val.HostFBO,
        }
    }
}

/// Standard data that hosts provide to all programs
#[derive(Debug)]
pub struct FFGLData {
    pub created_at: Instant,
    pub viewport: FFGLViewportStruct,
    pub host_time: SystemTime,
    pub host_beat: SetBeatinfoStruct,
}

impl FFGLData {
    pub fn new(viewport: &FFGLViewportStruct) -> FFGLData {
        Self {
            created_at: Instant::now(),
            viewport: *viewport,
            host_time: SystemTime::now(),
            host_beat: SetBeatinfoStruct {
                bpm: 120.0,
                barPhase: 0.0,
            },
        }
    }

    pub fn set_beat(&mut self, beat: SetBeatinfoStruct) {
        self.host_beat = beat;
    }

    pub fn set_time(&mut self, host_millis: f64) {
        self.host_time = UNIX_EPOCH + Duration::from_secs_f64(host_millis / 1000.0)
    }

    pub fn get_dimensions(&self) -> (u32, u32) {
        (self.viewport.width, self.viewport.height)
    }
}
