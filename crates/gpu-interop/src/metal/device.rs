//! Generic Metal device wrapper.
//!
//! Holds only the Metal device and command queue -- no shader libraries or
//! pipeline states.  Consumers (e.g. `ffgl-gpu`) are responsible for loading
//! their own shaders and building pipelines on top of this device.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandQueue, MTLCreateSystemDefaultDevice, MTLDevice};
use tracing::{debug, error};

/// A generic Metal device with a single command queue.
///
/// Created via [`MetalDevice::new()`] which obtains the system default Metal
/// device.  This struct intentionally contains *no* shader library or pipeline
/// state -- those belong to higher-level crates that know what shaders they
/// need.
pub struct MetalDevice {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    command_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
}

impl MetalDevice {
    /// Create a new Metal device using the system default GPU.
    ///
    /// Returns `None` if Metal is unavailable (e.g. no discrete/integrated GPU
    /// or a very old Mac).
    pub fn new() -> Option<Self> {
        let device = MTLCreateSystemDefaultDevice()?;
        debug!("Metal device: {}", device.name());

        let command_queue = match device.newCommandQueue() {
            Some(q) => q,
            None => {
                error!("Failed to create Metal command queue");
                return None;
            }
        };

        Some(Self {
            device,
            command_queue,
        })
    }

    /// Borrow the underlying `MTLDevice`.
    pub fn device(&self) -> &ProtocolObject<dyn MTLDevice> {
        &self.device
    }

    /// Borrow the command queue.
    pub fn command_queue(&self) -> &ProtocolObject<dyn MTLCommandQueue> {
        &self.command_queue
    }
}
