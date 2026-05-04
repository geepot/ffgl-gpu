//! This module provides a simplified way to implement a plugin.
//! 1. Implement [SimpleFFGLInstance] for your plugin
//! 2. Call [crate::plugin_main] with a [SimpleFFGLHandler] and your instance type, such as:
//!    ```rust plugin_main!(SimpleFFGLHandler<MyInstanceType>);```

use super::FFGLHandler;

use crate::parameters::ParamInfo;

use crate::{FFGLData, GLInput};

use super::FFGLInstance;

/// This is a handler that just delegates to a SimpleFFGLInstance
pub struct SimpleFFGLHandler<T: SimpleFFGLInstance> {
    pub(crate) _marker: std::marker::PhantomData<T>,
}

/// Implement this trait for a plugin without any static state
pub trait SimpleFFGLInstance: FFGLInstance + Send + Sync {
    fn new(inst_data: &FFGLData) -> Self;

    fn num_params() -> usize {
        0
    }
    fn param_info(_index: usize) -> &'static dyn ParamInfo {
        panic!("No params")
    }

    fn plugin_info() -> crate::info::PluginInfo;

    fn get_param(&self, _index: usize) -> f32 {
        panic!("No params")
    }
    fn set_param(&mut self, _index: usize, _value: f32) {
        panic!("No params")
    }

    /// Whether the param at `index` should currently be visible in the
    /// host's UI. Override to hide params that are irrelevant given the
    /// current state of other params. Default: always visible. See
    /// [`FFGLInstance::param_visible`] for the full contract.
    fn param_visible(&self, _index: usize) -> bool {
        true
    }

    /// Custom-formatted display string for the param at `index`.
    /// See [`FFGLInstance::param_display_value`] for details.
    fn param_display_value(&self, _index: usize) -> Option<String> {
        None
    }

    /// Consume pending parameter events. Override to push value changes to host.
    fn consume_param_events(&mut self, _max_events: usize) -> Vec<(u32, u64)> {
        vec![]
    }

    /// Called by [crate::conversions::Op::ProcessOpenGL] to draw the plugin
    fn draw(&mut self, inst_data: &FFGLData, frame_data: GLInput);
}

impl<T: SimpleFFGLInstance> FFGLInstance for T {
    fn get_param(&self, index: usize) -> f32 {
        SimpleFFGLInstance::get_param(self, index)
    }

    fn set_param(&mut self, index: usize, value: f32) {
        SimpleFFGLInstance::set_param(self, index, value)
    }

    fn param_visible(&self, index: usize) -> bool {
        SimpleFFGLInstance::param_visible(self, index)
    }

    fn param_display_value(&self, index: usize) -> Option<String> {
        SimpleFFGLInstance::param_display_value(self, index)
    }

    fn consume_param_events(&mut self, max_events: usize) -> Vec<(u32, u64)> {
        SimpleFFGLInstance::consume_param_events(self, max_events)
    }

    fn draw(&mut self, inst_data: &FFGLData, frame_data: GLInput) {
        SimpleFFGLInstance::draw(self, inst_data, frame_data)
    }
}

impl<T: SimpleFFGLInstance> FFGLHandler for SimpleFFGLHandler<T> {
    type Instance = T;
    type NewInstanceError = std::convert::Infallible;

    fn init() -> Self {
        Self {
            _marker: std::marker::PhantomData,
        }
    }

    fn num_params(&self) -> usize {
        T::num_params()
    }

    fn param_info(&'static self, index: usize) -> &'static dyn ParamInfo {
        T::param_info(index)
    }

    fn plugin_info(&self) -> crate::info::PluginInfo {
        T::plugin_info()
    }

    fn new_instance(&self, inst_data: &FFGLData) -> Result<Self::Instance, Self::NewInstanceError> {
        Ok(T::new(inst_data))
    }
}
