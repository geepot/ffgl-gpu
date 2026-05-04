//! This module provides the main traits for creating FFGL plugins.
//! Use [simplified] for a simpler way to create plugins.

use parameters::ParamInfo;

use std;

use std::error::Error;
use std::fmt::Debug;

use crate::inputs::FFGLData;

use crate::{info, inputs::GLInput, parameters};

#[doc(hidden)]
pub struct Instance<T> {
    pub(crate) data: FFGLData,
    pub(crate) renderer: T,
    /// True after the first `Op::GetParameterEvents` poll. Used by the
    /// dispatcher in `entry.rs` to inject a one-shot batch of
    /// visibility events for any param that's currently hidden — so
    /// the host re-queries them once the instance exists, since its
    /// pre-instance visibility query saw no state and got the
    /// "default visible" fallback.
    pub(crate) first_events_polled: bool,
}

impl<I> Debug for Instance<I> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Instance")
            .field("data", &self.data)
            .field("renderer", &std::any::type_name::<I>())
            .finish()
    }
}

/// This type is created once per instance of a plugin
pub trait FFGLInstance {
    fn get_param(&self, index: usize) -> f32;
    fn set_param(&mut self, index: usize, value: f32);

    /// Whether the parameter at `index` should currently be visible in
    /// the host's UI. Override to return `false` when a param is
    /// irrelevant given the current state of other params (e.g. hide
    /// "High Color" when a Color Mode dropdown is set to "Solid").
    /// Default: always visible.
    ///
    /// To trigger the host to re-query visibility after a state
    /// change, push a `(param_index, FF_EVENT_FLAG_VISIBILITY)` pair
    /// onto the event queue exposed by [`consume_param_events`].
    fn param_visible(&self, _index: usize) -> bool {
        true
    }

    /// Custom-formatted display string for the parameter at `index`,
    /// used when the host wants to render something other than the
    /// raw float (e.g. "5.7 px" for a slider that's internally a
    /// 0.5–16.0 pixel range). Default `None` — the host uses its
    /// own formatting.
    fn param_display_value(&self, _index: usize) -> Option<String> {
        None
    }

    /// Consume pending parameter events. Returns (param_index, event_flags) pairs.
    /// Called by the host via GetParameterEvents. Default: no events.
    fn consume_param_events(&mut self, _max_events: usize) -> Vec<(u32, u64)> {
        vec![]
    }

    /// Called by [crate::conversions::Op::ProcessOpenGL] to draw the plugin
    fn draw(&mut self, inst_data: &FFGLData, frame_data: GLInput);
}

/// This type is created once per plugin load.
/// You can use it to store static state and create instances
pub trait FFGLHandler: Send + Sync {
    type Instance: FFGLInstance;
    type NewInstanceError: Error + Send + Sync + 'static;

    /// Only called once per plugin
    fn init() -> Self;

    fn num_params(&'static self) -> usize;

    fn param_info(&'static self, index: usize) -> &'static dyn ParamInfo;

    fn plugin_info(&'static self) -> info::PluginInfo;

    fn new_instance(
        &'static self,
        inst_data: &FFGLData,
    ) -> Result<Self::Instance, Self::NewInstanceError>;
}

pub mod simplified;
