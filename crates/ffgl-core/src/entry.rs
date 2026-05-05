//! Primary entry point of the FFGL plugin. This is the function that is called by the host.
//! You can use [crate::plugin_main] to automate calling this entry function from the FFGL ABI.

use crate::ffi::copy_str_to_host_buffer;
use crate::ffi::*;
use crate::info;
use crate::FFGLData;

use crate::handler::{FFGLHandler, FFGLInstance};
use crate::log::try_init_default_subscriber;
use crate::parameters::ParamInfo;

use std::cell::RefCell;
use std::sync::OnceLock;
use std::{any::Any, ffi::CString};

// Thread-local buffer for `Op::GetParameterDisplay` — the FFGL contract
// is "return a char* the host reads synchronously, valid until the next
// call." A single thread-local CString satisfies that on the FFGL
// single-threaded host model. Per-call overwrite is fine because the
// host copies the string out before issuing the next op.
thread_local! {
    static DISPLAY_BUFFER: RefCell<CString> = RefCell::new(CString::default());
}

use crate::conversions::*;

use crate::handler;
use anyhow::{Context, Error};

fn param<H: FFGLHandler>(handler: &'static H, index: FFGLVal) -> &'static dyn ParamInfo {
    handler.param_info(unsafe { index.num as usize })
}

static INFO: OnceLock<info::PluginInfo> = OnceLock::new();
static INFO_STRUCT: OnceLock<PluginInfoStruct> = OnceLock::new();
static ABOUT: OnceLock<CString> = OnceLock::new();
static DESCRIPTION: OnceLock<CString> = OnceLock::new();

/// Wrapper to allow `PluginExtendedInfoStruct` in a `OnceLock`.
///
/// SAFETY: The raw pointers inside `PluginExtendedInfoStruct` point to
/// `&'static CStr` data allocated once and never freed. They are never written
/// to or deallocated, so sharing across threads is safe.
struct SyncExtendedInfo(PluginExtendedInfoStruct);
unsafe impl Send for SyncExtendedInfo {}
unsafe impl Sync for SyncExtendedInfo {}

static INFO_STRUCT_EXTENDED: OnceLock<SyncExtendedInfo> = OnceLock::new();
static INITIALIZED: OnceLock<()> = OnceLock::new();
static HANDLER: OnceLock<Box<dyn Any + Send + Sync>> = OnceLock::new();

use tracing::debug_span;
use tracing::trace_span;
use tracing::{debug, info, trace};

/// backtrace didn't seem to work. Maybe a problem with FFI. This is a hacky way to get the source
macro_rules! e {
    ($($arg:tt)*) => {{
        format!("{orig}\nSOURCE {file}:{line}:{column}", orig=format!($($arg)*),
        file = file!(),
        line = line!(),
        column = column!(),)
    }}
}

pub fn default_ffgl_entry<H: FFGLHandler + 'static>(
    function: Op,
    mut input_value: FFGLVal,
    instance: Option<&mut handler::Instance<H::Instance>>,
) -> Result<FFGLVal, Error> {
    INITIALIZED.get_or_init(|| {
        // Initialize the logger
        let _ = try_init_default_subscriber();
    });

    let handler = HANDLER
        .get_or_init(|| Box::new(H::init()))
        .downcast_ref::<H>()
        .context(e!("Handler type mismatch"))?;

    // Initialize plugin info if not already initialized
    let plugin_info = INFO.get_or_init(|| handler.plugin_info());

    let name = plugin_info.name_str();

    let _span = if !function.is_noisy() {
        debug_span!("entry", "fn" = ?function, name, "in" = unsafe { input_value.num })
    } else {
        trace_span!("entry", "fn" = ?function, name, "in" = unsafe { input_value.num })
    }
    .entered();

    // Initialize about and description strings
    let about = ABOUT
        .get_or_init(|| CString::new(plugin_info.about.clone()).expect("Invalid about string"));
    let description = DESCRIPTION.get_or_init(|| {
        CString::new(plugin_info.description.clone()).expect("Invalid description string")
    });

    // Initialize info structs
    let _info_struct = INFO_STRUCT.get_or_init(|| {
        INFO_STRUCT_EXTENDED
            .get_or_init(|| SyncExtendedInfo(info::plugin_info_extended(about, description)));
        info::plugin_info(
            // Safety: [u8; 4] and [i8; 4] have the same layout
            unsafe { &*(&plugin_info.unique_id as *const [u8; 4] as *const [i8; 4]) },
            // Safety: [u8; 16] and [i8; 16] have the same layout
            unsafe { &*(&plugin_info.name as *const [u8; 16] as *const [i8; 16]) },
            plugin_info.ty,
        )
    });

    let resp = match function {
        Op::GetPluginCaps => {
            let cap_num = unsafe { input_value.num };
            let cap: Option<PluginCapacity> = num::FromPrimitive::from_u32(cap_num);

            let result = match cap {
                Some(PluginCapacity::MinInputFrames) => FFGLVal { num: 0 },
                Some(PluginCapacity::MaxInputFrames) => FFGLVal { num: 1 },

                Some(PluginCapacity::ProcessOpenGl) => SupportVal::Supported.into(),
                Some(PluginCapacity::SetTime) => SupportVal::Supported.into(),

                Some(PluginCapacity::TopLeftTextureOrientation) => SupportVal::Supported.into(),

                _ => SupportVal::Unsupported.into(),
            };

            debug!(r = unsafe { result.num }, cap_num, "GetPluginCaps");

            result
        }

        Op::EnablePluginCap => {
            let cap_num = unsafe { input_value.num };
            let cap: Option<PluginCapacity> = num::FromPrimitive::from_u32(cap_num);

            let result: FFGLVal = match cap {
                Some(PluginCapacity::TopLeftTextureOrientation) => SuccessVal::Success.into(),
                _ => SuccessVal::Fail.into(),
            };

            debug!(r = unsafe { result.num }, cap_num, "EnablePluginCap");

            result
        }

        Op::GetNumParameters => FFGLVal {
            num: handler.num_params() as u32,
        },

        Op::GetParameterDefault => param(handler, input_value).default_val().into(),
        Op::GetParameterGroup => {
            let input: &GetStringStruct = unsafe { input_value.as_ref() };
            let buffer = input.stringBuffer;

            let group = H::param_info(handler, input.parameterNumber as usize).group();

            unsafe {
                copy_str_to_host_buffer(
                    buffer.address as *mut u8,
                    buffer.maxToWrite as usize,
                    group,
                )
            };

            debug!(g = group);

            SuccessVal::Success.into()
        }
        Op::GetParameterDisplayName => {
            let input: &GetStringStruct = unsafe { input_value.as_ref() };
            let buffer = input.stringBuffer;

            let display_name =
                H::param_info(handler, input.parameterNumber as usize).display_name();

            unsafe {
                copy_str_to_host_buffer(
                    buffer.address as *mut u8,
                    buffer.maxToWrite as usize,
                    display_name,
                )
            };

            debug!(d = display_name);

            SuccessVal::Success.into()
        }
        Op::GetParameterName => param(handler, input_value).name().into(),
        Op::GetParameterType => param(handler, input_value).param_type().into(),

        Op::GetParameter => instance
            .context(e!("No instance"))?
            .renderer
            .get_param(unsafe { input_value.num } as usize)
            .into(),

        Op::SetParameter => {
            let input: &SetParameterStruct = unsafe { input_value.as_ref() };
            let index = input.ParameterNumber;

            let index_usize = index as usize;

            // dunno why they store this in a u32, whatever..
            let new_value = f32::from_bits(unsafe { input.NewParameterValue.UIntValue });

            instance
                .context(e!("No instance"))?
                .renderer
                .set_param(index_usize, new_value);

            SuccessVal::Success.into()
        }
        Op::GetParameterRange => {
            let input: &mut GetRangeStruct = unsafe { (input_value).as_mut() };

            let index = input.parameterNumber;
            let p = handler.param_info(index as usize);

            input.range = RangeStruct {
                min: p.min(),
                max: p.max(),
            };

            SuccessVal::Success.into()
        }

        Op::GetNumParameterElements => {
            let index = unsafe { input_value.num } as usize;
            (handler.param_info(index).num_elements() as u32).into()
        }

        Op::GetParameterElementName => {
            let input: &mut GetParameterElementNameStruct = unsafe { input_value.as_mut() };

            let param_index = input.ParameterNumber;
            let elm_index = input.ElementNumber;

            handler
                .param_info(param_index as usize)
                .element_name(elm_index as usize)
                .into()
        }

        Op::GetParameterElementValue => {
            let input: &mut GetParameterElementValueStruct = unsafe { input_value.as_mut() };

            let param_index = input.ParameterNumber;
            let elm_index = input.ElementNumber;

            handler
                .param_info(param_index as usize)
                .element_value(elm_index as usize)
                .into()
        }

        Op::GetNumElementSeparators => 0u32.into(),

        Op::GetInfo => INFO_STRUCT.get().context(e!("No info"))?.into(),

        Op::GetExtendedInfo => {
            let ext = INFO_STRUCT_EXTENDED.get().context(e!("No extended info"))?;
            (&ext.0).into()
        }

        Op::InstantiateGL => {
            let viewport: &FFGLViewportStruct = unsafe { input_value.as_ref() };

            let data = FFGLData::new(viewport);
            let renderer = H::new_instance(handler, &data)
                .context("Failed to instantiate renderer")
                .context(format!(
                    "For {}",
                    std::str::from_utf8(&plugin_info.name).unwrap()
                ))?;

            let inst = handler::Instance {
                data,
                renderer,
                first_events_polled: false,
            };

            info!(
                id = ?plugin_info.unique_id,
                "Created INSTANCE:\n{inst:#?}",
            );

            FFGLVal::from_static(Box::leak(Box::<handler::Instance<H::Instance>>::new(inst)))
        }

        Op::DeinstantiateGL => {
            let inst = instance.context(e!("No instance"))?;

            debug!(?inst, "DEINSTGL");
            unsafe {
                drop(Box::from_raw(inst as *mut handler::Instance<H::Instance>));
            }

            SuccessVal::Success.into()
        }

        Op::ProcessOpenGL => {
            let gl_process_info: &ProcessOpenGLStruct = unsafe { input_value.as_ref() };

            let handler::Instance { data, renderer, .. } = instance.context(e!("No instance"))?;
            let gl_input = gl_process_info.into();

            renderer.draw(data, gl_input);

            SuccessVal::Success.into()
        }

        Op::SetTime => {
            let seconds: f64 = *unsafe { input_value.as_ref() };
            instance.context(e!("No instance"))?.data.set_time(seconds);
            SuccessVal::Success.into()
        }

        // This can be called before GLInitialize.
        Op::SetBeatInfo => {
            let beat_info: &SetBeatinfoStruct = unsafe { input_value.as_ref() };
            if let Some(inst) = instance {
                inst.data.set_beat(*beat_info);
                SuccessVal::Success.into()
            } else {
                SuccessVal::Fail.into()
            }
        }

        Op::Resize => {
            let viewport: &FFGLViewportStruct = unsafe { input_value.as_ref() };

            let handler::Instance { data, .. } = instance.context(e!("No instance"))?;
            data.viewport = *viewport;

            debug!(v = ?viewport, "RESIZE");
            SuccessVal::Success.into()
        }

        Op::Connect => SuccessVal::Success.into(),

        Op::GetParameterDisplay => {
            // Plugins can override the host-rendered slider value via
            // `param_display_value` to show e.g. "5.7 px" instead of
            // the raw 0.5–16.0 internal float. None / no instance =
            // tell the host to fall back to its own formatting.
            let idx = unsafe { input_value.num } as usize;
            let display = match instance {
                Some(inst) => inst.renderer.param_display_value(idx),
                None => None,
            };
            match display {
                Some(s) => DISPLAY_BUFFER.with(|buf| {
                    *buf.borrow_mut() = CString::new(s).unwrap_or_default();
                    let ptr = buf.borrow().as_ptr();
                    FFGLVal::from(ptr)
                }),
                None => SuccessVal::Fail.into(),
            }
        }

        Op::GetParameterVisibility => {
            // Resolume queries visibility while building the panel
            // template at plugin LOAD time, before any instance has
            // been created (`Op::InstantiateGL` only fires when the
            // user drops the effect onto a layer). The host CACHES
            // the pre-instance answer and only re-queries on
            // FF_EVENT_FLAG_VISIBILITY events that the plugin pushes —
            // and event draining is gated on layer rendering. So if
            // the user drops the effect on an idle (non-playing)
            // layer, no events will ever drain and the panel renders
            // whatever we returned at panel build, period.
            //
            // Returning a static `true` for every pre-instance query
            // means the panel always shows everything visible until
            // the layer plays. To make the panel render correctly on
            // idle layers (the common case for unmapped clips), the
            // handler exposes `param_default_visible` — the plugin's
            // visibility computation for DEFAULT param values, no
            // instance state needed. We use that pre-instance and
            // fall through to per-instance `param_visible` once an
            // instance exists.
            let idx = unsafe { input_value.num } as usize;
            let visible = match instance {
                Some(inst) => inst.renderer.param_visible(idx),
                None => handler.param_default_visible(idx),
            };
            FFGLVal {
                num: if visible { FF_TRUE } else { FF_FALSE },
            }
        }

        Op::GetParameterEvents => {
            let events_struct: &mut GetParamEventsStruct =
                unsafe { (input_value).as_mut() };
            let max = events_struct.numEvents as usize;
            // Pre-instance poll: empty queue (no instance state exists).
            // First post-instance poll: synthesize visibility events for
            // every currently-hidden param. Resolume queries visibility
            // ONCE at panel load (before instance creation), where our
            // fallback returns "visible" — so by the time the user sees
            // the panel after dropping the effect, hidden params would
            // erroneously show until something else triggered a re-query.
            // Auto-emitting on the first events poll closes that gap and
            // works without any plugin-side opt-in.
            let events: Vec<(u32, u64)> = match instance {
                Some(inst) => {
                    let mut events = Vec::new();
                    if !inst.first_events_polled {
                        inst.first_events_polled = true;
                        let n = handler.num_params();
                        for i in 0..n {
                            if events.len() >= max { break; }
                            if !inst.renderer.param_visible(i) {
                                events.push((i as u32, FF_EVENT_FLAG_VISIBILITY));
                            }
                        }
                    }
                    let remaining = max.saturating_sub(events.len());
                    if remaining > 0 {
                        events.extend(inst.renderer.consume_param_events(remaining));
                    }
                    events
                }
                None => Vec::new(),
            };
            let buf = events_struct.events;
            for (i, (param_num, flags)) in events.iter().enumerate() {
                unsafe {
                    (*buf.add(i)).ParameterNumber = *param_num;
                    (*buf.add(i)).eventFlags = *flags;
                }
            }
            events_struct.numEvents = events.len() as u32;
            SuccessVal::Success.into()
        }

        Op::Instantiate | Op::Deinstantiate | Op::ProcessFrame | Op::ProcessFrameCopy => {
            SuccessVal::Fail.into()
        }

        Op::InitialiseV2 => SuccessVal::Success.into(),
        Op::Initialise => SuccessVal::Success.into(),
        Op::Deinitialise => SuccessVal::Success.into(),

        _ => SuccessVal::Fail.into(),
    };

    if !function.is_noisy() {
        debug!(r = unsafe { resp.num }, "DONE");
    } else {
        trace!(r = unsafe { resp.num }, "DONE");
    }

    Ok(resp)
}
