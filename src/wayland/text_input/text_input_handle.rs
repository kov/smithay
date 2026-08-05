use std::mem;
use std::sync::{Arc, Mutex};

use tracing::debug;
use wayland_protocols::wp::text_input::zv3::server::zwp_text_input_v3::{
    self, ChangeCause, ContentHint, ContentPurpose, ZwpTextInputV3,
};
use wayland_server::backend::{ClientId, ObjectId};
use wayland_server::{Dispatch, Resource, protocol::wl_surface::WlSurface};

use crate::input::SeatHandler;
use crate::utils::{Logical, Rectangle};
use crate::wayland::input_method::InputMethodHandle;

use super::TextInputManagerState;

/// What a `zwp_text_input_v3` client asked for, as plain data.
///
/// Emitted only to a compositor-internal input method registered with
/// [`TextInputHandle::set_internal_input_method`]. A Wayland `zwp_input_method_v2` client gets
/// the same information through its own protocol objects instead; both are driven from the same
/// place so the two cannot drift.
///
/// These arrive in the order the client's atomic `commit` applied them, followed by
/// [`TextInputEvent::Done`].
#[derive(Debug, Clone, PartialEq)]
pub enum TextInputEvent {
    /// The client enabled text input on the focused surface. The surface is
    /// [`TextInputHandle::focus`].
    Enabled,
    /// The client disabled text input. No further state applies until the next `Enabled`.
    Disabled,
    /// Text around the cursor. `cursor` and `anchor` are **byte** offsets into `text`.
    SurroundingText {
        /// The surrounding text itself.
        text: String,
        /// Byte offset of the cursor within `text`.
        cursor: u32,
        /// Byte offset of the selection anchor within `text`.
        anchor: u32,
    },
    /// Why the surrounding text changed.
    TextChangeCause(ChangeCause),
    /// What kind of text the client expects.
    ContentType {
        /// Behavior hints.
        hint: ContentHint,
        /// The purpose of the field.
        purpose: ContentPurpose,
    },
    /// Where the cursor is, in surface-local coordinates — where a candidate popup goes.
    CursorRectangle(Rectangle<i32, Logical>),
    /// End of one atomic batch. Everything since the previous `Done` applies together.
    Done,
}

/// A compositor-internal input method: somewhere to hand [`TextInputEvent`]s.
///
/// Deliberately a plain callback rather than a trait method taking `&mut D`. An internal input
/// method is nearly always talking to something off-thread (IBus over D-Bus, say), so the sink
/// is a channel send; requiring compositor state here would buy nothing and would put a trait
/// bound on every `Dispatch` impl in the tree.
pub type InternalInputMethod = Arc<dyn Fn(TextInputEvent) + Send + Sync>;

#[derive(Default)]
pub(crate) struct TextInput {
    instances: Vec<Instance>,
    focus: Option<WlSurface>,
    active_text_input_id: Option<ObjectId>,
    internal_im: Option<InternalInputMethod>,
}

impl std::fmt::Debug for TextInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TextInput")
            .field("instances", &self.instances)
            .field("focus", &self.focus)
            .field("active_text_input_id", &self.active_text_input_id)
            .field("internal_im", &self.internal_im.is_some())
            .finish()
    }
}

impl TextInput {
    fn with_focused_client_all_text_inputs<F>(&mut self, mut f: F)
    where
        F: FnMut(&ZwpTextInputV3, &WlSurface, u32),
    {
        if let Some(surface) = self.focus.as_ref().filter(|surface| surface.is_alive()) {
            for text_input in self.instances.iter() {
                let instance_id = text_input.instance.id();
                if instance_id.same_client_as(&surface.id()) {
                    f(&text_input.instance, surface, text_input.serial);
                }
            }
        };
    }

    fn with_active_text_input<F>(&mut self, mut f: F)
    where
        F: FnMut(&ZwpTextInputV3, &WlSurface, u32),
    {
        let active_id = match &self.active_text_input_id {
            Some(active_text_input_id) => active_text_input_id,
            None => return,
        };

        let surface = match self.focus.as_ref().filter(|surface| surface.is_alive()) {
            Some(surface) => surface,
            None => return,
        };

        let surface_id = surface.id();
        if let Some(text_input) = self
            .instances
            .iter()
            .filter(|instance| instance.instance.id().same_client_as(&surface_id))
            .find(|instance| &instance.instance.id() == active_id)
        {
            f(&text_input.instance, surface, text_input.serial);
        }
    }
}

/// Handle to text input instances
#[derive(Default, Debug, Clone)]
pub struct TextInputHandle {
    pub(crate) inner: Arc<Mutex<TextInput>>,
}

impl TextInputHandle {
    pub(super) fn add_instance(&self, instance: &ZwpTextInputV3) {
        let mut inner = self.inner.lock().unwrap();
        inner.instances.push(Instance {
            instance: instance.clone(),
            serial: 0,
            pending_state: Default::default(),
        });
    }

    fn increment_serial(&self, text_input: &ZwpTextInputV3) {
        if let Some(instance) = self
            .inner
            .lock()
            .unwrap()
            .instances
            .iter_mut()
            .find(|instance| instance.instance == *text_input)
        {
            instance.serial += 1
        }
    }

    /// Register a compositor-internal input method, or clear it with `None`.
    ///
    /// Without one, and without a `zwp_input_method_v2` client, every text-input request is
    /// discarded — the client is told nothing is listening and its own composition (dead keys,
    /// Compose) is the only thing that runs. Registering here makes the compositor the input
    /// method: it starts receiving [`TextInputEvent`]s and may drive the client with
    /// [`Self::with_active_text_input`] and [`Self::done`].
    pub fn set_internal_input_method(&self, sink: Option<InternalInputMethod>) {
        self.inner.lock().unwrap().internal_im = sink;
    }

    /// Whether a compositor-internal input method is registered.
    pub fn has_internal_input_method(&self) -> bool {
        self.inner.lock().unwrap().internal_im.is_some()
    }

    /// The internal sink, cloned out so the caller can invoke it without holding the lock.
    ///
    /// Calling a compositor callback with the mutex held is a deadlock waiting to happen: the
    /// sink is free to turn around and ask this same handle a question.
    fn internal_sink(&self) -> Option<InternalInputMethod> {
        self.inner.lock().unwrap().internal_im.clone()
    }

    /// Return the currently focused surface.
    pub fn focus(&self) -> Option<WlSurface> {
        self.inner.lock().unwrap().focus.clone()
    }

    /// Advance the focus for the client to `surface`.
    ///
    /// This doesn't send any 'enter' or 'leave' events.
    pub fn set_focus(&self, surface: Option<WlSurface>) {
        self.inner.lock().unwrap().focus = surface;
    }

    /// Send `leave` on the text-input instance for the currently focused
    /// surface.
    pub fn leave(&self) {
        let mut inner = self.inner.lock().unwrap();
        // Leaving clears the active text input.
        inner.active_text_input_id = None;
        // NOTE: we implement it in a symmetrical way with `enter`.
        inner.with_focused_client_all_text_inputs(|text_input, focus, _| {
            text_input.leave(focus);
        });
    }

    /// Send `enter` on the text-input instance for the currently focused
    /// surface.
    pub fn enter(&self) {
        let mut inner = self.inner.lock().unwrap();
        // NOTE: protocol states that if we have multiple text inputs enabled, `enter` must
        // be send for each of them.
        inner.with_focused_client_all_text_inputs(|text_input, focus, _| {
            text_input.enter(focus);
        });
    }

    /// The `discard_state` is used when the input-method signaled that
    /// the state should be discarded and wrong serial sent.
    pub fn done(&self, discard_state: bool) {
        let mut inner = self.inner.lock().unwrap();
        inner.with_active_text_input(|text_input, _, serial| {
            if discard_state {
                debug!("discarding text-input state due to serial");
                // Discarding is done by sending non-matching serial.
                text_input.done(0);
            } else {
                text_input.done(serial);
            }
        });
    }

    /// Access the text-input instances for the currently focused surface.
    pub fn with_focused_text_input<F>(&self, mut f: F)
    where
        F: FnMut(&ZwpTextInputV3, &WlSurface),
    {
        let mut inner = self.inner.lock().unwrap();
        inner.with_focused_client_all_text_inputs(|ti, surface, _| {
            f(ti, surface);
        });
    }

    /// Access the active text-input instance for the currently focused surface.
    pub fn with_active_text_input<F>(&self, mut f: F)
    where
        F: FnMut(&ZwpTextInputV3, &WlSurface),
    {
        let mut inner = self.inner.lock().unwrap();
        inner.with_active_text_input(|ti, surface, _| {
            f(ti, surface);
        });
    }

    /// Call the callback with the serial of the active text_input or with the passed
    /// `default` one when empty.
    pub(crate) fn active_text_input_serial_or_default<F>(&self, default: u32, mut callback: F)
    where
        F: FnMut(u32),
    {
        let mut inner = self.inner.lock().unwrap();
        let mut should_default = true;
        inner.with_active_text_input(|_, _, serial| {
            should_default = false;
            callback(serial);
        });
        if should_default {
            callback(default)
        }
    }
}

/// User data of ZwpTextInputV3 object
#[derive(Debug)]
pub struct TextInputUserData {
    pub(super) handle: TextInputHandle,
    pub(crate) input_method_handle: InputMethodHandle,
}

impl<D> Dispatch<ZwpTextInputV3, TextInputUserData, D> for TextInputManagerState
where
    D: Dispatch<ZwpTextInputV3, TextInputUserData>,
    D: SeatHandler,
    D: 'static,
{
    fn request(
        state: &mut D,
        _client: &wayland_server::Client,
        resource: &ZwpTextInputV3,
        request: zwp_text_input_v3::Request,
        data: &TextInputUserData,
        _dhandle: &wayland_server::DisplayHandle,
        _data_init: &mut wayland_server::DataInit<'_, D>,
    ) {
        // Always increment serial to not desync with clients.
        if matches!(request, zwp_text_input_v3::Request::Commit) {
            data.handle.increment_serial(resource);
        }

        // A compositor-internal input method counts as an IME: it is the thing that will turn
        // these requests into preedit and commit strings.
        let internal_im = data.handle.internal_sink();

        // Discard requests without any active input method instance.
        if !data.input_method_handle.has_instance() && internal_im.is_none() {
            debug!("discarding text-input request without IME running");
            return;
        }

        let focus = match data.handle.focus() {
            Some(focus) if focus.id().same_client_as(&resource.id()) => focus,
            _ => {
                debug!("discarding text-input request for unfocused client");
                return;
            }
        };

        let mut guard = data.handle.inner.lock().unwrap();
        let pending_state = match guard.instances.iter_mut().find_map(|instance| {
            if instance.instance == *resource {
                Some(&mut instance.pending_state)
            } else {
                None
            }
        }) {
            Some(pending_state) => pending_state,
            None => {
                debug!("got request for untracked text-input");
                return;
            }
        };

        match request {
            zwp_text_input_v3::Request::Enable => {
                pending_state.enable = Some(true);
            }
            zwp_text_input_v3::Request::Disable => {
                pending_state.enable = Some(false);
            }
            zwp_text_input_v3::Request::SetSurroundingText { text, cursor, anchor } => {
                pending_state.surrounding_text = Some((text, cursor as u32, anchor as u32));
            }
            zwp_text_input_v3::Request::SetTextChangeCause { cause } => {
                // Guard against clients sending us unknown values from future versions.
                let cause = cause.into_result().unwrap_or(ChangeCause::Other);
                pending_state.text_change_cause = Some(cause);
            }
            zwp_text_input_v3::Request::SetContentType { hint, purpose } => {
                // Guard against clients sending us unknown values from future versions.
                let hint = ContentHint::from_bits_truncate(u32::from(hint));
                let purpose = purpose.into_result().unwrap_or(ContentPurpose::Normal);
                pending_state.content_type = Some((hint, purpose));
            }
            zwp_text_input_v3::Request::SetCursorRectangle { x, y, width, height } => {
                pending_state.cursor_rectangle = Some(Rectangle::new((x, y).into(), (width, height).into()));
            }
            zwp_text_input_v3::Request::Commit => {
                let mut new_state = mem::take(pending_state);
                let _ = pending_state;
                let active_text_input_id = &mut guard.active_text_input_id;

                if active_text_input_id.is_some() && *active_text_input_id != Some(resource.id()) {
                    debug!("discarding text_input request since we already have an active one");
                    return;
                }

                // The internal input method is notified alongside the Wayland one throughout,
                // rather than in a branch of its own, so the two can never fall out of step.
                let notify = |event: TextInputEvent| {
                    if let Some(sink) = internal_im.as_ref() {
                        sink(event);
                    }
                };

                match new_state.enable {
                    Some(true) => {
                        *active_text_input_id = Some(resource.id());
                        // Drop the guard before calling to other subsystem.
                        drop(guard);
                        data.input_method_handle.activate_input_method(state, &focus);
                        notify(TextInputEvent::Enabled);
                    }
                    Some(false) => {
                        *active_text_input_id = None;
                        // Drop the guard before calling to other subsystem.
                        drop(guard);
                        data.input_method_handle.deactivate_input_method(state);
                        notify(TextInputEvent::Disabled);
                        return;
                    }
                    None => {
                        if *active_text_input_id != Some(resource.id()) {
                            debug!("discarding text_input requests before enabling it");
                            return;
                        }

                        // Drop the guard before calling to other subsystems later on.
                        drop(guard);
                    }
                }

                if let Some((text, cursor, anchor)) = new_state.surrounding_text.take() {
                    notify(TextInputEvent::SurroundingText {
                        text: text.clone(),
                        cursor,
                        anchor,
                    });
                    data.input_method_handle.with_instance(move |input_method| {
                        input_method.object.surrounding_text(text, cursor, anchor)
                    });
                }

                if let Some(cause) = new_state.text_change_cause.take() {
                    notify(TextInputEvent::TextChangeCause(cause));
                    data.input_method_handle.with_instance(move |input_method| {
                        input_method.object.text_change_cause(cause);
                    });
                }

                if let Some((hint, purpose)) = new_state.content_type.take() {
                    notify(TextInputEvent::ContentType { hint, purpose });
                    data.input_method_handle.with_instance(move |input_method| {
                        input_method.object.content_type(hint, purpose);
                    });
                }

                if let Some(rect) = new_state.cursor_rectangle.take() {
                    notify(TextInputEvent::CursorRectangle(rect));
                    data.input_method_handle
                        .set_text_input_rectangle::<D>(state, rect);
                }

                notify(TextInputEvent::Done);
                data.input_method_handle.with_instance(|input_method| {
                    input_method.done();
                });
            }
            zwp_text_input_v3::Request::Destroy => {
                // Nothing to do
            }
            _ => unreachable!(),
        }
    }

    fn destroyed(state: &mut D, _client: ClientId, text_input: &ZwpTextInputV3, data: &TextInputUserData) {
        let destroyed_id = text_input.id();
        let deactivate_im = {
            let mut inner = data.handle.inner.lock().unwrap();
            inner.instances.retain(|inst| inst.instance.id() != destroyed_id);
            let destroyed_focused = inner
                .focus
                .as_ref()
                .map(|focus| focus.id().same_client_as(&destroyed_id))
                .unwrap_or(true);

            // Deactivate IM when we either lost focus entirely or destroyed text-input for the
            // currently focused client.
            destroyed_focused
                && !inner
                    .instances
                    .iter()
                    .any(|inst| inst.instance.id().same_client_as(&destroyed_id))
        };

        if deactivate_im {
            data.input_method_handle.deactivate_input_method(state);
        }
    }
}

#[derive(Debug)]
struct Instance {
    instance: ZwpTextInputV3,
    serial: u32,
    pending_state: TextInputState,
}

#[derive(Debug, Default)]
struct TextInputState {
    enable: Option<bool>,
    surrounding_text: Option<(String, u32, u32)>,
    content_type: Option<(ContentHint, ContentPurpose)>,
    cursor_rectangle: Option<Rectangle<i32, Logical>>,
    text_change_cause: Option<ChangeCause>,
}
