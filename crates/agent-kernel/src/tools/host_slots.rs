//! Host-owned capability slots on [`ToolCtx`].
//!
//! Every slot here is a capability the *host* decides whether to provide:
//! the question channel, the UI sink, the subagent spawner, the owning session
//! and the plugin router. Product backends live in `agent-capabilities`;
//! the context stores whichever services its host installs in `Extensions`.
//!
//! Keeping these out of `registry` is what lets the tool contract be extracted
//! without naming product types.

use std::sync::Arc;

use super::registry::{ToolCtx, ToolError};

/// The structured-question channel — `ask_question` parks a oneshot here,
/// emits `QuestionAsked` through `ui_tx`, and awaits the answer with the
/// same cancel poll as `wait_for_decision`. Headless contexts get
/// `ui_tx: None` → the tool refuses instead of deadlocking.
pub struct AskChannel {
    /// The session's UI event channel — `None` where nobody can answer
    /// (delegated subagents, tests).
    pub ui_tx: Option<crate::channels::UiSink>,
    pending: std::sync::Mutex<Option<(u64, tokio::sync::oneshot::Sender<String>)>>,
    next_id: std::sync::atomic::AtomicU64,
}

impl AskChannel {
    pub fn new(ui_tx: Option<crate::channels::UiSink>) -> Self {
        Self {
            ui_tx,
            pending: std::sync::Mutex::new(None),
            next_id: std::sync::atomic::AtomicU64::new(1),
        }
    }

    /// Interactive? `None` ui_tx means headless (subagent, test) — the
    /// tool checks this BEFORE building a request so a child agent can't
    /// park on a oneshot nobody can ever answer.
    pub fn is_available(&self) -> bool {
        self.ui_tx.is_some()
    }

    /// Emit the card, park the answer oneshot, await it (or cancellation).
    pub async fn ask(
        &self,
        question: String,
        options: Vec<agent_ipc::events::AskOption>,
        cancel: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    ) -> Result<String, ToolError> {
        let Some(ui_tx) = &self.ui_tx else {
            return Err(ToolError::Failed(
                "ask_question unavailable — no UI channel in this context".into(),
            ));
        };
        let id = self
            .next_id
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let (tx, rx) = tokio::sync::oneshot::channel();
        *self.pending.lock().unwrap() = Some((id, tx));
        if ui_tx
            .send(agent_ipc::events::UiEvent::QuestionAsked {
                request_id: id,
                question,
                options,
            })
            .is_err()
        {
            *self.pending.lock().unwrap() = None;
            return Err(ToolError::Failed(
                "ask_question failed — UI event channel is closed".into(),
            ));
        }
        let mut rx = std::pin::pin!(rx);
        loop {
            tokio::select! {
                r = &mut rx => {
                    return r.map_err(|_| ToolError::Failed(
                        "question channel closed before an answer arrived".into(),
                    ));
                }
                _ = tokio::time::sleep(std::time::Duration::from_millis(100)) => {
                    if cancel.as_ref().is_some_and(|c| c.load(std::sync::atomic::Ordering::Relaxed)) {
                        *self.pending.lock().unwrap() = None;
                        return Err(ToolError::Failed("cancelled".into()));
                    }
                }
            }
        }
    }

    /// `AnswerQuestion` writes here — resolves the parked oneshot when the
    /// id matches (a stale answer for a cleared question is ignored).
    pub fn answer(&self, request_id: u64, answer: String) -> bool {
        match self.pending.lock().unwrap().take() {
            Some((id, tx)) if id == request_id => tx.send(answer).is_ok(),
            Some(p) => {
                *self.pending.lock().unwrap() = Some(p);
                false
            }
            None => false,
        }
    }

    /// Cancel path — dropping the oneshot wakes `ask()` with an error.
    pub fn clear(&self) {
        *self.pending.lock().unwrap() = None;
    }
}

/// The interactive question channel. Installed by the host that owns a user
/// to answer; absent means headless (a subagent, a test), and `ask_question`
/// refuses instead of parking on an answer nobody can give.
pub struct AskSlot(pub Arc<AskChannel>);

/// The session UI event sink. Tools that fan out (batch items, subagent
/// progress) emit their internal calls here so they stay visible and
/// auditable instead of running silently; absent in headless contexts.
pub struct UiSlot(pub crate::channels::UiSink);

/// Subagent spawner — `delegate` runs a fresh-context child engine. Absent
/// where delegation is unavailable (tests, child contexts).
pub struct SubagentSlot(pub crate::tools::delegate::SubagentSpawner);

/// The owning session's id and per-workspace store. Tools that keep
/// per-session scratch state (`todo`) write next to the session's history
/// file; `delegate` uses the id to route steering to the right owner.
pub struct SessionSlot(pub i64);

/// Plugin router handle. `batch_execute` consults it to tell a plugin tool from
/// an unknown name. Absent where no plugin router is installed.
pub struct PluginSlot(pub crate::engine::PluginHandle);

impl ToolCtx {
    /// The question channel, or a refusal when the context has no one to ask.
    pub fn ask(&self) -> Result<Arc<AskChannel>, ToolError> {
        self.ext
            .get::<AskSlot>()
            .map(|s| s.0.clone())
            .ok_or_else(|| {
                ToolError::Failed("ask_question requires an interactive user session".into())
            })
    }

    /// Install the question channel. Hosts keep their own handle to the same
    /// `Arc`, so answers routed through the host reach the parked tool call.
    pub fn install_ask(&mut self, ask: Arc<AskChannel>) {
        self.ext.insert(Arc::new(AskSlot(ask)));
    }

    /// The session UI event sink, if this context has one. Tools that fan out
    /// emit their internal call events here; `None` means headless, where they
    /// skip emitting rather than inventing a sink.
    pub fn ui(&self) -> Option<crate::channels::UiSink> {
        self.ext.get::<UiSlot>().map(|s| s.0.clone())
    }

    /// Install the session UI event sink.
    pub fn install_ui(&mut self, ui: crate::channels::UiSink) {
        self.ext.insert(Arc::new(UiSlot(ui)));
    }

    /// The subagent spawner, if this context can delegate.
    pub fn subagent(&self) -> Option<Arc<SubagentSlot>> {
        self.ext.get::<SubagentSlot>()
    }

    /// Install the spawner that backs `delegate`.
    pub fn install_subagent(&mut self, spawner: crate::tools::delegate::SubagentSpawner) {
        self.ext.insert(Arc::new(SubagentSlot(spawner)));
    }

    /// Attach the owning session's id. The kernel only needs the number — it
    /// scopes a delegated child's steering to its owner; where a session's
    /// state is persisted is the host's business (see `TodoStorage`).
    pub fn with_session(mut self, id: i64) -> Self {
        self.ext.insert(Arc::new(SessionSlot(id)));
        self
    }

    /// The owning session's id, if attached.
    pub fn session(&self) -> Option<i64> {
        self.ext.get::<SessionSlot>().map(|s| s.0)
    }

    /// The plugin router handle, if one is installed.
    pub fn plugins(&self) -> Option<crate::engine::PluginHandle> {
        self.ext.get::<PluginSlot>().map(|s| s.0.clone())
    }

    /// The engine's `set_plugins` replaces its `Arc<ToolCtx>` wholesale —
    /// no interior mutability on the ctx itself.
    pub fn with_plugins(&self, plugins: crate::engine::PluginHandle) -> ToolCtx {
        let cancel = self
            .cancel
            .clone()
            .unwrap_or_else(|| Arc::new(std::sync::atomic::AtomicBool::new(false)));
        let mut next = self.with_cancel(cancel);
        next.ext.insert(Arc::new(PluginSlot(plugins)));
        next
    }
}
