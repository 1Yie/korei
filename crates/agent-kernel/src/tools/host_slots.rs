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

/// The user's answer to one approval card.
#[derive(Debug, Clone, Default)]
pub struct ToolVerdict {
    pub approved: bool,
    /// Which offered [`agent_ipc::events::ApprovalAction`] was picked, if the
    /// card had any. `None` is a plain 允许.
    pub action: Option<String>,
}

/// The host's answer to "may this path be granted?" — `action` is the id the
/// card offered, `path` the path the tool refused. Installed by a host that
/// can widen a session's reach (linked workspaces); without it the kernel has
/// nowhere to send the question, so a refusal stays a plain refusal.
pub struct ScopeGrantSlot(
    pub Arc<dyn Fn(&str, &str, &str) -> Result<(), String> + Send + Sync>,
);

use super::registry::{ToolCtx, ToolError};

/// How long a question may stay unanswered before the kernel gives it back —
/// a UI that never answers must not park the turn forever.
const DEFAULT_ASK_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(600);

/// How many timed-out questions one turn is remembered for the repeat guard.
const TIMED_OUT_QUESTIONS: usize = 32;

/// Render a timeout for the model-facing texts: whole seconds when the
/// setting is whole, otherwise its fractional value.
pub(crate) fn secs_label(d: std::time::Duration) -> String {
    if d.subsec_nanos() == 0 {
        d.as_secs().to_string()
    } else {
        format!("{:.3}", d.as_secs_f64())
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_string()
    }
}

/// What came back from [`AskChannel::ask`].
#[derive(Debug, Clone)]
pub enum AskOutcome {
    /// The user answered; the text is what they clicked or typed.
    Answered(String),
    /// Nobody answered before the channel's timeout.
    TimedOut {
        /// How long the card waited.
        after: std::time::Duration,
        /// The same question already timed out, so no card was shown.
        repeated: bool,
    },
}

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
    /// How long a question may stay unanswered — zero means no deadline.
    timeout: std::sync::Mutex<std::time::Duration>,
    /// Normalized texts that already timed out: the same question is handed
    /// the timeout again instead of a second card nobody will answer.
    timed_out: std::sync::Mutex<Vec<String>>,
}

impl AskChannel {
    pub fn new(ui_tx: Option<crate::channels::UiSink>) -> Self {
        Self {
            ui_tx,
            pending: std::sync::Mutex::new(None),
            next_id: std::sync::atomic::AtomicU64::new(1),
            timeout: std::sync::Mutex::new(DEFAULT_ASK_TIMEOUT),
            timed_out: std::sync::Mutex::new(Vec::new()),
        }
    }

    /// Bound how long a question may stay unanswered. Zero = no deadline.
    pub fn set_timeout(&self, d: std::time::Duration) {
        *self.timeout.lock().unwrap_or_else(|e| e.into_inner()) = d;
    }

    /// Forget which questions timed out. The host calls this at turn start so
    /// a question asked in a new turn is shown again.
    pub fn reset_timed_out(&self) {
        self.timed_out
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
    }

    /// Interactive? `None` ui_tx means headless (subagent, test) — the
    /// tool checks this BEFORE building a request so a child agent can't
    /// park on a oneshot nobody can ever answer.
    pub fn is_available(&self) -> bool {
        self.ui_tx.is_some()
    }

    /// Emit the card, park the answer oneshot, await it (or cancellation, or
    /// the timeout).
    pub async fn ask(
        &self,
        question: String,
        options: Vec<agent_ipc::events::AskOption>,
        cancel: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    ) -> Result<AskOutcome, ToolError> {
        let Some(ui_tx) = &self.ui_tx else {
            return Err(ToolError::Failed(
                "ask_question unavailable — no UI channel in this context".into(),
            ));
        };
        let timeout = *self.timeout.lock().unwrap_or_else(|e| e.into_inner());
        let normalized = question
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .to_lowercase();
        if self
            .timed_out
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .any(|q| q == &normalized)
        {
            // The same question can only run out the clock again — hand the
            // model the timeout now instead of drawing a card nobody answers.
            return Ok(AskOutcome::TimedOut {
                after: std::time::Duration::ZERO,
                repeated: true,
            });
        }
        let id = self
            .next_id
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let (tx, rx) = tokio::sync::oneshot::channel();
        *self.pending.lock().unwrap_or_else(|e| e.into_inner()) = Some((id, tx));
        if ui_tx
            .send(agent_ipc::events::UiEvent::QuestionAsked {
                request_id: id,
                question,
                options,
            })
            .is_err()
        {
            *self.pending.lock().unwrap_or_else(|e| e.into_inner()) = None;
            return Err(ToolError::Failed(
                "ask_question failed — UI event channel is closed".into(),
            ));
        }
        let mut rx = std::pin::pin!(rx);
        let deadline = (!timeout.is_zero()).then(|| tokio::time::Instant::now() + timeout);
        loop {
            tokio::select! {
                r = &mut rx => {
                    return r.map(AskOutcome::Answered).map_err(|_| ToolError::Failed(
                        "question channel closed before an answer arrived".into(),
                    ));
                }
                _ = tokio::time::sleep(std::time::Duration::from_millis(100)) => {
                    if cancel.as_ref().is_some_and(|c| c.load(std::sync::atomic::Ordering::Relaxed)) {
                        *self.pending.lock().unwrap_or_else(|e| e.into_inner()) = None;
                        return Err(ToolError::Failed("cancelled".into()));
                    }
                    let Some(deadline) = deadline else { continue };
                    if tokio::time::Instant::now() < deadline {
                        continue;
                    }
                    // Exactly one side wins: if `answer` already took the
                    // entry, the text is on its way and must still be
                    // returned; otherwise the parked card is ours to close.
                    let answered_first = {
                        let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
                        match pending.as_ref() {
                            Some((parked, _)) if *parked == id => {
                                *pending = None;
                                false
                            }
                            // Our card is no longer parked — cleared by the host.
                            Some(_) | None => true,
                        }
                    };
                    if answered_first {
                        return rx.await.map(AskOutcome::Answered).map_err(|_| ToolError::Failed(
                            "question channel closed before an answer arrived".into(),
                        ));
                    }
                    {
                        let mut timed_out =
                            self.timed_out.lock().unwrap_or_else(|e| e.into_inner());
                        if timed_out.len() >= TIMED_OUT_QUESTIONS {
                            timed_out.remove(0);
                        }
                        timed_out.push(normalized);
                    }
                    let _ = ui_tx.send(agent_ipc::events::UiEvent::QuestionTimedOut {
                        request_id: id,
                    });
                    return Ok(AskOutcome::TimedOut { after: timeout, repeated: false });
                }
            }
        }
    }

    /// `AnswerQuestion` writes here — resolves the parked oneshot when the
    /// id matches (a stale answer for a cleared question is ignored).
    pub fn answer(&self, request_id: u64, answer: String) -> bool {
        match self.pending.lock().unwrap_or_else(|e| e.into_inner()).take() {
            Some((id, tx)) if id == request_id => tx.send(answer).is_ok(),
            Some(p) => {
                *self.pending.lock().unwrap_or_else(|e| e.into_inner()) = Some(p);
                false
            }
            None => false,
        }
    }

    /// Cancel path — dropping the oneshot wakes `ask()` with an error.
    pub fn clear(&self) {
        *self.pending.lock().unwrap_or_else(|e| e.into_inner()) = None;
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

    /// Install the grant callback. The kernel offers `action` as an approval
    /// choice, so the ids it passes back are the ones the host put on the card.
    pub fn install_scope_grant(
        &mut self,
        grant: Arc<dyn Fn(&str, &str, &str) -> Result<(), String> + Send + Sync>,
    ) {
        self.ext.insert(Arc::new(ScopeGrantSlot(grant)));
    }

    /// Can the host grant more? Absent → a refusal stays a refusal and no
    /// approval card is shown for it.
    pub fn has_scope_grant(&self) -> bool {
        self.ext.get::<ScopeGrantSlot>().is_some()
    }

    /// Perform the chosen grant. `None` when no host installed one.
    /// `access` is `"read"` or `"readwrite"`: a grant for reading a
    /// dependency is not the same permission as one for editing it.
    pub fn grant_scope(&self, action: &str, path: &str, access: &str) -> Option<Result<(), String>> {
        self.ext.get::<ScopeGrantSlot>().map(|s| (s.0)(action, path, access))
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
