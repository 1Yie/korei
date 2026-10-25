//! `AgentState` machine + the ReAct turn loop.
//!
//! ```text
//! Idle→ScanningWorkspace→Reasoning→StreamingToken
//!     →{Finished | AwaitingToolConfirmation→ExecutingTool→Reasoning}
//! ```
//!
//! `Failed` is terminal per turn, not per session. The loop samples, runs the
//! returned tool calls, appends their results as `tool` messages and re-samples
//! until a turn ends with none pending.

use std::collections::{HashSet, VecDeque};
use std::sync::Arc;

use agent_ipc::{AgentState, UiEvent};
use agent_llm_core::sampler::{SampleRequest, Sampler, SamplerEvent};
use agent_llm_core::types::{ChatMessage, StreamChunk, ToolCallAssembler};
use tokio::sync::mpsc;
use tracing::instrument;

use crate::compaction::{self, CompactionSuppressor};
use crate::permissions::{Decision, PermissionGate};
use crate::tools::{ToolCtx, ToolRegistry};

/// The engine's observable side — channels + sink for UI events.
pub struct EngineIo {
    pub ui_tx: crate::channels::UiSink,
    /// Mid-turn steering texts — the host forwards its steer command
    /// into this channel; the engine drains it between tool calls.
    pub steer_rx: mpsc::Receiver<String>,
    /// Host-pushed notices (a background job finishing, a watch firing).
    /// Unlike `steer_rx` this carries no user intent: the engine appends each
    /// text as a host row, never as a user message and never as a steer. A
    /// running turn drains it between tool calls; an idle host wakes the
    /// model on it with [`Engine::run_notice_turn`].
    pub notice_rx: mpsc::Receiver<String>,
    /// Cooperative cancel flag — the UI/actor sets it (bypassing the command
    /// pump so it lands mid-turn); the engine checks it between chunks and
    /// before each tool dispatch. `Arc` shared with `SessionActor::cancel`.
    pub cancel: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

/// Cancel sentinel — the `Err` value and `Failed` reason (never rendered
/// verbatim; the status bar localizes `Failed`). The user-facing line is
/// `CANCEL_TEXT`.
const CANCEL_ERR: &str = "turn cancelled by user";

/// The single system line shown for a cancelled turn.
const CANCEL_TEXT: &str = "已被用户中断";

/// How often a running tool's cancel flag is polled. A blocking `bash` has to
/// stop within a fraction of a second of the user hitting Cancel, not at its
/// own timeout.
const CANCEL_POLL: std::time::Duration = std::time::Duration::from_millis(100);

/// How long an approval card may stay unanswered before the kernel denies the
/// call itself — a UI that never answers must not park the turn forever.
const DEFAULT_APPROVAL_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(300);

/// Correlation id for an approval card. Process-wide rather than per engine:
/// a delegated child paints its card on the parent's stream, so the click
/// arrives at the parent's session and only a shared id space tells the two
/// engines' requests apart.
fn next_request_id() -> u64 {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

/// How an approval wait ended. A cancel, a timeout and a deliberate refusal
/// are three different things: only the last one is the user's decision.
enum DecisionWait {
    /// The user answered the card.
    Verdict(crate::tools::registry::ToolVerdict),
    /// The turn was cancelled while the card was open.
    Cancelled,
    /// Nobody answered within `approval_timeout`.
    TimedOut,
}

/// The model-facing text for an approval that ran out the clock. The model
/// must not read the timeout as a deliberate refusal and route around it.
fn approval_timeout_text(name: &str, secs: &str, repeated: bool) -> String {
    let text = format!(
        "[approval timeout] {name} was NOT run. The user did not respond to this permission \
         request within {secs}s, so it was automatically denied by the safety policy. This is a \
         timeout, not an active refusal by the user. Do not request the same permission again \
         without new justification, and do not work around the denial by switching tools or \
         splitting the operation. Prefer an alternative within the current permission scope; if \
         you cannot continue safely, explain the limitation."
    );
    if repeated {
        format!("{text} The identical request already timed out earlier in this turn.")
    } else {
        text
    }
}

/// The approval card's short UI line when it timed out.
const APPROVAL_TIMEOUT_UI: &str = "审批超时，已自动拒绝";

/// Await `fut`, giving up as soon as the user cancels.
///
/// Dropping the future IS the kill: every sandbox backend spawns its child
/// with `kill_on_drop`, and bwrap runs with `--unshare-pid --die-with-parent`,
/// so a long build dies with the dropped future instead of running on after
/// the turn was abandoned.
async fn until_cancelled<F: std::future::Future>(
    cancel: &std::sync::atomic::AtomicBool,
    fut: F,
) -> Option<F::Output> {
    let mut fut = std::pin::pin!(fut);
    loop {
        tokio::select! {
            out = &mut fut => return Some(out),
            _ = tokio::time::sleep(CANCEL_POLL) => {
                if cancel.load(std::sync::atomic::Ordering::Relaxed) {
                    return None;
                }
            }
        }
    }
}

/// Per-turn outcome handed back to `SessionActor`.
#[derive(Debug)]
pub struct TurnOutcome {
    /// The turn's answer: the text of the round that ended the turn, after
    /// the response hooks rewrote it — the reply the user read, never the
    /// narration of the tool rounds that led up to it.
    pub text: String,
    /// Usage from the last `Done` chunk.
    /// (prompt, completion, cached) — cached ⊂ prompt.
    pub usage: Option<(u32, u32, u32)>,
    /// Tool calls executed this turn (for hunk attribution / UI list).
    pub tool_calls_run: usize,
    /// Steering texts injected mid-turn, in order. Provenance for the memory
    /// distiller — a mid-turn correction is the highest-value fact seed, and
    /// without this the `TurnRecord.steered_with` branch was dead code.
    pub steers: Vec<String>,
    /// Tool names the user denied via an approval card this turn — the other
    /// correction signal the distiller turns into facts.
    pub denials: Vec<String>,
}

/// Result of a manual `/compact` — `NothingToDo` is deliberately NOT an
/// error: the user asked, but the history already fits the verbatim suffix
/// budget (or is too short to split), so there is no work rather than a
/// failure to report.
#[derive(Debug)]
pub enum CompactOutcome {
    /// A pass ran and rewrote the history.
    Compacted(compaction::CompactionOutcome),
    /// Nothing worth compacting right now.
    NothingToDo,
}

/// The engine's view of one quiet round, built when the round goes quiet.
///
/// It holds clones of the engine's shared handles, not a borrow of the engine,
/// so the loop can keep its `&mut self` while the policy runs.
struct EngineTurnView {
    mode: Arc<std::sync::RwLock<String>>,
    /// Tool calls run so far this turn (the loop's `tool_calls_run`).
    tool_calls_run: usize,
    /// Opened by host notices, not a user prompt — see `TurnView::is_notice_turn`.
    notice_turn: bool,
}

impl crate::turn_policy::TurnView for EngineTurnView {
    fn mode(&self) -> String {
        self.mode.read().map(|m| m.clone()).unwrap_or_default()
    }
    fn did_tool_work(&self) -> bool {
        self.tool_calls_run > 0
    }
    fn is_notice_turn(&self) -> bool {
        self.notice_turn
    }
}

/// Land one policy [`Effect`](crate::turn_policy::Effect): history row first,
/// then the UI event, exactly as the policy handed them over.
fn land_effect(effect: crate::turn_policy::Effect, history: &mut Vec<ChatMessage>, io: &EngineIo) {
    if let Some(row) = effect.history {
        history.push(row);
    }
    if let Some(event) = effect.ui {
        let _ = io.ui_tx.send(event);
    }
}

/// Handle to an optional live router, replaceable by its host.
pub use crate::plugins::{PluginHandle, PluginRouter};

/// One host-declared agent mode: an id the session slot carries, the registry
/// that mode dispatches against, and whether plugin tools are advertised in
/// it. The kernel names no modes — the host assembles its own table (the
/// coding product: plan/goal/office; a second product: whatever it wants).
pub struct ModeEntry {
    pub id: String,
    pub registry: Arc<ToolRegistry>,
    /// Plugin exports ride the request's `tools` array in this mode. A
    /// read-only mode (plan) turns them off — the permission gate treats an
    /// unknown tool name as a write, which is the behaviour we want for MCP
    /// calls in every other mode.
    pub allow_plugins: bool,
}

pub struct Engine {
    sampler: Sampler,
    /// The registry the engine was built with — the default for any mode id
    /// without an entry (and the empty-id "build" mode).
    registry_full: Arc<ToolRegistry>,
    /// Host-declared modes — `set_agent_mode` swaps `registry` between these
    /// by id. Unknown/empty ids fall back to `registry_full`.
    modes: Vec<ModeEntry>,
    ctx: Arc<ToolCtx>,
    model: String,
    temperature: f32,
    /// Agent mode — shared slot like `permissions` so a mid-turn composer
    /// switch applies at the next sampling round instead of next session.
    /// Holds a mode id from the host's table ("" = the full registry).
    agent_mode: Arc<std::sync::RwLock<String>>,
    /// Pending mid-turn steering text drained from `io.steer_rx`.
    steer_queue: VecDeque<String>,
    /// Steers actually injected into this turn — reset per `run_turn`, and the
    /// source for `TurnOutcome::steers` (the error paths keep the record too,
    /// so a cancelled-after-steering turn still reaches the distiller).
    steers_this_turn: Vec<String>,
    /// Tools the user denied via an approval card this turn — same provenance
    /// role as `steers_this_turn` for `TurnOutcome::denials`.
    denials_this_turn: Vec<String>,
    /// Approvals that timed out this turn, keyed by tool name + canonical
    /// args: the same call is not shown a second card that nobody will answer.
    timed_out_approvals: HashSet<(String, String)>,
    /// Permission gate consulted before every tool dispatch.
    permissions: Arc<std::sync::RwLock<PermissionGate>>,
    /// Decision slot shared with SessionActor — the engine can't hold a
    /// receiver `&mut self`-locked across `run_turn`, so the actor writes
    /// `Some((request_id, approved))` here and the engine polls it. The
    /// `request_id` correlates a verdict to its `ApprovalRequested`; stale
    /// ids are ignored instead of approving a different tool (P1-a).
    decision: Arc<std::sync::Mutex<Option<(u64, crate::tools::registry::ToolVerdict)>>>,
    /// Context window for the active model — compaction triggers at
    /// `compact_at` × window (user-settable: 70/80/90%).
    context_window: usize,
    /// Fraction of `context_window` that triggers compaction.
    compact_at: f32,
    /// How long a pending approval waits for a verdict before the kernel
    /// denies the call itself — zero means no deadline.
    approval_timeout: std::time::Duration,
    /// Per-turn tool-loop bound — sessions use the default; delegated
    /// subagents get a smaller budget.
    max_tool_rounds: usize,
    /// The active model's declared input modalities (`ModelConfig.input`,
    /// e.g. `["text", "image"]`). `"image"` gates whether user-attached
    /// images ride the wire as real parts or degrade to path references.
    model_input: Vec<String>,
    /// Compaction retry-storm suppression.
    compaction_suppressor: CompactionSuppressor,
    /// Last `prompt_tokens` the provider actually reported — a real count
    /// that corrects the chars-based estimate (which under-reads CJK and
    /// tool-call payloads). Reset to the post-compaction estimate after a
    /// successful compact so the stale pre-compact figure can't retrigger.
    last_prompt_tokens: usize,
    /// Ordered hook chain (veto/mutate before+after tools).
    hooks: crate::hooks::HookChain,
    /// Plugin tool router — built-in names win; plugins fill the rest.
    /// Shared with the `SessionManager`, which can REPLACE the manager at
    /// runtime (a plugin added in settings). Read per request, never cached, so
    /// a live session picks up a reload without being restarted.
    plugin_router: Option<PluginHandle>,
    /// Product rules applied when a round goes quiet. [`NoopPolicy`] by
    /// default — the host installs its own (the coding product's
    /// `agent-coding` policy) before the first turn.
    policy: Box<dyn crate::turn_policy::TurnPolicy>,
    /// Thinking / reasoning intensity level (e.g. "off", "low", "medium",
    /// "high", "max"). Shared slot like `permissions` — the bridge reads it
    /// to report a session's live level back to the UI.
    thinking_level: Arc<std::sync::RwLock<Option<String>>>,
    /// Model-specific thinking level mapping from pi-agent config schema.
    thinking_level_map: Option<std::collections::HashMap<String, Option<String>>>,
    /// Per-model wire settings from config (`maxTokens`, `samplingParams`,
    /// model-level `compat`) — swapped alongside the model on a hot-swap so
    /// the next request carries the new model's own caps.
    model_params: agent_llm_core::ModelParams,
}

impl Engine {
    pub fn new(
        provider: Arc<dyn agent_llm_core::LlmProvider>,
        registry: Arc<ToolRegistry>,
        ctx: Arc<ToolCtx>,
        model: impl Into<String>,
        temperature: f32,
    ) -> Self {
        Self {
            sampler: Sampler::new(provider),
            registry_full: registry,
            modes: Vec::new(),
            ctx,
            model: model.into(),
            temperature,
            agent_mode: Arc::new(std::sync::RwLock::new(String::new())),
            steer_queue: VecDeque::new(),
            steers_this_turn: Vec::new(),
            denials_this_turn: Vec::new(),
            timed_out_approvals: HashSet::new(),
            permissions: Arc::new(std::sync::RwLock::new(PermissionGate::from_mode_str(
                "default",
            ))),
            decision: Arc::new(std::sync::Mutex::new(None)),
            context_window: 256_000,
            compact_at: crate::compaction::COMPACT_AT,
            approval_timeout: DEFAULT_APPROVAL_TIMEOUT,
            max_tool_rounds: usize::MAX,
            model_input: Vec::new(),
            compaction_suppressor: CompactionSuppressor::default(),
            last_prompt_tokens: 0,
            hooks: crate::hooks::HookChain::new(),
            plugin_router: None,
            policy: Box::new(crate::turn_policy::NoopPolicy),
            thinking_level: Arc::new(std::sync::RwLock::new(None)),
            thinking_level_map: None,
            model_params: agent_llm_core::ModelParams::EMPTY,
        }
    }

    /// Install the active model's per-model wire settings (`maxTokens`,
    /// `samplingParams`, model-level `compat`) — the session actor resolves
    /// these from config at spawn and on every model switch.
    pub fn set_model_params(&mut self, params: agent_llm_core::ModelParams) {
        self.model_params = params;
    }

    /// The active model's per-model wire settings.
    pub fn model_params(&self) -> &agent_llm_core::ModelParams {
        &self.model_params
    }

    /// Bound the tool loop per turn — unlimited by default.
    ///
    /// A count of round-trips is not evidence of a runaway: a legitimate turn
    /// can need hundreds (a repo-wide refactor, a review that reads forty
    /// files), and cutting it off mid-work costs more than the loop it was
    /// meant to catch. What does bound a turn is the context window and the
    /// user keeping the brake. A caller that pays per call, or that runs
    /// unattended, sets its own bound here.
    pub fn set_max_tool_rounds(&mut self, rounds: usize) {
        self.max_tool_rounds = rounds;
    }

    /// Set reasoning intensity. Explicit model maps constrain every caller;
    /// map-less generic engines may still supply an explicit provider effort.
    pub fn set_thinking_level(&mut self, level: Option<String>) {
        let level = if self.thinking_level_map.is_some() {
            agent_llm_core::normalize_thinking_level(
                level.as_deref(),
                self.thinking_level_map.as_ref(),
            )
        } else {
            level
        };
        if let Ok(mut w) = self.thinking_level.write() {
            *w = level;
        }
    }

    /// Restore a model-bound selection; no map means no selectable level.
    pub fn set_model_thinking_level(&mut self, level: Option<String>) {
        self.set_thinking_level(agent_llm_core::normalize_thinking_level(
            level.as_deref(),
            self.thinking_level_map.as_ref(),
        ));
    }

    /// Current thinking level if set.
    pub fn thinking_level(&self) -> Option<String> {
        self.thinking_level.read().ok().and_then(|l| l.clone())
    }

    /// Access the shared thinking-level slot — SessionManager keeps a clone
    /// on the session handle so `model_info` can report the live value.
    pub fn thinking_level_shared(&self) -> Arc<std::sync::RwLock<Option<String>>> {
        self.thinking_level.clone()
    }

    /// Set model-specific thinking level mapping.
    pub fn set_thinking_level_map(
        &mut self,
        map: Option<std::collections::HashMap<String, Option<String>>>,
    ) {
        self.thinking_level_map = map;
        // Model switches and config reloads must update the shared live slot
        // before model_info or the next request can observe an old tier.
        self.set_model_thinking_level(self.thinking_level());
    }

    /// Resolves the effective wire reasoning effort ("low", "medium", "high", "max" or None).
    pub fn resolve_reasoning_effort(&self) -> Option<String> {
        // Capability gate first — a wire that can't express reasoning (or
        // has it compat-disabled via `supports_reasoning_effort = false`)
        // gets no effort regardless of the thinking level the user picked.
        if !self.sampler.provider().capabilities().reasoning {
            return None;
        }
        let guard = self.thinking_level.read().ok()?;
        let level = guard.as_deref()?;
        if level == "off" {
            return None;
        }
        if let Some(map) = &self.thinking_level_map {
            // The final shared-slot gate also rejects undeclared/null tiers.
            return map
                .get(level)?
                .as_ref()
                .filter(|value| !value.trim().is_empty())
                .cloned();
        }
        // Map-less generic kernel callers can set an explicit provider effort.
        // Model-bound host paths use set_model_thinking_level instead.
        Some(level.to_string())
    }

    /// Install the host's external tool router.
    pub fn set_plugins(&mut self, handle: PluginHandle) {
        self.ctx = Arc::new(self.ctx.with_plugins(handle.clone()));
        self.plugin_router = Some(handle);
    }

    /// The router currently installed, if any. A read lock, cloned out — the
    /// handle outlives any single reload.
    fn plugins(&self) -> Option<Arc<dyn PluginRouter>> {
        self.plugin_router
            .as_ref()
            .and_then(|h| h.read().ok().and_then(|g| g.clone()))
    }

    /// The request's `tools` array: built-ins, plus every enabled plugin's
    /// exports as `plugin_id__tool`. The names are sanitized by the plugin
    /// layer into the wire's `function.name` charset — an illegal character
    /// here (`:` in the old spelling) makes a strict upstream reject the whole
    /// request as `invalid_argument`. A mode that opts plugins out (plan)
    /// advertises built-ins only — the permission gate treats an unknown tool
    /// name as a write, which is the behaviour we want for MCP calls in every
    /// other mode.
    fn request_tools_schema(&self) -> serde_json::Value {
        let builtin = self.active_registry().request_schema();
        if !self.plugins_allowed() {
            return builtin;
        }
        let Some(mgr) = self.plugins() else {
            return builtin;
        };
        let serde_json::Value::Array(mut tools) = builtin else {
            return builtin;
        };
        tools.extend(mgr.exported_tools());
        serde_json::Value::Array(tools)
    }

    /// SessionActor installs the hook chain (registered plugins/built-ins).
    pub fn set_hooks(&mut self, chain: crate::hooks::HookChain) {
        self.hooks = chain;
    }

    /// Install the product rules for quiet rounds (and the per-turn contracts).
    /// Must happen before the first turn — the default [`NoopPolicy`] ends
    /// every quiet round immediately.
    pub fn set_policy(&mut self, policy: Box<dyn crate::turn_policy::TurnPolicy>) {
        self.policy = policy;
    }

    /// Declare a mode the session slot may carry. Ids repeat = later wins.
    pub fn add_mode(&mut self, entry: ModeEntry) {
        self.modes.retain(|m| m.id != entry.id);
        self.modes.push(entry);
    }

    /// The active model's context window — SessionActor sets it from config.
    pub fn set_context_window(&mut self, window: usize) {
        self.context_window = window;
    }

    /// Current context-window bound — model config value or the 256k
    /// default (same bound `Usage` events report).
    pub fn context_window(&self) -> usize {
        self.context_window
    }

    /// Compaction trigger as a fraction of the window — the settings UI's
    /// 70/80/90% choices write here at actor spawn.
    pub fn set_compact_at(&mut self, frac: f32) {
        self.compact_at = frac.clamp(0.5, 0.95);
    }

    pub fn compact_at(&self) -> f32 {
        self.compact_at
    }

    /// Bound how long a pending approval may wait for a verdict. Zero = no
    /// deadline. The default keeps the long-standing 300 s bound.
    pub fn set_approval_timeout(&mut self, d: std::time::Duration) {
        self.approval_timeout = d;
    }

    /// SessionActor installs the permission gate (mode + repo rules).
    pub fn set_permissions(&mut self, gate: PermissionGate) {
        if let Ok(mut w) = self.permissions.write() {
            *w = gate;
        }
    }

    /// Adopt a gate the host already shares. The subagent spawner reads the
    /// same slot, so a mode switch mid-run reaches live children instead of
    /// leaving them on the mode they were born with.
    pub fn set_permissions_shared(&mut self, shared: Arc<std::sync::RwLock<PermissionGate>>) {
        self.permissions = shared;
    }

    /// Access the shared permission gate handle for direct mid-turn updates.
    pub fn permissions_writer(&self) -> Arc<std::sync::RwLock<PermissionGate>> {
        self.permissions.clone()
    }

    /// Switch the agent mode — swaps the active registry immediately; the
    /// next sampling round sees the new tool set. An unknown id keeps the
    /// full registry (the mode slot may carry a host id the engine was not
    /// told about; dispatching the full set is the safe fallback).
    pub fn set_agent_mode(&mut self, mode: impl Into<String>) {
        if let Ok(mut w) = self.agent_mode.write() {
            *w = mode.into();
        }
    }

    /// Shared mode slot — the actor polls it in `run_turn` (a mid-turn
    /// `SetAgentMode` writes the slot without needing `&mut Engine`).
    /// True when the current mode's entry opts plugins out — a read-only
    /// mode advertises built-ins only, since the permission gate treats an
    /// unknown tool name as a write.
    fn plugins_allowed(&self) -> bool {
        let id = self.agent_mode();
        self.modes
            .iter()
            .find(|m| m.id == id)
            .map(|m| m.allow_plugins)
            .unwrap_or(true)
    }

    pub fn agent_mode_writer(&self) -> Arc<std::sync::RwLock<String>> {
        self.agent_mode.clone()
    }

    /// Current agent mode id — the actor reads this for the prompt swap.
    /// Empty string = the full registry (the host's "build" mode).
    pub fn agent_mode(&self) -> String {
        self.agent_mode
            .read()
            .map(|m| m.clone())
            .unwrap_or_default()
    }

    /// The registry for the mode currently in the slot — resolved per call
    /// so a mid-turn `SetAgentMode` applies at the very next dispatch.
    fn active_registry(&self) -> Arc<ToolRegistry> {
        let id = self.agent_mode();
        let reg = self
            .modes
            .iter()
            .find(|m| m.id == id)
            .map(|m| m.registry.clone())
            .unwrap_or_else(|| self.registry_full.clone());
        // Refresh the ctx slot — `batch_execute` dispatches through the
        // same mode-scoped registry this call resolved.
        *self.ctx.active_registry.write().unwrap() = Some(reg.clone());
        reg
    }

    /// The shared decision slot — SessionActor clones it to deliver
    /// `ToolDecision { request_id, approved }` while the engine is
    /// mid-`run_turn`.
    pub fn decision_slot(
        &self,
    ) -> Arc<std::sync::Mutex<Option<(u64, crate::tools::registry::ToolVerdict)>>> {
        self.decision.clone()
    }

    /// This engine's child id, when it runs as a delegated subagent. An
    /// approval card for it belongs to that child's capsule, not the parent's
    /// turn.
    fn child_id(&self) -> Option<u64> {
        self.ctx
            .ext
            .get::<crate::tools::delegate::ChildIdentity>()
            .map(|c| c.id)
    }

    /// Make a pending card reachable from wherever the user clicks it. A child
    /// paints its card on the parent's stream, so the verdict arrives at the
    /// parent's session — this registers the slot it actually has to land in.
    fn route_child_approval(
        &self,
        request_id: u64,
    ) -> Option<crate::tools::delegate::ApprovalRoute> {
        self.child_id().map(|_| {
            crate::tools::delegate::route_child_approval(request_id, self.decision.clone())
        })
    }

    /// `AnswerQuestion` resolves here — same bypass-the-pump pattern as
    /// `decision_slot`: the command writes directly while the turn's tool
    /// is parked awaiting the answer.
    pub fn ask_channel(&self) -> Arc<crate::tools::registry::AskChannel> {
        // The owning session installs the channel before building the engine;
        // a bare engine without one is a wiring bug, not a runtime condition.
        self.ctx
            .ask()
            .expect("session installs the ask channel before the engine is built")
    }

    /// Hot-swap provider for the next turn.
    pub fn set_provider(&mut self, provider: Arc<dyn agent_llm_core::LlmProvider>) {
        self.sampler.set_provider(provider);
    }

    /// Install the resolved provider keys the egress masker scrubs from every
    /// outbound request. Called whenever the config is (re)loaded.
    pub fn set_secrets(&mut self, secrets: Vec<String>) {
        self.sampler.set_masker(secrets);
    }

    /// Install the degrade chain for the primary named `primary`.
    pub fn set_fallbacks(
        &mut self,
        primary: &str,
        chain: Vec<(String, Arc<dyn agent_llm_core::LlmProvider>)>,
    ) {
        self.sampler.set_primary_name(primary);
        self.sampler.set_fallbacks(chain);
    }

    pub fn set_model(&mut self, model: impl Into<String>) {
        self.model = model.into();
    }

    /// Hot-swap the active model's declared input modalities
    /// (`ModelConfig.input`) — SetModel passes the fresh list through.
    pub fn set_model_input(&mut self, input: Vec<String>) {
        self.model_input = input;
    }

    /// Does the active model accept image input? `"image"` in `input` —
    /// a missing/empty list means text-only (the honest default; config
    /// authors mark vision explicitly).
    fn supports_images(&self) -> bool {
        self.model_input
            .iter()
            .any(|i| i.eq_ignore_ascii_case("image"))
    }

    /// Drain host-pushed notices into history as `NoticeKind::Host` rows. A
    /// notice is not a steer: it carries no user intent, so it never draws a
    /// user bubble and never reaches `steers_this_turn`; nor is it a system
    /// instruction — adapters fence it at user privilege, because its text
    /// can carry untrusted tool output.
    fn drain_notices(&mut self, io: &mut EngineIo, history: &mut Vec<ChatMessage>) {
        while let Ok(text) = io.notice_rx.try_recv() {
            history.push(ChatMessage::host_notice(text));
        }
    }

    /// Drain any pending steering texts between tool calls — marks the
    /// turn `steered` for UI provenance.
    fn drain_steering(&mut self, io: &mut EngineIo, history: &mut Vec<ChatMessage>) {
        while let Ok(text) = io.steer_rx.try_recv() {
            self.steer_queue.push_back(text);
        }
        if !self.steer_queue.is_empty() {
            // Notify the UI the turn was steered (provenance marker).
            let line = "已插入引导指令".to_string();
            let _ = io.ui_tx.send(UiEvent::SystemMessage(line.clone()));
            history.push(ChatMessage::notice(line));
        }
    }

    /// Steers injected into the turn that just ran. Read after an `Err` return
    /// too — a steer followed by a cancel is exactly the correction the memory
    /// distiller most wants.
    pub fn steers_this_turn(&self) -> &[String] {
        &self.steers_this_turn
    }

    /// Tools denied via approval cards in the turn that just ran — read after
    /// an `Err` return too, like `steers_this_turn`.
    pub fn denials_this_turn(&self) -> &[String] {
        &self.denials_this_turn
    }

    /// Run one full user turn: prompt → sample → tools → re-sample … → done.
    ///
    /// `history` is the session's conversation; the engine appends the user
    /// turn, tool calls, and tool results to it in place.
    #[instrument(skip_all, name = "turn")]
    pub async fn run_turn(
        &mut self,
        io: &mut EngineIo,
        history: &mut Vec<ChatMessage>,
        user_text: String,
    ) -> Result<TurnOutcome, String> {
        // `<attached-image>` markers carry the composer's staged uploads —
        // they ride the wire as real image parts when the model declares
        // vision, else degrade to path references plus a notice so the
        // user knows the attachment was dropped.
        let images = extract_attached_images(&user_text);
        // Fresh provenance record for this turn — see `TurnOutcome::steers`.
        self.steers_this_turn.clear();
        self.denials_this_turn.clear();
        self.timed_out_approvals.clear();
        self.push_user_message(io, history, user_text, images);
        self.run_rounds(io, history, false).await
    }

    /// Wake the model on host notices while no user turn is running — a
    /// background job finished after the last turn ended. The turn opens on
    /// the notices themselves (`notices` plus whatever else is already
    /// queued on `notice_rx`); there is no user message, so nothing in it
    /// reads as the user's words or counts as a user turn.
    #[instrument(skip_all, name = "notice_turn")]
    pub async fn run_notice_turn(
        &mut self,
        io: &mut EngineIo,
        history: &mut Vec<ChatMessage>,
        notices: Vec<String>,
    ) -> Result<TurnOutcome, String> {
        self.steers_this_turn.clear();
        self.denials_this_turn.clear();
        self.timed_out_approvals.clear();
        for text in notices {
            history.push(ChatMessage::host_notice(text));
        }
        self.drain_notices(io, history);
        // A notice turn tells the policy so: contracts that push work (a
        // goal to advance, a plan to submit) must not fire on a wake-up.
        self.run_rounds(io, history, true).await
    }

    fn push_user_message(
        &mut self,
        io: &mut EngineIo,
        history: &mut Vec<ChatMessage>,
        user_text: String,
        images: Vec<agent_llm_core::types::ImageRef>,
    ) {
        let mut user_msg = ChatMessage::user(user_text);
        if !images.is_empty() {
            if self.supports_images() {
                user_msg = user_msg.with_images(images);
            } else {
                let line = format!(
                    "模型 {} 不支持图像输入 — 已忽略 {} 张图片",
                    self.model,
                    images.len()
                );
                let _ = io.ui_tx.send(UiEvent::SystemMessage(line.clone()));
                history.push(ChatMessage::notice(line));
            }
        }
        history.push(user_msg);
    }

    /// The sample → tools → re-sample loop shared by user and notice turns.
    async fn run_rounds(
        &mut self,
        io: &mut EngineIo,
        history: &mut Vec<ChatMessage>,
        notice_turn: bool,
    ) -> Result<TurnOutcome, String> {
        self.set_state(io, AgentState::Reasoning);

        // The per-round sanitize + compaction check below own both the
        // turn-start pass and every mid-turn tool round (one code path).

        // Every round's text in order — the fallback for a closing round that
        // streamed nothing, so the turn still answers with what it said.
        let mut text = String::new();
        let mut usage = None;
        let mut tool_calls_run = 0usize;
        let mut tool_rounds = 0usize;
        let mut force_no_tools = false;
        // Optional bound — unlimited unless a caller asked for one. Round-trip
        // count is a poor proxy for a runaway; see `set_max_tool_rounds`.
        let max_tool_rounds = self.max_tool_rounds;
        // A fresh turn starts from a clean policy state — a leftover
        // `goal_complete`/`submit_plan` from an earlier turn must not
        // short-circuit this one (the contracts are per-turn: each goal
        // prompt ends declared).
        self.policy.begin_turn();

        loop {
            // P0-C4: cooperative cancel — checked at the top of every
            // sample/tool round so a Cancel lands between model calls even
            // if no chunk is flowing.
            if io.cancel.load(std::sync::atomic::Ordering::Relaxed) {
                self.set_state(io, AgentState::Failed(CANCEL_ERR.into()));
                let _ = io.ui_tx.send(UiEvent::SystemMessage(CANCEL_TEXT.into()));
                history.push(ChatMessage::notice(CANCEL_TEXT));
                return Err(CANCEL_ERR.into());
            }

            self.drain_notices(io, history);
            self.drain_steering(io, history);
            while let Some(steer) = self.steer_queue.pop_front() {
                // Provenance: this steer is being injected into THIS turn.
                self.steers_this_turn.push(steer.clone());
                // Echo the steer as a user bubble — the live stream never
                // showed it otherwise (the composer doesn't echo), while a
                // reloaded view rebuilt it from history. Its own event, not
                // `UserPrompt`: a steer joins the running turn.
                let _ = io.ui_tx.send(UiEvent::UserSteer(steer.clone()));
                // Steering can carry `<attached-image>` markers too — same
                // extraction as the turn-start path; a text-only model
                // just keeps the path reference (no mid-turn notice spam).
                let steer_images = extract_attached_images(&steer);
                let mut steer_msg = ChatMessage::user_steer(&steer);
                if !steer_images.is_empty() && self.supports_images() {
                    steer_msg = steer_msg.with_images(steer_images);
                }
                history.push(steer_msg);
            }

            let native_tc = self.sampler.provider().native_tool_calls();
            // Compact BEFORE the budget-fit: a single bulky tool result can
            // jump the estimate past the 90% budget in one round, and
            // `fit_conversation_to_budget` would silently drop the oldest
            // messages (taking the just-returned tool output with them)
            // before the trigger ever saw an over-threshold fill. The check
            // works on the raw history — flattening is only needed for the
            // wire, not the estimate.
            // The same trigger at the TOP OF EVERY ROUND, not just at turn
            // start: a long agentic turn (tool → result → tool …) grows the
            // context without ever returning to a prompt, and the only other
            // guard is `fit_conversation_to_budget` silently dropping the
            // oldest messages at 90%. Compacting here keeps the summary (and
            // the card) instead of losing history to truncation.
            self.auto_compact_if_needed(io, history).await;
            // Sanitize every round (image policy + budget-fit). Text-
            // protocol flatten is a wire-only step below — the live
            // history keeps structured `tool_calls` either way.
            compaction::sanitize_for_sample(
                history,
                self.context_window.saturating_mul(9) / 10,
                self.supports_images(),
            );
            // Text-protocol flatten goes on a *wire-only copy*: the
            // persisted history keeps structured `tool_calls`, so the
            // snapshot replays real calls and the resume sanitizer's
            // `strip_call_echo` can never confuse an intentional flatten
            // with a provider echo (they were the same text).
            let mut wire_history;
            let wire_messages: &[ChatMessage] = if native_tc {
                history
            } else {
                wire_history = history.clone();
                compaction::flatten_tool_calls_for_wire(&mut wire_history);
                &wire_history
            };

            let mut assembler = ToolCallAssembler::new();
            let mut round_text = String::new();
            // This round's reasoning trace — accumulated for the *stored*
            // assistant row (the live stream only ever saw the coalesced
            // deltas), so a reopened session replays the same 思考过程 block.
            let mut round_reasoning = String::new();
            let mut round_provider_state = None;
            // First/last reasoning-delta stamps — the block's true span
            // persists on the assistant row as `duration_ms` so a replayed
            // view shows the real `· 20s` instead of nothing.
            let mut reasoning_first: Option<i64> = None;
            let mut reasoning_last = 0i64;
            let mut saw_done = false;
            // Delta coalescing — the UI doesn't need one event per SSE
            // chunk (often 1-5 chars). Buffer text/reasoning deltas and
            // flush at ~120 bytes or after 40ms, and always flush the other
            // side when the stream switches content type (or starts a tool
            // call) so the UI sees the blocks in stream order.
            // Shared by the chunk callback and the idle tick, hence the lock (the
            // boxed future must be Send; it is never contended).
            let coalesce = std::sync::Mutex::new(DeltaBuffers::new());
            // A provider-level `StreamChunk::Error` (e.g. devin's
            // `internal_server_error … upstream error`) arrives inside the
            // stream, not as a transport `Err` — without this flag the turn
            // would fall through to the no-calls branch and persist an EMPTY
            // assistant message into history, corrupting every later request.
            let mut stream_error: Option<String> = None;
            // Sampler policy notices (retry / thinking-degrade) shown live —
            // collected here and persisted after the sample so a reloaded
            // view replays them too.
            let mut policy_notes: Vec<String> = Vec::new();
            // Set by `on_event` on `Retrying`, consumed by `on_chunk` on the
            // retried attempt's first chunk. A retry resends the SAME messages
            // and the provider regenerates from scratch, so every per-attempt
            // accumulator must drop the dead attempt: otherwise history
            // persists `partial + regenerated` text, `assembler` finishes
            // attempt-1's half-written tool calls inside attempt 2, and a
            // stale `stream_error` fails a turn whose retry actually
            // succeeded. Lazily (not at event time) so a retry that never
            // produces a chunk still keeps the partial as salvage. Atomic
            // rather than Cell: the boxed future needs Send.
            let retry_reset = std::sync::atomic::AtomicBool::new(false);

            let effort = self.resolve_reasoning_effort();
            let schema = self.request_tools_schema();
            let tools_opt = if force_no_tools { None } else { Some(&schema) };
            let req = SampleRequest {
                model: &self.model,
                temperature: self.temperature,
                tools: tools_opt,
                reasoning_effort: effort.as_deref(),
                params: &self.model_params,
            };

            // Race the sample against the cancel flag — a Cancel mid-stream
            // drops the in-flight request instead of waiting for the next
            // chunk or the 300 s idle timeout. The flag is cloned out so the
            // watcher doesn't borrow `io` while the chunk closure does.
            let cancel_flag = io.cancel.clone();
            let cancel_watch = async move {
                while !cancel_flag.load(std::sync::atomic::Ordering::Relaxed) {
                    tokio::time::sleep(std::time::Duration::from_millis(25)).await;
                }
            };

            let res = tokio::select! {
                biased;
                _ = cancel_watch => {
                    Err(agent_llm_core::sampler::SampleError::Stream(CANCEL_ERR.into()))
                }
                r = self.sampler.sample(
                    req,
                    wire_messages,
                    |chunk| {
                        if retry_reset.swap(false, std::sync::atomic::Ordering::Relaxed) {
                            round_text.clear();
                            round_reasoning.clear();
                            round_provider_state = None;
                            reasoning_first = None;
                            reasoning_last = 0;
                            coalesce.lock().unwrap_or_else(|e| e.into_inner()).clear();
                            assembler = ToolCallAssembler::new();
                            saw_done = false;
                            stream_error = None;
                            usage = None;
                        }
                        match chunk {
                            StreamChunk::ReasoningDelta(t) => {
                                // Text is over once reasoning starts: flush it first.
                                let mut c = coalesce.lock().unwrap_or_else(|e| e.into_inner());
                                if let Some(text) = c.take_text(true) {
                                    let _ = io.ui_tx.send(UiEvent::TextDelta { text, parent: None, child_id: None });
                                }
                                round_reasoning.push_str(t);
                                c.reasoning.push_str(t);
                                let now = agent_llm_core::types::now_ms();
                                if reasoning_first.is_none() {
                                    reasoning_first = Some(now);
                                }
                                reasoning_last = now;
                                if let Some(text) = c.take_reasoning(false) {
                                    let _ = io.ui_tx.send(UiEvent::ReasoningDelta {
                                        text,
                                        parent: None,
                                        child_id: None,
                                    });
                                }
                            }
                            StreamChunk::ContentDelta(t) => {
                                // Reasoning is over once text starts: flush it first.
                                let mut c = coalesce.lock().unwrap_or_else(|e| e.into_inner());
                                if let Some(text) = c.take_reasoning(true) {
                                    let _ = io.ui_tx.send(UiEvent::ReasoningDelta { text, parent: None, child_id: None });
                                }
                                round_text.push_str(t);
                                c.text.push_str(t);
                                if let Some(text) = c.take_text(false) {
                                    let _ = io.ui_tx.send(UiEvent::TextDelta { text, parent: None, child_id: None });
                                }
                            }
                            StreamChunk::ToolCallDelta { .. } => {
                                // A tool call's argument stream can run for seconds with
                                // no idle tick, so the preamble must not wait for it.
                                let mut c = coalesce.lock().unwrap_or_else(|e| e.into_inner());
                                if let Some(text) = c.take_text(true) {
                                    let _ = io.ui_tx.send(UiEvent::TextDelta { text, parent: None, child_id: None });
                                }
                                if let Some(text) = c.take_reasoning(true) {
                                    let _ = io.ui_tx.send(UiEvent::ReasoningDelta { text, parent: None, child_id: None });
                                }
                                assembler.feed(chunk);
                            }
                            StreamChunk::Done { prompt_tokens, completion_tokens, cached_tokens } => {
                                saw_done = true;
                                // Flush any coalesced deltas BEFORE the
                                // terminal event lands so the UI never
                                // renders trailing text after Finished.
                                if let Some(text) = coalesce.lock().unwrap_or_else(|e| e.into_inner()).take_text(true) {
                                    let _ = io.ui_tx.send(UiEvent::TextDelta { text, parent: None, child_id: None });
                                }
                                if let Some(text) = coalesce.lock().unwrap_or_else(|e| e.into_inner()).take_reasoning(true) {
                                    let _ = io.ui_tx.send(UiEvent::ReasoningDelta { text, parent: None, child_id: None });
                                }
                                if let (Some(p), Some(c)) = (prompt_tokens, completion_tokens) {
                                    let cached = cached_tokens.unwrap_or(0);
                                    usage = Some((*p, *c, cached));
                                    // Real count from the wire — feeds the
                                    // compaction trigger next turn so the
                                    // estimate stops being the only signal.
                                    self.last_prompt_tokens = *p as usize;
                                    let _ = io.ui_tx.send(UiEvent::Usage {
                                        prompt_tokens: *p,
                                        completion_tokens: *c,
                                        context_window: self.context_window as u32,
                                        cached_tokens: cached,
                                    });
                                }
                            }
                            StreamChunk::ProviderState(state) => {
                                // Adapter-owned replay data is not UI output.
                                round_provider_state = Some(state.clone());
                            }
                            StreamChunk::Keepalive => {}
                            StreamChunk::Error(e) => {
                                stream_error = Some(e.clone());
                                if let Some(text) = coalesce.lock().unwrap_or_else(|e| e.into_inner()).take_text(true) {
                                    let _ = io.ui_tx.send(UiEvent::TextDelta { text, parent: None, child_id: None });
                                }
                                if let Some(text) = coalesce.lock().unwrap_or_else(|e| e.into_inner()).take_reasoning(true) {
                                    let _ = io.ui_tx.send(UiEvent::ReasoningDelta { text, parent: None, child_id: None });
                                }
                                // The `*[error: e]*` marker + turn-end `Error` event
                                // report this — no extra SystemMessage (it printed
                                // the same text a second time live).
                            }
                        }
                    },
                    |ev| {
                        // A retry or a provider switch restarts the attempt: the
                        // partial output so far belongs to the failed attempt.
                        if matches!(
                            ev,
                            SamplerEvent::Retrying { .. } | SamplerEvent::FellBack { .. }
                        ) {
                            retry_reset.store(true, std::sync::atomic::Ordering::Relaxed);
                        }
                        let t = ev.to_string();
                        let _ = io.ui_tx.send(UiEvent::SystemMessage(t.clone()));
                        policy_notes.push(t);
                    },
                    || {
                        // The stream is quiet. Whatever it already delivered is
                        // still sitting in the coalescing buffers (a flush only
                        // happens when the NEXT chunk arrives), so a model that
                        // stalls mid-sentence would freeze the reply on screen
                        // until the user cancels. Push it out now.
                        let mut c = coalesce.lock().unwrap_or_else(|e| e.into_inner());
                        if let Some(text) = c.take_text(false) {
                            let _ = io.ui_tx.send(UiEvent::TextDelta { text, parent: None, child_id: None });
                        }
                        if let Some(text) = c.take_reasoning(false) {
                            let _ = io.ui_tx.send(UiEvent::ReasoningDelta { text, parent: None, child_id: None });
                        }
                    },
                ) => r,
            };

            // Flush any sub-threshold deltas left after the sample
            // returned — a stream that ended cleanly before hitting 120
            // chars would otherwise strand its tail in the buffer.
            {
                let mut c = coalesce.lock().unwrap_or_else(|e| e.into_inner());
                if let Some(text) = c.take_text(true) {
                    let _ = io.ui_tx.send(UiEvent::TextDelta {
                        text,
                        parent: None,
                        child_id: None,
                    });
                }
                if let Some(text) = c.take_reasoning(true) {
                    let _ = io.ui_tx.send(UiEvent::ReasoningDelta {
                        text,
                        parent: None,
                        child_id: None,
                    });
                }
            }

            // Persist the sampler policy notices — they replayed live,
            // so the snapshot must carry them (cancel/error paths return
            // early below; drain before them).
            for t in policy_notes.drain(..) {
                history.push(ChatMessage::notice(t));
            }

            // A cancel mid-sample unwinds here — the partial text and the
            // cancel line persist separately so a reloaded view replays
            // exactly what the live stream showed (text, then the notice).
            if io.cancel.load(std::sync::atomic::Ordering::Relaxed) {
                if !round_text.is_empty() {
                    history.push(
                        ChatMessage::assistant(round_text.clone())
                            .with_reasoning(Some(std::mem::take(&mut round_reasoning)))
                            .with_duration_ms(reasoning_first.map(|f| reasoning_last - f)),
                    );
                }
                history.push(ChatMessage::notice(CANCEL_TEXT));
                self.set_state(io, AgentState::Failed(CANCEL_ERR.into()));
                let _ = io.ui_tx.send(UiEvent::SystemMessage(CANCEL_TEXT.into()));
                return Err(CANCEL_ERR.into());
            }

            if let Err(e) = res {
                // Stream salvage (hardening §4): partial text + interrupt
                // line + error line all persist — replay matches the live
                // stream line-for-line.
                if !round_text.is_empty() {
                    history.push(
                        ChatMessage::assistant(round_text.clone())
                            .with_reasoning(Some(std::mem::take(&mut round_reasoning)))
                            .with_duration_ms(reasoning_first.map(|f| reasoning_last - f)),
                    );
                    let line = "流传输中断 — 已保留部分内容";
                    let _ = io.ui_tx.send(UiEvent::SystemMessage(line.into()));
                    history.push(ChatMessage::notice(line));
                }
                self.set_state(io, AgentState::Failed(e.to_string()));
                let _ = io.ui_tx.send(UiEvent::Error(e.to_string()));
                history.push(ChatMessage::notice_error(e.to_string()));
                return Err(e.to_string());
            }
            // Provider surfaced an in-stream error (upstream 5xx, rate limit,
            // model refusal, …) — treat the turn as failed. Salvage the
            // partial text and persist the error line (the `⚠` the live
            // stream drew) so a reloaded view loses nothing.
            if let Some(e) = stream_error {
                if !round_text.is_empty() {
                    history.push(
                        ChatMessage::assistant(round_text.clone())
                            .with_reasoning(Some(std::mem::take(&mut round_reasoning)))
                            .with_duration_ms(reasoning_first.map(|f| reasoning_last - f)),
                    );
                }
                history.push(ChatMessage::notice_error(e.clone()));
                self.set_state(io, AgentState::Failed(e.clone()));
                let _ = io.ui_tx.send(UiEvent::Error(e.clone()));
                return Err(e);
            }
            debug_assert!(saw_done, "sampler guarantees Done exactly once");

            text.push_str(&round_text);
            let reasoning = std::mem::take(&mut round_reasoning);
            // The block's real span — first to last reasoning delta. `None`
            // when the round produced no trace.
            let reasoning_ms = reasoning_first.map(|f| reasoning_last - f);
            let mut calls = assembler.finish();

            // Degenerate calls — a truncated/announcement delta can produce
            // a slot with a real provider `id` but no `name` (deepseek sends
            // id-only `tool_calls` deltas). Persisting it in the assistant
            // row's `tool_calls` while skipping dispatch orphans the entry:
            // strict backends reject every later request with
            // `No tool output found for tool call …` (400). Drop the call
            // BEFORE the assistant row is built so no dangling id ever
            // reaches the wire — and `name:""` is itself malformed anyway.
            let bad_calls = calls.iter().filter(|c| c.name.trim().is_empty()).count();
            if bad_calls > 0 {
                calls.retain(|c| !c.name.trim().is_empty());
                let line = format!("已跳过 {bad_calls} 个格式异常的空工具调用");
                let _ = io.ui_tx.send(UiEvent::SystemMessage(line.clone()));
                history.push(ChatMessage::notice(line));
            }

            if calls.is_empty() {
                let final_text = if round_text.trim().is_empty() && !text.trim().is_empty() {
                    text.clone()
                } else {
                    round_text
                };

                if final_text.trim().is_empty() && tool_calls_run > 0 && !force_no_tools {
                    force_no_tools = true;
                    history.push(ChatMessage::user_hidden(
                        "Please provide your final answer and summary to the user based on the tool results above.",
                    ));
                    continue;
                }

                // The quiet round goes to the installed policy: it may end the
                // turn (with an optional check report and an optional
                // deliverable payload — the plan card) or push a hidden nudge.
                // Plan cards are taken by the policy, not the loop: the card
                // lands after the final text, and the policy holds the
                // controller the signal rides on.
                let view = EngineTurnView {
                    mode: self.agent_mode.clone(),
                    tool_calls_run,
                    notice_turn,
                };
                let verdict = self.policy.on_quiet_round(&view).await;
                let (before_final, after_final) = match verdict {
                    crate::turn_policy::Verdict::Nudge { hidden } => {
                        if !final_text.is_empty() {
                            history.push(
                                ChatMessage::assistant(final_text)
                                    .with_reasoning(Some(reasoning))
                                    .with_provider_state(round_provider_state.take())
                                    .with_duration_ms(reasoning_ms),
                            );
                        }
                        history.push(ChatMessage::user_hidden(hidden));
                        continue;
                    }
                    crate::turn_policy::Verdict::Finish {
                        before_final,
                        after_final,
                    } => (before_final, after_final),
                };
                for effect in before_final {
                    land_effect(effect, history, io);
                }
                let mut final_text = final_text;
                self.hooks.run_on_response(&mut final_text).await;

                history.push(ChatMessage {
                    role: agent_llm_core::Role::Assistant,
                    content: if final_text.is_empty() {
                        None
                    } else {
                        Some(final_text.clone())
                    },
                    tool_calls: None,
                    tool_call_id: None,
                    is_error: None,
                    notice: None,
                    ts: Some(agent_llm_core::types::now_ms()),
                    images: Vec::new(),
                    reasoning: Some(reasoning),
                    duration_ms: reasoning_ms,
                    provider_state: round_provider_state.take(),
                });
                self.set_state(io, AgentState::Finished);
                let _ = io.ui_tx.send(UiEvent::AssistantMessage(final_text.clone()));
                for effect in after_final {
                    land_effect(effect, history, io);
                }
                // The outcome carries the answer the user just read — the
                // accumulated `text` only ever fed the empty-round fallback
                // above, so a caller must not read it as the reply.
                return Ok(TurnOutcome {
                    text: final_text,
                    usage,
                    tool_calls_run,
                    steers: self.steers_this_turn.clone(),
                    denials: self.denials_this_turn.clone(),
                });
            }

            history.push(ChatMessage {
                role: agent_llm_core::Role::Assistant,
                content: if round_text.is_empty() {
                    None
                } else {
                    Some(round_text.clone())
                },
                tool_calls: Some(calls.clone()),
                tool_call_id: None,
                is_error: None,
                notice: None,
                ts: Some(agent_llm_core::types::now_ms()),
                images: Vec::new(),
                reasoning: Some(reasoning),
                duration_ms: reasoning_ms,
                provider_state: round_provider_state.take(),
            });

            tool_rounds += 1;

            for (call_idx, call) in calls.iter().enumerate() {
                // P0-C4: cancel between tool dispatches — a destructive tool
                // must not fire after the user already hit Cancel.
                if io.cancel.load(std::sync::atomic::Ordering::Relaxed) {
                    // This call's card has not been drawn yet —
                    // `ToolCallStarted` comes further down — so there is no
                    // spinner to settle. A `ToolCallFinished` here would
                    // invent a failed card for a call that never showed.
                    return self.abort_cancelled(
                        io,
                        history,
                        &calls,
                        call_idx,
                        "cancelled before dispatch",
                        None,
                    );
                }
                // Skip degenerate calls — a text-protocol echo can produce an
                // empty-name or empty-args call that must never dispatch
                // (it'd surface as `unknown tool ''` and poison history).
                if call.name.trim().is_empty() {
                    // Unreachable since the pre-persist filter above — kept
                    // as a pairing guard: a skipped call must still emit a
                    // tool row, or its `tool_calls` entry dangles and strict
                    // providers reject the next request (deepseek 400).
                    let msg = "skipped malformed empty tool call".to_string();
                    let _ = io.ui_tx.send(UiEvent::ToolCallFinished {
                        name: call.name.clone(),
                        ok: false,
                        content: msg.clone(),
                        ui_type: None,
                        parent: None,
                        child_id: None,
                    });
                    history.push(ChatMessage::tool_result_err(call.id.clone(), msg));
                    continue;
                }
                tool_calls_run += 1;

                let args: serde_json::Value = serde_json::from_str(&call.arguments)
                    .unwrap_or_else(|_| serde_json::json!({ "_malformed": call.arguments }));

                // Primary locator (path/command/pattern) for the tool call —
                // preserved in full so audit/confirmation tooltips can display
                // the complete command or path without losing critical arguments.
                // A tool that reviews its own calls (a desktop tool says exactly
                // what it will do) supplies the summary; the rest get the
                // generic locator below.
                let mut review = self
                    .active_registry()
                    .spec(&call.name)
                    .and_then(|s| s.review.clone())
                    .map(|f| f(&args, &self.ctx))
                    .unwrap_or_default();
                let review_summary = review.summary.take();
                let args_preview = review_summary.clone().unwrap_or_else(|| {
                    args.get("path")
                        .or_else(|| args.get("command"))
                        .or_else(|| args.get("pattern"))
                        .or_else(|| args.get("query"))
                        .or_else(|| args.get("file"))
                        .or_else(|| args.get("action"))
                        .or_else(|| args.get("url"))
                        .or_else(|| args.get("target"))
                        .and_then(|v| v.as_str())
                        .map(|s| {
                            let s = s.trim();
                            if s.chars().count() > 4096 {
                                let mut chars = s.chars();
                                let prefix: String = chars.by_ref().take(4096).collect();
                                format!("{prefix}...")
                            } else {
                                s.to_string()
                            }
                        })
                        .unwrap_or_else(|| {
                            if let Some(obj) = args.as_object() {
                                for v in obj.values() {
                                    if let Some(s) = v.as_str() {
                                        let s = s.trim();
                                        if !s.is_empty() {
                                            if s.chars().count() > 4096 {
                                                let mut chars = s.chars();
                                                let prefix: String =
                                                    chars.by_ref().take(4096).collect();
                                                return format!("{prefix}...");
                                            } else {
                                                return s.to_string();
                                            }
                                        }
                                    }
                                }
                            }
                            String::new()
                        })
                });
                let _ = io.ui_tx.send(UiEvent::ToolCallStarted {
                    name: call.name.clone(),
                    args_preview,
                    parent: None,
                    child_id: None,
                });
                self.set_state(
                    io,
                    AgentState::ExecutingTool {
                        tool_name: call.name.clone(),
                    },
                );

                // before_tool hooks (veto/mutate, pre-gate)
                let mut call_mut = call.clone();
                if let Err(msg) = self.hooks.run_before_tool(&mut call_mut).await {
                    let _ = io.ui_tx.send(UiEvent::ToolCallFinished {
                        name: call.name.clone(),
                        ok: false,
                        content: msg.clone(),
                        ui_type: None,
                        parent: None,
                        child_id: None,
                    });
                    history.push(ChatMessage::tool_result_err(call.id.clone(), msg));
                    continue;
                }
                let call = &call_mut; // hooks may have rewritten args
                                      // Gate + dispatch must see the *rewritten* arguments; `args`
                                      // above is the model's own request, kept for the step card's
                                      // preview and the audit trail.
                let args: serde_json::Value = serde_json::from_str(&call.arguments)
                    .unwrap_or_else(|_| serde_json::json!({ "_malformed": call.arguments }));

                // Permission gate before dispatch
                let registry = self.active_registry();
                let is_readonly = registry.is_readonly(&call.name);
                // A tool's declared execution class drives the permission mode.
                let class = registry.spec(&call.name).map(|s| s.class);
                // Anything not declared read-only may write, and a plugin/MCP
                // tool has no built-in spec at all — so the baseline keys off
                // `readonly`, not `class`, and an unknown name counts as a write.
                // The baseline itself is the installed policy's business: it is
                // taken inside `before_dispatch`, lazily before the first write.
                let diff_summary = review_summary.clone().unwrap_or_else(|| {
                    args.get("path")
                        .and_then(|v| v.as_str())
                        .map(|p| format!("{p}"))
                        .unwrap_or_else(|| call.name.clone())
                });

                // P1-c: staged writers are dispatched EARLY — they return a
                // `Commit` instead of acting inside the tool, so we can
                // show the REAL diff + fuzzy flag on the approval card and only
                // commit the write after the user approves. Read-only and bash
                // tools keep the classic gate→dispatch flow (they have side
                // effects we can't dry-run). `ToolSpec::staged` is the single
                // source of truth, and documents why it is not
                // `ToolClass::WorkspaceMutation` (serena).
                let is_write = self
                    .active_registry()
                    .spec(&call.name)
                    .is_some_and(|s| s.staged);

                // Dry-run a staged tool to harvest diff/fuzzy/commit.
                // `dispatch` on these tools is side-effect-free (returns a
                // Commit), so it's safe to run pre-approval.
                let staged: Option<crate::tools::registry::ToolResult> = if is_write {
                    match self
                        .active_registry()
                        .dispatch(&call.name, args.clone(), self.ctx.clone())
                        .await
                    {
                        Ok(res) => Some(res),
                        Err(e) => {
                            // A write's path is checked by this dry run, so a
                            // refusal the host can lift becomes a question —
                            // and one grant makes the same call go through.
                            let retried = match &e {
                                agent_tool::ToolError::OutOfScope { path, suggestion } => Some(
                                    self.ask_scope_grant(io, call, path, suggestion.as_deref())
                                        .await,
                                ),
                                _ => None,
                            };
                            let outcome = match retried {
                                Some(Ok(())) => {
                                    self.active_registry()
                                        .dispatch(&call.name, args.clone(), self.ctx.clone())
                                        .await
                                }
                                Some(Err(msg)) => Err(agent_tool::ToolError::Failed(msg)),
                                None => Err(e),
                            };
                            match outcome {
                                Ok(res) => Some(res),
                                Err(e) => {
                                    let msg = e.to_string();
                                    let _ = io.ui_tx.send(UiEvent::ToolCallFinished {
                                        name: call.name.clone(),
                                        ok: false,
                                        content: msg.clone(),
                                        ui_type: None,
                                        parent: None,
                                        child_id: None,
                                    });
                                    history
                                        .push(ChatMessage::tool_result_err(call.id.clone(), msg));
                                    continue;
                                }
                            }
                        }
                    }
                } else {
                    None
                };
                let real_diff = staged
                    .as_ref()
                    .map(|r| r.content.clone())
                    .or_else(|| review_summary.clone())
                    .unwrap_or_else(|| {
                        args.get("replace")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string()
                    });
                let real_fuzzy = staged.as_ref().map(|r| r.fuzzy).unwrap_or(false);

                // A call the tool already knows cannot run is refused here,
                // before the card: an approval for it would change nothing, and
                // a question with only one possible answer is noise.
                if let Some(reason) = review.deny.take() {
                    self.denials_this_turn.push(call.name.clone());
                    let _ = io.ui_tx.send(UiEvent::ToolCallFinished {
                        name: call.name.clone(),
                        ok: false,
                        content: reason.clone(),
                        ui_type: None,
                        parent: None,
                        child_id: None,
                    });
                    history.push(ChatMessage::tool_result_err(call.id.clone(), reason));
                    continue;
                }

                let action = review.action.take();
                let decision = {
                    let gate = self.permissions.read().unwrap_or_else(|e| e.into_inner());
                    let d = gate.decide(
                        &call.name,
                        class,
                        is_readonly,
                        review.risk,
                        diff_summary.clone(),
                    );
                    match &action {
                        Some(a) => gate.escalate(&call.name, a, d, diff_summary.clone()),
                        None => d,
                    }
                };

                match decision {
                    Decision::Deny { reason } => {
                        let _ = io.ui_tx.send(UiEvent::ToolCallFinished {
                            name: call.name.clone(),
                            ok: false,
                            content: reason.clone(),
                            ui_type: None,
                            parent: None,
                            child_id: None,
                        });
                        history.push(ChatMessage::tool_result_err(call.id.clone(), reason));
                        continue;
                    }
                    Decision::Ask { diff_summary } => {
                        let fingerprint = (call.name.clone(), args.to_string());
                        // A call whose approval already ran out the clock this
                        // turn is not shown a second card: the same wait would
                        // only run out again.
                        if self.timed_out_approvals.contains(&fingerprint) {
                            self.denials_this_turn.push(call.name.clone());
                            let msg = approval_timeout_text(
                                &call.name,
                                &crate::tools::host_slots::secs_label(self.approval_timeout),
                                true,
                            );
                            let _ = io.ui_tx.send(UiEvent::ToolCallFinished {
                                name: call.name.clone(),
                                ok: false,
                                content: APPROVAL_TIMEOUT_UI.to_string(),
                                ui_type: None,
                                parent: None,
                                child_id: None,
                            });
                            history.push(ChatMessage::tool_result_err(call.id.clone(), msg));
                            continue;
                        }
                        // Acquire any platform grant before asking for in-app
                        // approval. A refusal ends the call before the card.
                        if let Some(prepare) = review.prepare.take() {
                            if let Err(msg) = prepare.await {
                                let _ = io.ui_tx.send(UiEvent::ToolCallFinished {
                                    name: call.name.clone(),
                                    ok: false,
                                    content: msg.clone(),
                                    ui_type: None,
                                    parent: None,
                                    child_id: None,
                                });
                                history.push(ChatMessage::tool_result_err(call.id.clone(), msg));
                                continue;
                            }
                        }
                        // Pause in AwaitingToolConfirmation; the UI's
                        // ToolDecision writes the shared slot via
                        // decision_slot(). P1-a: a fresh request_id ties
                        // this pending approval to its verdict — a stale
                        // decision can't approve this tool.
                        let request_id = next_request_id();
                        let _route = self.route_child_approval(request_id);
                        self.set_state(
                            io,
                            AgentState::AwaitingToolConfirmation {
                                tool_name: call.name.clone(),
                                diff_summary: diff_summary.clone(),
                            },
                        );
                        // P1-c: show the REAL diff + fuzzy flag the staged
                        // run produced — the card reflects what gets written.
                        let _ = io.ui_tx.send(UiEvent::ApprovalRequested {
                            request_id,
                            tool_name: call.name.clone(),
                            diff: real_diff.clone(),
                            fuzzy: real_fuzzy,
                            actions: Vec::new(),
                            child_id: self.child_id(),
                        });
                        let wait = self.wait_for_decision(request_id, &io.cancel).await;
                        self.set_state(
                            io,
                            AgentState::ExecutingTool {
                                tool_name: call.name.clone(),
                            },
                        );
                        match wait {
                            DecisionWait::Verdict(v) if v.approved => {}
                            DecisionWait::Verdict(_) => {
                                let msg = format!("user denied {call_name}", call_name = call.name);
                                self.denials_this_turn.push(call.name.clone());
                                let _ = io.ui_tx.send(UiEvent::ToolCallFinished {
                                    name: call.name.clone(),
                                    ok: false,
                                    content: msg.clone(),
                                    ui_type: None,
                                    parent: None,
                                    child_id: None,
                                });
                                history.push(ChatMessage::tool_result_err(call.id.clone(), msg));
                                continue;
                            }
                            // A cancel is the user abandoning the turn, not a
                            // refusal of this tool — nothing to record as a denial.
                            DecisionWait::Cancelled => {
                                let msg = format!(
                                    "{} was not run — the turn was cancelled while awaiting \
                                     approval",
                                    call.name
                                );
                                let _ = io.ui_tx.send(UiEvent::ToolCallFinished {
                                    name: call.name.clone(),
                                    ok: false,
                                    content: msg.clone(),
                                    ui_type: None,
                                    parent: None,
                                    child_id: None,
                                });
                                history.push(ChatMessage::tool_result_err(call.id.clone(), msg));
                                continue;
                            }
                            DecisionWait::TimedOut => {
                                self.denials_this_turn.push(call.name.clone());
                                self.timed_out_approvals.insert(fingerprint);
                                let msg = approval_timeout_text(
                                    &call.name,
                                    &crate::tools::host_slots::secs_label(self.approval_timeout),
                                    false,
                                );
                                let _ = io.ui_tx.send(UiEvent::ApprovalTimedOut {
                                    request_id,
                                    child_id: self.child_id(),
                                });
                                let _ = io.ui_tx.send(UiEvent::ToolCallFinished {
                                    name: call.name.clone(),
                                    ok: false,
                                    content: APPROVAL_TIMEOUT_UI.to_string(),
                                    ui_type: None,
                                    parent: None,
                                    child_id: None,
                                });
                                history.push(ChatMessage::tool_result_err(call.id.clone(), msg));
                                continue;
                            }
                        }
                        // K3: approval and cancel/policy change can race —
                        // `wait_for_decision` returns a verdict that was already
                        // in the slot. Re-check both right before side effects.
                        let post_deny: Option<String> = if io
                            .cancel
                            .load(std::sync::atomic::Ordering::Relaxed)
                        {
                            Some(format!("{} cancelled before it ran", call.name))
                        } else if let Decision::Deny { reason } = {
                            let gate = self.permissions.read().unwrap_or_else(|e| e.into_inner());
                            let d = gate.decide(
                                &call.name,
                                class,
                                is_readonly,
                                review.risk,
                                diff_summary.clone(),
                            );
                            match &action {
                                Some(a) => {
                                    // A revoke that landed while the card was open
                                    // still wins; an action that was approved by
                                    // this card is not re-asked here.
                                    match gate.escalate(&call.name, a, d, diff_summary.clone()) {
                                        Decision::Ask { .. } => Decision::Allow,
                                        d => d,
                                    }
                                }
                                None => d,
                            }
                        } {
                            Some(reason)
                        } else {
                            None
                        };
                        if let Some(msg) = post_deny {
                            let _ = io.ui_tx.send(UiEvent::ToolCallFinished {
                                name: call.name.clone(),
                                ok: false,
                                content: msg.clone(),
                                ui_type: None,
                                parent: None,
                                child_id: None,
                            });
                            history.push(ChatMessage::tool_result_err(call.id.clone(), msg));
                            continue;
                        }
                    }
                    Decision::Allow => {}
                }

                if let Some(msg) = self.policy.before_dispatch(&call.name, is_readonly).await {
                    let _ = io.ui_tx.send(UiEvent::ToolCallFinished {
                        name: call.name.clone(),
                        ok: false,
                        content: msg.clone(),
                        ui_type: None,
                        parent: None,
                        child_id: None,
                    });
                    history.push(ChatMessage::tool_result_err(call.id.clone(), msg));
                    continue;
                }

                // Staged write tools already hold their result; dispatch the
                // rest now (read-only/bash). The clock starts here — it
                // covers dispatch + deferred-write commits + after_tool
                // hooks, i.e. everything the row's `· 3s` label meant live.
                let call_started = std::time::Instant::now();
                let (mut content, ui_type, mut ok, commit, result_images) = if let Some(res) =
                    staged
                {
                    let ui_type = res.ui_type.map(|s| s.to_string());
                    (res.content, ui_type, true, res.commit, res.images)
                } else {
                    // A cancel-aware tool watches `ctx.cancel` itself and ends
                    // with its own terminal text — which is the result that
                    // has to be recorded (a cancelled child names itself
                    // there). Racing it here would drop that text for a
                    // generic line, and the UI could no longer tie the card
                    // to the child's transcript.
                    let cancel_aware = registry.spec(&call.name).is_some_and(|s| s.cancel_aware);
                    let dispatched = if cancel_aware {
                        self.active_registry()
                            .dispatch(&call.name, args.clone(), self.ctx.clone())
                            .await
                    } else {
                        // A `bash` (or the test suite under it) can run for
                        // minutes. Race it against the flag so Cancel lands
                        // now, not after the tool's own timeout.
                        let Some(dispatched) = until_cancelled(
                            &io.cancel,
                            self.active_registry().dispatch(
                                &call.name,
                                args.clone(),
                                self.ctx.clone(),
                            ),
                        )
                        .await
                        else {
                            return self.abort_cancelled(
                                io,
                                history,
                                &calls,
                                call_idx,
                                "cancelled while running",
                                Some(call.name.clone()),
                            );
                        };
                        dispatched
                    };
                    match dispatched {
                        Ok(res) => (
                            res.content,
                            res.ui_type.map(|s| s.to_string()),
                            true,
                            res.commit,
                            res.images,
                        ),
                        Err(e) => {
                            // Built-in miss → try the plugin router (built-in
                            // names are reserved and can't be shadowed).
                            // External tools use the same result contract.
                            // Plan mode dispatches read-only built-ins only.
                            // A call that is not staged has no dry run: its
                            // path is checked when it runs. A refusal the host
                            // can lift becomes the same grant question, then
                            // one more attempt with the wider reach.
                            if let agent_tool::ToolError::OutOfScope { path, suggestion } = &e {
                                let granted = self
                                    .ask_scope_grant(io, call, path, suggestion.as_deref())
                                    .await
                                    .is_ok();
                                if granted {
                                    match self
                                        .active_registry()
                                        .dispatch(&call.name, args.clone(), self.ctx.clone())
                                        .await
                                    {
                                        Ok(res) => (
                                            res.content,
                                            res.ui_type.map(|s| s.to_string()),
                                            true,
                                            res.commit,
                                            res.images,
                                        ),
                                        Err(e) => (e.to_string(), None, false, None, Vec::new()),
                                    }
                                } else {
                                    (e.to_string(), None, false, None, Vec::new())
                                }
                            } else if !self.plugins_allowed() {
                                (e.to_string(), None, false, None, Vec::new())
                            } else if let Some(mgr) = self.plugins() {
                                // Report the plugin's own failure. Its message names the
                                // real cause (server error, dead bridge); the built-in
                                // "unknown tool" text would send the model hunting for a
                                // tool that exists. A plugin round-trip is a network
                                // wait too, so it gets the same cancel race.
                                let Some(routed) = until_cancelled(
                                    &io.cancel,
                                    mgr.dispatch_tool_call(&call.name, args),
                                )
                                .await
                                else {
                                    return self.abort_cancelled(
                                        io,
                                        history,
                                        &calls,
                                        call_idx,
                                        "cancelled while running",
                                        Some(call.name.clone()),
                                    );
                                };
                                match routed {
                                    Some(Ok(out)) => (
                                        out.content,
                                        out.ui_type.map(|s| s.to_string()),
                                        true,
                                        out.commit,
                                        out.images,
                                    ),
                                    // An unowned name preserves the registry's real error.
                                    None => (e.to_string(), None, false, None, Vec::new()),
                                    Some(Err(msg)) => (msg, None, false, None, Vec::new()),
                                }
                            } else {
                                (e.to_string(), None, false, None, Vec::new())
                            }
                        }
                    }
                };

                // P1-c: apply the effect the tool staged — the engine owns it
                // so it lands only after the permission decision. `verify`
                // first (the preview must still hold), then `apply`; either
                // failing abandons the commit and fails the call with the
                // tool's own reason. What the effect IS stays the tool's
                // business — the engine never looks inside.
                if let Some(commit) = commit {
                    let agent_tool::Commit { verify, apply } = commit;
                    match verify() {
                        Err(reason) => {
                            ok = false;
                            content = reason;
                        }
                        Ok(()) => {
                            if let Err(reason) = apply.await {
                                ok = false;
                                content = reason;
                            }
                        }
                    }
                }

                // after_tool hooks (may mutate output)
                self.hooks.run_after_tool(&call, &mut content).await;

                let _ = io.ui_tx.send(UiEvent::ToolCallFinished {
                    name: call.name.clone(),
                    ok,
                    content: content.clone(),
                    ui_type,
                    parent: None,
                    child_id: None,
                });

                // `ok` survives into the snapshot via `is_error` so a
                // reloaded view replays the red capsule instead of a green
                // one — the live `ToolCallFinished.ok` carried it.
                let result_msg = if ok {
                    ChatMessage::tool_result(call.id.clone(), content)
                } else {
                    ChatMessage::tool_result_err(call.id.clone(), content)
                };
                // A tool that produced images (a screenshot) hands them to the
                // model alongside its text. Adapters encode them per their wire
                // — Anthropic carries real blocks inside `tool_result`, the
                // text-only tool protocols get a synthetic user turn after it.
                // Gated on vision like the composer path: a text-only model
                // keeps the file path the tool already printed.
                let result_msg = if result_images.is_empty() || !self.supports_images() {
                    result_msg
                } else {
                    result_msg.tool_result_with_images(result_images)
                };
                // The call's wall time persists with the row — a replayed
                // tools step re-derives its span from `ts - duration_ms`.
                history.push(
                    result_msg.with_duration_ms(Some(call_started.elapsed().as_millis() as i64)),
                );
            }

            // If we have reached or exceeded the tool limit, force the next round to be a synthesis round without tools
            if tool_rounds >= max_tool_rounds && !force_no_tools {
                let msg = format!("已达单轮工具调用上限 ({max_tool_rounds} 轮)，正在汇总已收集的信息生成最终回答...");
                let _ = io.ui_tx.send(UiEvent::SystemMessage(msg.clone()));
                history.push(ChatMessage::notice(msg));
                history.push(ChatMessage::user_hidden(
                    "You have reached the tool execution limit for this turn. Do NOT request any more tool calls. Please synthesize all your findings and provide your complete, detailed response/answer to the user now.",
                ));
                force_no_tools = true;
            }

            self.set_state(io, AgentState::Reasoning);
        }
    }

    /// Abandon the turn because the user cancelled.
    ///
    /// Every call from `from` on still gets a tool result — an assistant row
    /// carrying `tool_calls` with unanswered results orphans the next request
    /// (strict providers reject it) and makes a reloaded view silently drop
    /// chips the live view had. `running` names the call whose dispatch was
    /// interrupted, so its card settles instead of spinning forever — `None`
    /// when the cancel landed before that call's `ToolCallStarted`.
    fn abort_cancelled(
        &mut self,
        io: &mut EngineIo,
        history: &mut Vec<ChatMessage>,
        calls: &[agent_llm_core::types::ToolCall],
        from: usize,
        label: &str,
        running: Option<String>,
    ) -> Result<TurnOutcome, String> {
        if let Some(name) = running {
            let _ = io.ui_tx.send(UiEvent::ToolCallFinished {
                name,
                ok: false,
                content: CANCEL_TEXT.into(),
                ui_type: None,
                parent: None,
                child_id: None,
            });
        }
        for pending_call in calls.iter().skip(from) {
            history.push(ChatMessage::tool_result_err(pending_call.id.clone(), label));
        }
        history.push(ChatMessage::notice(CANCEL_TEXT));
        self.set_state(io, AgentState::Failed(CANCEL_ERR.into()));
        let _ = io.ui_tx.send(UiEvent::SystemMessage(CANCEL_TEXT.into()));
        Err(CANCEL_ERR.into())
    }

    /// Auto-compaction check — the turn-start pass and every mid-turn tool
    /// round share this. Fires when the estimated fill crosses `compact_at`
    /// (the user-settable 70/80/90% of the window); the suppressor prevents
    /// a retry storm after a failure, with a 95% emergency escape.
    async fn auto_compact_if_needed(&mut self, io: &mut EngineIo, history: &mut Vec<ChatMessage>) {
        // The provider's reported `prompt_tokens` is the better signal when
        // present — the chars-based estimate under-reads CJK history.
        let est = compaction::estimate_tokens(history).max(self.last_prompt_tokens);
        let fill = est as f32 / self.context_window as f32;
        if !compaction::should_compact_at(est, self.context_window, self.compact_at)
            || !self.compaction_suppressor.check_at_fill(fill)
        {
            return;
        }
        self.set_state(io, AgentState::Compacting);
        match self.compact_history(io, history, /*manual*/ false).await {
            Ok(_) => {
                // The card row + `UiEvent::Compacted` already went out from
                // `compact_history` — no second system line here (that
                // duplicate was the whole visible feedback before).
                self.compaction_suppressor.on_success();
            }
            Err(e) => {
                self.compaction_suppressor.on_failure();
                let line = format!("上下文压缩失败: {e}");
                let _ = io.ui_tx.send(UiEvent::SystemMessage(line.clone()));
                history.push(ChatMessage::notice(line));
            }
        }
        self.set_state(io, AgentState::Reasoning);
    }

    /// Compaction: split history keeping ~25% as the verbatim suffix →
    /// summarize the prefix into a dense `NOTE` via the current provider →
    /// splice the note in place of the prefix. A prior NOTE folds into the
    /// new one (the compactor is told to merge, not stack).
    async fn compact_history(
        &mut self,
        io: &mut EngineIo,
        history: &mut Vec<ChatMessage>,
        manual: bool,
    ) -> Result<Option<compaction::CompactionOutcome>, String> {
        let before_tokens = compaction::estimate_tokens(history);
        // `None` = nothing to do (too few messages, or the whole history
        // already fits the ~25% suffix budget). Not an error — the caller
        // decides how to report it.
        let Some(plan) = compaction::plan(history, self.context_window) else {
            return Ok(None);
        };
        // The running card can appear immediately — the pass below is a real
        // provider request on the pre-splice prefix.
        let _ = io.ui_tx.send(UiEvent::CompactionStarted { manual });

        // `history[0]` is the rendered kernel system prompt — it survives
        // compaction verbatim, so feeding it to the compactor only burns
        // input tokens and risks protocol language leaking into the note.
        // UI-only rows are cards, not conversation — feeding their JSON
        // payload to the compactor is pure noise.
        let prefix_text: String = history[1..plan.prefix_end]
            .iter()
            .filter(|m| !m.notice.as_ref().is_some_and(|n| n.is_ui_only()))
            .map(compaction::render_for_summary)
            .collect::<Vec<_>>()
            .join("\n---\n");
        let summarize_history = vec![
            ChatMessage::system(
                "You are a context compactor. Compress the conversation below into a dense \
                 note (facts, decisions, file:line touched, tool outcomes) that preserves \
                 everything needed to continue the task. If the input already contains a \
                 [BEGIN COMPACTED CONTEXT] note, merge its content into the new note — do \
                 not stack summaries. Output only the note.",
            ),
            ChatMessage::user(prefix_text),
        ];

        let mut note = String::new();
        // Same retry-reset rule as the main loop: a retry regenerates the
        // summary from scratch, so the dead attempt's text must not survive
        // into `note`.
        let retry_reset = std::sync::atomic::AtomicBool::new(false);
        let req = SampleRequest {
            model: &self.model,
            temperature: 0.0,
            tools: None,
            reasoning_effort: None,
            // Summarization is a real request on the same model — it must
            // respect the same output cap and per-model compat.
            params: &self.model_params,
        };
        self.sampler
            .sample(
                req,
                &summarize_history,
                |chunk| {
                    if retry_reset.swap(false, std::sync::atomic::Ordering::Relaxed) {
                        note.clear();
                    }
                    if let StreamChunk::ContentDelta(t) = chunk {
                        note.push_str(t);
                    }
                },
                |ev| {
                    if matches!(
                        ev,
                        SamplerEvent::Retrying { .. } | SamplerEvent::FellBack { .. }
                    ) {
                        retry_reset.store(true, std::sync::atomic::Ordering::Relaxed);
                    }
                },
                || {},
            )
            .await
            .map_err(|e| e.to_string())?;

        if note.trim().is_empty() {
            return Err("compactor returned empty note".into());
        }
        let removed_messages = plan.prefix_end.saturating_sub(1);
        compaction::apply(history, &plan, note.clone());
        // The pre-compact `prompt_tokens` is stale — reseed the actual-usage
        // signal with the post-splice estimate or the trigger would
        // immediately re-fire on the stale (large) figure.
        let after_tokens = compaction::estimate_tokens(history);
        self.last_prompt_tokens = after_tokens;
        // Persist the display row at the point in the stream where the
        // compaction happened (mid-turn: right after the live user turn) —
        // replay then draws the card in the same place the live event did.
        history.push(ChatMessage::compaction(
            before_tokens as u32,
            after_tokens as u32,
            removed_messages as u32,
            manual,
            &note,
        ));
        // Card + meter in one event: the stream gets the summary, the
        // title-bar fill drops to the post-splice figure now instead of at
        // the next sample's `Usage`.
        let _ = io.ui_tx.send(UiEvent::Compacted {
            before_tokens: before_tokens as u32,
            after_tokens: after_tokens as u32,
            removed_messages: removed_messages as u32,
            context_window: self.context_window as u32,
            manual,
            note: note.clone(),
        });
        Ok(Some(compaction::CompactionOutcome {
            before_tokens,
            after_tokens,
            removed_messages,
            note,
        }))
    }

    /// Manual `/compact` — bypasses the threshold AND the suppressor (the
    /// user asked for it now), but a result still updates the suppression
    /// state so a failed manual attempt doesn't leave the storm guard off.
    pub async fn compact_now(
        &mut self,
        io: &mut EngineIo,
        history: &mut Vec<ChatMessage>,
    ) -> Result<CompactOutcome, String> {
        match self.compact_history(io, history, /*manual*/ true).await {
            Ok(Some(outcome)) => {
                self.compaction_suppressor.on_success();
                Ok(CompactOutcome::Compacted(outcome))
            }
            Ok(None) => Ok(CompactOutcome::NothingToDo),
            Err(e) => {
                self.compaction_suppressor.on_failure();
                Err(e)
            }
        }
    }

    /// A tool refused a path outside every root this session may reach, and
    /// the host can still grant one: put the grants on the approval card, then
    /// let the host apply the pick. `Ok(())` means "granted — run the same call
    /// again"; the caller retries at most once, so a host that grants nothing
    /// useful cannot loop the engine.
    ///
    /// Without a host slot this stays a plain refusal: the kernel has no idea
    /// what a directory grant would even mean.
    async fn ask_scope_grant(
        &mut self,
        io: &mut EngineIo,
        call: &agent_llm_core::types::ToolCall,
        path: &str,
        suggestion: Option<&str>,
    ) -> Result<(), String> {
        const LINK_SESSION: &str = "link:session";
        const LINK_PERSIST: &str = "link:workspace";
        if !self.ctx.has_scope_grant() {
            return Err(format!("path escapes workspace: {path}"));
        }
        // A read that escaped asks for read access, not for a writable
        // checkout: the person answering this card was reading a file, and
        // widening it to "edit anything under that project" is a grant they
        // never saw. A write tool — and `bash`, which can write — still asks
        // for the wider grant.
        let readonly = self.active_registry().is_readonly(&call.name);
        let access = if readonly { "read" } else { "readwrite" };
        let grant_dir = suggestion.map_or_else(|| path.to_string(), str::to_string);
        let scope_word = if readonly { "只读" } else { "读写" };
        // The host decides what a link means; the kernel only names the two
        // scopes every host that links directories offers.
        let actions = vec![
            agent_ipc::events::ApprovalAction {
                id: LINK_SESSION.into(),
                label: format!("关联（本会话 · {scope_word}）"),
                description: Some(format!(
                    "本次会话可以{scope_word}访问 {grant_dir}，会话结束即失效"
                )),
            },
            agent_ipc::events::ApprovalAction {
                id: LINK_PERSIST.into(),
                label: format!("关联并记住（{scope_word}）"),
                description: Some(format!(
                    "这个工作区的每个会话都可以{scope_word}访问 {grant_dir}"
                )),
            },
        ];
        // What the user is being asked, in the order they need it: the
        // directory that would be granted (and how far), then the path that
        // was refused.
        let diff = format!("关联目录：{grant_dir}（{scope_word}）\n被拒绝的路径：{path}");
        let request_id = next_request_id();
        let _route = self.route_child_approval(request_id);
        self.set_state(
            io,
            AgentState::AwaitingToolConfirmation {
                tool_name: call.name.clone(),
                diff_summary: diff.clone(),
            },
        );
        let _ = io.ui_tx.send(UiEvent::ApprovalRequested {
            request_id,
            tool_name: call.name.clone(),
            diff,
            fuzzy: false,
            actions,
            child_id: self.child_id(),
        });
        let wait = self.wait_for_decision(request_id, &io.cancel).await;
        self.set_state(
            io,
            AgentState::ExecutingTool {
                tool_name: call.name.clone(),
            },
        );
        let verdict = match wait {
            DecisionWait::Verdict(v) => v,
            DecisionWait::Cancelled => {
                return Err(format!("access grant for {path} was cancelled"));
            }
            DecisionWait::TimedOut => {
                let _ = io.ui_tx.send(UiEvent::ApprovalTimedOut {
                    request_id,
                    child_id: self.child_id(),
                });
                return Err(format!(
                    "[approval timeout] access to {path} was not granted: the user did not \
                     respond within {}s, so the request was automatically denied.",
                    crate::tools::host_slots::secs_label(self.approval_timeout)
                ));
            }
        };
        if !verdict.approved {
            return Err(format!("user did not grant access to {path}"));
        }
        let Some(action) = verdict.action else {
            return Err(format!("user did not grant access to {path}"));
        };
        if action != LINK_SESSION && action != LINK_PERSIST {
            return Err(format!("`{action}` is not a grant this session offers"));
        }
        match self.ctx.grant_scope(&action, path, access) {
            Some(Ok(())) => Ok(()),
            Some(Err(msg)) => Err(msg),
            None => Err("this host cannot link directories".into()),
        }
    }

    /// Poll the shared decision slot until SessionActor writes a verdict
    /// for THIS `request_id`, the turn is cancelled, or the approval timeout
    /// fires (fail-closed: a timeout denies the call).
    ///
    /// The verdict slot holds `Option<(request_id, verdict)>`. A verdict
    /// whose id doesn't match is a stale click on an already-resolved card —
    /// it is cleared and ignored rather than allowed to approve this tool
    /// (P1-a). A timeout takes the slot under the same lock, so a verdict
    /// that lands afterwards has nowhere to go and the next card (a new id)
    /// cannot be answered by it. Cancelling does not wait out the timeout.
    async fn wait_for_decision(
        &self,
        request_id: u64,
        cancel: &std::sync::Arc<std::sync::atomic::AtomicBool>,
    ) -> DecisionWait {
        let deadline = (!self.approval_timeout.is_zero())
            .then(|| std::time::Instant::now() + self.approval_timeout);
        loop {
            // Poison-safe: a panicked writer must not deadlock the engine.
            // Take the verdict out of the slot in a short scope so the
            // MutexGuard is never held across `.await` (not `Send`).
            let verdict = {
                let mut slot = self.decision.lock().unwrap_or_else(|e| e.into_inner());
                // `take` — the verdict is not `Copy`, and a stale one must not
                // stay in the slot where it could answer a later request.
                match slot.take() {
                    Some((id, verdict)) if id == request_id => Some(verdict),
                    Some(_) => None,
                    None => None,
                }
            };
            if let Some(verdict) = verdict {
                return DecisionWait::Verdict(verdict);
            }
            if cancel.load(std::sync::atomic::Ordering::Relaxed) {
                return DecisionWait::Cancelled;
            }
            if deadline.is_some_and(|d| std::time::Instant::now() > d) {
                // Close the card under the slot's own lock: whatever arrives
                // after this point is a late click and is discarded.
                let _ = self
                    .decision
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .take();
                return DecisionWait::TimedOut;
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
    }

    fn set_state(&self, io: &EngineIo, state: AgentState) {
        // Lifecycle hooks observe the transition fire-and-forget: a hook must
        // never delay the state the UI is about to render.
        self.hooks.observe_nowait(&state);
        let _ = io.ui_tx.send(UiEvent::StateChanged(state));
    }
}

/// Pull `<attached-image path="…"/>` markers out of a prompt — the
/// composer's image-attachment channel, mirroring `<attached-file>`.
/// The marker itself stays in `content` (path provenance the model can
/// act on with `read`/`bash`); the returned refs become real wire parts
/// when the model declares `"image"` input.
fn extract_attached_images(text: &str) -> Vec<agent_llm_core::types::ImageRef> {
    const TAG: &str = "<attached-image path=\"";
    let mut out = Vec::new();
    let mut rest = text;
    while let Some(start) = rest.find(TAG) {
        let after = &rest[start + TAG.len()..];
        let Some(end) = after.find('"') else { break };
        if let Some(img) =
            agent_llm_core::types::ImageRef::for_path(std::path::PathBuf::from(&after[..end]))
        {
            out.push(img);
        }
        rest = &after[end..];
    }
    out
}

/// Coalescing bounds for streamed text/reasoning deltas: the bounded UI
/// channel must not see a per-token burst, but a slow model must not leave
/// the UI idle until a size threshold fills either.
const DELTA_FLUSH_BYTES: usize = 120;
const DELTA_FLUSH_AFTER: std::time::Duration = std::time::Duration::from_millis(40);

/// The text and reasoning deltas waiting to be sent to the UI, with the time
/// each side last flushed.
struct DeltaBuffers {
    text: String,
    reasoning: String,
    text_flushed_at: std::time::Instant,
    reasoning_flushed_at: std::time::Instant,
}

impl DeltaBuffers {
    fn new() -> Self {
        let now = std::time::Instant::now();
        Self {
            text: String::new(),
            reasoning: String::new(),
            text_flushed_at: now,
            reasoning_flushed_at: now,
        }
    }

    fn clear(&mut self) {
        self.text.clear();
        self.reasoning.clear();
    }

    /// The pending text if it is due (or `force`d), leaving the buffer empty.
    fn take_text(&mut self, force: bool) -> Option<String> {
        let due = force && !self.text.is_empty()
            || should_flush_delta(self.text.len(), self.text_flushed_at.elapsed());
        if !due {
            return None;
        }
        self.text_flushed_at = std::time::Instant::now();
        Some(std::mem::take(&mut self.text))
    }

    fn take_reasoning(&mut self, force: bool) -> Option<String> {
        let due = force && !self.reasoning.is_empty()
            || should_flush_delta(self.reasoning.len(), self.reasoning_flushed_at.elapsed());
        if !due {
            return None;
        }
        self.reasoning_flushed_at = std::time::Instant::now();
        Some(std::mem::take(&mut self.reasoning))
    }
}

/// Flush a pending delta once it is big enough OR has waited long enough.
/// Evaluated when a chunk arrives, so an empty buffer never flushes.
fn should_flush_delta(pending_bytes: usize, since_last_flush: std::time::Duration) -> bool {
    pending_bytes >= DELTA_FLUSH_BYTES
        || (pending_bytes > 0 && since_last_flush >= DELTA_FLUSH_AFTER)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delta_flushes_on_size_or_age_but_never_when_empty() {
        use std::time::Duration;
        let young = Duration::from_millis(5);
        let old = Duration::from_millis(40);
        assert!(!should_flush_delta(0, old), "empty buffer must not flush");
        assert!(!should_flush_delta(10, young), "small + young waits");
        assert!(should_flush_delta(10, old), "small but old flushes");
        assert!(
            should_flush_delta(DELTA_FLUSH_BYTES, young),
            "full flushes immediately"
        );
    }

    #[test]
    fn test_thinking_level_resolution() {
        let mut engine = Engine::new(
            Arc::new(agent_llm_core::provider::UnconfiguredProvider),
            Arc::new(ToolRegistry::with_builtins()),
            Arc::new(ToolCtx::new()),
            "devin/swe-2",
            1.0,
        );

        assert_eq!(engine.resolve_reasoning_effort(), None);

        engine.set_thinking_level(Some("off".into()));
        assert_eq!(engine.resolve_reasoning_effort(), None);

        engine.set_thinking_level(Some("medium".into()));
        assert_eq!(engine.resolve_reasoning_effort().as_deref(), Some("medium"));
        engine.set_model_thinking_level(Some("xhigh".into()));
        assert_eq!(engine.thinking_level(), None);

        // Custom thinking-level map (like Devin SWE-2).
        let mut map = std::collections::HashMap::new();
        map.insert("off".into(), None);
        map.insert("minimal".into(), None);
        map.insert("low".into(), None);
        map.insert("medium".into(), Some("medium".into()));
        map.insert("high".into(), Some("high".into()));
        map.insert("max".into(), Some("max".into()));
        engine.set_thinking_level_map(Some(map));

        engine.set_thinking_level(Some("medium".into()));
        assert_eq!(engine.resolve_reasoning_effort().as_deref(), Some("medium"));

        engine.set_thinking_level(Some("high".into()));
        assert_eq!(engine.resolve_reasoning_effort().as_deref(), Some("high"));

        engine.set_thinking_level(Some("max".into()));
        assert_eq!(engine.resolve_reasoning_effort().as_deref(), Some("max"));

        engine.set_thinking_level(Some("low".into()));
        // Null disables the tier; normalize to the model's supported default.
        assert_eq!(engine.thinking_level().as_deref(), Some("medium"));
        assert_eq!(engine.resolve_reasoning_effort().as_deref(), Some("medium"));

        engine.set_thinking_level(Some("off".into()));
        assert_eq!(engine.resolve_reasoning_effort(), None);
    }

    #[test]
    fn model_switch_normalizes_a_stale_thinking_level() {
        let mut engine = Engine::new(
            Arc::new(agent_llm_core::provider::UnconfiguredProvider),
            Arc::new(ToolRegistry::with_builtins()),
            Arc::new(ToolCtx::new()),
            "model-b",
            1.0,
        );
        engine.set_thinking_level_map(Some(std::collections::HashMap::from([(
            "xhigh".into(),
            Some("xhigh".into()),
        )])));
        engine.set_thinking_level(Some("xhigh".into()));
        engine.set_thinking_level_map(Some(std::collections::HashMap::from([
            ("low".into(), Some("low".into())),
            ("high".into(), Some("high".into())),
            ("max".into(), Some("max".into())),
        ])));
        assert_eq!(engine.thinking_level().as_deref(), Some("low"));
        assert_eq!(engine.resolve_reasoning_effort().as_deref(), Some("low"));
        engine.set_thinking_level(Some("xhigh".into()));
        assert_eq!(engine.thinking_level().as_deref(), Some("low"));
        engine.set_thinking_level_map(None);
        assert_eq!(engine.thinking_level(), None);
        assert_eq!(engine.resolve_reasoning_effort(), None);
    }

    #[test]
    fn undeclared_thinking_level_never_reaches_the_wire() {
        let mut engine = Engine::new(
            Arc::new(agent_llm_core::provider::UnconfiguredProvider),
            Arc::new(ToolRegistry::with_builtins()),
            Arc::new(ToolCtx::new()),
            "model-a",
            1.0,
        );
        engine.set_thinking_level_map(Some(std::collections::HashMap::from([
            ("low".into(), Some("low".into())),
            ("high".into(), Some("high".into())),
            ("max".into(), Some("max".into())),
        ])));
        // A stale/shared-slot value must also be caught at the wire boundary.
        *engine.thinking_level_shared().write().unwrap() = Some("xhigh".into());
        assert_eq!(engine.resolve_reasoning_effort(), None);
    }

    #[test]
    fn extracts_attached_image_markers() {
        let text = "看看这张图\n\n<attached-image path=\"/ws/.husk/attachments/cat-a1b2.png\" name=\"cat.png\"/>\n\n<attached-image path=\"/ws/.husk/attachments/doc-c3d4.txt\"/>";
        let imgs = extract_attached_images(text);
        assert_eq!(imgs.len(), 1); // .txt isn't an image type
        assert_eq!(imgs[0].media_type, "image/png");
        assert!(imgs[0].path.ends_with("cat-a1b2.png"));
        assert!(extract_attached_images("plain text").is_empty());
    }
}
