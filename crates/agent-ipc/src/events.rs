//! The two event enums that cross the kernel↔frontend channel.
//!
//! `AgentEvent` (sampler→session feedback) is kernel-internal, carries
//! `agent_llm::StreamChunk` and never crosses a transport — see
//! `agent-kernel/src/channels.rs`.

use serde::{Deserialize, Serialize};

/// Kernel state machine — mirrored to the UI so the status strip can render
/// the current phase without guessing.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgentState {
    Idle,
    ScanningWorkspace,
    /// Waiting on the first SSE byte.
    Reasoning,
    /// Text deltas flowing.
    StreamingToken,
    /// A write/destructive tool is paused for user review.
    AwaitingToolConfirmation {
        tool_name: String,
        diff_summary: String,
    },
    /// A plugin wants a capability beyond its manifest.
    AwaitingPluginConsent {
        plugin_id: String,
        capability: String,
    },
    /// P2: generalizes tool/plugin/capture approvals.
    AwaitingConsent {
        kind: String,
    },
    /// P2: CoW fork-and-verify planning.
    Branching {
        candidates: u8,
    },
    ExecutingTool {
        tool_name: String,
    },
    /// A compaction pass is in flight (auto near the window edge, or a
    /// manual `/compact`) — the UI shows a running card + status row.
    Compacting,
    Finished,
    Failed(String),
}

impl AgentState {
    /// Is the kernel mid-turn (any non-terminal, non-idle state)?
    pub fn is_active(&self) -> bool {
        !matches!(
            self,
            AgentState::Idle | AgentState::Finished | AgentState::Failed(_)
        )
    }

    /// Short human label for the status bar — deliberately NOT the Debug
    /// dump: `AwaitingToolConfirmation` carries a whole `diff_summary`
    /// (hundreds of chars) that would flood the bar, so map each variant
    /// to a compact, single-line description instead.
    pub fn label(&self) -> String {
        match self {
            AgentState::Idle => "idle".into(),
            AgentState::ScanningWorkspace => "scanning".into(),
            AgentState::Reasoning => "reasoning".into(),
            AgentState::StreamingToken => "streaming".into(),
            AgentState::AwaitingToolConfirmation { tool_name, .. } => {
                format!("confirm {tool_name}")
            }
            AgentState::AwaitingPluginConsent {
                plugin_id,
                capability,
            } => {
                format!("consent {plugin_id}:{capability}")
            }
            AgentState::AwaitingConsent { kind } => format!("consent {kind}"),
            AgentState::Branching { candidates } => format!("branching ×{candidates}"),
            AgentState::ExecutingTool { tool_name } => format!("running {tool_name}"),
            AgentState::Compacting => "compacting".into(),
            AgentState::Finished => "done".into(),
            AgentState::Failed(e) => format!("failed: {}", e.chars().take(30).collect::<String>()),
        }
    }
}

/// One offered answer on an `ask_question` card.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AskOption {
    pub label: String,
    #[serde(default)]
    pub description: Option<String>,
}

/// One thing the user may do instead of a plain yes/no — a grant the host
/// offers when a tool refused a path (associate another directory with the
/// session). `id` is opaque to the UI: it echoes back in `ToolDecision` and
/// only the host that offered it knows what it means.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalAction {
    pub id: String,
    pub label: String,
    #[serde(default)]
    pub description: Option<String>,
}

/// Kernel → UI events. Every variant is displayable data — the frontend never
/// re-parses text to figure out what happened.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UiEvent {
    /// A state-machine transition.
    StateChanged(AgentState),
    /// The user prompt echoed back — the stream renders it as a user block.
    UserPrompt(String),
    /// A mid-turn steer echoed back. Drawn as a user bubble like
    /// [`UiEvent::UserPrompt`], but it belongs to the turn already running and
    /// never opens one. Carries the steer's own words, without the prefix the
    /// model reads.
    UserSteer(String),
    /// A token delta (already throttled by the bridge — NOT per SSE chunk).
    /// `parent` is set on a delegated child's deltas so the UI nests them under
    /// that child's capsule instead of mixing them into the turn's draft.
    TextDelta {
        text: String,
        #[serde(default)]
        parent: Option<String>,
        /// Same child marker as `ToolCallStarted::child_id` — a nested delta
        /// names the subagent it belongs to, not just its label.
        #[serde(default)]
        child_id: Option<u64>,
    },
    /// Reasoning-trace delta (separate visual stream; same `parent` rule).
    ReasoningDelta {
        text: String,
        #[serde(default)]
        parent: Option<String>,
        /// Same child marker as `ToolCallStarted::child_id`.
        #[serde(default)]
        child_id: Option<u64>,
    },
    /// A tool began executing. `args_preview` is a single-line summary of
    /// the call's target (a path / command / pattern) shown on the capsule —
    /// the full output only appears when the capsule is expanded.
    ToolCallStarted {
        name: String,
        args_preview: String,
        /// Set on calls emitted INSIDE another tool (`batch_execute`
        /// items) — the UI nests them under the parent capsule instead
        /// of counting them as top-level calls.
        #[serde(default)]
        parent: Option<String>,
        /// Set on a delegated subagent's card: the id a `SteerChild` names.
        #[serde(default)]
        child_id: Option<u64>,
    },
    /// A tool finished (or failed); `content` is the truncated model-facing
    /// text, `ui_type` is the optional typed card hint (`diff`/`table`/…).
    ToolCallFinished {
        name: String,
        ok: bool,
        content: String,
        ui_type: Option<String>,
        /// Same nesting marker as `ToolCallStarted::parent`.
        #[serde(default)]
        parent: Option<String>,
        /// Same child marker as `ToolCallStarted::child_id`.
        #[serde(default)]
        child_id: Option<u64>,
    },
    /// `ask_question` paused the turn for structured input — the card
    /// offers the model's options plus a free-text field. Resolved by
    /// `AnswerQuestion`.
    QuestionAsked {
        request_id: u64,
        question: String,
        options: Vec<AskOption>,
    },
    /// The matching `AnswerQuestion` resolved the parked card — emitted so
    /// the frontend can drop `pendingQuestion` from the view buffer itself,
    /// rather than hiding it behind a per-mount flag that a session switch
    /// would reset (the "answered card comes back" bug).
    QuestionAnswered { request_id: u64 },
    /// The question's deadline expired before anyone answered — the kernel
    /// already closed the card and handed the model a timeout. The UI marks
    /// the card as timed out instead of leaving it answerable.
    QuestionTimedOut { request_id: u64 },
    /// A destructive op paused for review — `diff` is the unified diff for
    /// the approval card; `fuzzy` flags an approximate patch match.
    /// `request_id` is the correlation token the frontend echoes back in
    /// `ToolDecision` so a stale verdict can't land on a different pending
    /// approval (P1-a).
    ApprovalRequested {
        request_id: u64,
        tool_name: String,
        diff: String,
        fuzzy: bool,
        /// Alternatives to 允许/拒绝, offered when the tool's refusal is a
        /// permission the host can grant (a path outside the workspace). An
        /// empty list means an ordinary approve/deny card. Older readers see
        /// the absent field as empty.
        #[serde(default)]
        actions: Vec<ApprovalAction>,
        /// The delegated child that asked, when the card belongs to one. The
        /// child paints its card on the parent's stream, so without this the
        /// click has no way to name whose turn is parked. Absent = the
        /// session's own turn.
        #[serde(default)]
        child_id: Option<u64>,
    },
    /// The approval card's deadline expired unanswered — the request was
    /// denied automatically (`ToolCallFinished` carries the result). The UI
    /// marks the card as timed out so a late click can't look meaningful.
    ApprovalTimedOut {
        request_id: u64,
        #[serde(default)]
        child_id: Option<u64>,
    },
    /// Final assistant text for the turn (complete, post-streaming).
    AssistantMessage(String),
    /// The session's tool whitelist just gained `tool_name` — emitted when
    /// `ApproveSessionTool` lands so the UI can show the matching overlay
    /// (computer-use's "AI is controlling your screen" HUD + edge glow).
    /// `stopped:false` on grant, `stopped:true` on revoke.
    SessionToolWhitelisted {
        tool_name: String,
        #[serde(default)]
        stopped: bool,
    },
    /// System/degrade notice (sampler retry, fallback, compaction…).
    SystemMessage(String),
    /// Token-usage update for the meter. `context_window` rides along so
    /// the UI can render fill percentage without re-deriving the bound.
    Usage {
        prompt_tokens: u32,
        completion_tokens: u32,
        context_window: u32,
        cached_tokens: u32,
    },
    /// Session-fatal error.
    Error(String),
    /// A compaction pass finished — older turns were folded into a summary
    /// note. The UI draws a compaction card (before → after tokens, how
    /// many messages were merged, the summary itself) and re-bases the
    /// context meter on `after_tokens` immediately, instead of waiting for
    /// the next sample's `Usage`.
    Compacted {
        before_tokens: u32,
        after_tokens: u32,
        removed_messages: u32,
        context_window: u32,
        /// `true` = the user ran `/compact`, `false` = the automatic trigger.
        /// The UI keeps an automatic pass inside the turn it happened in, but
        /// gives a manual one its own block — it is a user action between
        /// turns, not part of the previous answer.
        manual: bool,
        note: String,
    },
    /// A compaction pass started — emitted right before the summarization
    /// sample so the UI can draw the running card while the (potentially
    /// slow) pass runs. `manual` mirrors `Compacted`; no accounting yet.
    CompactionStarted { manual: bool },
    /// A product-defined event. The kernel forwards it untouched; `kind`
    /// names it for the host's frontends and `payload` is its data.
    Custom {
        kind: String,
        payload: serde_json::Value,
    },
    /// A `Retry` command rewound the session to the last user prompt —
    /// the UI drops the retried turn's items before the fresh
    /// `UserPrompt` echo lands on the same stream.
    TurnRetry,
    /// The session's queued-prompt list changed — emitted on every
    /// `SetQueued` write and after each item pops off the drain, so the
    /// composer always renders the kernel's list (single source of truth).
    /// Carries the full list — queue ops are rare and the list is small.
    QueuedPrompts { items: Vec<String> },
}
