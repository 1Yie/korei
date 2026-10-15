//! Turn policy seam: what happens when a model round goes quiet.
//!
//! The turn loop in `engine.rs` owns sampling, tool dispatch and history. The
//! product rules that push a quiet turn back to work (plan submission, goal
//! declaration, architecture rework) live behind [`TurnPolicy`], so the loop
//! does not name them — the coding product's rule set is `agent-coding`'s
//! `CodingPolicy`, installed by the host.
//!
//! The policy owns its per-turn state (followup counters, the architecture
//! baseline, the goal/plan controllers) behind interior mutability: the engine
//! calls [`TurnPolicy::begin_turn`] at turn start and never sees any of it.
//! What crosses the seam is generic — the current mode id, whether tools ran,
//! and an opaque report/payload pair the engine renders without knowing what
//! produced them.

/// One thing a policy asks the engine to land: a row for the conversation
/// history and/or an event for the UI. The engine appends and sends them in
/// order without looking inside — what they mean (a plan card, a check
/// report) is the policy's product, and the UI event's wire shape is the
/// host's contract.
#[derive(Default)]
pub struct Effect {
    pub history: Option<agent_llm_core::ChatMessage>,
    pub ui: Option<agent_ipc::UiEvent>,
}

/// The policy's answer when the model finished a round without tool calls.
pub enum Verdict {
    /// The turn may end. `before_final` effects land just ahead of the
    /// assistant's final message, `after_final` ones right behind it.
    Finish {
        before_final: Vec<Effect>,
        after_final: Vec<Effect>,
    },
    /// The turn must continue. The engine records the quiet round's text, then
    /// pushes `hidden` as the nudge the model sees next.
    Nudge { hidden: String },
}

impl Verdict {
    /// A plain finish — nothing extra to land.
    pub fn finish() -> Self {
        Verdict::Finish {
            before_final: Vec::new(),
            after_final: Vec::new(),
        }
    }
}

/// What a policy may read about the turn at its quiet point, without holding
/// the engine. Mode ids are the host's ([`crate::engine::ModeEntry`]); the
/// policy compares them against the ids its own contract tools were built for.
pub trait TurnView: Send + Sync {
    /// The mode currently in the shared slot.
    fn mode(&self) -> String;
    /// Whether this turn ran at least one tool call.
    fn did_tool_work(&self) -> bool;
}

/// Product rules applied to a turn that went quiet.
#[async_trait::async_trait]
pub trait TurnPolicy: Send + Sync {
    /// Called when a turn starts. A policy keeps per-turn counters and
    /// snapshots behind interior mutability; this is where it clears them —
    /// including any lifecycle controllers it holds, so a leftover
    /// `goal_complete`/`submit_plan` from an earlier turn cannot short-circuit
    /// this one (the contracts are per-turn).
    fn begin_turn(&self);

    /// Called for each tool call that cleared permission and is about to run.
    /// `is_readonly` lets a policy snapshot state lazily before the first
    /// write. Returns `Some(message)` to refuse the call: the engine reports the
    /// message as the call's failed result and does not dispatch.
    async fn before_dispatch(&self, tool: &str, is_readonly: bool) -> Option<String>;

    /// Decide whether a quiet round may end the turn. What the turn changed is
    /// the policy's own knowledge: it observes its tools' effects itself.
    async fn on_quiet_round(&self, view: &dyn TurnView) -> Verdict;
}

/// The default policy: every quiet round finishes, nothing is refused.
///
/// A second product assembled straight on the kernel gets this; a coding
/// session installs the coding product's policy (`agent-coding`'s
/// `CodingPolicy`) at spawn instead.
pub struct NoopPolicy;

#[async_trait::async_trait]
impl TurnPolicy for NoopPolicy {
    fn begin_turn(&self) {}
    async fn before_dispatch(&self, _tool: &str, _is_readonly: bool) -> Option<String> {
        None
    }
    async fn on_quiet_round(&self, _view: &dyn TurnView) -> Verdict {
        Verdict::finish()
    }
}
