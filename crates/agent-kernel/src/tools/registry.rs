//! The kernel's tool context and registry.
//!
//! The tool contract itself (specs, classes, results, errors, the deferred
//! write) lives in the `agent-tool` crate; this module instantiates it for
//! [`ToolCtx`], owns the context, and dispatches by name. Plugin tools merge in
//! under `plugin_id:name` (Stage 9).

use std::any::Any;
use std::collections::BTreeMap;
use std::sync::Arc;

// The tool contract lives in `agent-tool`. Re-exported so existing
// `crate::tools::registry::X` paths keep working, with the two context-generic
// items instantiated for this kernel's `ToolCtx`.
pub use agent_tool::{
    arg_str, arg_usize, schema_for, Args, Commit, Extensions, Review, ToolClass, ToolContext,
    ToolError, ToolResult,
};
/// A tool's exec, instantiated for [`ToolCtx`].
pub type ExecFn = agent_tool::ExecFn<ToolCtx>;
/// The kernel's tool spec, instantiated for [`ToolCtx`].
pub type ToolSpec = agent_tool::ToolSpec<ToolCtx>;

// Host-owned capability slots live in `host_slots`; re-exported so existing
// `crate::tools::registry::X` paths keep working.
pub use crate::tools::host_slots::*;

/// Shared context handed to every tool invocation.

#[derive(Default)]
pub struct ToolCtx {
    /// Parent turn's cooperative cancel flag — a delegated subagent polls
    /// it so a user cancel propagates into the delegation instead of
    /// orphaning a running child. `None` in unattached contexts.
    pub cancel: Option<Arc<std::sync::atomic::AtomicBool>>,
    /// Live view of the engine's active registry — refreshed on every
    /// `active_registry()` resolution so `batch_execute` dispatches through
    /// the same mode-scoped registry as single calls (plan mode can't be
    /// bypassed by batching a captured full registry).
    pub active_registry: std::sync::RwLock<Option<Arc<ToolRegistry>>>,
    /// Delegation depth — `delegate` bumps it in the child's ctx and the
    /// tool refuses past MAX_SUBAGENT_DEPTH, a second guard behind the
    /// registry-level exclusion of `delegate`.
    pub depth: u8,
    /// Typed host services — the goal and plan controllers live here, not as
    /// fields, so this crate does not name the modules that own them. See
    /// [`Extensions`]; the `GoalController`/`PlanController` slots are
    /// installed by the product crate (`agent-coding`) at assembly time, per
    /// root context and per delegated child.
    pub ext: Extensions,
    /// The todo list used when the host installed no `TodoStorageSlot`; owned
    /// by the context so it is freed with it. Scoped copies share it, children
    /// get their own.
    pub todo_fallback: Arc<crate::tools::todo::MemoryTodos>,
}

impl ToolCtx {
    pub fn new() -> Self {
        Self::default()
    }

    /// A required host service. A context built without it refuses the tool
    /// with a reason instead of panicking mid-turn.
    pub fn require<T: Any + Send + Sync>(&self, what: &str) -> Result<Arc<T>, ToolError> {
        self.ext
            .get::<T>()
            .ok_or_else(|| ToolError::Failed(format!("{what} is not available in this context")))
    }

    /// Same context with a different cancel flag — `batch_execute` scopes
    /// items to a batch deadline flag so a timeout can abort in-flight
    /// blocking work without touching the turn's cancel.
    pub fn with_cancel(&self, cancel: Arc<std::sync::atomic::AtomicBool>) -> ToolCtx {
        ToolCtx {
            cancel: Some(cancel),
            active_registry: std::sync::RwLock::new(self.active_registry.read().unwrap().clone()),
            depth: self.depth,
            ext: self.ext.clone(),
            todo_fallback: self.todo_fallback.clone(),
        }
    }
}

/// Capability adapters access the context through this product-free contract.
impl agent_tool::ToolContext for ToolCtx {
    fn extensions(&self) -> &Extensions {
        &self.ext
    }

    fn extensions_mut(&mut self) -> &mut Extensions {
        &mut self.ext
    }

    fn cancel_flag(&self) -> Option<Arc<std::sync::atomic::AtomicBool>> {
        self.cancel.clone()
    }
}

/// Name → spec map. `BTreeMap` keeps the `tools` array deterministic
/// (provider prompt-cache friendly).
#[derive(Default, Clone)]
pub struct ToolRegistry {
    specs: BTreeMap<&'static str, Arc<ToolSpec>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, spec: ToolSpec) {
        self.specs.insert(spec.name, Arc::new(spec));
    }

    /// A view of this registry keeping only tools that satisfy `keep` —
    /// agent modes build their registries from it (`plan` drops every
    /// non-readonly spec, `goal` adds the contract tools on top of full).
    pub fn filtered(&self, keep: impl Fn(&ToolSpec) -> bool) -> Self {
        let mut r = Self::new();
        for (name, spec) in &self.specs {
            if keep(spec) {
                r.specs.insert(name, spec.clone());
            }
        }
        r
    }

    /// Read-only view — every spec whose writes could touch the workspace
    /// drops out. `plan` mode dispatches against this.
    pub fn readonly_only(&self) -> Self {
        self.filtered(|s| s.readonly)
    }

    /// The `tools` array for an LLM request (OpenAI `function` shape).
    pub fn request_schema(&self) -> serde_json::Value {
        let tools: Vec<serde_json::Value> = self
            .specs
            .values()
            .map(|s| {
                serde_json::json!({
                    "type": "function",
                    "function": {
                        "name": s.name,
                        "description": s.schema["description"].clone(),
                        "parameters": s.schema["parameters"].clone(),
                    }
                })
            })
            .collect();
        serde_json::Value::Array(tools)
    }

    /// Dispatch by name. Unknown tool → model-facing error string so the
    /// model can self-correct instead of crashing the turn.
    pub async fn dispatch(
        &self,
        name: &str,
        args: Args,
        ctx: Arc<ToolCtx>,
    ) -> Result<ToolResult, ToolError> {
        match self.specs.get(name) {
            Some(spec) => (spec.exec)(args, ctx).await,
            None => Err(ToolError::Failed(format!(
                "unknown tool '{name}' — available: {}",
                self.specs.keys().copied().collect::<Vec<_>>().join(", ")
            ))),
        }
    }

    pub fn is_readonly(&self, name: &str) -> bool {
        self.specs.get(name).map(|s| s.readonly).unwrap_or(false)
    }

    /// Spec lookup — `batch_execute` filters eligible tools through this
    /// before running them itself.
    pub fn spec(&self, name: &str) -> Option<Arc<ToolSpec>> {
        self.specs.get(name).cloned()
    }

    pub fn len(&self) -> usize {
        self.specs.len()
    }
}

#[cfg(test)]
mod tests {
    #[derive(Debug, PartialEq)]
    struct Marker(u32);

    #[test]
    fn extension_round_trips_by_type_and_clones_share_it() {
        let mut ext = Extensions::default();
        assert!(ext.get::<Marker>().is_none());
        ext.insert(Arc::new(Marker(7)));
        assert_eq!(ext.get::<Marker>().map(|m| m.0), Some(7));
        // A clone is the same slot: the controller a ctx was built with is
        // the controller its `with_cancel` copy reads.
        let copy = ext.clone();
        assert!(Arc::ptr_eq(
            &ext.get::<Marker>().unwrap(),
            &copy.get::<Marker>().unwrap()
        ));
    }

    #[test]
    fn derived_ctx_shares_the_controller_but_a_child_gets_its_own() {
        // The contract controllers are no longer kernel types (the coding
        // crate owns them) — the scoping guarantee they exercise is generic:
        // a derived ctx shares the parent's slots, a fresh child does not.
        let mut parent = ToolCtx::new();
        parent.ext.insert(Arc::new(Marker(7)));
        let parent_goal = parent.require::<Marker>("contract").unwrap();
        // `with_cancel` is how batch and the turn scope a ctx: same slot.
        let scoped = parent.with_cancel(Arc::new(std::sync::atomic::AtomicBool::new(false)));
        let scoped_goal = scoped.require::<Marker>("contract").unwrap();
        assert!(Arc::ptr_eq(&parent_goal, &scoped_goal));
        // A subagent ctx is built fresh: its slots start empty, and whatever
        // the host installs into it is a different pair than the parent's.
        let mut child = ToolCtx::new();
        assert!(child.require::<Marker>("contract").is_err());
        child.ext.insert(Arc::new(Marker(9)));
        let child_goal = child.require::<Marker>("contract").unwrap();
        assert!(!Arc::ptr_eq(&parent_goal, &child_goal));
    }

    #[test]
    fn ctx_without_controllers_refuses_instead_of_panicking() {
        let bare = ToolCtx {
            ext: Extensions::default(),
            ..ToolCtx::new()
        };
        let err = bare
            .require::<Marker>("contract control")
            .err()
            .expect("missing service must be an error");
        assert!(err
            .to_string()
            .contains("contract control is not available"));
    }

    #[test]
    fn ask_is_refused_without_a_host_channel_and_shared_after_cancel_scope() {
        let mut ctx = ToolCtx::new();
        // Headless by default: the tool gets a refusal, not a parked oneshot.
        assert!(matches!(ctx.ask(), Err(ToolError::Failed(_))));

        let channel = Arc::new(AskChannel::new(None));
        ctx.install_ask(channel.clone());
        assert!(Arc::ptr_eq(&ctx.ask().unwrap(), &channel));
        // A batch item's cancel-scoped copy answers on the same channel.
        let scoped = ctx.with_cancel(Arc::new(std::sync::atomic::AtomicBool::new(false)));
        assert!(Arc::ptr_eq(&scoped.ask().unwrap(), &channel));
    }

    #[test]
    fn ui_sink_is_absent_until_installed_and_shared_after_cancel_scope() {
        let mut ctx = ToolCtx::new();
        // Headless by default: fan-out tools skip emitting instead of panicking.
        assert!(ctx.ui().is_none());

        let (sink, _rx) = crate::channels::UiSink::channel();
        ctx.install_ui(sink);
        assert!(ctx.ui().is_some());
        // A batch item's cancel-scoped copy emits on the same sink.
        let scoped = ctx.with_cancel(Arc::new(std::sync::atomic::AtomicBool::new(false)));
        assert!(scoped.ui().is_some());
    }

    #[test]
    fn subagent_is_absent_until_installed_and_shared_after_cancel_scope() {
        let mut ctx = ToolCtx::new();
        assert!(ctx.subagent().is_none());

        // Never called: the test only checks presence, not delegation.
        struct Idle;
        #[async_trait::async_trait]
        impl agent_llm_core::LlmProvider for Idle {
            fn id(&self) -> &'static str {
                "idle"
            }
            async fn chat_stream(
                &self,
                _model: &str,
                _messages: &[agent_llm_core::ChatMessage],
                _tools: Option<serde_json::Value>,
                _temperature: f32,
                _reasoning_effort: Option<&str>,
                _params: &agent_llm_core::ModelParams,
            ) -> anyhow::Result<agent_llm_core::BoxStream<agent_llm_core::StreamChunk>>
            {
                anyhow::bail!("idle provider")
            }
        }
        let registry = Arc::new(ToolRegistry::new());
        let route = Arc::new(std::sync::RwLock::new(crate::tools::delegate::ChildRoute {
            provider: Arc::new(Idle),
            model: String::new(),
            temperature: 0.0,
            context_window: 0,
            thinking_level: None,
            thinking_level_map: None,
            model_params: agent_llm_core::ModelParams::default(),
            model_input: Vec::new(),
            models: std::collections::HashMap::new(),
        }));
        ctx.install_subagent(crate::tools::delegate::SubagentSpawner::new(
            route,
            registry.clone(),
            registry,
        ));
        let scoped = ctx.with_cancel(Arc::new(std::sync::atomic::AtomicBool::new(false)));
        assert!(scoped.subagent().is_some());
    }

    #[test]
    fn session_is_absent_until_attached_and_shared_after_cancel_scope() {
        let ctx = ToolCtx::new();
        assert!(ctx.session().is_none());

        let ctx = ctx.with_session(7);
        assert_eq!(ctx.session(), Some(7));
        let scoped = ctx.with_cancel(Arc::new(std::sync::atomic::AtomicBool::new(false)));
        assert_eq!(scoped.session(), Some(7));
    }

    #[test]
    fn plugins_are_scoped_to_the_derived_context_not_the_parent() {
        let parent = ToolCtx::new();
        assert!(parent.plugins().is_none());

        let handle: crate::engine::PluginHandle = Arc::new(std::sync::RwLock::new(None));
        let routed = parent.with_plugins(handle.clone());
        // The engine swaps the whole ctx on set_plugins; the parent keeps none.
        assert!(parent.plugins().is_none());
        assert!(Arc::ptr_eq(&routed.plugins().unwrap(), &handle));
    }

    #[test]
    fn extension_keys_by_type_not_by_value() {
        struct Other;
        let mut ext = Extensions::default();
        ext.insert(Arc::new(Marker(1)));
        assert!(ext.get::<Other>().is_none());
    }

    use super::*;

    #[test]
    fn builtins_are_only_the_generic_conversation_tools() {
        let r = ToolRegistry::with_builtins();
        let schema = r.request_schema();
        let mut names: Vec<_> = schema
            .as_array()
            .unwrap()
            .iter()
            .map(|s| s["function"]["name"].as_str().unwrap())
            .collect();
        names.sort();
        assert_eq!(
            names,
            ["ask_question", "batch_execute", "delegate", "link_workspace", "todo"]
        );
    }

    #[test]
    fn unknown_tool_is_not_readonly() {
        // The turn-start baseline keys off `is_readonly`: a plugin/MCP tool
        // has no built-in spec, so it must count as a potential write.
        let reg = ToolRegistry::with_builtins();
        assert!(!reg.is_readonly("plugin__tool_that_does_not_exist"));
        assert!(reg.is_readonly("todo"));
    }

    /// A child may track tasks and batch calls, but cannot recurse or reach a human.
    #[test]
    fn child_builtins_exclude_recursion_and_human_tools() {
        let child = ToolRegistry::child_builtins();
        for banned in ["delegate", "ask_question"] {
            assert!(
                child.spec(banned).is_none(),
                "child registry must not hold `{banned}`"
            );
        }
        assert!(child.spec("todo").is_some());
        assert!(child.spec("batch_execute").is_some());
        // The parent still has the full set, so the exclusion is the child's alone.
        assert!(ToolRegistry::with_builtins().spec("delegate").is_some());
    }
}
