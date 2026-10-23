//! The tool contract: what a tool declares, what it returns, and the error it
//! raises. Nothing here knows about a product.
//!
//! [`ToolSpec`] is generic over the context type `C` so this crate does not
//! name the host's context; the kernel instantiates it with its own `ToolCtx`.
//! Classes, results, deferred writes, errors and the argument/schema helpers
//! are context-free.

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::Arc;

use futures::future::BoxFuture;

/// Arguments as delivered by the model — a JSON object.
pub type Args = serde_json::Value;

/// Typed extension slots for a tool context. A tool that needs a host-owned
/// service reads it by type, so the context itself carries no product fields
/// and this crate never has to name the types that provide them.
///
/// Keyed by `TypeId`: one value per type. Cloning shares the same `Arc`s, so a
/// `with_cancel` copy of the ctx sees the same controller as its parent.
#[derive(Clone, Default)]
pub struct Extensions {
    map: HashMap<TypeId, Arc<dyn Any + Send + Sync>>,
    /// Slots a delegated child context may share with its parent. Everything
    /// else (per-context controllers, observers) stays with the parent: a child
    /// never inherits `ext` wholesale.
    inheritable: std::collections::HashSet<TypeId>,
}

impl Extensions {
    pub fn insert<T: Any + Send + Sync>(&mut self, value: Arc<T>) {
        self.map.insert(TypeId::of::<T>(), value);
    }

    /// Insert a slot a child context should share with its parent (a sandbox,
    /// a desktop backend, a memory handle — capabilities of the *host*, not of
    /// one context). The child gets the same `Arc`.
    pub fn insert_shared<T: Any + Send + Sync>(&mut self, value: Arc<T>) {
        self.inheritable.insert(TypeId::of::<T>());
        self.insert(value);
    }

    pub fn get<T: Any + Send + Sync>(&self) -> Option<Arc<T>> {
        self.map
            .get(&TypeId::of::<T>())
            .and_then(|v| Arc::clone(v).downcast::<T>().ok())
    }

    /// Borrow a slot for an accessor whose result lives as long as the table.
    pub fn get_ref<T: Any + Send + Sync>(&self) -> Option<&T> {
        self.map
            .get(&TypeId::of::<T>())
            .and_then(|v| v.as_ref().downcast_ref::<T>())
    }

    /// The slots meant for a child: only those inserted with
    /// [`insert_shared`](Self::insert_shared), sharing the parent's `Arc`s.
    pub fn inheritable(&self) -> Extensions {
        let mut child = Extensions::default();
        for id in &self.inheritable {
            if let Some(v) = self.map.get(id) {
                child.map.insert(*id, v.clone());
                child.inheritable.insert(*id);
            }
        }
        child
    }
}

/// Generic host services and cancellation available to a tool.
///
/// The kernel's `ToolCtx` implements this; capability adapters name the trait
/// so they can be written, tested and shipped without depending on the kernel.
pub trait ToolContext: Send + Sync + 'static {
    fn extensions(&self) -> &Extensions;
    fn extensions_mut(&mut self) -> &mut Extensions;

    /// The parent turn's cooperative cancel flag, if one is installed.
    fn cancel_flag(&self) -> Option<Arc<std::sync::atomic::AtomicBool>> {
        None
    }

    /// A typed host service. Capability adapters use this table without the
    /// contract naming any backend or product type.
    fn service<T: Any + Send + Sync>(&self) -> Option<Arc<T>>
    where
        Self: Sized,
    {
        self.extensions().get::<T>()
    }
}

/// A deferred side effect a tool wants committed *after* the permission
/// decision. The tool does its analysis in `exec` (and may be dry-run before
/// approval, see [`ToolSpec::staged`]); the engine runs `verify` then `apply`
/// only once the call is approved. It never looks inside — what the effect is
/// (a file write, a remote command, a ticket) is the tool's business.
pub struct Commit {
    /// Re-check, right before applying, that what the preview was computed
    /// against still holds. `Err(reason)` abandons the whole commit and the
    /// reason becomes the call's failed result — all-or-nothing.
    pub verify: Box<dyn FnOnce() -> Result<(), String> + Send>,
    /// Perform the effect. `Err(reason)` fails the call with that text.
    pub apply: BoxFuture<'static, Result<(), String>>,
}

impl std::fmt::Debug for Commit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Commit")
    }
}

impl Commit {
    /// A commit with nothing to re-verify.
    pub fn new(
        apply: impl std::future::Future<Output = Result<(), String>> + Send + 'static,
    ) -> Self {
        Self {
            verify: Box::new(|| Ok(())),
            apply: Box::pin(apply),
        }
    }

    pub fn verified(
        mut self,
        verify: impl FnOnce() -> Result<(), String> + Send + 'static,
    ) -> Self {
        self.verify = Box::new(verify);
        self
    }
}

/// Tool execution result — `content` is model-facing text (already folded
/// under the truncation budget), `ui` is an optional typed card for the
/// DiffViewer/status strip.
#[derive(Debug)]
pub struct ToolResult {
    pub content: String,
    /// Typed UI card hint (e.g. `"diff"`, `"table"`, `"markdown"`), rendered
    /// as a rich card when present — plugins obey the same contract.
    pub ui_type: Option<&'static str>,
    /// True when the patch matched approximately — approval card must say so.
    pub fuzzy: bool,
    /// The deferred side effect the engine commits post-approval — `None` for
    /// tools that act inside `exec` or have nothing to commit.
    pub commit: Option<Commit>,
    /// Staged image refs this result carries as model input (a screenshot).
    /// The engine hangs them on the tool message; adapters encode or annotate
    /// them per their wire (only the vision-capable ones get real parts).
    pub images: Vec<agent_llm_core::types::ImageRef>,
}

impl ToolResult {
    pub fn text(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            ui_type: None,
            fuzzy: false,
            commit: None,
            images: Vec::new(),
        }
    }
}

/// Errors tools can raise. Message text is model-facing — ambiguity errors
/// must *teach the fix* ("add context lines"), per native-tools.md.
#[derive(Debug, thiserror::Error)]
pub enum ToolError {
    #[error("{0}")]
    Args(String),

    #[error("path escapes workspace: {0}")]
    PathEscape(String),

    /// The path is outside every root this context may reach — and the host
    /// can grant more. Unlike `PathEscape` this is a *question*, not a wall:
    /// the engine turns it into an approval whose actions are the grants the
    /// host offers, so one click makes the same call work. `suggestion` is the
    /// project root worth granting (the tool's best guess, may be absent).
    #[error("path escapes workspace: {path}")]
    OutOfScope {
        path: String,
        suggestion: Option<String>,
    },

    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    #[error("{0}")]
    Failed(String),
}

/// A tool's exec — `Arc<dyn Fn>` so a bridged tool (e.g. a Serena MCP
/// passthrough) can capture its bridge + remote name in a closure, while
/// plain builtins still wrap a free fn.
pub type ExecFn<C> =
    Arc<dyn Fn(Args, Arc<C>) -> BoxFuture<'static, Result<ToolResult, ToolError>> + Send + Sync>;

/// What a call actually does — the capability seam `batch_execute`,
/// permission policy, and (eventually) audit all key off. `readonly`
/// stays as the auto-approval hint; `class` is the execution taxonomy —
/// a tool can't sneak into an observation batch just by being readonly
/// (ask_question, goal signals, todo all prove the flags differ).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolClass {
    /// Pure reads — filesystem, network fetch, git status. Batch-eligible.
    Observation,
    /// Internal session state (`todo`) or instructions loaded into it
    /// (`skill`) — serial semantics, never batched.
    SessionMutation,
    /// Writes to the workspace (patches, file edits) — permission-gated.
    WorkspaceMutation,
    /// Process execution (bash, test runner).
    Process,
    /// Turn-control signals (goal contract) — the engine consumes them.
    Control,
    /// Blocks on a human decision (ask_question).
    HumanInteraction,
    /// Spawns orchestration (delegate, batch_execute).
    Orchestration,
}

/// A tool's assessment of one call, beyond its static `readonly` flag.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum CallRisk {
    /// Use the tool's static metadata and the permission mode.
    #[default]
    Normal,
    /// This call only observes, even if other calls of the same tool mutate.
    Readonly,
    /// Ask above the mode default unless the tool is explicitly allowed.
    RequiresApproval,
}

/// What a tool says about one call *before* it is approved — everything the
/// engine and the permission gate need to treat that call properly without
/// knowing the tool by name.
#[derive(Default)]
pub struct Review {
    pub risk: CallRisk,
    /// One line saying exactly what this call will do — the text the approval
    /// card shows, and the call's primary locator in the UI. `None` = the
    /// engine derives a generic one from the arguments.
    pub summary: Option<String>,
    /// The call's sub-action, when the tool has actions the gate treats
    /// differently (a desktop tool's `launch` vs a `click`). Opaque to the
    /// engine; handed to the gate's per-tool policy.
    pub action: Option<String>,
    /// An async step to run before the user is asked — a grant the platform
    /// must give first, so its prompt comes ahead of the app's card. `Err`
    /// ends the call (there is no point asking about an action that cannot
    /// run); the text becomes the failed result.
    pub prepare: Option<BoxFuture<'static, Result<(), String>>>,
}

/// How a tool reviews a call: arguments and the context in, a [`Review`] out.
/// Synchronous and side-effect free — it only describes.
pub type ReviewFn<C> = Arc<dyn Fn(&Args, &Arc<C>) -> Review + Send + Sync>;

/// One tool: name, schema for the LLM, readonly flag, exec fn.
#[derive(Clone)]
pub struct ToolSpec<C> {
    pub name: &'static str,
    /// JSON Schema object describing the tool's parameters.
    pub schema: serde_json::Value,
    /// Read-only tools auto-run in `default` permission mode.
    pub readonly: bool,
    /// Execution class — batch eligibility, policy, and audit key off it.
    pub class: ToolClass,
    /// Crosses the network boundary — batchable only under a tighter
    /// per-batch cap (SSRF/rate-limit amplification guard).
    pub network: bool,
    /// The tool stages its effect instead of performing it: `exec` touches
    /// nothing and returns a [`ToolResult::commit`], so the engine may dispatch
    /// it *before* the permission decision to harvest the real preview, then
    /// commit only after approval. Dispatching any other tool early would run
    /// it for real, so this is opt-in and deliberately not derived from
    /// [`ToolClass`] (a `WorkspaceMutation` tool may write directly).
    pub staged: bool,
    /// How this tool describes a call before approval. `None` for the common
    /// tool: the engine's generic summary applies.
    pub review: Option<ReviewFn<C>>,
    /// The tool observes the caller's cancel flag itself and ends with its own
    /// terminal text. The engine then awaits it instead of racing it against
    /// the flag, so the result the tool wrote is the one recorded — a racing
    /// abort would replace it with a generic line and drop what the tool knew
    /// (which child it stopped, for `delegate`).
    pub cancel_aware: bool,
    pub exec: ExecFn<C>,
}

/// Helper: pull a required string arg with a teaching error.
pub fn arg_str<'a>(args: &'a Args, key: &str) -> Result<&'a str, ToolError> {
    args.get(key)
        .and_then(|v| v.as_str())
        .ok_or_else(|| ToolError::Args(format!("missing required string arg '{key}'")))
}

/// Helper: pull an optional usize arg.
pub fn arg_usize(args: &Args, key: &str) -> Option<usize> {
    args.get(key).and_then(|v| v.as_u64()).map(|v| v as usize)
}

/// Derive a JSON Schema `parameters` object from a serde shape via schemars,
/// and attach the tool description separately.
pub fn schema_for<T: schemars::JsonSchema>(description: &str) -> serde_json::Value {
    let params = schemars::schema_for!(T);
    let params = serde_json::to_value(params).unwrap_or_default();
    serde_json::json!({ "description": description, "parameters": params })
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Marker(u32);

    /// The extension table is the contract's own storage; it must work with no
    /// kernel, no context and no capability types in sight.
    #[test]
    fn a_child_inherits_only_the_shared_slots() {
        struct Shared(u32);
        let mut parent = Extensions::default();
        parent.insert_shared(Arc::new(Shared(1)));
        parent.insert(Arc::new(Marker(2)));
        let child = parent.inheritable();
        assert_eq!(child.get::<Shared>().map(|s| s.0), Some(1));
        assert!(
            child.get::<Marker>().is_none(),
            "per-context slots stay put"
        );
        // Shared means the same Arc, so a handle swap on the parent is seen.
        assert!(Arc::ptr_eq(
            &parent.get::<Shared>().unwrap(),
            &child.get::<Shared>().unwrap()
        ));
        // The child can pass them on to its own children.
        assert!(child.inheritable().get::<Shared>().is_some());
    }

    #[test]
    fn extensions_round_trip_without_a_context() {
        let mut ext = Extensions::default();
        assert!(ext.get::<Marker>().is_none());
        ext.insert(Arc::new(Marker(7)));
        assert_eq!(ext.get::<Marker>().map(|m| m.0), Some(7));
    }

    /// A spec is generic over the context, so a product that has no kernel
    /// `ToolCtx` can still declare tools.
    #[test]
    fn a_spec_can_be_built_for_any_context() {
        struct Host;
        let spec: ToolSpec<Host> = ToolSpec {
            name: "demo",
            schema: serde_json::json!({"type": "object"}),
            readonly: true,
            class: ToolClass::Observation,
            network: false,
            staged: false,
            review: None,
            cancel_aware: false,
            exec: Arc::new(|_args: Args, _ctx: Arc<Host>| {
                Box::pin(async move { Ok(ToolResult::text("ok")) })
            }),
        };
        assert_eq!(spec.name, "demo");
        assert_eq!(spec.class, ToolClass::Observation);
        assert!(spec.readonly);
    }
}
