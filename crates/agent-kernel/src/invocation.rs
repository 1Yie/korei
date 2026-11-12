//! Turn-borrowed discovery, authorization, execution and audit boundary.

use std::collections::HashSet;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use agent_ipc::events::{AgentState, UiEvent};
use agent_llm_core::types::{ImageRef, ToolCall};
use agent_tool::{CallRisk, ToolError, ToolResult};
use async_trait::async_trait;
use futures::{future::BoxFuture, stream::FuturesUnordered, FutureExt, StreamExt};
use serde_json::Value;
use tokio::sync::Mutex as AsyncMutex;

use super::{args_preview_of, next_request_id, DecisionWait, Engine, EngineIo};
use crate::permissions::Decision;
use crate::plugins::PluginRouter;
use crate::tools::registry::{ToolClass, ToolSpec};

/// Per-operation audit I/O budget, including independent terminal cleanup.
pub const AUDIT_IO_BUDGET: Duration = Duration::from_secs(5);
const MAX_AUDIT_RESULT_BYTES: usize = 16 * 1024;
const MAX_METADATA_BYTES: usize = 1024;

struct TerminalGuard {
    ui: crate::channels::UiSink,
    call: ToolCall,
    parent: Option<String>,
    finished: bool,
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        if !self.finished {
            let _ = self.ui.send(UiEvent::ToolCallFinished {
                call_id: self.call.id.clone(),
                name: self.call.name.clone(),
                ok: false,
                content: "invocation aborted".into(),
                ui_type: None,
                parent: self.parent.clone(),
                child_id: None,
            });
        }
    }
}

/// Shared by every invocation within one turn; create a fresh value for each turn.
#[derive(Default)]
pub struct InvocationTurn {
    approval: AsyncMutex<()>,
    policy: AsyncMutex<()>,
    mutation: AsyncMutex<()>,
    timed_out: Mutex<HashSet<(String, String)>>,
}

#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct InvocationMetadata {
    pub invocation_id: String,
    pub parent_call_id: Option<String>,
    pub execution_id: Option<String>,
    pub agent_id: Option<String>,
    pub session_id: Option<String>,
    pub trace_id: Option<String>,
}

#[derive(Clone, Debug)]
pub struct InvocationRequest {
    pub call: ToolCall,
    pub metadata: InvocationMetadata,
    pub observation_only: bool,
    pub rejection: Option<String>,
}

impl InvocationRequest {
    pub fn new(call: ToolCall) -> Self {
        Self {
            metadata: InvocationMetadata {
                invocation_id: call.id.clone(),
                ..Default::default()
            },
            call,
            observation_only: false,
            rejection: None,
        }
    }
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct InvocationStart {
    pub metadata: InvocationMetadata,
    pub requested_call: ToolCall,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub enum InvocationStatus {
    Succeeded,
    Failed,
    Denied,
    Cancelled,
    TimedOut,
}

/// Decision provenance for hosts. Only `UserRejected` comes from a negative
/// verdict for the identified human approval request.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub enum InvocationStopReason {
    UserRejected { request_id: u64 },
    PolicyDenied,
    ApprovalTimedOut { request_id: Option<u64> },
    InvocationDeadline,
    Cancelled,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct InvocationAudit {
    pub metadata: InvocationMetadata,
    pub reviewed_call: Option<ToolCall>,
    pub authorized_call: Option<ToolCall>,
    pub status: InvocationStatus,
    pub stop_reason: Option<InvocationStopReason>,
    pub effects_may_have_occurred: bool,
    pub result_preview: String,
    pub result_truncated: bool,
    pub result_actual_bytes: usize,
    pub approval_request_id: Option<u64>,
    pub scope_approval_request_id: Option<u64>,
    pub duration_ms: i64,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub enum InvocationPhase {
    Prepare,
    StagedPreview,
    Dispatch,
    ScopeGrant {
        action: String,
        path: String,
        access: String,
    },
    Commit,
}

/// Acknowledged before entering the named operation. Prepare and staged
/// preview have no tool-write authorization; scope approval authorizes only
/// the directory grant. Acknowledgement is an intent, not proof of completion.
#[derive(Clone, Debug, serde::Serialize)]
pub struct InvocationEffect {
    pub metadata: InvocationMetadata,
    pub reviewed_call: ToolCall,
    pub authorized_call: Option<ToolCall>,
    pub phase: InvocationPhase,
    pub approval_request_id: Option<u64>,
    pub scope_approval_request_id: Option<u64>,
}

/// Hosts requiring durable audit must acknowledge durable records here.
/// `None` on the gateway preserves ReAct's event-only behavior.
/// Begin/pre-effect observe cancellation and the invocation deadline, with a
/// five-second I/O cap. Finish has its own five-second cleanup grace, even
/// after cancellation. Timed-out futures are dropped, never spawned.
/// Finish is attempted once after an acknowledged begin. I/O failure leaves
/// durable write completion uncertain; `audit_error` never implies rollback.
#[async_trait]
pub trait AuditSink: Send + Sync {
    async fn begin(&self, record: &InvocationStart) -> Result<(), String>;
    async fn pre_effect(&self, record: &InvocationEffect) -> Result<(), String>;
    async fn finish(&self, record: &InvocationAudit) -> Result<(), String>;
}

pub struct InvocationOutcome {
    pub status: InvocationStatus,
    pub content: String,
    pub ui_type: Option<String>,
    pub images: Vec<ImageRef>,
    pub audit: InvocationAudit,
    /// Tool status and effects remain available when durable finish fails.
    pub audit_error: Option<String>,
    pub attempted: bool,
    pub dispatched: bool,
    /// ReAct turn-suppression hint, including review refusal and approval
    /// timeout. Human decision provenance is in `audit.stop_reason`.
    pub denied: bool,
}

impl InvocationOutcome {
    pub fn ok(&self) -> bool {
        self.status == InvocationStatus::Succeeded && self.audit_error.is_none()
    }
}

#[derive(Clone, Debug)]
pub struct ToolMetadata {
    pub name: String,
    pub description: String,
    pub readonly: bool,
    pub staged: bool,
}

#[derive(Clone, Debug)]
pub struct ToolDescription {
    pub metadata: ToolMetadata,
    pub parameters: Value,
}

enum ResolvedTool {
    Builtin(Arc<ToolSpec>),
    Plugin {
        router: Arc<dyn PluginRouter>,
        description: ToolDescription,
    },
}

impl ResolvedTool {
    fn readonly(&self) -> bool {
        match self {
            Self::Builtin(spec) => spec.readonly,
            Self::Plugin { .. } => false,
        }
    }

    fn staged(&self) -> bool {
        matches!(self, Self::Builtin(spec) if spec.staged)
    }

    fn class(&self) -> Option<ToolClass> {
        match self {
            Self::Builtin(spec) => Some(spec.class),
            Self::Plugin { .. } => None,
        }
    }

    fn schema(&self) -> &Value {
        match self {
            Self::Builtin(spec) => spec.schema.get("parameters").unwrap_or(&spec.schema),
            Self::Plugin { description, .. } => &description.parameters,
        }
    }
}

/// Immutable engine configuration is borrowed, never forced into `'static`.
/// Mutable turn phases are shared by all handles borrowing the same turn.
pub struct InvocationGateway<'turn> {
    engine: &'turn Engine,
    io: &'turn EngineIo,
    turn: &'turn InvocationTurn,
    audit: Option<&'turn dyn AuditSink>,
    parent: Option<String>,
    deadline: Option<Instant>,
}

impl Engine {
    /// Start a tool turn after the previous turn's borrowed gateways are released.
    pub fn begin_invocation_turn(&mut self) -> InvocationTurn {
        self.policy.begin_turn();
        InvocationTurn::default()
    }

    pub fn invocation_gateway<'turn>(
        &'turn self,
        io: &'turn EngineIo,
        turn: &'turn InvocationTurn,
        audit: Option<&'turn dyn AuditSink>,
    ) -> InvocationGateway<'turn> {
        InvocationGateway {
            engine: self,
            io,
            turn,
            audit,
            parent: None,
            deadline: None,
        }
    }
}

fn clipped(text: &str, max: usize) -> String {
    if text.len() <= max {
        return text.to_string();
    }
    let mut end = max.saturating_sub(32);
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}\n[output truncated]", &text[..end])
}

fn plugin_description(value: Value) -> Result<ToolDescription, String> {
    let function = value.get("function").unwrap_or(&value);
    let name = function
        .get("name")
        .and_then(Value::as_str)
        .filter(|name| !name.is_empty() && name.len() <= MAX_METADATA_BYTES)
        .ok_or("plugin tool has no valid name")?;
    let parameters = function
        .get("parameters")
        .cloned()
        .ok_or_else(|| format!("plugin tool `{name}` has no schema"))?;
    Ok(ToolDescription {
        metadata: ToolMetadata {
            name: name.into(),
            description: clipped(
                function
                    .get("description")
                    .and_then(Value::as_str)
                    .unwrap_or(""),
                MAX_METADATA_BYTES,
            ),
            readonly: false,
            staged: false,
        },
        parameters,
    })
}

impl InvocationGateway<'_> {
    pub fn with_deadline(mut self, deadline: Instant) -> Self {
        self.deadline = Some(
            self.deadline
                .map_or(deadline, |current| current.min(deadline)),
        );
        self
    }
    /// Compact metadata only; schemas are obtained one at a time with `describe`.
    pub fn list(&self) -> Result<Vec<ToolMetadata>, String> {
        let registry = self.engine.active_registry();
        let mut tools = Vec::new();
        let mut names = HashSet::new();
        for spec in registry.specs() {
            let name = spec.name;
            names.insert(name.to_string());
            tools.push(ToolMetadata {
                name: name.to_string(),
                description: clipped(
                    spec.schema["description"].as_str().unwrap_or(""),
                    MAX_METADATA_BYTES,
                ),
                readonly: spec.readonly,
                staged: spec.staged,
            });
        }
        if self.engine.plugins_allowed() {
            if let Some(router) = self.engine.plugins() {
                for value in router.exported_tools() {
                    let description = plugin_description(value)?;
                    if names.insert(description.metadata.name.clone()) {
                        tools.push(description.metadata);
                    }
                }
            }
        }
        Ok(tools)
    }

    pub fn describe(&self, name: &str) -> Result<ToolDescription, String> {
        let registry = self.engine.active_registry();
        let resolved = self.resolve(&registry, name)?;
        crate::schema::validator(resolved.schema())?;
        Ok(match resolved {
            ResolvedTool::Builtin(spec) => ToolDescription {
                metadata: ToolMetadata {
                    name: spec.name.to_string(),
                    description: clipped(
                        spec.schema["description"].as_str().unwrap_or(""),
                        MAX_METADATA_BYTES,
                    ),
                    readonly: spec.readonly,
                    staged: spec.staged,
                },
                parameters: spec
                    .schema
                    .get("parameters")
                    .unwrap_or(&spec.schema)
                    .clone(),
            },
            ResolvedTool::Plugin { description, .. } => description,
        })
    }

    fn resolve(
        &self,
        registry: &crate::tools::registry::ToolRegistry,
        name: &str,
    ) -> Result<ResolvedTool, String> {
        if let Some(spec) = registry.spec(name) {
            return Ok(ResolvedTool::Builtin(spec));
        }
        if self.engine.plugins_allowed() {
            if let Some(router) = self.engine.plugins() {
                for value in router.exported_tools() {
                    let description = plugin_description(value)?;
                    if description.metadata.name == name {
                        return Ok(ResolvedTool::Plugin {
                            router,
                            description,
                        });
                    }
                }
            }
        }
        Err(format!(
            "unknown tool '{name}' — available: {}",
            registry
                .specs()
                .map(|spec| spec.name)
                .collect::<Vec<_>>()
                .join(", ")
        ))
    }

    fn ensure_current(&self, name: &str, resolved: &ResolvedTool) -> Result<(), String> {
        let registry = self.engine.active_registry();
        let current = self.resolve(&registry, name)?;
        let unchanged = match (resolved, &current) {
            (ResolvedTool::Builtin(before), ResolvedTool::Builtin(after)) => {
                Arc::ptr_eq(before, after)
            }
            (
                ResolvedTool::Plugin {
                    router: before,
                    description: before_description,
                },
                ResolvedTool::Plugin {
                    router: after,
                    description: after_description,
                },
            ) => {
                Arc::ptr_eq(before, after)
                    && before_description.parameters == after_description.parameters
            }
            _ => false,
        };
        if unchanged {
            Ok(())
        } else {
            Err(format!("tool registration changed before dispatch: {name}"))
        }
    }

    fn permission_decision(
        &self,
        resolved: &ResolvedTool,
        call: &ToolCall,
        risk: CallRisk,
        diff: &str,
        action: Option<&str>,
    ) -> Decision {
        let gate = self
            .engine
            .permissions
            .read()
            .unwrap_or_else(|e| e.into_inner());
        let decision = gate.decide(
            &call.name,
            resolved.class(),
            resolved.readonly(),
            risk,
            diff,
        );
        let decision = match action {
            Some(action) => gate.escalate(&call.name, action, decision, diff),
            None => decision,
        };
        match decision {
            Decision::Ask { .. } if !gate.can_ask() => Decision::Deny {
                reason: format!(
                    "`{}` requires human approval — unavailable in this context",
                    call.name
                ),
            },
            decision => decision,
        }
    }

    pub fn invoke(&self, request: InvocationRequest) -> BoxFuture<'_, InvocationOutcome> {
        async move {
            let args =
                crate::schema::parse_arguments(&request.call.arguments).unwrap_or(Value::Null);
            let _ = self.io.ui_tx.send(UiEvent::ToolCallQueued {
                name: request.call.name.clone(),
                args_preview: args_preview_of(&args),
                call_id: request.call.id.clone(),
                parent: self.parent.clone(),
                child_id: None,
            });
            self.invoke_queued(request).await
        }
        .boxed()
    }

    pub(super) fn invoke_queued(
        &self,
        request: InvocationRequest,
    ) -> BoxFuture<'_, InvocationOutcome> {
        async move {
            let mut terminal = TerminalGuard {
                ui: self.io.ui_tx.clone(),
                call: request.call.clone(),
                parent: self.parent.clone(),
                finished: false,
            };
            let started = Instant::now();
            let mut outcome = InvocationOutcome {
                status: InvocationStatus::Failed,
                content: String::new(),
                ui_type: None,
                images: Vec::new(),
                audit: InvocationAudit {
                    metadata: InvocationMetadata {
                        invocation_id: clipped(&request.metadata.invocation_id, MAX_METADATA_BYTES),
                        parent_call_id: request
                            .metadata
                            .parent_call_id
                            .as_deref()
                            .map(|s| clipped(s, MAX_METADATA_BYTES)),
                        execution_id: request
                            .metadata
                            .execution_id
                            .as_deref()
                            .map(|s| clipped(s, MAX_METADATA_BYTES)),
                        agent_id: request
                            .metadata
                            .agent_id
                            .as_deref()
                            .map(|s| clipped(s, MAX_METADATA_BYTES)),
                        session_id: request
                            .metadata
                            .session_id
                            .as_deref()
                            .map(|s| clipped(s, MAX_METADATA_BYTES)),
                        trace_id: request
                            .metadata
                            .trace_id
                            .as_deref()
                            .map(|s| clipped(s, MAX_METADATA_BYTES)),
                    },
                    reviewed_call: None,
                    authorized_call: None,
                    status: InvocationStatus::Failed,
                    stop_reason: None,
                    effects_may_have_occurred: false,
                    result_preview: String::new(),
                    result_truncated: false,
                    result_actual_bytes: 0,
                    approval_request_id: None,
                    scope_approval_request_id: None,
                    duration_ms: 0,
                },
                audit_error: None,
                attempted: false,
                dispatched: false,
                denied: false,
            };
            let metadata_ok = !request.metadata.invocation_id.is_empty()
                && [
                    Some(&request.metadata.invocation_id),
                    request.metadata.parent_call_id.as_ref(),
                    request.metadata.execution_id.as_ref(),
                    request.metadata.agent_id.as_ref(),
                    request.metadata.session_id.as_ref(),
                    request.metadata.trace_id.as_ref(),
                    Some(&request.call.id),
                    Some(&request.call.name),
                ]
                .into_iter()
                .flatten()
                .all(|value| value.len() <= MAX_METADATA_BYTES);
            let mut began = false;
            if !metadata_ok || request.call.arguments.len() > crate::schema::MAX_ARGUMENT_BYTES {
                outcome.content = "invocation metadata or arguments exceed limits".into();
            } else if let Some(audit) = self.audit {
                let record = InvocationStart {
                    metadata: request.metadata.clone(),
                    requested_call: request.call.clone(),
                };
                match self.audit_before(audit.begin(&record), &mut outcome).await {
                    Ok(()) => began = true,
                    Err(error) => {
                        let error = clipped(&error, MAX_AUDIT_RESULT_BYTES);
                        outcome.content = format!("audit begin failed: {error}");
                        outcome.audit_error = Some(error);
                    }
                }
            }
            if outcome.content.is_empty() {
                if let Some(reason) = &request.rejection {
                    outcome.status = InvocationStatus::Denied;
                    outcome.ui_type = Some("rejected".into());
                    outcome.content = clipped(reason, MAX_AUDIT_RESULT_BYTES);
                } else if let Err(error) = self
                    .execute(&request.call, request.observation_only, &mut outcome)
                    .await
                {
                    outcome.content = error;
                }
            }
            // Cancellation between calls preserves an already completed result.
            if outcome.dispatched && !self.io.cancel.load(std::sync::atomic::Ordering::Relaxed) {
                if let Some(call) = outcome.audit.authorized_call.clone() {
                    let mut content = std::mem::take(&mut outcome.content);
                    let after = self.engine.hooks.run_after_tool(&call, &mut content);
                    // After hooks cannot hold terminal cleanup open.
                    let _ = tokio::time::timeout(AUDIT_IO_BUDGET, self.active(after, &mut outcome))
                        .await;
                    outcome.content = content;
                }
            }
            outcome.audit.duration_ms = started.elapsed().as_millis().min(i64::MAX as u128) as i64;
            outcome.audit.status = outcome.status.clone();
            if outcome.status == InvocationStatus::Denied && outcome.audit.stop_reason.is_none() {
                outcome.audit.stop_reason = Some(InvocationStopReason::PolicyDenied);
            }
            outcome.audit.result_truncated = outcome.content.len() > MAX_AUDIT_RESULT_BYTES;
            outcome.audit.result_actual_bytes = outcome.content.len();
            outcome.audit.result_preview = clipped(&outcome.content, MAX_AUDIT_RESULT_BYTES);
            if began {
                if let Some(audit) = self.audit {
                    let finish =
                        tokio::time::timeout(AUDIT_IO_BUDGET, audit.finish(&outcome.audit))
                            .await
                            .unwrap_or_else(|_| Err("audit finish I/O budget exceeded".into()));
                    if let Err(error) = finish {
                        let error = clipped(&error, MAX_AUDIT_RESULT_BYTES);
                        outcome.audit_error = Some(match outcome.audit_error.take() {
                            Some(before) => format!("{before}; {error}"),
                            None => error,
                        });
                    }
                }
            }
            let _ = self.io.ui_tx.send(UiEvent::ToolCallFinished {
                call_id: request.call.id.clone(),
                name: request.call.name.clone(),
                ok: outcome.ok(),
                content: if matches!(
                    outcome.audit.stop_reason,
                    Some(InvocationStopReason::ApprovalTimedOut { .. })
                ) {
                    super::APPROVAL_TIMEOUT_UI.to_string()
                } else {
                    clipped(&outcome.content, MAX_AUDIT_RESULT_BYTES)
                },
                ui_type: outcome.ui_type.clone(),
                parent: self.parent.clone(),
                child_id: None,
            });
            terminal.finished = true;
            outcome
        }
        .boxed()
    }

    fn check_cancel(&self, outcome: &mut InvocationOutcome) -> Result<(), String> {
        if self
            .deadline
            .is_some_and(|d| tokio::time::Instant::now().into_std() >= d)
        {
            outcome.status = InvocationStatus::TimedOut;
            outcome.audit.stop_reason = Some(InvocationStopReason::InvocationDeadline);
            return Err("invocation deadline exceeded".into());
        }
        if self.io.cancel.load(std::sync::atomic::Ordering::Relaxed) {
            outcome.status = InvocationStatus::Cancelled;
            outcome.audit.stop_reason = Some(InvocationStopReason::Cancelled);
            Err("cancelled while running".into())
        } else {
            Ok(())
        }
    }

    async fn active<F: std::future::Future>(
        &self,
        future: F,
        outcome: &mut InvocationOutcome,
    ) -> Result<F::Output, String> {
        self.check_cancel(outcome)?;
        tokio::pin!(future);
        loop {
            tokio::select! {
                biased;
                _ = async {
                    match self.deadline {
                        Some(d) => tokio::time::sleep_until(d.into()).await,
                        None => futures::future::pending().await,
                    }
                } => { self.check_cancel(outcome)?; }
                _ = tokio::time::sleep(Duration::from_millis(25)) => {
                    self.check_cancel(outcome)?;
                }
                value = &mut future => {
                    self.check_cancel(outcome)?;
                    return Ok(value);
                }
            }
        }
    }

    async fn audit_before<F: std::future::Future<Output = Result<(), String>>>(
        &self,
        future: F,
        outcome: &mut InvocationOutcome,
    ) -> Result<(), String> {
        let result = tokio::time::timeout(AUDIT_IO_BUDGET, self.active(future, outcome))
            .await
            .unwrap_or_else(|_| Err("audit I/O budget exceeded".into()))
            .and_then(|result| result);
        if let Err(error) = &result {
            outcome.audit_error = Some(clipped(error, MAX_AUDIT_RESULT_BYTES));
        }
        result
    }

    async fn pre_effect(
        &self,
        call: &ToolCall,
        phase: InvocationPhase,
        authorized: bool,
        outcome: &mut InvocationOutcome,
    ) -> Result<(), String> {
        self.check_cancel(outcome)?;
        if let Some(audit) = self.audit {
            let record = InvocationEffect {
                metadata: outcome.audit.metadata.clone(),
                reviewed_call: call.clone(),
                authorized_call: authorized.then(|| call.clone()),
                phase,
                approval_request_id: outcome.audit.approval_request_id,
                scope_approval_request_id: outcome.audit.scope_approval_request_id,
            };
            self.audit_before(audit.pre_effect(&record), outcome)
                .await?;
        }
        self.check_cancel(outcome)
    }

    async fn dispatch(
        &self,
        resolved: &ResolvedTool,
        name: &str,
        args: Value,
        outcome: &mut InvocationOutcome,
    ) -> Result<ToolResult, ToolError> {
        use crate::tools::dispatch::{
            NestedDispatcher, NestedRequest, NESTED_CONCURRENCY, NESTED_QUEUE,
        };
        self.check_cancel(outcome).map_err(ToolError::Failed)?;
        let (sender, mut receiver) = tokio::sync::mpsc::channel::<NestedRequest>(NESTED_QUEUE);
        let mut ctx = self.engine.ctx.with_cancel(self.io.cancel.clone());
        ctx.ext.insert(Arc::new(NestedDispatcher { sender }));
        let ctx = Arc::new(ctx);
        let future: BoxFuture<'_, Result<ToolResult, ToolError>> = match resolved {
            ResolvedTool::Builtin(spec) => (spec.exec)(args, ctx),
            ResolvedTool::Plugin { router, .. } => async move {
                router
                    .dispatch_tool_call(name, args)
                    .await
                    .unwrap_or_else(|| Err(format!("unknown tool: {name}")))
                    .map_err(ToolError::Failed)
            }
            .boxed(),
        };
        let cancel_aware = matches!(resolved, ResolvedTool::Builtin(spec) if spec.cancel_aware);
        let mut tool = Some(future);
        let mut result = None;
        let mut accepting = true;
        let mut pending = FuturesUnordered::<BoxFuture<'_, usize>>::new();
        let mut flags =
            std::collections::HashMap::<usize, Arc<std::sync::atomic::AtomicBool>>::new();
        let mut next_id = 0usize;
        let parent_metadata = outcome.audit.metadata.clone();
        let parent_call = outcome
            .audit
            .reviewed_call
            .as_ref()
            .map(|call| call.id.clone())
            .unwrap_or_else(|| parent_metadata.invocation_id.clone());
        loop {
            if self.check_cancel(outcome).is_err() {
                receiver.close();
                accepting = false;
                for flag in flags.values() {
                    flag.store(true, std::sync::atomic::Ordering::Relaxed);
                }
                if !cancel_aware && tool.is_some() {
                    tool = None;
                    result = Some(Err(ToolError::Failed(
                        if outcome.status == InvocationStatus::TimedOut {
                            "invocation deadline exceeded".into()
                        } else {
                            "cancelled while running".into()
                        },
                    )));
                }
            }
            if tool.is_none() && pending.is_empty() {
                return result.expect("completed tool");
            }
            tokio::select! {
                output = async { tool.as_mut().expect("active tool").await }, if tool.is_some() => {
                    tool = None;
                    result = Some(output);
                    accepting = false;
                    receiver.close();
                    for flag in flags.values() { flag.store(true, std::sync::atomic::Ordering::Relaxed); }
                }
                Some(request) = receiver.recv(), if accepting && pending.len() < NESTED_CONCURRENCY => {
                    if request.reply.is_closed() { continue; }
                    let key = next_id;
                    next_id += 1;
                    flags.insert(key, request.cancel.clone());
                    let mut metadata = parent_metadata.clone();
                    metadata.invocation_id = request.call.id.clone();
                    metadata.parent_call_id = Some(parent_call.clone());
                    if metadata.agent_id.is_none() { metadata.agent_id = self.engine.child_id().map(|id| id.to_string()); }
                    if metadata.session_id.is_none() { metadata.session_id = self.engine.ctx.session().map(|id| id.to_string()); }
                    let parent = name.to_string();
                    pending.push(async move {
                        let (_steer, steer_rx) = tokio::sync::mpsc::channel(1);
                        let (_notice, notice_rx) = tokio::sync::mpsc::channel(1);
                        let io = EngineIo { ui_tx: self.io.ui_tx.clone(), steer_rx, notice_rx, cancel: request.cancel };
                        let gateway = InvocationGateway {
                            engine: self.engine, io: &io, turn: self.turn, audit: self.audit,
                            parent: Some(parent),
                            deadline: Some(self.deadline.map_or(request.deadline, |d| d.min(request.deadline))),
                        };
                        let outcome = gateway.invoke(InvocationRequest {
                            call: request.call, metadata, observation_only: true, rejection: request.rejection,
                        }).await;
                        let _ = request.reply.send(outcome);
                        key
                    }.boxed());
                }
                Some(key) = pending.next(), if !pending.is_empty() => { flags.remove(&key); }
                _ = tokio::time::sleep(std::time::Duration::from_millis(25)) => {}
            }
        }
    }

    async fn dispatch_with_scope(
        &self,
        resolved: &ResolvedTool,
        call: &ToolCall,
        args: Value,
        risk: CallRisk,
        action: Option<&str>,
        diff: &str,
        approved: &mut bool,
        outcome: &mut InvocationOutcome,
    ) -> Result<ToolResult, String> {
        let phase = if resolved.staged() {
            InvocationPhase::StagedPreview
        } else {
            InvocationPhase::Dispatch
        };
        self.authorize_operation(
            resolved,
            call,
            &args,
            risk,
            diff,
            action,
            approved,
            phase.clone(),
            !resolved.staged(),
            outcome,
        )
        .await?;
        outcome.audit.effects_may_have_occurred |=
            !resolved.staged() && !(resolved.readonly() || risk == CallRisk::Readonly);
        match self
            .dispatch(resolved, &call.name, args.clone(), outcome)
            .await
        {
            Err(ToolError::OutOfScope { path, suggestion }) => {
                self.scope_allowed(resolved, call, risk, diff, action, outcome)?;
                let approval = self.active(self.turn.approval.lock(), outcome).await?;
                self.check_cancel(outcome)?;
                self.scope_allowed(resolved, call, risk, diff, action, outcome)?;
                let (grant_action, access, request_id) = self
                    .engine
                    .ask_scope_grant(
                        self.io,
                        call,
                        &path,
                        suggestion.as_deref(),
                        resolved.readonly() || risk == CallRisk::Readonly,
                        self.deadline,
                        &mut outcome.status,
                        &mut outcome.audit,
                    )
                    .await?;
                outcome.audit.scope_approval_request_id = Some(request_id);
                self.check_cancel(outcome)?;
                drop(approval);
                self.scope_allowed(resolved, call, risk, diff, action, outcome)?;
                self.authorize_operation(
                    resolved,
                    call,
                    &args,
                    risk,
                    diff,
                    action,
                    approved,
                    InvocationPhase::ScopeGrant {
                        action: grant_action.clone(),
                        path: path.clone(),
                        access: access.clone(),
                    },
                    true,
                    outcome,
                )
                .await?;
                self.scope_allowed(resolved, call, risk, diff, action, outcome)?;
                self.check_cancel(outcome)?;
                outcome.audit.effects_may_have_occurred = true;
                match self.engine.ctx.grant_scope(&grant_action, &path, &access) {
                    Some(result) => result?,
                    None => return Err("this host cannot link directories".into()),
                }
                self.authorize_operation(
                    resolved,
                    call,
                    &args,
                    risk,
                    diff,
                    action,
                    approved,
                    phase,
                    !resolved.staged(),
                    outcome,
                )
                .await?;
                self.check_cancel(outcome)?;
                outcome.audit.effects_may_have_occurred |=
                    !resolved.staged() && !(resolved.readonly() || risk == CallRisk::Readonly);
                self.dispatch(resolved, &call.name, args, outcome)
                    .await
                    .map_err(|e| e.to_string())
            }
            result => result.map_err(|e| e.to_string()),
        }
    }

    fn scope_allowed(
        &self,
        resolved: &ResolvedTool,
        call: &ToolCall,
        risk: CallRisk,
        diff: &str,
        action: Option<&str>,
        outcome: &mut InvocationOutcome,
    ) -> Result<(), String> {
        self.check_cancel(outcome)?;
        self.ensure_current(&call.name, resolved)?;
        if !self
            .engine
            .permissions
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .can_ask()
        {
            outcome.status = InvocationStatus::Denied;
            return Err(
                "directory access requires human approval — unavailable in this context".into(),
            );
        }
        if let Decision::Deny { reason } =
            self.permission_decision(resolved, call, risk, diff, action)
        {
            outcome.status = InvocationStatus::Denied;
            return Err(reason);
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    async fn authorize_operation(
        &self,
        resolved: &ResolvedTool,
        call: &ToolCall,
        args: &Value,
        risk: CallRisk,
        diff: &str,
        action: Option<&str>,
        approved: &mut bool,
        phase: InvocationPhase,
        needs_execution_approval: bool,
        outcome: &mut InvocationOutcome,
    ) -> Result<(), String> {
        loop {
            self.check_cancel(outcome)?;
            self.ensure_current(&call.name, resolved)?;
            match self.permission_decision(resolved, call, risk, diff, action) {
                Decision::Deny { reason } => {
                    outcome.status = InvocationStatus::Denied;
                    return Err(reason);
                }
                Decision::Ask { .. } if needs_execution_approval && !*approved => {
                    *approved = self
                        .authorize(resolved, call, args, risk, diff, action, false, outcome)
                        .await?;
                }
                _ => {}
            }
            let authorized = matches!(phase, InvocationPhase::Dispatch | InvocationPhase::Commit);
            self.pre_effect(call, phase.clone(), authorized, outcome)
                .await?;
            self.ensure_current(&call.name, resolved)?;
            match self.permission_decision(resolved, call, risk, diff, action) {
                Decision::Deny { reason } => {
                    outcome.status = InvocationStatus::Denied;
                    return Err(reason);
                }
                Decision::Ask { .. } if needs_execution_approval && !*approved => continue,
                _ => {}
            }
            self.check_cancel(outcome)?;
            if authorized {
                outcome.audit.authorized_call = Some(call.clone());
            }
            return Ok(());
        }
    }

    async fn execute(
        &self,
        original: &ToolCall,
        observation_only: bool,
        outcome: &mut InvocationOutcome,
    ) -> Result<(), String> {
        if let Err(error) = self.check_cancel(outcome) {
            return Err(if outcome.status == InvocationStatus::Cancelled {
                "cancelled before dispatch".into()
            } else {
                error
            });
        }
        outcome.attempted = true;
        let registry = self.engine.active_registry();
        let args = crate::schema::parse_arguments(&original.arguments)?;
        let _ = self.io.ui_tx.send(UiEvent::ToolCallStarted {
            name: original.name.clone(),
            args_preview: args_preview_of(&args),
            call_id: original.id.clone(),
            parent: self.parent.clone(),
            child_id: None,
        });
        let initial = self.resolve(&registry, &original.name)?;
        if observation_only && (initial.class() != Some(ToolClass::Observation) || initial.staged())
        {
            outcome.status = InvocationStatus::Denied;
            return Err("nested calls require a non-staged Observation tool".into());
        }
        crate::schema::validate(initial.schema(), &args)?;
        self.check_cancel(outcome)?;
        let mut call = original.clone();
        self.active(self.engine.hooks.run_before_tool(&mut call), outcome)
            .await??;
        call.id.clone_from(&original.id);
        if call.name.len() > MAX_METADATA_BYTES {
            return Err("tool name exceeds invocation limit".into());
        }
        if call.arguments.len() > crate::schema::MAX_ARGUMENT_BYTES {
            return Err("tool arguments exceed 64 KiB".into());
        }
        outcome.audit.reviewed_call = Some(call.clone());
        let args = crate::schema::parse_arguments(&call.arguments)?;
        let registry = self.engine.active_registry();
        let resolved = self.resolve(&registry, &call.name)?;
        if observation_only
            && (resolved.class() != Some(ToolClass::Observation) || resolved.staged())
        {
            outcome.status = InvocationStatus::Denied;
            return Err("nested calls require a non-staged Observation tool".into());
        }
        crate::schema::validate(resolved.schema(), &args)?;
        self.check_cancel(outcome)?;
        let (risk, review_summary, action, prepare) = match &resolved {
            ResolvedTool::Builtin(spec) => {
                if let Some(review) = &spec.review {
                    let review = review(&args, &self.engine.ctx);
                    if let Some(reason) = review.deny {
                        outcome.status = InvocationStatus::Denied;
                        outcome.denied = true;
                        return Err(reason);
                    }
                    (review.risk, review.summary, review.action, review.prepare)
                } else {
                    (CallRisk::Normal, None, None, None)
                }
            }
            ResolvedTool::Plugin { .. } => (CallRisk::Normal, None, None, None),
        };
        let readonly = resolved.readonly() || risk == CallRisk::Readonly;
        if observation_only && !readonly {
            outcome.status = InvocationStatus::Denied;
            return Err("nested Observation tool must be readonly".into());
        }
        // Orchestration must release the parent phase before waiting for children.
        let orchestration = resolved.class() == Some(ToolClass::Orchestration);
        let _mutation = if resolved.staged() || (!readonly && !orchestration) {
            Some(self.active(self.turn.mutation.lock(), outcome).await?)
        } else {
            None
        };
        self.check_cancel(outcome)?;
        self.ensure_current(&call.name, &resolved)?;
        if let Some(prepare) = prepare {
            let mut approved = false;
            self.authorize_operation(
                &resolved,
                &call,
                &args,
                risk,
                "",
                action.as_deref(),
                &mut approved,
                InvocationPhase::Prepare,
                false,
                outcome,
            )
            .await?;
            self.active(prepare, outcome).await??;
        }
        let mut approved = false;
        let staged = if resolved.staged() {
            Some(
                self.dispatch_with_scope(
                    &resolved,
                    &call,
                    args.clone(),
                    risk,
                    action.as_deref(),
                    "",
                    &mut approved,
                    outcome,
                )
                .await?,
            )
        } else {
            None
        };
        let diff = staged
            .as_ref()
            .map(|r| r.content.clone())
            .or(review_summary)
            .unwrap_or_else(|| {
                args.get("path")
                    .and_then(Value::as_str)
                    .unwrap_or(&call.name)
                    .to_string()
            });
        approved = self
            .authorize(
                &resolved,
                &call,
                &args,
                risk,
                &diff,
                action.as_deref(),
                staged.as_ref().is_some_and(|result| result.fuzzy),
                outcome,
            )
            .await?;
        self.check_cancel(outcome)?;
        {
            let _phase = self.active(self.turn.policy.lock(), outcome).await?;
            self.check_cancel(outcome)?;
            if let Some(reason) = self
                .active(
                    self.engine.policy.before_dispatch(&call.name, readonly),
                    outcome,
                )
                .await?
            {
                outcome.status = InvocationStatus::Denied;
                return Err(reason);
            }
        }
        self.check_cancel(outcome)?;
        self.ensure_current(&call.name, &resolved)?;
        match self.permission_decision(&resolved, &call, risk, &diff, action.as_deref()) {
            Decision::Deny { reason } => {
                outcome.status = InvocationStatus::Denied;
                return Err(reason);
            }
            Decision::Ask { .. } if !approved => {
                approved = self
                    .authorize(
                        &resolved,
                        &call,
                        &args,
                        risk,
                        &diff,
                        action.as_deref(),
                        staged.as_ref().is_some_and(|result| result.fuzzy),
                        outcome,
                    )
                    .await?;
                self.check_cancel(outcome)?;
                self.ensure_current(&call.name, &resolved)?;
            }
            _ => {}
        }
        outcome.audit.authorized_call = Some(call.clone());
        outcome.dispatched = true;
        self.engine.set_state(
            self.io,
            AgentState::ExecutingTool {
                tool_name: call.name.clone(),
            },
        );
        let result = if let Some(result) = staged {
            result
        } else {
            self.dispatch_with_scope(
                &resolved,
                &call,
                args.clone(),
                risk,
                action.as_deref(),
                &diff,
                &mut approved,
                outcome,
            )
            .await?
        };
        let ToolResult {
            content,
            ui_type,
            commit,
            images,
            ..
        } = result;
        outcome.ui_type = ui_type.map(str::to_string);
        outcome.images = images;
        if let Some(commit) = commit {
            self.check_cancel(outcome)?;
            let agent_tool::Commit { verify, apply } = commit;
            verify()?;
            self.authorize_operation(
                &resolved,
                &call,
                &args,
                risk,
                &diff,
                action.as_deref(),
                &mut approved,
                InvocationPhase::Commit,
                true,
                outcome,
            )
            .await?;
            self.check_cancel(outcome)?;
            outcome.audit.effects_may_have_occurred = true;
            self.active(apply, outcome).await??;
        }
        outcome.content = content;
        outcome.status = InvocationStatus::Succeeded;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    async fn authorize(
        &self,
        resolved: &ResolvedTool,
        call: &ToolCall,
        args: &Value,
        risk: CallRisk,
        diff: &str,
        action: Option<&str>,
        fuzzy: bool,
        outcome: &mut InvocationOutcome,
    ) -> Result<bool, String> {
        // Hold only the interaction phase: one UI decision slot, no execution lock.
        let _approval = self.active(self.turn.approval.lock(), outcome).await?;
        self.check_cancel(outcome)?;
        let gate = &self.engine.permissions;
        let decision = self.permission_decision(resolved, call, risk, diff, action);
        match decision {
            Decision::Allow => Ok(false),
            Decision::Deny { reason } => {
                outcome.status = InvocationStatus::Denied;
                Err(reason)
            }
            Decision::Ask { diff_summary } => {
                let mut canonical = args.clone();
                canonical.sort_all_objects();
                let key = (call.name.clone(), canonical.to_string());
                if self
                    .turn
                    .timed_out
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .contains(&key)
                {
                    outcome.denied = true;
                    outcome.status = InvocationStatus::TimedOut;
                    outcome.audit.stop_reason =
                        Some(InvocationStopReason::ApprovalTimedOut { request_id: None });
                    return Err(super::approval_timeout_text(
                        &call.name,
                        &crate::tools::host_slots::secs_label(self.engine.approval_timeout),
                        true,
                    ));
                }
                if self
                    .turn
                    .timed_out
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .len()
                    >= 32
                {
                    return Err("approval timeout tracking limit reached for this turn".into());
                }
                let request_id = next_request_id();
                outcome.audit.approval_request_id = Some(request_id);
                let _route = self.engine.route_child_approval(request_id);
                self.engine.set_state(
                    self.io,
                    AgentState::AwaitingToolConfirmation {
                        tool_name: call.name.clone(),
                        diff_summary: diff_summary.clone(),
                    },
                );
                let actions = Vec::new();
                let _ = self.io.ui_tx.send(UiEvent::ApprovalRequested {
                    request_id,
                    tool_name: call.name.clone(),
                    diff: diff_summary,
                    fuzzy,
                    actions,
                    child_id: self.engine.child_id(),
                });
                match self
                    .engine
                    .wait_for_decision(request_id, &self.io.cancel, self.deadline)
                    .await
                {
                    DecisionWait::Cancelled => {
                        outcome.status = InvocationStatus::Cancelled;
                        outcome.audit.stop_reason = Some(InvocationStopReason::Cancelled);
                        Err(format!(
                            "{} was not run — the turn was cancelled while awaiting approval",
                            call.name
                        ))
                    }
                    DecisionWait::TimedOut => {
                        outcome.denied = true;
                        outcome.audit.stop_reason = Some(InvocationStopReason::ApprovalTimedOut {
                            request_id: Some(request_id),
                        });
                        self.turn
                            .timed_out
                            .lock()
                            .unwrap_or_else(|e| e.into_inner())
                            .insert(key);
                        let _ = self.io.ui_tx.send(UiEvent::ApprovalTimedOut {
                            request_id,
                            child_id: self.engine.child_id(),
                        });
                        outcome.status = InvocationStatus::TimedOut;
                        Err(super::approval_timeout_text(
                            &call.name,
                            &crate::tools::host_slots::secs_label(self.engine.approval_timeout),
                            false,
                        ))
                    }
                    DecisionWait::InvocationDeadline => {
                        outcome.status = InvocationStatus::TimedOut;
                        outcome.audit.stop_reason = Some(InvocationStopReason::InvocationDeadline);
                        Err("invocation deadline exceeded".into())
                    }
                    DecisionWait::Verdict(verdict) => {
                        if !verdict.approved {
                            outcome.status = InvocationStatus::Denied;
                            outcome.denied = true;
                            outcome.audit.stop_reason =
                                Some(InvocationStopReason::UserRejected { request_id });
                            return Err(format!("user denied {}", call.name));
                        }
                        self.check_cancel(outcome)?;
                        outcome.audit.approval_request_id = Some(request_id);
                        let denied = {
                            let gate = gate.read().unwrap_or_else(|e| e.into_inner());
                            let decision = gate.decide(
                                &call.name,
                                resolved.class(),
                                resolved.readonly(),
                                risk,
                                diff,
                            );
                            match action {
                                Some(action) => gate.escalate(&call.name, action, decision, diff),
                                None => decision,
                            }
                        };
                        if let Decision::Deny { reason } = denied {
                            outcome.status = InvocationStatus::Denied;
                            return Err(reason);
                        }
                        Ok(true)
                    }
                }
            }
        }
    }
}
