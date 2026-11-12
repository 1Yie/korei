use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc, Mutex,
};
use std::time::Duration;

use agent_ipc::UiEvent;
use agent_kernel::channels::{UiReceiver, UiSink};
use agent_kernel::engine::{Engine, EngineIo, ModeEntry};
use agent_kernel::hooks::{AgentHook, HookChain};
use agent_kernel::invocation::{
    AuditSink, InvocationAudit, InvocationEffect, InvocationMetadata, InvocationPhase,
    InvocationRequest, InvocationStart, InvocationStatus, InvocationStopReason, InvocationTurn,
};
use agent_kernel::permissions::{Decision, PermissionGate, PermissionMode, PermissionRules};
use agent_kernel::plugins::{PluginHandle, PluginRouter};
use agent_kernel::tools::registry::{
    Commit, Review, ToolClass, ToolCtx, ToolError, ToolRegistry, ToolResult, ToolSpec, ToolVerdict,
};
use agent_kernel::turn_policy::{TurnPolicy, TurnView, Verdict};
use agent_llm_core::types::ToolCall;
use agent_testkit::ScriptedProvider;
use agent_tool::CallRisk;
use async_trait::async_trait;
use futures::FutureExt;
use serde_json::{json, Value};

fn tool(name: &'static str, readonly: bool, runs: Arc<AtomicUsize>) -> ToolSpec {
    ToolSpec {
        name,
        schema: json!({
            "description": "Probe",
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
                "additionalProperties": false
            }
        }),
        readonly,
        class: if readonly {
            ToolClass::Observation
        } else {
            ToolClass::WorkspaceMutation
        },
        network: false,
        staged: false,
        review: None,
        cancel_aware: false,
        exec: Arc::new(move |args, ctx| {
            let runs = runs.clone();
            async move {
                assert!(ctx.cancel.is_some());
                runs.fetch_add(1, Ordering::SeqCst);
                Ok(ToolResult::text(args["value"].to_string()))
            }
            .boxed()
        }),
    }
}

fn harness(specs: Vec<ToolSpec>) -> (Engine, EngineIo, UiReceiver) {
    harness_with_ctx(specs, ToolCtx::new())
}

fn harness_with_ctx(specs: Vec<ToolSpec>, ctx: ToolCtx) -> (Engine, EngineIo, UiReceiver) {
    let mut registry = ToolRegistry::new();
    for spec in specs {
        registry.register(spec);
    }
    let mut engine = Engine::new(
        Arc::new(ScriptedProvider::new()),
        Arc::new(registry),
        Arc::new(ctx),
        "scripted",
        0.0,
    );
    engine.set_permissions(PermissionGate::from_mode_str("bypass"));
    let (ui_tx, rx) = UiSink::channel();
    let (_steer, steer_rx) = tokio::sync::mpsc::channel(8);
    let (_notice, notice_rx) = tokio::sync::mpsc::channel(8);
    let io = EngineIo {
        ui_tx,
        steer_rx,
        notice_rx,
        cancel: Arc::new(AtomicBool::new(false)),
    };
    (engine, io, rx)
}

fn request(id: &str, name: &str, args: &str) -> InvocationRequest {
    InvocationRequest::new(ToolCall {
        id: id.into(),
        name: name.into(),
        arguments: args.into(),
    })
}

use agent_kernel::invocation_scope::{InvocationIdentity, InvocationScope, ScopedChildServices};

#[derive(Clone)]
struct ScopeProbe {
    value: usize,
    phases: Arc<Mutex<Vec<String>>>,
}

fn scope_probe(ctx: &ToolCtx, phase: &str) -> usize {
    let probe = ctx.require::<ScopeProbe>("scope probe").unwrap();
    probe
        .phases
        .lock()
        .unwrap()
        .push(format!("{phase}:{}", probe.value));
    probe.value
}

fn test_scope(
    ctx: ToolCtx,
    hooks: HookChain,
    plugins: Option<PluginHandle>,
    sink: Option<Arc<dyn AuditSink>>,
    deadline: Option<std::time::Instant>,
) -> InvocationScope {
    InvocationScope::new(
        Arc::new(ctx),
        InvocationIdentity {
            execution_id: "trusted-execution".into(),
            agent_id: "trusted-agent".into(),
            session_id: "trusted-session".into(),
            trace_id: Some("trusted-trace".into()),
            parent_call_id: Some("trusted-parent".into()),
        },
        hooks,
        plugins,
        sink,
        deadline,
    )
}

struct ScopeHooks;

#[async_trait]
impl AgentHook for ScopeHooks {
    fn id(&self) -> &str {
        "scope"
    }
    async fn before_tool_execute(&self, ctx: &ToolCtx, call: &mut ToolCall) -> anyhow::Result<()> {
        let value = scope_probe(ctx, "before");
        if call.name == "stage" {
            call.arguments = json!({"value":value}).to_string();
        }
        Ok(())
    }
    async fn after_tool_execute(
        &self,
        ctx: &ToolCtx,
        _: &ToolCall,
        _: &mut String,
    ) -> anyhow::Result<()> {
        scope_probe(ctx, "after");
        Ok(())
    }
    async fn on_state_transition(
        &self,
        ctx: &ToolCtx,
        _: &agent_ipc::AgentState,
        _: &agent_ipc::AgentState,
    ) {
        scope_probe(ctx, "state");
    }
}

struct ScopeBudget {
    starts: Arc<AtomicUsize>,
    calls: Arc<AtomicUsize>,
}

#[async_trait]
impl TurnPolicy for ScopeBudget {
    fn begin_turn(&self) {
        self.starts.fetch_add(1, Ordering::SeqCst);
    }
    async fn before_dispatch(&self, ctx: &ToolCtx, _: &str, _: bool) -> Option<String> {
        scope_probe(ctx, "policy");
        (self.calls.fetch_add(1, Ordering::SeqCst) > 0).then(|| "parent budget exhausted".into())
    }
    async fn after_execute(&self, ctx: &ToolCtx, _: &ToolCall, _: &ToolResult) {
        scope_probe(ctx, "policy-after");
    }
    async fn on_quiet_round(&self, _: &dyn TurnView) -> Verdict {
        Verdict::finish()
    }
}

#[tokio::test]
async fn scoped_stages_use_one_binding_and_scopes_share_the_parent_budget() {
    let phases = Arc::new(Mutex::new(Vec::new()));
    let starts = Arc::new(AtomicUsize::new(0));
    let calls = Arc::new(AtomicUsize::new(0));
    let mut stage = tool("stage", false, Arc::new(AtomicUsize::new(0)));
    stage.staged = true;
    stage.review = Some(Arc::new(|args, ctx| {
        assert_eq!(args["value"], scope_probe(ctx, "review"));
        let ctx = ctx.clone();
        Review {
            prepare: Some(
                async move {
                    scope_probe(&ctx, "prepare");
                    Ok(())
                }
                .boxed(),
            ),
            ..Default::default()
        }
    }));
    stage.exec = Arc::new(|args, ctx| {
        async move {
            assert_eq!(args["value"], scope_probe(&ctx, "dispatch"));
            Ok(ToolResult {
                commit: Some(Commit::new(async move {
                    scope_probe(&ctx, "commit");
                    Ok(())
                })),
                ..ToolResult::text("preview")
            })
        }
        .boxed()
    });
    let (mut engine, io, _) = harness(vec![stage]);
    let root_hooks = HookChain::new();
    root_hooks.replace(vec![Arc::new(ScopeHooks)]);
    engine.set_hooks(root_hooks);
    engine.set_policy(Box::new(ScopeBudget {
        starts: starts.clone(),
        calls: calls.clone(),
    }));
    let turn = engine.begin_invocation_turn();
    let sink = Arc::new(EffectAudit::default());
    for value in [7, 8] {
        let mut ctx = ToolCtx::new();
        ctx.ext.insert(Arc::new(ScopeProbe {
            value,
            phases: phases.clone(),
        }));
        let hooks = HookChain::new();
        hooks.replace(vec![Arc::new(ScopeHooks)]);
        let scope = test_scope(ctx, hooks, None, Some(sink.clone()), None);
        let mut req = request(&format!("stage-{value}"), "stage", r#"{"value":1}"#);
        req.metadata.execution_id = Some("forged".into());
        req.metadata.agent_id = Some("forged".into());
        req.metadata.parent_call_id = Some("forged".into());
        let out = engine
            .invocation_gateway(&io, &turn, None)
            .with_scope(&scope)
            .invoke(req)
            .await;
        assert_eq!(out.ok(), value == 7, "{}", out.content);
        assert_eq!(
            out.audit.metadata.execution_id.as_deref(),
            Some("trusted-execution")
        );
        assert_eq!(
            out.audit.metadata.agent_id.as_deref(),
            Some("trusted-agent")
        );
        assert_eq!(
            out.audit.metadata.parent_call_id.as_deref(),
            Some("trusted-parent")
        );
        scope.close().await.unwrap();
    }
    assert_eq!(starts.load(Ordering::SeqCst), 1);
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    let phases = phases.lock().unwrap();
    for phase in [
        "before",
        "review",
        "prepare",
        "dispatch",
        "policy",
        "state",
        "commit",
        "policy-after",
        "after",
    ] {
        assert!(phases.contains(&format!("{phase}:7")), "{phases:?}");
    }
    assert!(!phases.contains(&"commit:8".into()));
    let effects = sink.effects.lock().unwrap();
    assert!(effects
        .iter()
        .any(|e| e.phase == InvocationPhase::Prepare && e.authorized_call.is_none()));
    assert!(effects
        .iter()
        .any(|e| e.phase == InvocationPhase::Commit && e.authorized_call.is_some()));
    assert_eq!(sink.finishes.lock().unwrap().len(), 2);
}

#[tokio::test]
async fn scoped_discovery_and_plugin_dispatch_never_fall_back_to_root() {
    let root_runs = Arc::new(AtomicUsize::new(0));
    let scoped_runs = Arc::new(AtomicUsize::new(0));
    let (mut engine, io, _) = harness(Vec::new());
    let root: PluginHandle = Default::default();
    *root.write().unwrap() = Some(Arc::new(Router {
        name: "root",
        runs: root_runs.clone(),
    }));
    engine.set_plugins(root);
    let scoped: PluginHandle = Default::default();
    *scoped.write().unwrap() = Some(Arc::new(Router {
        name: "scoped",
        runs: scoped_runs.clone(),
    }));
    let scope = test_scope(ToolCtx::new(), HookChain::new(), Some(scoped), None, None);
    let turn = engine.begin_invocation_turn();
    let gateway = engine
        .invocation_gateway(&io, &turn, None)
        .with_scope(&scope);
    assert_eq!(gateway.list().unwrap()[0].name, "scoped");
    assert!(gateway.describe("root").is_err());
    assert!(!gateway
        .invoke(request("root", "root", r#"{"value":1}"#))
        .await
        .ok());
    assert!(gateway
        .invoke(request("scoped", "scoped", r#"{"value":1}"#))
        .await
        .ok());
    assert_eq!(root_runs.load(Ordering::SeqCst), 0);
    assert_eq!(scoped_runs.load(Ordering::SeqCst), 1);
    scope.close().await.unwrap();
    assert!(gateway.list().is_err());
    assert!(!gateway
        .invoke(request("late", "scoped", r#"{"value":1}"#))
        .await
        .ok());
    assert_eq!(scoped_runs.load(Ordering::SeqCst), 1);
}

struct ScopedStateWait {
    entered: Arc<tokio::sync::Notify>,
    dropped: Arc<AtomicUsize>,
    effects: Arc<AtomicUsize>,
}

#[async_trait]
impl AgentHook for ScopedStateWait {
    fn id(&self) -> &str {
        "scoped-state-wait"
    }
    async fn on_state_transition(
        &self,
        _: &ToolCtx,
        _: &agent_ipc::AgentState,
        _: &agent_ipc::AgentState,
    ) {
        let _drop = AuditDrop(&self.dropped);
        self.entered.notify_one();
        futures::future::pending::<()>().await;
        self.effects.fetch_add(1, Ordering::SeqCst);
    }
}

#[tokio::test(start_paused = true)]
async fn scoped_state_hooks_are_cancelled_inline_and_close_waits_for_terminal_audit() {
    let runs = Arc::new(AtomicUsize::new(0));
    let entered = Arc::new(tokio::sync::Notify::new());
    let dropped = Arc::new(AtomicUsize::new(0));
    let effects = Arc::new(AtomicUsize::new(0));
    let (engine, io, mut rx) = harness(vec![tool("probe", true, runs.clone())]);
    let hooks = HookChain::new();
    hooks.replace(vec![Arc::new(ScopedStateWait {
        entered: entered.clone(),
        dropped: dropped.clone(),
        effects: effects.clone(),
    })]);
    let sink = Arc::new(EffectAudit {
        pending: Some("finish"),
        ..Default::default()
    });
    let scope = test_scope(
        ToolCtx::new(),
        hooks.clone(),
        None,
        Some(sink.clone()),
        None,
    );
    hooks.observe_nowait(scope.ctx().clone(), &agent_ipc::AgentState::Reasoning);
    let turn = InvocationTurn::default();
    let gateway = engine
        .invocation_gateway(&io, &turn, None)
        .with_scope(&scope);
    let stop = async {
        entered.notified().await;
        scope.close().await.unwrap();
        assert_eq!(sink.dropped.load(Ordering::SeqCst), 1);
    };
    let (out, ()) = tokio::join!(
        gateway.invoke(request("hook", "probe", r#"{"value":1}"#)),
        stop
    );
    assert_eq!(out.status, InvocationStatus::Cancelled);
    assert_eq!(dropped.load(Ordering::SeqCst), 1);
    assert_eq!(effects.load(Ordering::SeqCst), 0);
    assert_eq!(runs.load(Ordering::SeqCst), 0);
    assert_eq!(terminals(&mut rx), 1);
    assert!(!io.cancel.load(Ordering::SeqCst));
}

#[tokio::test]
async fn scoped_async_grants_recheck_authority_and_stop_before_late_effects() {
    for case in ["success", "cancel", "deadline", "revoked", "rejected"] {
        let attempts = Arc::new(AtomicUsize::new(0));
        let grants = Arc::new(AtomicUsize::new(0));
        let granted = Arc::new(AtomicBool::new(false));
        let entered = Arc::new(tokio::sync::Notify::new());
        let release = Arc::new(tokio::sync::Notify::new());
        let dropped = Arc::new(AtomicUsize::new(0));
        let sink = Arc::new(EffectAudit::default());
        let mut probe = tool("read", true, attempts.clone());
        let count = attempts.clone();
        let access = granted.clone();
        probe.exec = Arc::new(move |_, _| {
            let count = count.clone();
            let access = access.clone();
            async move {
                count.fetch_add(1, Ordering::SeqCst);
                if access.load(Ordering::SeqCst) {
                    Ok(ToolResult::text("granted"))
                } else {
                    Err(ToolError::OutOfScope {
                        path: "/scoped/file".into(),
                        suggestion: None,
                    })
                }
            }
            .boxed()
        });
        let (mut engine, io, mut rx) = harness(vec![probe]);
        let mut ctx = ToolCtx::new();
        let (entered_grant, release_grant, grant_drop, grant_count, audit) = (
            entered.clone(),
            release.clone(),
            dropped.clone(),
            grants.clone(),
            sink.clone(),
        );
        ctx.install_scope_grant(Arc::new(move |_, path, _| {
            let (entered, release, dropped, grants, granted, audit) = (
                entered_grant.clone(),
                release_grant.clone(),
                grant_drop.clone(),
                grant_count.clone(),
                granted.clone(),
                audit.clone(),
            );
            Box::pin(async move {
                let _drop = AuditDrop(&dropped);
                assert_eq!(path, "/scoped/file");
                assert!(audit
                    .effects
                    .lock()
                    .unwrap()
                    .iter()
                    .any(|e| matches!(e.phase, InvocationPhase::ScopeGrant { .. })));
                entered.notify_one();
                release.notified().await;
                grants.fetch_add(1, Ordering::SeqCst);
                granted.store(true, Ordering::SeqCst);
                Ok(())
            })
        }));
        let deadline =
            (case == "deadline").then(|| std::time::Instant::now() + Duration::from_millis(100));
        let scope = test_scope(ctx, HookChain::new(), None, Some(sink.clone()), deadline);
        let permissions = engine.permissions_writer();
        let turn = engine.begin_invocation_turn();
        let gateway = engine
            .invocation_gateway(&io, &turn, None)
            .with_scope(&scope);
        let decisions = engine.decision_slot();
        let control = async {
            let request_id = loop {
                if let Some(UiEvent::ApprovalRequested { request_id, .. }) = rx.try_recv() {
                    *decisions.lock().unwrap() = Some((
                        request_id,
                        ToolVerdict {
                            approved: case != "rejected",
                            action: Some("link:session".into()),
                        },
                    ));
                    break request_id;
                }
                tokio::time::sleep(Duration::from_millis(1)).await;
            };
            if case == "rejected" {
                return request_id;
            }
            entered.notified().await;
            match case {
                "cancel" => scope.cancel(),
                "deadline" => {}
                "revoked" => {
                    *permissions.write().unwrap() = PermissionGate::new(
                        PermissionMode::Bypass,
                        PermissionRules {
                            deny: ["read".to_string()].into_iter().collect(),
                            ..Default::default()
                        },
                    );
                    release.notify_one();
                }
                _ => release.notify_one(),
            }
            request_id
        };
        let (out, approval_id) = tokio::join!(
            gateway.invoke(request(case, "read", r#"{"value":1}"#)),
            control
        );
        assert_eq!(out.ok(), case == "success", "{case}: {}", out.content);
        assert_eq!(
            attempts.load(Ordering::SeqCst),
            if case == "success" { 2 } else { 1 },
            "{case}"
        );
        assert_eq!(
            grants.load(Ordering::SeqCst),
            usize::from(matches!(case, "success" | "revoked")),
            "{case}"
        );
        assert_eq!(
            dropped.load(Ordering::SeqCst),
            usize::from(case != "rejected"),
            "{case}"
        );
        match case {
            "rejected" => assert_eq!(
                out.audit.stop_reason,
                Some(InvocationStopReason::UserRejected {
                    request_id: approval_id
                })
            ),
            "cancel" => assert_eq!(out.audit.stop_reason, Some(InvocationStopReason::Cancelled)),
            "deadline" => assert_eq!(
                out.audit.stop_reason,
                Some(InvocationStopReason::InvocationDeadline)
            ),
            _ => {}
        }
        scope.close().await.unwrap();
        assert_eq!(sink.finishes.lock().unwrap().len(), 1);
        assert!(!io.cancel.load(Ordering::SeqCst));
    }
}

#[tokio::test]
async fn scoped_nested_calls_share_context_audit_and_reject_late_dispatch() {
    use agent_kernel::tools::dispatch::NestedDispatcher;
    let phases = Arc::new(Mutex::new(Vec::new()));
    let captured = Arc::new(Mutex::new(None::<NestedDispatcher>));
    let mut probe = tool("probe", true, Arc::new(AtomicUsize::new(0)));
    let saved = captured.clone();
    probe.exec = Arc::new(move |_, ctx| {
        let saved = saved.clone();
        async move {
            assert_eq!(scope_probe(&ctx, "nested"), 42);
            *saved.lock().unwrap() = Some((*ctx.ext.get::<NestedDispatcher>().unwrap()).clone());
            Ok(ToolResult::text("scoped nested"))
        }
        .boxed()
    });
    let (engine, io, _) = harness(vec![agent_kernel::tools::batch::spec(), probe]);
    let mut ctx = ToolCtx::new();
    ctx.ext.insert(Arc::new(ScopeProbe {
        value: 42,
        phases: phases.clone(),
    }));
    let hooks = HookChain::new();
    hooks.replace(vec![Arc::new(ScopeHooks)]);
    let sink = Arc::new(EffectAudit::default());
    let scope = test_scope(ctx, hooks, None, Some(sink.clone()), None);
    let turn = InvocationTurn::default();
    let out = engine
        .invocation_gateway(&io, &turn, None)
        .with_scope(&scope)
        .invoke(batch_request(json!([{"tool":"probe","args":{"value":1}}])))
        .await;
    assert!(out.ok(), "{}", out.content);
    assert!(out.content.contains("scoped nested"), "{}", out.content);
    let audits = sink.finishes.lock().unwrap().clone();
    assert_eq!(audits.len(), 2);
    let nested = audits
        .iter()
        .find(|a| a.metadata.invocation_id != out.audit.metadata.invocation_id)
        .unwrap();
    assert_eq!(
        nested.metadata.execution_id,
        out.audit.metadata.execution_id
    );
    assert_eq!(
        nested.metadata.parent_call_id.as_deref(),
        Some(out.audit.metadata.invocation_id.as_str())
    );
    scope.close().await.unwrap();
    let late = captured
        .lock()
        .unwrap()
        .take()
        .unwrap()
        .invoke(
            ToolCall {
                id: "late".into(),
                name: "probe".into(),
                arguments: r#"{"value":1}"#.into(),
            },
            Arc::new(AtomicBool::new(false)),
            std::time::Instant::now() + Duration::from_secs(1),
            None,
        )
        .await;
    assert!(late.is_err());
    assert_eq!(
        phases
            .lock()
            .unwrap()
            .iter()
            .filter(|p| p.as_str() == "nested:42")
            .count(),
        1
    );
}

#[tokio::test]
async fn scoped_background_children_require_bound_services_keep_own_authority_and_drain() {
    use agent_kernel::tools::delegate::{self, ChildRoute, SubagentSpawner};
    use agent_llm_core::types::StreamChunk;
    for factory in [false, true] {
        let phases = Arc::new(Mutex::new(Vec::new()));
        let entered = Arc::new(tokio::sync::Notify::new());
        let dropped = Arc::new(AtomicUsize::new(0));
        let critical_runs = Arc::new(AtomicUsize::new(0));
        let child_provider = Arc::new(ScriptedProvider::new());
        for (id, name) in [("child-critical", "critical"), ("child-read", "read")] {
            child_provider.push_script(vec![
                StreamChunk::ToolCallDelta {
                    slot: 0,
                    id: Some(id.into()),
                    name: Some(name.into()),
                    args_delta: r#"{"value":1}"#.into(),
                },
                StreamChunk::Done {
                    prompt_tokens: None,
                    completion_tokens: None,
                    cached_tokens: None,
                },
            ]);
        }
        let mut critical = tool("critical", true, critical_runs.clone());
        critical.review = Some(Arc::new(|_, ctx| {
            assert_eq!(scope_probe(ctx, "child-review"), 43);
            Review {
                risk: CallRisk::Critical,
                ..Default::default()
            }
        }));
        let mut read = tool("read", true, Arc::new(AtomicUsize::new(0)));
        let (read_entered, read_dropped) = (entered.clone(), dropped.clone());
        read.exec = Arc::new(move |_, ctx| {
            let (entered, dropped) = (read_entered.clone(), read_dropped.clone());
            async move {
                let _drop = AuditDrop(&dropped);
                assert_eq!(scope_probe(&ctx, "child-dispatch"), 43);
                entered.notify_one();
                futures::future::pending::<()>().await;
                Ok(ToolResult::text("unreachable"))
            }
            .boxed()
        });
        let mut child_registry = ToolRegistry::new();
        child_registry.register(critical);
        child_registry.register(read);
        let child_registry = Arc::new(child_registry);
        let gate = Arc::new(std::sync::RwLock::new(PermissionGate::from_mode_str(
            "bypass",
        )));
        let mut spawner = SubagentSpawner::new(
            Arc::new(std::sync::RwLock::new(ChildRoute {
                provider: child_provider,
                model: "scripted".into(),
                temperature: 0.0,
                context_window: 100_000,
                thinking_level: None,
                thinking_level_map: None,
                model_params: Default::default(),
                model_input: vec![],
                models: Default::default(),
            })),
            child_registry.clone(),
            child_registry,
        )
        .with_parent_gate(gate.clone());
        if factory {
            spawner = spawner
                .with_scoped_services(Arc::new(|parent, child| {
                    let marker = parent.require::<ScopeProbe>("parent probe").unwrap();
                    assert_eq!(scope_probe(parent, "child-factory"), 42);
                    child.ext.insert(Arc::new(ScopeProbe {
                        value: 43,
                        phases: marker.phases.clone(),
                    }));
                    let hooks = HookChain::new();
                    hooks.replace(vec![Arc::new(ScopeHooks)]);
                    Ok(ScopedChildServices {
                        hooks,
                        plugins: None,
                    })
                }))
                .with_policy(Arc::new(|parent, child| {
                    assert_eq!(scope_probe(parent, "child-policy-parent"), 42);
                    assert_eq!(scope_probe(child, "child-policy-context"), 43);
                    Box::new(agent_kernel::turn_policy::NoopPolicy)
                }));
        }
        let session = if factory { -5011 } else { -5010 };
        let mut ctx = ToolCtx::new().with_session(session);
        ctx.ext.insert(Arc::new(ScopeProbe {
            value: 42,
            phases: phases.clone(),
        }));
        ctx.install_subagent(spawner);
        let global_cancel = Arc::new(AtomicBool::new(false));
        delegate::install_background_cancel(&mut ctx, global_cancel.clone());
        let (mut engine, io, _) = harness(vec![delegate::spec(), delegate::answer_child_spec()]);
        engine.set_permissions_shared(gate);
        let hooks = HookChain::new();
        hooks.replace(vec![Arc::new(ScopeHooks)]);
        let sink = Arc::new(EffectAudit::default());
        let scope = test_scope(ctx, hooks, None, Some(sink.clone()), None);
        let turn = engine.begin_invocation_turn();
        let gateway = engine
            .invocation_gateway(&io, &turn, None)
            .with_scope(&scope);
        let out = gateway
            .invoke(request(
                "delegate-call",
                "delegate",
                r#"{"task":"inspect","background":true}"#,
            ))
            .await;
        if !factory {
            assert!(!out.ok());
            assert!(
                out.content.contains("scoped child services factory"),
                "{}",
                out.content
            );
            scope.close().await.unwrap();
            assert!(delegate::active_children(session).is_empty());
            continue;
        }
        assert!(out.ok(), "{}", out.content);
        tokio::time::timeout(Duration::from_secs(2), entered.notified())
            .await
            .unwrap();
        let child_id = delegate::active_children(session)[0];
        assert_eq!(critical_runs.load(Ordering::SeqCst), 0);
        let mut other_ctx = ToolCtx::new().with_session(session);
        other_ctx.install_subagent(scope.ctx().subagent().unwrap().0.clone());
        let other = test_scope(other_ctx, HookChain::new(), None, None, None);
        let foreign = engine
            .invocation_gateway(&io, &turn, None)
            .with_scope(&other)
            .invoke(request(
                "foreign",
                "answer_child",
                &json!({"child_id":child_id,"answer":"continue"}).to_string(),
            ))
            .await;
        assert!(!foreign.ok());
        assert!(
            foreign.content.contains("in this invocation scope"),
            "{}",
            foreign.content
        );
        other.close().await.unwrap();
        scope.close().await.unwrap();
        assert!(delegate::active_children(session).is_empty());
        assert!(delegate::active_child_infos(session).is_empty());
        assert_eq!(dropped.load(Ordering::SeqCst), 1);
        assert!(!global_cancel.load(Ordering::SeqCst));
        assert!(!io.cancel.load(Ordering::SeqCst));
        let late = gateway
            .invoke(request(
                "late-child",
                "delegate",
                r#"{"task":"late","background":true}"#,
            ))
            .await;
        assert!(!late.ok());
        let finishes = sink.finishes.lock().unwrap();
        let child_audits: Vec<_> = finishes
            .iter()
            .filter(|a| a.metadata.parent_call_id.as_deref() == Some("delegate-call"))
            .collect();
        assert_eq!(child_audits.len(), 2, "{finishes:?}");
        for audit in child_audits {
            assert_eq!(
                audit.metadata.execution_id.as_deref(),
                Some("trusted-execution")
            );
            assert_eq!(
                audit.metadata.agent_id.as_deref(),
                Some(child_id.to_string().as_str())
            );
        }
        assert!(finishes
            .iter()
            .any(|a| a.metadata.invocation_id == "child-critical"
                && a.status == InvocationStatus::Denied));
        assert!(finishes
            .iter()
            .any(|a| a.metadata.invocation_id == "child-read"
                && a.status == InvocationStatus::Cancelled));
        assert!(phases.lock().unwrap().contains(&"state:43".to_string()));
    }
}

fn terminals(rx: &mut UiReceiver) -> usize {
    let mut count = 0;
    while let Some(event) = rx.try_recv() {
        if matches!(event, UiEvent::ToolCallFinished { .. }) {
            count += 1;
        }
    }
    count
}

#[tokio::test]
async fn invalid_json_schema_and_external_references_never_execute() {
    let runs = Arc::new(AtomicUsize::new(0));
    let mut bad = tool("external", true, runs.clone());
    bad.schema = json!({"$ref": "file:///etc/passwd"});
    let mut http = tool("http", true, runs.clone());
    http.schema = json!({"$ref": "https://example.invalid/schema"});
    let mut invalid = tool("invalid", true, runs.clone());
    invalid.schema = json!({"type": "bogus"});
    let (engine, io, mut rx) = harness(vec![tool("probe", true, runs.clone()), bad, http, invalid]);
    let turn = InvocationTurn::default();
    let gateway = engine.invocation_gateway(&io, &turn, None);
    for (id, name, args) in [
        ("json", "probe", "{"),
        ("schema", "probe", r#"{"value":"wrong"}"#),
        ("missing", "probe", "{}"),
        ("external", "external", "{}"),
        ("http", "http", "{}"),
        ("invalid", "invalid", "{}"),
    ] {
        assert!(!gateway.invoke(request(id, name, args)).await.ok());
    }
    assert_eq!(runs.load(Ordering::SeqCst), 0);
    assert_eq!(terminals(&mut rx), 6);
    assert!(gateway.describe("external").is_err());
    assert!(gateway.describe("invalid").is_err());
    assert_eq!(
        gateway.describe("probe").unwrap().parameters["type"],
        "object"
    );
}

#[tokio::test]
async fn local_references_validate_and_schema_size_and_depth_are_bounded() {
    let runs = Arc::new(AtomicUsize::new(0));
    let mut local = tool("local", true, runs.clone());
    local.schema = json!({
        "$defs": {"integer": {"type": "integer"}},
        "type": "object",
        "properties": {"value": {"$ref": "#/$defs/integer"}},
        "required": ["value"]
    });
    let mut large = tool("large", true, runs.clone());
    large.schema = json!({"description": "x".repeat(256 * 1024), "type": "object"});
    let mut deep = tool("deep", true, runs.clone());
    let mut nested = json!({"type": "integer"});
    for _ in 0..70 {
        nested = json!({"type":"object","properties":{"value":nested}});
    }
    deep.schema = nested;
    let (engine, io, _) = harness(vec![local, large, deep]);
    let turn = InvocationTurn::default();
    let gateway = engine.invocation_gateway(&io, &turn, None);
    assert!(gateway
        .invoke(request("local-ok", "local", r#"{"value":1}"#))
        .await
        .ok());
    assert!(!gateway
        .invoke(request("local-invalid", "local", r#"{"value":"wrong"}"#))
        .await
        .ok());
    assert!(!gateway
        .invoke(request("large", "large", r#"{"value":1}"#))
        .await
        .ok());
    assert!(!gateway
        .invoke(request("deep", "deep", r#"{"value":1}"#))
        .await
        .ok());
    assert_eq!(runs.load(Ordering::SeqCst), 1);
}

struct Rewrite(&'static str);
#[async_trait]
impl AgentHook for Rewrite {
    fn id(&self) -> &str {
        "rewrite"
    }
    async fn before_tool_execute(&self, _ctx: &ToolCtx, call: &mut ToolCall) -> anyhow::Result<()> {
        call.arguments = self.0.into();
        call.id = "forged".into();
        Ok(())
    }
}

#[tokio::test]
async fn hook_arguments_are_revalidated_reviewed_and_authorized() {
    let runs = Arc::new(AtomicUsize::new(0));
    let reviewed = Arc::new(AtomicUsize::new(0));
    let mut probe = tool("probe", true, runs.clone());
    let review_count = reviewed.clone();
    probe.review = Some(Arc::new(move |args, _| {
        review_count.fetch_add(1, Ordering::SeqCst);
        Review {
            risk: if args["value"] == 2 {
                CallRisk::Critical
            } else {
                CallRisk::Normal
            },
            ..Default::default()
        }
    }));
    let (mut engine, io, mut rx) = harness(vec![probe]);
    engine.set_permissions(PermissionGate::from_mode_str("dontAsk"));
    let hooks = HookChain::new();
    hooks.replace(vec![Arc::new(Rewrite(r#"{"value":2}"#))]);
    engine.set_hooks(hooks.clone());
    let turn = InvocationTurn::default();
    let outcome = engine
        .invocation_gateway(&io, &turn, None)
        .invoke(request("original", "probe", r#"{"value":1}"#))
        .await;
    assert_eq!(outcome.status, InvocationStatus::Denied);
    assert_eq!(reviewed.load(Ordering::SeqCst), 1);
    assert_eq!(runs.load(Ordering::SeqCst), 0);
    assert_eq!(terminals(&mut rx), 1);

    let (mut engine, io, _) = harness(vec![tool("probe", true, runs.clone())]);
    let hooks = HookChain::new();
    hooks.replace(vec![Arc::new(Rewrite(r#"{"value":"invalid"}"#))]);
    engine.set_hooks(hooks);
    let outcome = engine
        .invocation_gateway(&io, &turn, None)
        .invoke(request("schema", "probe", r#"{"value":1}"#))
        .await;
    assert!(!outcome.ok());
    assert_eq!(runs.load(Ordering::SeqCst), 0);

    let (mut engine, io, mut rx) = harness(vec![tool("probe", true, runs.clone())]);
    let hooks = HookChain::new();
    hooks.replace(vec![Arc::new(Rewrite(r#"{"value":3}"#))]);
    engine.set_hooks(hooks);
    let outcome = engine
        .invocation_gateway(&io, &turn, None)
        .invoke(request("identity", "probe", r#"{"value":1}"#))
        .await;
    assert!(outcome.ok());
    assert_eq!(outcome.content, "3");
    assert_eq!(outcome.audit.authorized_call.unwrap().id, "identity");
    while let Some(event) = rx.try_recv() {
        if let UiEvent::ToolCallFinished { call_id, .. } = event {
            assert_eq!(call_id, "identity");
        }
    }
}

struct Router {
    name: &'static str,
    runs: Arc<AtomicUsize>,
}
#[async_trait]
impl PluginRouter for Router {
    fn exported_tools(&self) -> Vec<Value> {
        vec![json!({"type":"function","function":{
            "name":self.name,"description":"External probe",
            "parameters":{"type":"object","properties":{"value":{"type":"integer"}},"required":["value"]}
        }})]
    }
    async fn dispatch_tool_call(
        &self,
        name: &str,
        _args: Value,
    ) -> Option<Result<ToolResult, String>> {
        if name != self.name {
            return None;
        }
        self.runs.fetch_add(1, Ordering::SeqCst);
        Some(Ok(ToolResult::text("plugin")))
    }
}

#[tokio::test]
async fn discovery_and_dispatch_follow_live_mode_and_plugin_view_without_fallback() {
    let builtin_runs = Arc::new(AtomicUsize::new(0));
    let plugin_runs = Arc::new(AtomicUsize::new(0));
    let mut builtin = tool("probe", true, builtin_runs);
    builtin.exec = Arc::new(|_, _| async { Err(ToolError::Failed("real failure".into())) }.boxed());
    let (mut engine, io, _) = harness(vec![builtin]);
    let handle: PluginHandle = Default::default();
    *handle.write().unwrap() = Some(Arc::new(Router {
        name: "probe",
        runs: plugin_runs.clone(),
    }));
    engine.set_plugins(handle.clone());
    engine.add_mode(ModeEntry {
        id: "restricted".into(),
        registry: Arc::new(ToolRegistry::new()),
        allow_plugins: false,
    });
    let mode = engine.agent_mode_writer();
    let turn = InvocationTurn::default();
    let gateway = engine.invocation_gateway(&io, &turn, None);
    assert_eq!(gateway.list().unwrap().len(), 1);
    assert!(gateway.describe("probe").unwrap().metadata.readonly);
    let result = gateway
        .invoke(request("builtin", "probe", r#"{"value":1}"#))
        .await;
    assert_eq!(result.content, "real failure");
    assert_eq!(plugin_runs.load(Ordering::SeqCst), 0);
    *handle.write().unwrap() = Some(Arc::new(Router {
        name: "external",
        runs: plugin_runs.clone(),
    }));
    assert_eq!(gateway.list().unwrap().len(), 2);
    assert!(gateway
        .invoke(request("invalid-plugin", "external", "{}"))
        .await
        .content
        .contains("invalid"));
    assert!(gateway
        .invoke(request("plugin", "external", r#"{"value":1}"#))
        .await
        .ok());
    *mode.write().unwrap() = "restricted".into();
    assert!(gateway.list().unwrap().is_empty());
    assert!(gateway.describe("external").is_err());
    assert!(!gateway
        .invoke(request("restricted", "external", r#"{"value":1}"#))
        .await
        .ok());
    assert_eq!(plugin_runs.load(Ordering::SeqCst), 1);
}

struct BorrowedAudit<'a> {
    records: &'a Mutex<Vec<InvocationAudit>>,
    fail_begin: bool,
    fail_finish: bool,
}
#[async_trait]
impl AuditSink for BorrowedAudit<'_> {
    async fn pre_effect(&self, _: &InvocationEffect) -> Result<(), String> {
        Ok(())
    }
    async fn begin(&self, record: &InvocationStart) -> Result<(), String> {
        assert_eq!(record.metadata.execution_id.as_deref(), Some("execution"));
        if self.fail_begin {
            Err("begin unavailable".into())
        } else {
            Ok(())
        }
    }
    async fn finish(&self, record: &InvocationAudit) -> Result<(), String> {
        self.records.lock().unwrap().push(record.clone());
        if self.fail_finish {
            Err("finish unavailable".into())
        } else {
            Ok(())
        }
    }
}

#[tokio::test]
async fn borrowed_audit_fails_closed_at_begin_and_reports_effects_on_finish_failure() {
    let runs = Arc::new(AtomicUsize::new(0));
    let (engine, io, mut rx) = harness(vec![tool("write", false, runs.clone())]);
    let turn = InvocationTurn::default();
    let records = Mutex::new(Vec::new());
    let mut req = request("call", "write", r#"{"value":1}"#);
    req.metadata = InvocationMetadata {
        invocation_id: "call".into(),
        parent_call_id: Some("parent".into()),
        execution_id: Some("execution".into()),
        agent_id: Some("agent".into()),
        session_id: Some("session".into()),
        trace_id: Some("trace".into()),
    };
    let sink = BorrowedAudit {
        records: &records,
        fail_begin: true,
        fail_finish: false,
    };
    let outcome = engine
        .invocation_gateway(&io, &turn, Some(&sink))
        .invoke(req.clone())
        .await;
    assert!(!outcome.ok());
    assert!(outcome.content.starts_with("audit begin failed:"));
    assert!(!outcome.audit.effects_may_have_occurred);
    assert_eq!(runs.load(Ordering::SeqCst), 0);
    assert!(records.lock().unwrap().is_empty());

    let sink = BorrowedAudit {
        records: &records,
        fail_begin: false,
        fail_finish: true,
    };
    let outcome = engine
        .invocation_gateway(&io, &turn, Some(&sink))
        .invoke(req)
        .await;
    assert!(!outcome.ok());
    assert_eq!(outcome.status, InvocationStatus::Succeeded);
    assert!(outcome.audit_error.is_some());
    assert!(outcome.audit.effects_may_have_occurred);
    assert_eq!(outcome.content, "1");
    assert_eq!(runs.load(Ordering::SeqCst), 1);
    let records = records.lock().unwrap();
    assert_eq!(
        records[0].metadata.parent_call_id.as_deref(),
        Some("parent")
    );
    assert_eq!(records[0].result_preview, "1");
    assert_eq!(terminals(&mut rx), 2);
}

struct Baseline {
    started: AtomicBool,
    ready: Arc<AtomicBool>,
}
#[async_trait]
impl TurnPolicy for Baseline {
    fn begin_turn(&self) {}
    async fn before_dispatch(&self, _: &ToolCtx, _: &str, _: bool) -> Option<String> {
        if !self.started.swap(true, Ordering::SeqCst) {
            tokio::time::sleep(Duration::from_millis(20)).await;
            self.ready.store(true, Ordering::SeqCst);
        }
        assert!(self.ready.load(Ordering::SeqCst));
        None
    }
    async fn on_quiet_round(&self, _: &dyn TurnView) -> Verdict {
        Verdict::Finish {
            before_final: Vec::new(),
            after_final: Vec::new(),
        }
    }
}

#[tokio::test]
async fn readonly_calls_overlap_after_a_serial_policy_phase() {
    let ready = Arc::new(AtomicBool::new(false));
    let barrier = Arc::new(tokio::sync::Barrier::new(2));
    let mut probe = tool("read", true, Arc::new(AtomicUsize::new(0)));
    let checked = ready.clone();
    probe.exec = Arc::new(move |_, _| {
        let barrier = barrier.clone();
        let checked = checked.clone();
        async move {
            assert!(checked.load(Ordering::SeqCst));
            barrier.wait().await;
            Ok(ToolResult::text("read"))
        }
        .boxed()
    });
    let (mut engine, io, _) = harness(vec![probe]);
    engine.set_policy(Box::new(Baseline {
        started: AtomicBool::new(false),
        ready,
    }));
    let turn = InvocationTurn::default();
    let gateway = engine.invocation_gateway(&io, &turn, None);
    let results = tokio::time::timeout(Duration::from_secs(2), async {
        tokio::join!(
            gateway.invoke(request("a", "read", r#"{"value":1}"#)),
            gateway.invoke(request("b", "read", r#"{"value":2}"#))
        )
    })
    .await
    .expect("readonly execution must overlap");
    assert!(results.0.ok() && results.1.ok());
}

#[tokio::test]
async fn mutating_calls_are_serialized() {
    let active = Arc::new(AtomicUsize::new(0));
    let max = Arc::new(AtomicUsize::new(0));
    let mut probe = tool("write", false, Arc::new(AtomicUsize::new(0)));
    let current = active.clone();
    let peak = max.clone();
    probe.exec = Arc::new(move |_, _| {
        let current = current.clone();
        let peak = peak.clone();
        async move {
            let count = current.fetch_add(1, Ordering::SeqCst) + 1;
            peak.fetch_max(count, Ordering::SeqCst);
            tokio::time::sleep(Duration::from_millis(20)).await;
            current.fetch_sub(1, Ordering::SeqCst);
            Ok(ToolResult::text("wrote"))
        }
        .boxed()
    });
    let (engine, io, _) = harness(vec![probe]);
    let turn = InvocationTurn::default();
    let gateway = engine.invocation_gateway(&io, &turn, None);
    let results = tokio::join!(
        gateway.invoke(request("a", "write", r#"{"value":1}"#)),
        gateway.invoke(request("b", "write", r#"{"value":2}"#))
    );
    assert!(results.0.ok() && results.1.ok());
    assert_eq!(max.load(Ordering::SeqCst), 1);
    assert_eq!(active.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn concurrent_approvals_keep_ids_and_verdicts_separate() {
    let runs = Arc::new(AtomicUsize::new(0));
    let mut probe = tool("write", true, runs.clone());
    probe.review = Some(Arc::new(|_, _| Review {
        risk: CallRisk::Critical,
        ..Default::default()
    }));
    let (mut engine, io, mut rx) = harness(vec![probe]);
    engine.set_permissions(PermissionGate::from_mode_str("default"));
    let slot = engine.decision_slot();
    let turn = InvocationTurn::default();
    let gateway = engine.invocation_gateway(&io, &turn, None);
    let invoke = async {
        tokio::join!(
            gateway.invoke(request("a", "write", r#"{"value":1}"#)),
            gateway.invoke(request("b", "write", r#"{"value":2}"#))
        )
    };
    let decisions = async {
        let mut approvals = Vec::new();
        let mut finished = 0;
        while finished < 2 {
            match rx.recv().await.unwrap() {
                UiEvent::ApprovalRequested { request_id, .. } => {
                    assert!(!approvals.contains(&request_id));
                    approvals.push(request_id);
                    *slot.lock().unwrap() = Some((
                        request_id + 10_000,
                        ToolVerdict {
                            approved: true,
                            action: None,
                        },
                    ));
                    tokio::time::sleep(Duration::from_millis(70)).await;
                    *slot.lock().unwrap() = Some((
                        request_id,
                        ToolVerdict {
                            approved: approvals.len() == 2,
                            action: None,
                        },
                    ));
                }
                UiEvent::ToolCallFinished { .. } => finished += 1,
                _ => {}
            }
        }
        assert_eq!(approvals.len(), 2);
    };
    let ((a, b), ()) = tokio::time::timeout(Duration::from_secs(3), async {
        tokio::join!(invoke, decisions)
    })
    .await
    .expect("approval queue must finish");
    assert_eq!(usize::from(a.ok()) + usize::from(b.ok()), 1);
    assert_eq!(runs.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn staged_commit_requires_approval_and_verification() {
    let applied = Arc::new(AtomicUsize::new(0));
    let mut staged = tool("stage", false, applied.clone());
    staged.staged = true;
    let effects = applied.clone();
    staged.exec = Arc::new(move |_, _| {
        let effects = effects.clone();
        async move {
            Ok(ToolResult {
                commit: Some(
                    Commit::new(async move {
                        effects.fetch_add(1, Ordering::SeqCst);
                        Ok(())
                    })
                    .verified(|| Err("stale preview".into())),
                ),
                ..ToolResult::text("preview")
            })
        }
        .boxed()
    });
    let (mut engine, io, _) = harness(vec![staged]);
    let turn = InvocationTurn::default();
    engine.set_permissions(PermissionGate::from_mode_str("dontAsk"));
    let outcome = engine
        .invocation_gateway(&io, &turn, None)
        .invoke(request("denied", "stage", r#"{"value":1}"#))
        .await;
    assert_eq!(outcome.status, InvocationStatus::Denied);
    assert_eq!(applied.load(Ordering::SeqCst), 0);
    engine.set_permissions(PermissionGate::from_mode_str("bypass"));
    let outcome = engine
        .invocation_gateway(&io, &turn, None)
        .invoke(request("stale", "stage", r#"{"value":1}"#))
        .await;
    assert!(!outcome.ok());
    assert_eq!(outcome.content, "stale preview");
    assert!(!outcome.audit.effects_may_have_occurred);
    assert_eq!(applied.load(Ordering::SeqCst), 0);
}

#[test]
fn critical_risk_always_requires_a_human_and_preserves_existing_precedence() {
    for mode in [
        PermissionMode::Default,
        PermissionMode::AcceptEdits,
        PermissionMode::Auto,
        PermissionMode::Bypass,
        PermissionMode::DontAsk,
    ] {
        for allow in [Vec::new(), vec!["*"], vec!["probe"], vec!["*", "probe"]] {
            let rules = PermissionRules {
                allow: allow.into_iter().map(str::to_string).collect(),
                ..Default::default()
            };
            let gate = PermissionGate::new(mode, rules);
            let decision = gate.decide(
                "probe",
                Some(ToolClass::WorkspaceMutation),
                true,
                CallRisk::Critical,
                "critical",
            );
            assert!(if mode == PermissionMode::DontAsk {
                matches!(decision, Decision::Deny { .. })
            } else {
                matches!(decision, Decision::Ask { .. })
            });
            let child = PermissionGate::child_of(&gate, false);
            assert!(matches!(
                child.decide("probe", None, true, CallRisk::Critical, ""),
                Decision::Deny { .. }
            ));
            let mut deny = PermissionRules::default();
            deny.deny.insert("probe".into());
            deny.allow.insert("probe".into());
            let gate = PermissionGate::new(mode, deny);
            assert!(matches!(
                gate.decide("probe", None, true, CallRisk::Critical, ""),
                Decision::Deny { .. }
            ));
        }
    }
    let gate = PermissionGate::from_mode_str("bypass");
    assert_eq!(
        gate.decide("probe", None, false, CallRisk::Normal, ""),
        Decision::Allow
    );
}

#[tokio::test]
async fn scope_grant_retries_the_same_builtin_once() {
    let granted = Arc::new(AtomicBool::new(false));
    let attempts = Arc::new(AtomicUsize::new(0));
    let mut ctx = ToolCtx::new();
    let scope = granted.clone();
    ctx.install_scope_grant(Arc::new(move |action, path, access| {
        assert_eq!(action, "link:session");
        assert_eq!(path, "/outside/file");
        assert_eq!(access, "read");
        scope.store(true, Ordering::SeqCst);
        Box::pin(async { Ok(()) })
    }));
    let mut probe = tool("read", true, attempts.clone());
    let count = attempts.clone();
    probe.exec = Arc::new(move |_, _| {
        let granted = granted.clone();
        let count = count.clone();
        async move {
            count.fetch_add(1, Ordering::SeqCst);
            if granted.load(Ordering::SeqCst) {
                Ok(ToolResult::text("read after grant"))
            } else {
                Err(ToolError::OutOfScope {
                    path: "/outside/file".into(),
                    suggestion: Some("/outside".into()),
                })
            }
        }
        .boxed()
    });
    let (engine, io, mut rx) = harness_with_ctx(vec![probe], ctx);
    let slot = engine.decision_slot();
    let turn = InvocationTurn::default();
    let gateway = engine.invocation_gateway(&io, &turn, None);
    let responder = async {
        loop {
            if let UiEvent::ApprovalRequested {
                request_id,
                actions,
                ..
            } = rx.recv().await.unwrap()
            {
                assert_eq!(actions.len(), 2);
                *slot.lock().unwrap() = Some((
                    request_id,
                    ToolVerdict {
                        approved: true,
                        action: Some("link:session".into()),
                    },
                ));
                break;
            }
        }
    };
    let (outcome, ()) = tokio::join!(
        gateway.invoke(request("scope", "read", r#"{"value":1}"#)),
        responder
    );
    assert!(outcome.ok(), "{}", outcome.content);
    assert_eq!(outcome.content, "read after grant");
    assert_eq!(attempts.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn staged_success_applies_once_and_cancel_before_dispatch_never_commits() {
    let applied = Arc::new(AtomicUsize::new(0));
    let mut staged = tool("stage", false, applied.clone());
    staged.staged = true;
    let effects = applied.clone();
    staged.exec = Arc::new(move |_, _| {
        let effects = effects.clone();
        async move {
            Ok(ToolResult {
                commit: Some(Commit::new(async move {
                    effects.fetch_add(1, Ordering::SeqCst);
                    Ok(())
                })),
                ..ToolResult::text("preview")
            })
        }
        .boxed()
    });
    let (engine, io, _) = harness(vec![staged]);
    let turn = InvocationTurn::default();
    let outcome = engine
        .invocation_gateway(&io, &turn, None)
        .invoke(request("commit", "stage", r#"{"value":1}"#))
        .await;
    assert!(outcome.ok());
    assert!(outcome.audit.effects_may_have_occurred);
    assert_eq!(applied.load(Ordering::SeqCst), 1);
    io.cancel.store(true, Ordering::SeqCst);
    let outcome = engine
        .invocation_gateway(&io, &turn, None)
        .invoke(request("cancel", "stage", r#"{"value":1}"#))
        .await;
    assert_eq!(outcome.status, InvocationStatus::Cancelled);
    assert!(!outcome.audit.effects_may_have_occurred);
    assert_eq!(applied.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn original_result_and_media_are_preserved_while_audit_preview_is_bounded() {
    let runs = Arc::new(AtomicUsize::new(0));
    let mut probe = tool("image", true, runs.clone());
    let calls = runs.clone();
    probe.exec = Arc::new(move |_, _| {
        let calls = calls.clone();
        async move {
            calls.fetch_add(1, Ordering::SeqCst);
            Ok(ToolResult {
                images: vec![
                    agent_llm_core::types::ImageRef::for_path("/tmp/probe.png".into()).unwrap(),
                ],
                ui_type: Some("markdown"),
                ..ToolResult::text("界".repeat(700_000))
            })
        }
        .boxed()
    });
    let (engine, io, _) = harness(vec![probe]);
    let turn = InvocationTurn::default();
    let gateway = engine.invocation_gateway(&io, &turn, None);
    let outcome = gateway
        .invoke(request("image", "image", r#"{"value":1}"#))
        .await;
    assert!(outcome.ok());
    assert_eq!(outcome.content.len(), 2_100_000);
    assert!(outcome.content.chars().all(|c| c == '界'));
    assert!(outcome.audit.result_preview.len() <= 16 * 1024);
    assert!(outcome.audit.result_truncated);
    assert_eq!(outcome.audit.result_actual_bytes, 2_100_000);
    assert_eq!(outcome.images.len(), 1);
    assert_eq!(outcome.ui_type.as_deref(), Some("markdown"));
    let large = gateway
        .invoke(request("large", "image", &" ".repeat(64 * 1024 + 1)))
        .await;
    assert!(!large.ok());
    let mut req = request("metadata", "image", r#"{"value":1}"#);
    req.metadata.trace_id = Some("x".repeat(2048));
    let large = gateway.invoke(req).await;
    assert!(!large.ok());
    assert!(large.audit.metadata.trace_id.unwrap().len() <= 1024);
    assert_eq!(runs.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn scope_permission_tightening_requires_separate_tool_approval() {
    let granted = Arc::new(AtomicBool::new(false));
    let runs = Arc::new(AtomicUsize::new(0));
    let mut ctx = ToolCtx::new();
    let scope = granted.clone();
    ctx.install_scope_grant(Arc::new(move |_, _, _| {
        scope.store(true, Ordering::SeqCst);
        Box::pin(async { Ok(()) })
    }));
    let mut probe = tool("probe", true, runs.clone());
    let calls = runs.clone();
    let access = granted.clone();
    probe.exec = Arc::new(move |_, _| {
        let calls = calls.clone();
        let access = access.clone();
        async move {
            calls.fetch_add(1, Ordering::SeqCst);
            if access.load(Ordering::SeqCst) {
                Ok(ToolResult::text("read"))
            } else {
                Err(ToolError::OutOfScope {
                    path: "/linked/file".into(),
                    suggestion: None,
                })
            }
        }
        .boxed()
    });
    let (engine, io, mut rx) = harness_with_ctx(vec![probe], ctx);
    let writer = engine.permissions_writer();
    let decision = engine.decision_slot();
    let turn = InvocationTurn::default();
    let gateway = engine.invocation_gateway(&io, &turn, None);
    let respond = async {
        let mut approvals = 0;
        while let Some(event) = rx.recv().await {
            if let UiEvent::ApprovalRequested {
                request_id,
                ref actions,
                ..
            } = event
            {
                approvals += 1;
                if !actions.is_empty() {
                    let mut rules = PermissionRules::default();
                    rules.ask.insert("probe".into());
                    writer.write().unwrap().merge_rules(rules);
                    *decision.lock().unwrap() = Some((
                        request_id,
                        ToolVerdict {
                            approved: true,
                            action: Some("link:session".into()),
                            ..Default::default()
                        },
                    ));
                } else {
                    assert_eq!(runs.load(Ordering::SeqCst), 1);
                    *decision.lock().unwrap() = Some((request_id, ToolVerdict::default()));
                }
            }
            if matches!(event, UiEvent::ToolCallFinished { .. }) {
                break;
            }
        }
        approvals
    };
    let (outcome, approvals) = tokio::join!(
        gateway.invoke(request("scope-tighten", "probe", r#"{"value":1}"#)),
        respond
    );
    assert_eq!(approvals, 2);
    assert_eq!(outcome.status, InvocationStatus::Denied);
    assert_eq!(runs.load(Ordering::SeqCst), 1);
}

#[tokio::test(start_paused = true)]
async fn invocation_deadline_interrupts_ordinary_approval() {
    let runs = Arc::new(AtomicUsize::new(0));
    let (mut engine, io, mut rx) = harness(vec![tool("probe", false, runs.clone())]);
    engine.set_permissions(PermissionGate::from_mode_str("default"));
    let turn = InvocationTurn::default();
    let gateway = engine
        .invocation_gateway(&io, &turn, None)
        .with_deadline(tokio::time::Instant::now().into_std() + Duration::from_secs(1));
    let invoke = gateway.invoke(request("deadline", "probe", r#"{"value":1}"#));
    let observe = async {
        while let Some(event) = rx.recv().await {
            if matches!(event, UiEvent::ApprovalRequested { .. }) {
                tokio::time::advance(Duration::from_secs(1)).await;
                break;
            }
        }
    };
    let (outcome, ()) = tokio::time::timeout(Duration::from_secs(2), async {
        tokio::join!(invoke, observe)
    })
    .await
    .expect("invocation deadline must bound approval wait");
    assert_eq!(outcome.status, InvocationStatus::TimedOut);
    assert!(!outcome.denied);
    assert_eq!(runs.load(Ordering::SeqCst), 0);
}

struct PausedPolicy {
    entered: Arc<tokio::sync::Notify>,
    release: Arc<tokio::sync::Notify>,
}

#[async_trait]
impl TurnPolicy for PausedPolicy {
    fn begin_turn(&self) {}
    async fn before_dispatch(&self, _: &ToolCtx, _: &str, _: bool) -> Option<String> {
        self.entered.notify_one();
        self.release.notified().await;
        None
    }
    async fn on_quiet_round(&self, _: &dyn TurnView) -> Verdict {
        Verdict::Finish {
            before_final: Vec::new(),
            after_final: Vec::new(),
        }
    }
}

#[tokio::test]
async fn mode_and_permission_changes_during_policy_wait_are_enforced_before_execution() {
    for mode_change in [false, true] {
        let runs = Arc::new(AtomicUsize::new(0));
        let (mut engine, io, _) = harness(vec![tool("probe", true, runs.clone())]);
        engine.add_mode(ModeEntry {
            id: "restricted".into(),
            registry: Arc::new(ToolRegistry::new()),
            allow_plugins: false,
        });
        let entered = Arc::new(tokio::sync::Notify::new());
        let release = Arc::new(tokio::sync::Notify::new());
        engine.set_policy(Box::new(PausedPolicy {
            entered: entered.clone(),
            release: release.clone(),
        }));
        let mode = engine.agent_mode_writer();
        let gate = engine.permissions_writer();
        let turn = InvocationTurn::default();
        let gateway = engine.invocation_gateway(&io, &turn, None);
        let tighten = async {
            entered.notified().await;
            if mode_change {
                *mode.write().unwrap() = "restricted".into();
            } else {
                let mut rules = PermissionRules::default();
                rules.deny.insert("probe".into());
                gate.write().unwrap().merge_rules(rules);
            }
            release.notify_one();
        };
        let (outcome, ()) = tokio::join!(
            gateway.invoke(request("wait", "probe", r#"{"value":1}"#)),
            tighten
        );
        assert!(!outcome.ok());
        assert_eq!(runs.load(Ordering::SeqCst), 0);
        assert!(!outcome.audit.effects_may_have_occurred);
    }
}

struct NestedHook {
    value: Value,
    rename: bool,
}
#[async_trait]
impl AgentHook for NestedHook {
    fn id(&self) -> &str {
        "nested"
    }
    async fn before_tool_execute(&self, _: &ToolCtx, call: &mut ToolCall) -> anyhow::Result<()> {
        if call.name == "probe" {
            call.arguments = json!({"value":self.value}).to_string();
            if self.rename {
                call.name = "write".into();
            }
        }
        Ok(())
    }
}

struct NestedPolicy {
    calls: Arc<AtomicUsize>,
    deny: bool,
}
#[async_trait]
impl TurnPolicy for NestedPolicy {
    fn begin_turn(&self) {}
    async fn before_dispatch(&self, _ctx: &ToolCtx, tool: &str, _: bool) -> Option<String> {
        if tool == "probe" {
            self.calls.fetch_add(1, Ordering::SeqCst);
            if self.deny {
                return Some("nested policy refused".into());
            }
        }
        None
    }
    async fn on_quiet_round(&self, _: &dyn TurnView) -> Verdict {
        Verdict::Finish {
            before_final: Vec::new(),
            after_final: Vec::new(),
        }
    }
}

struct NestedAudit<'a> {
    begins: &'a Mutex<Vec<InvocationStart>>,
    finishes: &'a Mutex<Vec<InvocationAudit>>,
    fail_inner: bool,
}
#[async_trait]
impl AuditSink for NestedAudit<'_> {
    async fn pre_effect(&self, _: &InvocationEffect) -> Result<(), String> {
        Ok(())
    }
    async fn begin(&self, record: &InvocationStart) -> Result<(), String> {
        self.begins.lock().unwrap().push(record.clone());
        if self.fail_inner && record.metadata.parent_call_id.as_deref() == Some("outer") {
            Err("nested audit unavailable".into())
        } else {
            Ok(())
        }
    }
    async fn finish(&self, record: &InvocationAudit) -> Result<(), String> {
        self.finishes.lock().unwrap().push(record.clone());
        Ok(())
    }
}

fn batch_request(calls: Value) -> InvocationRequest {
    let mut req = request(
        "outer",
        "batch_execute",
        &json!({"calls":calls}).to_string(),
    );
    req.metadata.execution_id = Some("execution".into());
    req.metadata.agent_id = Some("agent".into());
    req.metadata.session_id = Some("session".into());
    req
}

#[derive(Default)]
struct EffectAudit {
    effects: Mutex<Vec<InvocationEffect>>,
    finishes: Mutex<Vec<InvocationAudit>>,
    fail: Option<InvocationPhase>,
    pending: Option<&'static str>,
    inner_only: bool,
    entered: tokio::sync::Notify,
    dropped: AtomicUsize,
}

#[tokio::test(start_paused = true)]
async fn structured_stop_reason_distinguishes_human_policy_and_timeout() {
    for case in [
        "user",
        "review",
        "deny",
        "headless",
        "dontask",
        "approval-timeout",
        "deadline",
        "policy",
    ] {
        let runs = Arc::new(AtomicUsize::new(0));
        let mut probe = tool("probe", false, runs.clone());
        if case == "review" {
            probe.review = Some(Arc::new(|_, _| Review {
                deny: Some("review refusal".into()),
                ..Default::default()
            }));
        } else if case == "headless" {
            probe.review = Some(Arc::new(|_, _| Review {
                risk: CallRisk::Critical,
                ..Default::default()
            }));
        }
        let (mut engine, io, mut rx) = harness(vec![probe]);
        match case {
            "user" | "approval-timeout" | "deadline" => {
                engine.set_permissions(PermissionGate::from_mode_str("default"));
            }
            "headless" => engine.set_permissions(PermissionGate::for_subagent()),
            "dontask" | "deny" => {
                let mut rules = PermissionRules::default();
                if case == "deny" {
                    rules.deny.insert("probe".into());
                } else {
                    rules.ask.insert("probe".into());
                }
                engine.set_permissions(PermissionGate::new(
                    if case == "dontask" {
                        PermissionMode::DontAsk
                    } else {
                        PermissionMode::Bypass
                    },
                    rules,
                ));
            }
            "policy" => engine.set_policy(Box::new(NestedPolicy {
                calls: Arc::new(AtomicUsize::new(0)),
                deny: true,
            })),
            _ => {}
        }
        engine.set_approval_timeout(Duration::from_secs(1));
        let decision = engine.decision_slot();
        let sink = EffectAudit::default();
        let turn = InvocationTurn::default();
        let mut gateway = engine.invocation_gateway(&io, &turn, Some(&sink));
        if case == "deadline" {
            gateway = gateway
                .with_deadline(tokio::time::Instant::now().into_std() + Duration::from_millis(500));
        }
        let respond = async {
            let mut approval_id = None;
            while let Some(event) = rx.recv().await {
                match event {
                    UiEvent::ApprovalRequested { request_id, .. } => {
                        approval_id = Some(request_id);
                        if case == "user" {
                            *decision.lock().unwrap() = Some((request_id, ToolVerdict::default()));
                        } else {
                            tokio::time::advance(Duration::from_secs(1)).await;
                        }
                    }
                    UiEvent::ToolCallFinished { .. } => break,
                    _ => {}
                }
            }
            approval_id
        };
        let (outcome, approval_id) = tokio::time::timeout(Duration::from_secs(3), async {
            tokio::join!(
                gateway.invoke(request(case, "probe", r#"{"value":1}"#)),
                respond
            )
        })
        .await
        .expect("a decision terminates only the invocation");
        let expected = match case {
            "user" => InvocationStopReason::UserRejected {
                request_id: approval_id.unwrap(),
            },
            "approval-timeout" => InvocationStopReason::ApprovalTimedOut {
                request_id: approval_id,
            },
            "deadline" => InvocationStopReason::InvocationDeadline,
            _ => InvocationStopReason::PolicyDenied,
        };
        assert_eq!(
            outcome.audit.stop_reason.as_ref(),
            Some(&expected),
            "{case}"
        );
        assert_eq!(outcome.audit.approval_request_id, approval_id, "{case}");
        assert_eq!(
            sink.finishes.lock().unwrap()[0].stop_reason.as_ref(),
            Some(&expected)
        );
        assert!(
            !io.cancel.load(Ordering::SeqCst),
            "a refusal must not cancel the agent"
        );
        assert_eq!(runs.load(Ordering::SeqCst), 0);
        assert!(!outcome.audit.effects_may_have_occurred);
        let encoded = serde_json::to_value(&outcome.audit).unwrap();
        assert_eq!(
            encoded["stop_reason"],
            serde_json::to_value(expected).unwrap()
        );
        if case == "review" || case == "approval-timeout" {
            assert!(outcome.denied);
        }
    }
}

#[tokio::test(start_paused = true)]
async fn scope_grant_audit_permission_and_stop_boundaries_are_enforced() {
    for case in [
        "success",
        "audit",
        "cancel",
        "deadline",
        "deny",
        "dontask",
        "headless",
        "decline",
        "approval-timeout",
    ] {
        let grants = Arc::new(AtomicUsize::new(0));
        let runs = Arc::new(AtomicUsize::new(0));
        let mut ctx = ToolCtx::new();
        let granted = grants.clone();
        ctx.install_scope_grant(Arc::new(move |_, _, _| {
            granted.fetch_add(1, Ordering::SeqCst);
            Box::pin(async { Ok(()) })
        }));
        let mut probe = tool("read", true, runs.clone());
        let granted = grants.clone();
        let calls = runs.clone();
        probe.exec = Arc::new(move |_, _| {
            let granted = granted.clone();
            let calls = calls.clone();
            async move {
                calls.fetch_add(1, Ordering::SeqCst);
                if granted.load(Ordering::SeqCst) == 0 {
                    Err(ToolError::OutOfScope {
                        path: "/linked/file".into(),
                        suggestion: None,
                    })
                } else {
                    Ok(ToolResult::text("read"))
                }
            }
            .boxed()
        });
        let (mut engine, io, mut rx) = harness_with_ctx(vec![probe], ctx);
        if case == "approval-timeout" {
            engine.set_approval_timeout(Duration::from_secs(1));
        }
        let writer = engine.permissions_writer();
        let decision = engine.decision_slot();
        let phase = InvocationPhase::ScopeGrant {
            action: "link:session".into(),
            path: "/linked/file".into(),
            access: "read".into(),
        };
        let sink = EffectAudit {
            fail: (case == "audit").then_some(phase.clone()),
            ..Default::default()
        };
        let turn = InvocationTurn::default();
        let mut gateway = engine.invocation_gateway(&io, &turn, Some(&sink));
        if case == "deadline" {
            gateway = gateway
                .with_deadline(tokio::time::Instant::now().into_std() + Duration::from_secs(1));
        }
        let respond = async {
            while let Some(event) = rx.recv().await {
                match event {
                    UiEvent::ApprovalRequested { request_id, .. } => {
                        match case {
                            "cancel" => io.cancel.store(true, Ordering::SeqCst),
                            "deadline" => tokio::time::advance(Duration::from_secs(1)).await,
                            "approval-timeout" => {
                                tokio::time::advance(Duration::from_secs(1)).await;
                                continue;
                            }
                            "deny" => {
                                let mut rules = PermissionRules::default();
                                rules.deny.insert("read".into());
                                writer.write().unwrap().merge_rules(rules);
                            }
                            "dontask" => writer.write().unwrap().set_mode(PermissionMode::DontAsk),
                            "headless" => *writer.write().unwrap() = PermissionGate::for_subagent(),
                            _ => {}
                        }
                        *decision.lock().unwrap() = Some((
                            request_id,
                            ToolVerdict {
                                approved: case != "decline",
                                action: Some("link:session".into()),
                                ..Default::default()
                            },
                        ));
                    }
                    UiEvent::ToolCallFinished { .. } => break,
                    _ => {}
                }
            }
        };
        let (outcome, ()) = tokio::time::timeout(Duration::from_secs(3), async {
            tokio::join!(
                gateway.invoke(request("scope-boundary", "read", r#"{"value":1}"#)),
                respond
            )
        })
        .await
        .expect("scope must respect stop boundaries");
        assert_eq!(
            grants.load(Ordering::SeqCst),
            usize::from(case == "success"),
            "{case}"
        );
        assert_eq!(
            runs.load(Ordering::SeqCst),
            if case == "success" { 2 } else { 1 },
            "{case}"
        );
        assert_eq!(
            outcome.audit.effects_may_have_occurred,
            case == "success",
            "{case}"
        );
        assert_eq!(outcome.audit_error.is_some(), case == "audit", "{case}");
        let expected_stop = match case {
            "decline" => Some(InvocationStopReason::UserRejected {
                request_id: outcome.audit.scope_approval_request_id.unwrap(),
            }),
            "approval-timeout" => Some(InvocationStopReason::ApprovalTimedOut {
                request_id: outcome.audit.scope_approval_request_id,
            }),
            "deadline" => Some(InvocationStopReason::InvocationDeadline),
            "cancel" => Some(InvocationStopReason::Cancelled),
            "deny" | "dontask" | "headless" => Some(InvocationStopReason::PolicyDenied),
            _ => None,
        };
        assert_eq!(outcome.audit.stop_reason, expected_stop, "{case}");
        assert_eq!(
            sink.finishes.lock().unwrap()[0].stop_reason,
            expected_stop,
            "{case}"
        );
        if case == "cancel" {
            assert_eq!(outcome.status, InvocationStatus::Cancelled);
            assert!(!outcome.denied);
        }
        if case == "deadline" {
            assert_eq!(outcome.status, InvocationStatus::TimedOut);
            assert!(!outcome.denied);
        }
        if case == "success" {
            let records = sink.effects.lock().unwrap();
            assert_eq!(records.iter().filter(|r| r.phase == phase).count(), 1);
            let grant = records.iter().find(|r| r.phase == phase).unwrap();
            assert!(grant.authorized_call.is_none());
            assert!(grant.scope_approval_request_id.is_some());
            assert!(grant.approval_request_id.is_none());
            assert!(outcome.ok());
        }
        assert_eq!(sink.finishes.lock().unwrap().len(), 1);
    }
}

struct PendingInvocationHook {
    entered: Arc<tokio::sync::Notify>,
    after: bool,
}
#[async_trait]
impl AgentHook for PendingInvocationHook {
    fn id(&self) -> &str {
        "pending-boundary"
    }
    async fn before_tool_execute(&self, _: &ToolCtx, _: &mut ToolCall) -> anyhow::Result<()> {
        if !self.after {
            self.entered.notify_one();
            futures::future::pending::<()>().await;
        }
        Ok(())
    }
    async fn after_tool_execute(
        &self,
        _: &ToolCtx,
        _: &ToolCall,
        _: &mut String,
    ) -> anyhow::Result<()> {
        if self.after {
            self.entered.notify_one();
            futures::future::pending::<()>().await;
        }
        Ok(())
    }
}

#[tokio::test(start_paused = true)]
async fn hooks_prepare_policy_and_commit_waits_observe_cancel_and_deadline() {
    for phase in ["before", "prepare", "policy", "commit", "after"] {
        for stop in ["cancel", "deadline"] {
            let entered = Arc::new(tokio::sync::Notify::new());
            let runs = Arc::new(AtomicUsize::new(0));
            let mut probe = tool("probe", false, runs.clone());
            if phase == "prepare" {
                let entered = entered.clone();
                probe.review = Some(Arc::new(move |_, _| {
                    let entered = entered.clone();
                    Review {
                        prepare: Some(
                            async move {
                                entered.notify_one();
                                futures::future::pending().await
                            }
                            .boxed(),
                        ),
                        ..Default::default()
                    }
                }));
            }
            if phase == "commit" {
                let entered = entered.clone();
                probe.exec = Arc::new(move |_, _| {
                    let entered = entered.clone();
                    async move {
                        Ok(ToolResult {
                            commit: Some(Commit::new(async move {
                                entered.notify_one();
                                futures::future::pending().await
                            })),
                            ..ToolResult::text("preview")
                        })
                    }
                    .boxed()
                });
            }
            let (mut engine, io, mut rx) = harness(vec![probe]);
            if phase == "policy" {
                engine.set_policy(Box::new(PausedPolicy {
                    entered: entered.clone(),
                    release: Arc::new(tokio::sync::Notify::new()),
                }));
            }
            if phase == "before" || phase == "after" {
                let mut hooks = HookChain::new();
                hooks.register(Arc::new(PendingInvocationHook {
                    entered: entered.clone(),
                    after: phase == "after",
                }));
                engine.set_hooks(hooks);
            }
            let turn = InvocationTurn::default();
            let mut gateway = engine.invocation_gateway(&io, &turn, None);
            if stop == "deadline" {
                gateway = gateway
                    .with_deadline(tokio::time::Instant::now().into_std() + Duration::from_secs(1));
            }
            let stop_wait = async {
                entered.notified().await;
                if stop == "cancel" {
                    io.cancel.store(true, Ordering::SeqCst);
                }
                tokio::time::advance(Duration::from_secs(1)).await;
            };
            let (outcome, ()) = tokio::time::timeout(Duration::from_secs(3), async {
                tokio::join!(
                    gateway.invoke(request("phase-stop", "probe", r#"{"value":1}"#)),
                    stop_wait
                )
            })
            .await
            .expect("phase must be interruptible");
            assert_eq!(
                outcome.status,
                if stop == "cancel" {
                    InvocationStatus::Cancelled
                } else {
                    InvocationStatus::TimedOut
                },
                "{phase}/{stop}"
            );
            assert!(!outcome.denied);
            assert_eq!(terminals(&mut rx), 1);
            assert_eq!(
                outcome.audit.effects_may_have_occurred,
                phase == "commit" || phase == "after"
            );
        }
    }
}

#[tokio::test(start_paused = true)]
async fn approval_mutation_and_policy_lock_waits_observe_deadline() {
    for phase in ["approval", "mutation", "policy"] {
        let entered = Arc::new(tokio::sync::Notify::new());
        let runs = Arc::new(AtomicUsize::new(0));
        let mut probe = tool("probe", phase != "mutation", runs);
        if phase == "mutation" {
            let entered = entered.clone();
            probe.exec = Arc::new(move |_, _| {
                let entered = entered.clone();
                async move {
                    entered.notify_one();
                    futures::future::pending().await
                }
                .boxed()
            });
        }
        let (mut engine, io, mut rx) = harness(vec![probe]);
        if phase == "approval" {
            let mut rules = PermissionRules::default();
            rules.ask.insert("probe".into());
            engine.set_permissions(PermissionGate::new(PermissionMode::Bypass, rules));
        }
        if phase == "policy" {
            engine.set_policy(Box::new(PausedPolicy {
                entered: entered.clone(),
                release: Arc::new(tokio::sync::Notify::new()),
            }));
        }
        let turn = InvocationTurn::default();
        let first = engine.invocation_gateway(&io, &turn, None);
        let second = async {
            if phase == "approval" {
                while let Some(event) = rx.recv().await {
                    if matches!(event, UiEvent::ApprovalRequested { .. }) {
                        break;
                    }
                }
            } else {
                entered.notified().await;
            }
            let second = engine
                .invocation_gateway(&io, &turn, None)
                .with_deadline(tokio::time::Instant::now().into_std() + Duration::from_secs(1));
            let outcome = second
                .invoke(request("second", "probe", r#"{"value":1}"#))
                .await;
            io.cancel.store(true, Ordering::SeqCst);
            assert_eq!(outcome.status, InvocationStatus::TimedOut, "{phase}");
            assert!(!outcome.audit.effects_may_have_occurred);
        };
        let (outcome, ()) = tokio::time::timeout(Duration::from_secs(3), async {
            tokio::join!(
                first.invoke(request("first", "probe", r#"{"value":1}"#)),
                second
            )
        })
        .await
        .expect("lock wait must respect invocation deadline");
        assert_eq!(outcome.status, InvocationStatus::Cancelled);
    }
}

struct AuditDrop<'a>(&'a AtomicUsize);
impl Drop for AuditDrop<'_> {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

impl EffectAudit {
    async fn block(&self, part: &str, metadata: &InvocationMetadata) {
        if self.pending == Some(part) && (!self.inner_only || metadata.parent_call_id.is_some()) {
            let _drop = AuditDrop(&self.dropped);
            self.entered.notify_one();
            futures::future::pending::<()>().await;
        }
    }
}

#[async_trait]
impl AuditSink for EffectAudit {
    async fn begin(&self, record: &InvocationStart) -> Result<(), String> {
        self.block("begin", &record.metadata).await;
        Ok(())
    }
    async fn pre_effect(&self, record: &InvocationEffect) -> Result<(), String> {
        self.effects.lock().unwrap().push(record.clone());
        self.block("effect", &record.metadata).await;
        if self.fail.as_ref() == Some(&record.phase) {
            Err("pre-effect unavailable".into())
        } else {
            Ok(())
        }
    }
    async fn finish(&self, record: &InvocationAudit) -> Result<(), String> {
        self.finishes.lock().unwrap().push(record.clone());
        self.block("finish", &record.metadata).await;
        Ok(())
    }
}

#[tokio::test]
async fn pre_effect_records_reviewed_call_and_blocks_dispatch_on_failure() {
    for fail in [false, true] {
        let runs = Arc::new(AtomicUsize::new(0));
        let (mut engine, io, mut rx) = harness(vec![tool("probe", true, runs.clone())]);
        let mut hooks = HookChain::new();
        hooks.register(Arc::new(NestedHook {
            value: json!(9),
            rename: false,
        }));
        engine.set_hooks(hooks);
        let sink = EffectAudit {
            fail: fail.then_some(InvocationPhase::Dispatch),
            ..Default::default()
        };
        let turn = InvocationTurn::default();
        let mut req = request("audited", "probe", r#"{"value":1}"#);
        req.metadata.trace_id = Some("trace".into());
        req.metadata.execution_id = Some("execution".into());
        let outcome = engine
            .invocation_gateway(&io, &turn, Some(&sink))
            .invoke(req)
            .await;
        assert_eq!(outcome.ok(), !fail);
        assert_eq!(runs.load(Ordering::SeqCst), usize::from(!fail));
        assert_eq!(terminals(&mut rx), 1);
        let effects = sink.effects.lock().unwrap();
        assert_eq!(effects.len(), 1);
        assert_eq!(effects[0].reviewed_call.arguments, r#"{"value":9}"#);
        assert_eq!(
            effects[0].authorized_call.as_ref().unwrap().arguments,
            r#"{"value":9}"#
        );
        assert_eq!(effects[0].metadata.trace_id.as_deref(), Some("trace"));
        assert_eq!(
            effects[0].metadata.execution_id.as_deref(),
            Some("execution")
        );
        assert_eq!(sink.finishes.lock().unwrap().len(), 1);
        assert_eq!(outcome.audit_error.is_some(), fail);
    }
}

#[tokio::test]
async fn staged_prepare_preview_and_commit_have_distinct_pre_effect_records() {
    for fail in [
        None,
        Some(InvocationPhase::Prepare),
        Some(InvocationPhase::StagedPreview),
        Some(InvocationPhase::Commit),
    ] {
        let sink = Arc::new(EffectAudit {
            fail: fail.clone(),
            ..Default::default()
        });
        let prepared = Arc::new(AtomicUsize::new(0));
        let previewed = Arc::new(AtomicUsize::new(0));
        let committed = Arc::new(AtomicUsize::new(0));
        let mut probe = tool("write", false, previewed.clone());
        probe.staged = true;
        let prepares = prepared.clone();
        probe.review = Some(Arc::new(move |_, _| {
            let prepares = prepares.clone();
            Review {
                prepare: Some(
                    async move {
                        prepares.fetch_add(1, Ordering::SeqCst);
                        Ok(())
                    }
                    .boxed(),
                ),
                ..Default::default()
            }
        }));
        let previews = previewed.clone();
        let commits = committed.clone();
        let audit = sink.clone();
        probe.exec = Arc::new(move |_, _| {
            let previews = previews.clone();
            let commits = commits.clone();
            let audit = audit.clone();
            async move {
                assert_eq!(
                    audit.effects.lock().unwrap().last().unwrap().phase,
                    InvocationPhase::StagedPreview
                );
                previews.fetch_add(1, Ordering::SeqCst);
                Ok(ToolResult {
                    commit: Some(Commit::new(async move {
                        assert_eq!(
                            audit.effects.lock().unwrap().last().unwrap().phase,
                            InvocationPhase::Commit
                        );
                        commits.fetch_add(1, Ordering::SeqCst);
                        Ok(())
                    })),
                    ..ToolResult::text("preview")
                })
            }
            .boxed()
        });
        let (mut engine, io, mut rx) = harness(vec![probe]);
        engine.set_permissions(PermissionGate::from_mode_str("default"));
        let decision = engine.decision_slot();
        let turn = InvocationTurn::default();
        let gateway = engine.invocation_gateway(&io, &turn, Some(sink.as_ref()));
        let respond = async {
            while let Some(event) = rx.recv().await {
                match event {
                    UiEvent::ApprovalRequested { request_id, .. } => {
                        *decision.lock().unwrap() = Some((
                            request_id,
                            ToolVerdict {
                                approved: true,
                                ..Default::default()
                            },
                        ));
                    }
                    UiEvent::ToolCallFinished { .. } => break,
                    _ => {}
                }
            }
        };
        let (outcome, ()) = tokio::join!(
            gateway.invoke(request("staged-audit", "write", r#"{"value":1}"#)),
            respond
        );
        assert_eq!(
            prepared.load(Ordering::SeqCst),
            usize::from(fail != Some(InvocationPhase::Prepare))
        );
        assert_eq!(
            previewed.load(Ordering::SeqCst),
            usize::from(
                fail != Some(InvocationPhase::Prepare)
                    && fail != Some(InvocationPhase::StagedPreview)
            )
        );
        assert_eq!(
            committed.load(Ordering::SeqCst),
            usize::from(fail.is_none())
        );
        assert_eq!(outcome.audit.effects_may_have_occurred, fail.is_none());
        let records = sink.effects.lock().unwrap();
        for record in records.iter() {
            if record.phase == InvocationPhase::Commit {
                assert!(record.authorized_call.is_some());
                assert!(record.approval_request_id.is_some());
            } else {
                assert!(record.authorized_call.is_none());
                assert!(record.approval_request_id.is_none());
            }
        }
    }
}

#[tokio::test(start_paused = true)]
async fn pending_audit_operations_are_bounded_and_drop_their_futures() {
    for part in ["begin", "effect", "finish"] {
        for stop in ["cancel", "deadline", "io"] {
            let runs = Arc::new(AtomicUsize::new(0));
            let (engine, io, mut rx) = harness(vec![tool("write", false, runs.clone())]);
            let sink = EffectAudit {
                pending: Some(part),
                ..Default::default()
            };
            let turn = InvocationTurn::default();
            let mut gateway = engine.invocation_gateway(&io, &turn, Some(&sink));
            if stop == "deadline" {
                gateway = gateway
                    .with_deadline(tokio::time::Instant::now().into_std() + Duration::from_secs(1));
            }
            let stop_wait = async {
                sink.entered.notified().await;
                if stop == "cancel" {
                    io.cancel.store(true, Ordering::SeqCst);
                }
                // The start signal proves that the intended audit future is pending.
                tokio::time::advance(Duration::from_secs(6)).await;
            };
            let (outcome, ()) = tokio::time::timeout(Duration::from_secs(12), async {
                tokio::join!(
                    gateway.invoke(request("pending", "write", r#"{"value":1}"#)),
                    stop_wait
                )
            })
            .await
            .expect("audit must be bounded");
            assert!(outcome.audit_error.is_some(), "{part}/{stop}");
            assert_eq!(runs.load(Ordering::SeqCst), usize::from(part == "finish"));
            assert_eq!(outcome.audit.effects_may_have_occurred, part == "finish");
            if part != "finish" && stop == "cancel" {
                assert_eq!(outcome.status, InvocationStatus::Cancelled);
            }
            if part != "finish" && stop == "deadline" {
                assert_eq!(outcome.status, InvocationStatus::TimedOut);
            }
            assert_eq!(sink.dropped.load(Ordering::SeqCst), 1);
            assert_eq!(
                sink.finishes.lock().unwrap().len(),
                usize::from(part != "begin")
            );
            assert_eq!(terminals(&mut rx), 1);
        }
    }
}

#[tokio::test(start_paused = true)]
async fn parent_stop_drains_pending_inner_begin_effect_and_finish() {
    for part in ["begin", "effect", "finish"] {
        for stop in ["cancel", "deadline"] {
            let runs = Arc::new(AtomicUsize::new(0));
            let (engine, io, mut rx) = harness(vec![
                tool("probe", true, runs.clone()),
                agent_kernel::tools::batch::spec(),
            ]);
            let sink = EffectAudit {
                pending: Some(part),
                inner_only: true,
                ..Default::default()
            };
            let turn = InvocationTurn::default();
            let mut gateway = engine.invocation_gateway(&io, &turn, Some(&sink));
            if stop == "deadline" {
                gateway = gateway
                    .with_deadline(tokio::time::Instant::now().into_std() + Duration::from_secs(1));
            }
            let cancel = async {
                sink.entered.notified().await;
                if stop == "cancel" {
                    io.cancel.store(true, Ordering::SeqCst);
                } else {
                    tokio::time::advance(Duration::from_secs(1)).await;
                }
                tokio::time::advance(Duration::from_millis(100)).await;
                if part == "finish" {
                    tokio::time::advance(Duration::from_secs(6)).await;
                }
            };
            let (outcome, ()) = tokio::time::timeout(Duration::from_secs(12), async {
                tokio::join!(
                    gateway.invoke(batch_request(json!([{"tool":"probe","args":{"value":1}}]))),
                    cancel
                )
            })
            .await
            .unwrap_or_else(|_| panic!("nested audit must drain: {part}/{stop}"));
            assert_eq!(
                outcome.status,
                if stop == "cancel" {
                    InvocationStatus::Cancelled
                } else {
                    InvocationStatus::TimedOut
                }
            );
            assert_eq!(sink.dropped.load(Ordering::SeqCst), 1);
            assert_eq!(runs.load(Ordering::SeqCst), usize::from(part == "finish"));
            assert_eq!(terminals(&mut rx), 2);
            let records = sink.finishes.lock().unwrap();
            assert_eq!(
                records
                    .iter()
                    .filter(|r| r.metadata.parent_call_id.is_some())
                    .count(),
                usize::from(part != "begin")
            );
            assert_eq!(
                records
                    .iter()
                    .filter(|r| r.metadata.invocation_id == "outer")
                    .count(),
                1
            );
        }
    }
}

#[tokio::test]
async fn batch_inner_schema_hook_review_policy_and_audit_cannot_be_bypassed() {
    for case in ["schema", "review", "policy", "audit", "rename", "success"] {
        let runs = Arc::new(AtomicUsize::new(0));
        let policy_calls = Arc::new(AtomicUsize::new(0));
        let reviewed = Arc::new(AtomicUsize::new(0));
        let mut probe = tool("probe", true, runs.clone());
        let reviews = reviewed.clone();
        probe.review = Some(Arc::new(move |args, _| {
            assert_eq!(args["value"], 3);
            reviews.fetch_add(1, Ordering::SeqCst);
            Review {
                risk: if case == "review" {
                    CallRisk::Critical
                } else {
                    CallRisk::Normal
                },
                ..Default::default()
            }
        }));
        let (mut engine, io, mut rx) = harness(vec![
            agent_kernel::tools::batch::spec(),
            probe,
            tool("write", false, runs.clone()),
        ]);
        engine.set_permissions(PermissionGate::from_mode_str("dontAsk"));
        engine.set_policy(Box::new(NestedPolicy {
            calls: policy_calls.clone(),
            deny: case == "policy",
        }));
        let hooks = HookChain::new();
        hooks.replace(vec![Arc::new(NestedHook {
            value: if case == "schema" {
                json!("invalid")
            } else {
                json!(3)
            },
            rename: case == "rename",
        })]);
        engine.set_hooks(hooks);
        let turn = engine.begin_invocation_turn();
        let begins = Mutex::new(Vec::new());
        let finishes = Mutex::new(Vec::new());
        let sink = NestedAudit {
            begins: &begins,
            finishes: &finishes,
            fail_inner: case == "audit",
        };
        let outcome = engine
            .invocation_gateway(&io, &turn, Some(&sink))
            .invoke(batch_request(json!([{"tool":"probe","args":{"value":1}}])))
            .await;
        assert!(outcome.ok(), "{case}: {}", outcome.content);
        assert_eq!(
            runs.load(Ordering::SeqCst),
            usize::from(case == "success"),
            "{case}"
        );
        assert_eq!(
            reviewed.load(Ordering::SeqCst),
            usize::from(matches!(case, "review" | "policy" | "success")),
            "{case}"
        );
        assert_eq!(
            policy_calls.load(Ordering::SeqCst),
            usize::from(matches!(case, "policy" | "success")),
            "{case}"
        );
        let starts = begins.lock().unwrap();
        assert_eq!(starts.len(), 2, "{case}");
        let inner = starts
            .iter()
            .find(|r| r.metadata.parent_call_id.as_deref() == Some("outer"))
            .unwrap();
        assert_eq!(inner.metadata.execution_id.as_deref(), Some("execution"));
        assert_eq!(inner.metadata.agent_id.as_deref(), Some("agent"));
        assert_eq!(inner.metadata.session_id.as_deref(), Some("session"));
        let finishes = finishes.lock().unwrap();
        assert_eq!(
            finishes.len(),
            if case == "audit" { 1 } else { 2 },
            "{case}"
        );
        if case == "success" {
            let inner = finishes
                .iter()
                .find(|r| r.metadata.parent_call_id.as_deref() == Some("outer"))
                .unwrap();
            let authorized = inner.authorized_call.as_ref().unwrap();
            assert_eq!(
                serde_json::from_str::<Value>(&authorized.arguments).unwrap()["value"],
                3
            );
        }
        let mut terminals = std::collections::HashMap::new();
        while let Some(event) = rx.try_recv() {
            if let UiEvent::ToolCallFinished {
                call_id, parent, ..
            } = event
            {
                if call_id != "outer" {
                    assert_eq!(parent.as_deref(), Some("batch_execute"));
                }
                *terminals.entry(call_id).or_insert(0) += 1;
            }
        }
        assert_eq!(terminals.len(), 2, "{case}");
        assert!(terminals.values().all(|count| *count == 1));
    }
}

#[tokio::test]
async fn batch_inner_parallelism_partial_failure_and_missing_gateway_are_preserved() {
    let runs = Arc::new(AtomicUsize::new(0));
    let barrier = Arc::new(tokio::sync::Barrier::new(2));
    let mut probe = tool("probe", true, runs.clone());
    let count = runs.clone();
    probe.exec = Arc::new(move |_, _| {
        let barrier = barrier.clone();
        let count = count.clone();
        async move {
            count.fetch_add(1, Ordering::SeqCst);
            barrier.wait().await;
            Ok(ToolResult::text("parallel"))
        }
        .boxed()
    });
    let (mut engine, io, mut rx) = harness(vec![agent_kernel::tools::batch::spec(), probe]);
    let turn = engine.begin_invocation_turn();
    let outcome = tokio::time::timeout(
        Duration::from_secs(2),
        engine
            .invocation_gateway(&io, &turn, None)
            .invoke(batch_request(json!([
                {"tool":"probe","args":{"value":1}},
                {"tool":"probe","args":{"value":2}},
                {"tool":"probe","args":{"value":"invalid"}}
            ]))),
    )
    .await
    .expect("inner readonly calls must overlap");
    assert!(outcome.ok());
    assert_eq!(runs.load(Ordering::SeqCst), 2);
    assert!(outcome.content.contains("invalid tool arguments"));
    assert_eq!(terminals(&mut rx), 4);
    let batch = agent_kernel::tools::batch::spec();
    let error = (batch.exec)(
        json!({"calls":[{"tool":"probe","args":{"value":1}}]}),
        Arc::new(ToolCtx::new()),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(error.contains("requires an invocation gateway"), "{error}");
}

#[tokio::test]
async fn batch_cancellation_drains_inner_audit_and_emits_one_terminal_per_call() {
    struct Dropped(Arc<AtomicUsize>);
    impl Drop for Dropped {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    let started = Arc::new(AtomicUsize::new(0));
    let dropped = Arc::new(AtomicUsize::new(0));
    let entered = Arc::new(tokio::sync::Notify::new());
    let mut probe = tool("probe", true, started.clone());
    let calls = started.clone();
    let ended = dropped.clone();
    let notify = entered.clone();
    probe.exec = Arc::new(move |_, _| {
        let calls = calls.clone();
        let ended = ended.clone();
        let notify = notify.clone();
        async move {
            let _guard = Dropped(ended);
            if calls.fetch_add(1, Ordering::SeqCst) + 1 == 2 {
                notify.notify_one();
            }
            futures::future::pending::<Result<ToolResult, ToolError>>().await
        }
        .boxed()
    });
    let (mut engine, io, mut rx) = harness(vec![agent_kernel::tools::batch::spec(), probe]);
    let turn = engine.begin_invocation_turn();
    let begins = Mutex::new(Vec::new());
    let finishes = Mutex::new(Vec::new());
    let sink = NestedAudit {
        begins: &begins,
        finishes: &finishes,
        fail_inner: false,
    };
    let gateway = engine.invocation_gateway(&io, &turn, Some(&sink));
    let cancel = async {
        entered.notified().await;
        io.cancel.store(true, Ordering::SeqCst);
    };
    let (outcome, ()) = tokio::time::timeout(Duration::from_secs(2), async {
        tokio::join!(
            gateway.invoke(batch_request(json!([
                {"tool":"probe","args":{"value":1}}, {"tool":"probe","args":{"value":2}}
            ]))),
            cancel
        )
    })
    .await
    .expect("nested cancellation must drain");
    assert_eq!(outcome.status, InvocationStatus::Cancelled);
    assert_eq!(dropped.load(Ordering::SeqCst), 2);
    assert_eq!(finishes.lock().unwrap().len(), 3);
    assert_eq!(terminals(&mut rx), 3);
}

#[tokio::test]
async fn nested_slots_close_with_parent_and_child_dispatch_uses_child_permissions() {
    use agent_kernel::tools::dispatch::NestedDispatcher;
    let captured = Arc::new(Mutex::new(None::<NestedDispatcher>));
    let mut capture = tool("capture", true, Arc::new(AtomicUsize::new(0)));
    let destination = captured.clone();
    capture.exec = Arc::new(move |_, ctx| {
        let destination = destination.clone();
        async move {
            assert!(ctx.ext.inheritable().get::<NestedDispatcher>().is_none());
            *destination.lock().unwrap() =
                Some((*ctx.ext.get::<NestedDispatcher>().unwrap()).clone());
            Ok(ToolResult::text("captured"))
        }
        .boxed()
    });
    let (mut parent, io, _) = harness(vec![capture]);
    let turn = parent.begin_invocation_turn();
    assert!(parent
        .invocation_gateway(&io, &turn, None)
        .invoke(request("capture", "capture", r#"{"value":1}"#))
        .await
        .ok());
    let stale = captured.lock().unwrap().take().unwrap();
    let late = stale
        .invoke(
            ToolCall {
                id: "late".into(),
                name: "capture".into(),
                arguments: r#"{"value":1}"#.into(),
            },
            Arc::new(AtomicBool::new(false)),
            std::time::Instant::now() + Duration::from_secs(1),
            None,
        )
        .await;
    assert!(late.err().unwrap().contains("closed"));
    let runs = Arc::new(AtomicUsize::new(0));
    let mut probe = tool("probe", true, runs.clone());
    probe.review = Some(Arc::new(|_, _| Review {
        risk: CallRisk::Critical,
        ..Default::default()
    }));
    let mut ctx = ToolCtx::new();
    ctx.depth = 1;
    ctx.ext.insert(Arc::new(stale));
    let (mut child, io, _) = harness_with_ctx(vec![agent_kernel::tools::batch::spec(), probe], ctx);
    child.set_permissions(PermissionGate::child_of(
        &PermissionGate::from_mode_str("bypass"),
        false,
    ));
    let turn = child.begin_invocation_turn();
    let outcome = child
        .invocation_gateway(&io, &turn, None)
        .invoke(batch_request(json!([{"tool":"probe","args":{"value":1}}])))
        .await;
    assert!(outcome.ok(), "{}", outcome.content);
    assert!(
        outcome.content.contains("requires human approval"),
        "{}",
        outcome.content
    );
    assert_eq!(runs.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn expired_deadlines_never_dispatch() {
    let runs = Arc::new(AtomicUsize::new(0));
    let (mut engine, io, mut rx) = harness(vec![tool("probe", true, runs.clone())]);
    let turn = engine.begin_invocation_turn();
    let outcome = engine
        .invocation_gateway(&io, &turn, None)
        .with_deadline(std::time::Instant::now())
        .invoke(request("deadline", "probe", r#"{"value":1}"#))
        .await;
    assert_eq!(outcome.status, InvocationStatus::TimedOut);
    assert!(outcome.content.contains("deadline"));
    assert_eq!(runs.load(Ordering::SeqCst), 0);
    assert_eq!(terminals(&mut rx), 1);
}
