use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex,
};
use std::time::Duration;

use agent_ipc::{
    code_mode::{CodeModeEvent, CodeModeIdentity, CodeModePhase},
    AgentState, UiEvent,
};
use agent_kernel::{
    code_mode::{
        control_schema, CodeModeContext, CodeModeHandler, CodeModeOutcome, CodeModeRequest,
        EngineStrategy, MAX_MODEL_RESULT_BYTES,
    },
    engine::{Engine, EngineIo},
    hooks::{AgentHook, HookChain},
    invocation::{InvocationRequest, InvocationStopReason},
    invocation_scope::{is_invocation_scoped, InvocationIdentity, InvocationScope},
    permissions::PermissionGate,
    tools::registry::{ToolClass, ToolCtx, ToolRegistry, ToolResult, ToolSpec, ToolVerdict},
};
use agent_llm_core::{types::ToolCall, ChatMessage, StreamChunk};
use agent_testkit::ScriptedProvider;
use serde_json::{json, Value};

fn done() -> StreamChunk {
    StreamChunk::Done {
        prompt_tokens: Some(1),
        completion_tokens: Some(1),
        cached_tokens: None,
    }
}

fn call(slot: usize, name: &str, args: impl Into<String>) -> StreamChunk {
    StreamChunk::ToolCallDelta {
        slot,
        id: Some(format!("outer_{slot}")),
        name: Some(name.into()),
        args_delta: args.into(),
    }
}

fn program(slot: usize, attempts: usize) -> StreamChunk {
    call(
        slot,
        "execute_code",
        json!({"code":"return params", "params":{"attempts":attempts}}).to_string(),
    )
}

#[derive(Default)]
struct FakeHost {
    remaining: Mutex<Vec<usize>>,
    requests: Mutex<Vec<u64>>,
    stop: Option<InvocationStopReason>,
    projection: Option<String>,
    deadline: bool,
    cancel: bool,
    reported_extra: usize,
    scoped_hooks: Option<HookChain>,
}

#[async_trait::async_trait]
impl CodeModeHandler for FakeHost {
    async fn execute(&self, request: CodeModeRequest, ctx: CodeModeContext<'_>) -> CodeModeOutcome {
        self.remaining
            .lock()
            .unwrap()
            .push(ctx.remaining_tool_calls);
        self.requests.lock().unwrap().push(ctx.request_id);
        let identity = CodeModeIdentity {
            execution_id: format!("execution_{}", ctx.request_id),
            call_id: ctx.control_call.id.clone(),
            control_request_id: ctx.request_id,
            parent_call_id: ctx.parent_identity.and_then(|id| id.parent_call_id.clone()),
            session_id: "session".into(),
            agent_id: "agent".into(),
            trace_id: None,
        };
        let emit = |phase, summary: &str, ok| {
            ctx.io
                .ui_tx
                .send(
                    CodeModeEvent {
                        identity: identity.clone(),
                        phase,
                        request_id: None,
                        summary: summary.into(),
                        console: None,
                        ok,
                    }
                    .into_ui_event()
                    .unwrap(),
                )
                .unwrap();
        };
        emit(CodeModePhase::Started, "started", None);
        let scope = self.scoped_hooks.as_ref().map(|hooks| {
            InvocationScope::new(
                ctx.tool_ctx.clone(),
                InvocationIdentity {
                    execution_id: identity.execution_id.clone(),
                    agent_id: identity.agent_id.clone(),
                    session_id: identity.session_id.clone(),
                    trace_id: None,
                    parent_call_id: Some(ctx.control_call.id.clone()),
                },
                hooks.clone(),
                None,
                None,
                ctx.parent_deadline,
            )
        });
        let requested = request.params["attempts"].as_u64().unwrap_or(0) as usize;
        let mut attempted_calls = 0;
        let mut stop_reason = self.stop.clone();
        let mut is_error = stop_reason.is_some();
        // This fake host reserves before each dispatch. Production concurrent
        // dispatches require one execution-owned reservation ledger.
        for index in 0..requested.min(ctx.remaining_tool_calls) {
            let mut gateway = ctx.engine.invocation_gateway(ctx.io, ctx.turn, None);
            if let Some(scope) = &scope {
                gateway = gateway.with_scope(scope);
            }
            if self.deadline {
                gateway = gateway.with_deadline(std::time::Instant::now());
            }
            if self.cancel {
                ctx.io.cancel.store(true, Ordering::Release);
            }
            let outcome = gateway
                .invoke(InvocationRequest::new(ToolCall {
                    id: format!("inner_{}_{}", ctx.request_id, index),
                    name: "probe".into(),
                    arguments: "{}".into(),
                }))
                .await;
            attempted_calls += usize::from(outcome.attempted);
            is_error |= !outcome.ok();
            if matches!(
                outcome.audit.stop_reason,
                Some(
                    InvocationStopReason::UserRejected { .. }
                        | InvocationStopReason::InvocationDeadline
                        | InvocationStopReason::Cancelled
                )
            ) {
                stop_reason = outcome.audit.stop_reason;
                break;
            }
        }
        if let Some(scope) = scope {
            scope.close().await.unwrap();
        }
        emit(CodeModePhase::Progress, "completion drain finished", None);
        // Fake artifact and terminal audit acknowledgements precede Finished.
        emit(
            CodeModePhase::Finished,
            "artifacts and audit finalized",
            Some(!is_error),
        );
        CodeModeOutcome {
            model_result: self
                .projection
                .clone()
                .unwrap_or_else(|| json!({"attempted":attempted_calls}).to_string()),
            is_error,
            attempted_calls: attempted_calls.saturating_add(self.reported_extra),
            stop_reason,
        }
    }
}

struct Harness {
    engine: Engine,
    io: EngineIo,
    rx: agent_kernel::channels::UiReceiver,
    provider: Arc<ScriptedProvider>,
    capabilities: Arc<AtomicUsize>,
    host: Arc<FakeHost>,
    history: Vec<ChatMessage>,
}

impl Harness {
    fn new(host: FakeHost, scripts: Vec<Vec<StreamChunk>>) -> Self {
        let provider = Arc::new(ScriptedProvider::new());
        for mut script in scripts {
            script.push(done());
            provider.push_script(script);
        }
        provider.script_text("final");
        let capabilities = Arc::new(AtomicUsize::new(0));
        let mut registry = ToolRegistry::new();
        for name in ["probe", "execute_code"] {
            let count = capabilities.clone();
            registry.register(ToolSpec {
                name,
                schema: json!({"parameters":{"type":"object"}}),
                readonly: false,
                class: ToolClass::WorkspaceMutation,
                network: false,
                staged: false,
                review: None,
                cancel_aware: false,
                exec: Arc::new(move |_, _| {
                    count.fetch_add(1, Ordering::Relaxed);
                    Box::pin(async { Ok(ToolResult::text("RAW_CAPABILITY_OUTPUT")) })
                }),
            });
        }
        let (ui_tx, rx) = agent_kernel::channels::UiSink::channel();
        let mut tool_ctx = ToolCtx::new();
        tool_ctx.install_ui(ui_tx.clone());
        let mut engine = Engine::new(
            provider.clone(),
            Arc::new(registry),
            Arc::new(tool_ctx),
            "m",
            0.0,
        );
        assert_eq!(engine.strategy(), EngineStrategy::ReAct);
        engine.set_strategy(EngineStrategy::CodeMode);
        let host = Arc::new(host);
        engine.set_code_mode_handler(host.clone());
        engine.set_permissions(PermissionGate::from_mode_str("bypassPermissions"));
        Self {
            engine,
            io: EngineIo {
                ui_tx,
                steer_rx: tokio::sync::mpsc::channel(4).1,
                notice_rx: tokio::sync::mpsc::channel(4).1,
                cancel: Arc::new(false.into()),
            },
            rx,
            provider,
            capabilities,
            host,
            history: Vec::new(),
        }
    }

    async fn run(&mut self) -> Result<agent_kernel::engine::TurnOutcome, String> {
        tokio::time::timeout(
            Duration::from_secs(5),
            self.engine
                .run_turn(&mut self.io, &mut self.history, "hi".into()),
        )
        .await
        .expect("Code Mode must finish")
    }

    fn events(&mut self) -> Vec<UiEvent> {
        let mut events = Vec::new();
        while let Some(event) = self.rx.try_recv() {
            events.push(event);
        }
        events
    }
}

fn assert_outer(events: &[UiEvent], ids: &[&str]) {
    let mut previous_queue = 0;
    for id in ids {
        let positions = |kind: u8| {
            events
                .iter()
                .enumerate()
                .filter_map(move |(index, event)| {
                    let matched = match event {
                        UiEvent::ToolCallQueued { call_id, .. } => kind == 0 && call_id == id,
                        UiEvent::ToolCallStarted { call_id, .. } => kind == 1 && call_id == id,
                        UiEvent::ToolCallFinished { call_id, .. } => kind == 2 && call_id == id,
                        _ => false,
                    };
                    matched.then_some(index)
                })
                .collect::<Vec<_>>()
        };
        let (queued, started, finished) = (positions(0), positions(1), positions(2));
        assert_eq!(queued.len(), 1, "{id}: queued");
        assert_eq!(started.len(), 1, "{id}: started");
        assert_eq!(finished.len(), 1, "{id}: finished");
        assert!(queued[0] >= previous_queue);
        assert!(queued[0] < started[0] && started[0] < finished[0]);
        previous_queue = queued[0];
    }
}

#[tokio::test]
async fn control_uses_borrowed_host_gateway_and_only_projection_enters_history() {
    let mut h = Harness::new(FakeHost::default(), vec![vec![program(0, 1)]]);
    assert_eq!(h.run().await.unwrap().tool_calls_run, 1);
    assert_eq!(h.capabilities.load(Ordering::Relaxed), 1);
    assert_eq!(h.provider.tool_names.lock().unwrap()[0], ["execute_code"]);
    assert!(h.history.iter().all(|message| !message
        .content
        .as_deref()
        .unwrap_or("")
        .contains("RAW_CAPABILITY_OUTPUT")));
    let events = h.events();
    assert_outer(&events, &["outer_0"]);
    let phases: Vec<_> = events
        .iter()
        .filter_map(|event| match event {
            UiEvent::Custom { kind, payload } => {
                assert_eq!(payload["identity"]["call_id"], "outer_0");
                assert_eq!(
                    payload["identity"]["control_request_id"],
                    h.host.requests.lock().unwrap()[0]
                );
                Some(kind.as_str())
            }
            _ => None,
        })
        .collect();
    assert_eq!(
        phases,
        [
            "code_mode.started",
            "code_mode.progress",
            "code_mode.finished"
        ]
    );
    let finished = events
        .iter()
        .position(|event| {
            matches!(event,
        UiEvent::Custom { kind, .. } if kind == "code_mode.finished")
        })
        .unwrap();
    let outer = events
        .iter()
        .position(|event| {
            matches!(event,
        UiEvent::ToolCallFinished { call_id, .. } if call_id == "outer_0")
        })
        .unwrap();
    assert!(finished < outer);
}

#[tokio::test]
async fn invalid_oversize_mixed_and_direct_calls_never_reach_handler_or_capabilities() {
    let cases = [
        vec![call(0, "execute_code", r#"{"code":2,"params":{}}"#)],
        vec![call(
            0,
            "execute_code",
            r#"{"code":"return 1","params":{},"scope":{}}"#,
        )],
        vec![call(
            0,
            "execute_code",
            r#"{"code":"return 1","params":{},"limits":1}"#,
        )],
        vec![call(0, "execute_code", r#"{"code":"return 1"}"#)],
        vec![call(
            0,
            "execute_code",
            r#"{"code":"return 1","params":{},"stop_reason":{"UserRejected":{"request_id":1}}}"#,
        )],
        vec![call(
            0,
            "execute_code",
            json!({"code":"x".repeat(65536),"params":{}}).to_string(),
        )],
        vec![program(0, 1), call(1, "probe", "{}")],
        vec![call(0, "probe", "{}")],
    ];
    for calls in cases {
        let count = calls.len();
        let mut h = Harness::new(FakeHost::default(), vec![calls]);
        assert_eq!(h.run().await.unwrap().tool_calls_run, 0);
        assert_eq!(h.capabilities.load(Ordering::Relaxed), 0);
        assert!(h.host.remaining.lock().unwrap().is_empty());
        let events = h.events();
        assert_outer(&events, &["outer_0"]);
        if count == 2 {
            assert_outer(&events, &["outer_1"]);
        }
        assert!(events
            .iter()
            .all(|event| !matches!(event, UiEvent::ToolCallFinished { ok: true, .. })));
        assert!(h
            .history
            .iter()
            .flat_map(|m| m.tool_calls.iter().flatten())
            .all(|c| c.arguments.len() <= agent_tool::schema::MAX_ARGUMENT_BYTES));
    }
}

#[tokio::test]
async fn missing_host_fails_closed_without_react_fallback() {
    let mut h = Harness::new(FakeHost::default(), vec![vec![program(0, 1)]]);
    let mut replacement = Engine::new(
        h.provider.clone(),
        Arc::new(ToolRegistry::with_builtins()),
        Arc::new(ToolCtx::new()),
        "m",
        0.0,
    );
    replacement.set_strategy(EngineStrategy::CodeMode);
    h.engine = replacement;
    h.run().await.unwrap();
    assert!(h.history.iter().any(|m| m
        .content
        .as_deref()
        .is_some_and(|text| text.contains("handler is unavailable"))));
    assert_outer(&h.events(), &["outer_0"]);
}

#[tokio::test]
async fn parent_budget_is_shared_across_control_calls_and_forces_final() {
    let mut h = Harness::new(
        FakeHost::default(),
        vec![vec![program(0, 2), program(1, 9), program(2, 1)]],
    );
    h.engine.set_max_tool_rounds(3);
    h.engine.set_max_code_mode_tool_calls(Some(3));
    assert_eq!(h.run().await.unwrap().tool_calls_run, 3);
    assert_eq!(*h.host.remaining.lock().unwrap(), [3, 1]);
    assert_eq!(h.capabilities.load(Ordering::Relaxed), 3);
    assert!(!h.provider.calls.lock().unwrap()[1].2);
    assert_outer(&h.events(), &["outer_0", "outer_1", "outer_2"]);
}

#[tokio::test]
async fn zero_attempt_programs_still_consume_control_rounds() {
    let mut h = Harness::new(
        FakeHost::default(),
        vec![vec![program(0, 0)], vec![program(0, 0)]],
    );
    h.engine.set_max_tool_rounds(2);
    assert_eq!(h.run().await.unwrap().tool_calls_run, 0);
    assert_eq!(*h.host.remaining.lock().unwrap(), [usize::MAX, usize::MAX]);
    assert!(!h.provider.calls.lock().unwrap()[2].2);
}

#[tokio::test]
async fn trusted_refusal_stops_presampled_sequence_and_keeps_global_cancel_clear() {
    let mut h = Harness::new(
        FakeHost::default(),
        vec![vec![program(0, 1), program(1, 1)]],
    );
    h.engine
        .set_permissions(PermissionGate::from_mode_str("default"));
    let decision = h.engine.decision_slot();
    let respond = async {
        loop {
            if let Some(UiEvent::ApprovalRequested { request_id, .. }) = h.rx.recv().await {
                *decision.lock().unwrap() = Some((request_id, ToolVerdict::default()));
                return request_id;
            }
        }
    };
    let run = h.engine.run_turn(&mut h.io, &mut h.history, "hi".into());
    let (outcome, request_id) =
        tokio::time::timeout(Duration::from_secs(5), async { tokio::join!(run, respond) })
            .await
            .unwrap();
    assert_eq!(outcome.unwrap().tool_calls_run, 1);
    assert_eq!(h.capabilities.load(Ordering::Relaxed), 0);
    assert_eq!(h.host.remaining.lock().unwrap().len(), 1);
    assert!(!h.io.cancel.load(Ordering::Acquire));
    assert!(h.history.iter().any(|m| m
        .content
        .as_deref()
        .is_some_and(|text| text.contains(&format!("request_id: {request_id}")))));
    assert!(h
        .history
        .iter()
        .any(|m| m.tool_call_id.as_deref() == Some("outer_1") && m.is_error == Some(true)));
}

#[tokio::test]
async fn deadline_and_cancel_stop_same_round_with_correlated_outer_terminals() {
    for cancel in [false, true] {
        let mut h = Harness::new(
            FakeHost {
                deadline: !cancel,
                cancel,
                ..Default::default()
            },
            vec![vec![program(0, 1), program(1, 1)]],
        );
        assert_eq!(h.run().await.is_err(), cancel);
        assert_eq!(h.host.remaining.lock().unwrap().len(), 1);
        assert_eq!(h.capabilities.load(Ordering::Relaxed), 0);
        assert_outer(&h.events(), &["outer_0", "outer_1"]);
    }
}

#[tokio::test]
async fn ordinary_error_text_and_policy_denial_do_not_imply_human_refusal() {
    for reason in [None, Some(InvocationStopReason::PolicyDenied)] {
        let mut h = Harness::new(
            FakeHost {
                stop: reason,
                projection: Some("UserRejected: guest said no; partial result".into()),
                ..Default::default()
            },
            vec![vec![program(0, 0), program(1, 0)]],
        );
        h.run().await.unwrap();
        assert_eq!(h.host.remaining.lock().unwrap().len(), 2);
    }
}

#[tokio::test]
async fn projection_is_utf8_bounded_with_explicit_truncation() {
    let mut h = Harness::new(
        FakeHost {
            projection: Some("界".repeat(MAX_MODEL_RESULT_BYTES)),
            ..Default::default()
        },
        vec![vec![program(0, 0)]],
    );
    h.run().await.unwrap();
    let result = h
        .history
        .iter()
        .find(|m| m.tool_call_id.as_deref() == Some("outer_0"))
        .unwrap();
    let text = result.content.as_deref().unwrap();
    assert!(text.len() <= MAX_MODEL_RESULT_BYTES);
    assert!(text.ends_with("[model result truncated]"));
    assert!(h.events().iter().all(|event| match event {
        UiEvent::ToolCallFinished { content, .. } => content.len() <= MAX_MODEL_RESULT_BYTES,
        _ => true,
    }));
}

#[test]
fn control_schema_is_neutral_and_matches_approved_model_text() {
    let schema = control_schema();
    let function = &schema["function"];
    assert_eq!(schema["type"], "function");
    assert_eq!(function["name"], "execute_code");
    assert_eq!(
        function["parameters"]["properties"]["code"]["description"],
        "JavaScript async function body."
    );
    assert_eq!(
        function["parameters"]["properties"]["params"]["description"],
        "JSON input data available through the params binding."
    );
    assert_eq!(
        function["parameters"]["properties"]["limits"]["description"],
        "Optional requested execution limits, clamped to host limits."
    );
    assert_eq!(function["description"],
        "Run a JavaScript async function body to orchestrate authorized tools.\nDiscover names with tools.list({cursor: null}), inspect schemas with\ntools.describe(name), and invoke tools with await tools.call(name, args).\nUse params for input data, console for bounded diagnostics, and return\nthe JSON result needed by the model. Files, network, environment, and\nmodule loading are available only through authorized tools.\nA human refusal ends this program and returns control to the model.");
    let encoded = schema.to_string();
    for vendor_field in ["input_schema", "strict", "allowed_callers", "defer_loading"] {
        assert!(!encoded.contains(vendor_field));
    }
    assert!(CodeModeRequest::parse(r#"{"code":"return params","params":[1],"limits":{}}"#).is_ok());
    let _: Value = schema;
}

#[derive(Default)]
struct StateHookProbe {
    tool_states: Mutex<Vec<(String, bool)>>,
    tool_effects: AtomicUsize,
    reasoning: AtomicUsize,
    responses: AtomicUsize,
    finished: tokio::sync::Notify,
}

#[async_trait::async_trait]
impl AgentHook for StateHookProbe {
    fn id(&self) -> &str {
        "state-hook-probe"
    }

    async fn before_tool_execute(&self, _: &ToolCtx, _: &mut ToolCall) -> anyhow::Result<()> {
        self.tool_effects.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    async fn on_state_transition(&self, ctx: &ToolCtx, _: &AgentState, new: &AgentState) {
        if *new == AgentState::Reasoning {
            self.reasoning.fetch_add(1, Ordering::Relaxed);
        }
        if let AgentState::ExecutingTool { tool_name } = new {
            self.tool_states
                .lock()
                .unwrap()
                .push((tool_name.clone(), is_invocation_scoped(ctx)));
        }
        if *new == AgentState::Finished {
            self.finished.notify_one();
        }
    }

    async fn on_response(&self, _: &mut String) {
        self.responses.fetch_add(1, Ordering::Relaxed);
    }
}

#[tokio::test]
async fn outer_control_state_is_ui_only_before_host_scope_exists() {
    let mut h = Harness::new(FakeHost::default(), vec![vec![program(0, 0)]]);
    let probe = Arc::new(StateHookProbe::default());
    let mut hooks = HookChain::new();
    hooks.register(probe.clone());
    h.engine.set_hooks(hooks);
    h.run().await.unwrap();
    // The zero-dispatch fake host creates no execution scope. Finished drains
    // the root transition pump so a queued control effect cannot escape this check.
    tokio::time::timeout(Duration::from_secs(1), probe.finished.notified())
        .await
        .unwrap();
    assert!(
        probe.tool_states.lock().unwrap().is_empty(),
        "outer control must not execute root state hooks: {:?}",
        probe.tool_states.lock().unwrap()
    );
    assert_eq!(probe.tool_effects.load(Ordering::Relaxed), 0);
    assert!(probe.reasoning.load(Ordering::Relaxed) > 0);
    assert_eq!(probe.responses.load(Ordering::Relaxed), 1);
    let events = h.events();
    assert_outer(&events, &["outer_0"]);
    assert!(events.iter().any(|event| matches!(
        event,
        UiEvent::StateChanged(AgentState::ExecutingTool { tool_name })
            if tool_name == "execute_code"
    )));
}

#[tokio::test]
async fn react_tool_state_and_lifecycle_hooks_keep_root_context() {
    let mut h = Harness::new(FakeHost::default(), vec![vec![program(0, 0)]]);
    h.engine.set_strategy(EngineStrategy::ReAct);
    let probe = Arc::new(StateHookProbe::default());
    let mut hooks = HookChain::new();
    hooks.register(probe.clone());
    h.engine.set_hooks(hooks);
    h.run().await.unwrap();
    tokio::time::timeout(Duration::from_secs(1), probe.finished.notified())
        .await
        .unwrap();
    assert_eq!(
        *probe.tool_states.lock().unwrap(),
        [("execute_code".into(), false)]
    );
    assert_eq!(probe.tool_effects.load(Ordering::Relaxed), 1);
    assert!(probe.reasoning.load(Ordering::Relaxed) > 0);
    assert_eq!(probe.responses.load(Ordering::Relaxed), 1);
    assert_eq!(h.capabilities.load(Ordering::Relaxed), 1);
    assert!(h.host.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn code_mode_inner_state_and_tool_hooks_use_host_scope() {
    let root = Arc::new(StateHookProbe::default());
    let scoped = Arc::new(StateHookProbe::default());
    let mut scoped_hooks = HookChain::new();
    scoped_hooks.register(scoped.clone());
    let mut h = Harness::new(
        FakeHost {
            scoped_hooks: Some(scoped_hooks),
            ..Default::default()
        },
        vec![vec![program(0, 1)]],
    );
    let mut root_hooks = HookChain::new();
    root_hooks.register(root.clone());
    h.engine.set_hooks(root_hooks);
    assert_eq!(h.run().await.unwrap().tool_calls_run, 1);
    tokio::time::timeout(Duration::from_secs(1), root.finished.notified())
        .await
        .unwrap();
    assert!(root.tool_states.lock().unwrap().is_empty());
    assert_eq!(root.tool_effects.load(Ordering::Relaxed), 0);
    assert!(root.reasoning.load(Ordering::Relaxed) > 0);
    assert_eq!(root.responses.load(Ordering::Relaxed), 1);
    assert_eq!(
        *scoped.tool_states.lock().unwrap(),
        [("probe".into(), true)]
    );
    assert_eq!(scoped.tool_effects.load(Ordering::Relaxed), 1);
    assert_eq!(h.capabilities.load(Ordering::Relaxed), 1);
    assert_outer(&h.events(), &["outer_0"]);
}

#[tokio::test]
async fn tool_round_limit_does_not_become_an_inner_call_limit() {
    let mut h = Harness::new(FakeHost::default(), vec![vec![program(0, 3)]]);
    h.engine.set_max_tool_rounds(2);
    assert_eq!(h.run().await.unwrap().tool_calls_run, 3);
    assert_eq!(*h.host.remaining.lock().unwrap(), [usize::MAX]);
    assert!(h.provider.calls.lock().unwrap()[1].2);
}

#[tokio::test]
async fn call_cap_is_shared_across_assistant_rounds() {
    let mut h = Harness::new(
        FakeHost::default(),
        vec![vec![program(0, 2)], vec![program(0, 8)]],
    );
    h.engine.set_max_code_mode_tool_calls(Some(3));
    assert_eq!(h.run().await.unwrap().tool_calls_run, 3);
    assert_eq!(*h.host.remaining.lock().unwrap(), [3, 1]);
    assert!(!h.provider.calls.lock().unwrap()[2].2);
}

#[tokio::test]
async fn host_overreport_is_a_contract_error_and_stops_later_programs() {
    let mut h = Harness::new(
        FakeHost {
            reported_extra: 4,
            ..Default::default()
        },
        vec![vec![program(0, 0), program(1, 1)]],
    );
    h.engine.set_max_code_mode_tool_calls(Some(3));
    assert_eq!(h.run().await.unwrap().tool_calls_run, 4);
    assert_eq!(h.host.remaining.lock().unwrap().len(), 1);
    assert_eq!(h.capabilities.load(Ordering::Relaxed), 0);
    assert!(!h.provider.calls.lock().unwrap()[1].2);
    assert!(h.history.iter().any(|m| m
        .content
        .as_deref()
        .is_some_and(|text| text.contains("effects may have occurred"))));
}
