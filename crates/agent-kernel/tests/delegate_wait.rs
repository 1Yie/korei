use std::sync::Arc;
use std::time::Duration;

use agent_kernel::tools::delegate::{ChildRoute, SubagentSpawner};
use agent_kernel::tools::{ToolCtx, ToolRegistry};
use agent_testkit::ScriptedProvider;

fn context(
    provider: Arc<ScriptedProvider>,
    session: i64,
) -> (Arc<ToolCtx>, tokio::sync::mpsc::Receiver<String>) {
    let children = Arc::new(ToolRegistry::child_builtins());
    let route = Arc::new(std::sync::RwLock::new(ChildRoute {
        provider,
        model: "test".into(),
        temperature: 0.0,
        context_window: 256_000,
        thinking_level: None,
        thinking_level_map: None,
        model_params: Default::default(),
        model_input: Vec::new(),
        models: Default::default(),
    }));
    let (sender, notices) = tokio::sync::mpsc::channel(32);
    let mut ctx = ToolCtx::new()
        .with_session(session)
        .with_cancel(Arc::new(std::sync::atomic::AtomicBool::new(false)));
    ctx.install_subagent(SubagentSpawner::new(route, children.clone(), children));
    agent_kernel::tools::delegate::install_child_notifier(&mut ctx, move |text| {
        sender.try_send(text).unwrap();
    });
    (Arc::new(ctx), notices)
}

fn child_id(text: &str) -> u64 {
    text.split("child_id ")
        .nth(1)
        .unwrap()
        .split(')')
        .next()
        .unwrap()
        .parse()
        .unwrap()
}

#[tokio::test]
async fn parent_wait_stays_pending_until_the_child_report_is_delivered() {
    let registry = ToolRegistry::with_builtins();
    let wait = registry
        .spec("wait_child")
        .expect("parents need a real child-wait action");
    let provider = Arc::new(ScriptedProvider::new());
    let gate = Arc::new(tokio::sync::Notify::new());
    *provider.gate.lock().unwrap() = Some(gate.clone());
    provider.script_text("REVIEW-RESULT");
    let (ctx, mut notices) = context(provider, 11);
    let started = (registry.spec("delegate").unwrap().exec)(
        serde_json::json!({"task": "review independently", "background": true}),
        ctx.clone(),
    )
    .await
    .unwrap();
    let id = child_id(&started.content);
    let waiting = (wait.exec)(serde_json::json!({"child_id": id}), ctx);
    tokio::pin!(waiting);

    assert!(
        tokio::time::timeout(Duration::from_millis(30), &mut waiting)
            .await
            .is_err(),
        "the parent must stay parked while its child has no report"
    );
    assert!(notices.try_recv().is_err());
    gate.notify_one();
    let out = tokio::time::timeout(Duration::from_secs(3), waiting)
        .await
        .expect("the child's report releases the parent")
        .expect("waiting succeeds");
    assert!(out.content.contains(&id.to_string()), "{}", out.content);
    let report = notices.try_recv().expect("the report precedes the wake");
    assert!(report.contains("REVIEW-RESULT"), "{report}");
    assert!(notices.try_recv().is_err(), "the report is delivered once");
}

#[tokio::test]
async fn an_already_delivered_report_is_preserved_when_the_parent_starts_waiting() {
    let registry = ToolRegistry::with_builtins();
    let provider = Arc::new(ScriptedProvider::new());
    provider.script_text("EARLY-REPORT");
    let (ctx, mut notices) = context(provider, 12);
    let started = (registry.spec("delegate").unwrap().exec)(
        serde_json::json!({"task": "finish promptly", "background": true}),
        ctx.clone(),
    )
    .await
    .unwrap();
    let id = child_id(&started.content);
    let report = tokio::time::timeout(Duration::from_secs(3), notices.recv())
        .await
        .unwrap()
        .unwrap();
    assert!(report.contains("EARLY-REPORT"), "{report}");
    let out = tokio::time::timeout(
        Duration::from_secs(1),
        (registry.spec("wait_child").unwrap().exec)(serde_json::json!({"child_id": id}), ctx),
    )
    .await
    .expect("a report delivered before the wait cannot strand the parent")
    .unwrap();
    assert!(out.content.contains(&id.to_string()), "{}", out.content);
    assert!(!out.content.contains("EARLY-REPORT"));
    assert!(
        notices.try_recv().is_err(),
        "waiting must not repeat the report"
    );
}

fn ask_script(question: &str) -> Vec<agent_llm_core::types::StreamChunk> {
    use agent_llm_core::types::StreamChunk;
    vec![
        StreamChunk::ToolCallDelta {
            slot: 0,
            id: Some(format!("ask-{question}")),
            name: Some("ask_parent".into()),
            args_delta: serde_json::json!({"question": question}).to_string(),
        },
        StreamChunk::Done {
            prompt_tokens: None,
            completion_tokens: None,
            cached_tokens: None,
        },
    ]
}

#[tokio::test]
async fn each_question_and_the_final_report_releases_one_parent_wait() {
    let registry = ToolRegistry::with_builtins();
    let provider = Arc::new(ScriptedProvider::new());
    provider.push_script(ask_script("FIRST-QUESTION"));
    provider.push_script(ask_script("SECOND-QUESTION"));
    provider.script_text("FINAL-REPORT");
    let gate = Arc::new(tokio::sync::Notify::new());
    *provider.gate.lock().unwrap() = Some(gate.clone());
    let (ctx, mut notices) = context(provider.clone(), 13);
    let started = (registry.spec("delegate").unwrap().exec)(
        serde_json::json!({"task": "review with clarification", "background": true}),
        ctx.clone(),
    )
    .await
    .unwrap();
    let id = child_id(&started.content);
    let wait = registry.spec("wait_child").unwrap();
    for update in ["FIRST-QUESTION", "SECOND-QUESTION", "FINAL-REPORT"] {
        let waiting = (wait.exec)(serde_json::json!({"child_id": id}), ctx.clone());
        tokio::pin!(waiting);
        assert!(
            tokio::time::timeout(Duration::from_millis(30), &mut waiting)
                .await
                .is_err(),
            "each answer must reset the handoff before the child's next update"
        );
        gate.notify_one();
        tokio::time::timeout(Duration::from_secs(3), waiting)
            .await
            .unwrap()
            .unwrap();
        let notice = notices.try_recv().expect("the update precedes the wake");
        assert!(notice.contains(update), "{notice}");
        assert!(notices.try_recv().is_err(), "one notice per child update");
        if update != "FINAL-REPORT" {
            *provider.gate.lock().unwrap() = Some(gate.clone());
            (registry.spec("answer_child").unwrap().exec)(
                serde_json::json!({
                    "child_id": id, "answer": "continue the review", "background": true
                }),
                ctx.clone(),
            )
            .await
            .unwrap();
        }
    }
    assert_eq!(provider.calls.lock().unwrap().len(), 3);
    assert!(
        provider
            .tool_names
            .lock()
            .unwrap()
            .iter()
            .all(|names| !names.iter().any(|name| name == "wait_child")),
        "a child must never be offered its parent's wait action"
    );
}

#[tokio::test]
async fn cancelling_the_parent_wait_releases_it_without_stopping_the_background_child() {
    use agent_kernel::channels::UiSink;
    use agent_kernel::engine::{Engine, EngineIo};
    use agent_kernel::permissions::PermissionGate;
    use agent_llm_core::types::StreamChunk;
    use std::sync::atomic::Ordering;
    let registry = ToolRegistry::with_builtins();
    let provider = Arc::new(ScriptedProvider::new());
    provider.script_text("CHILD-SURVIVED");
    let gate = Arc::new(tokio::sync::Notify::new());
    *provider.gate.lock().unwrap() = Some(gate.clone());
    let (ctx, notice_rx) = context(provider, 14);
    let started = (registry.spec("delegate").unwrap().exec)(
        serde_json::json!({"task": "review independently", "background": true}),
        ctx.clone(),
    )
    .await
    .unwrap();
    let parent = Arc::new(ScriptedProvider::new());
    parent.push_script(vec![
        StreamChunk::ToolCallDelta {
            slot: 0,
            id: Some("wait-cancel".into()),
            name: Some("wait_child".into()),
            args_delta: serde_json::json!({"child_id": child_id(&started.content)}).to_string(),
        },
        StreamChunk::Done {
            prompt_tokens: None,
            completion_tokens: None,
            cached_tokens: None,
        },
    ]);
    parent.script_text("unexpected continuation");
    let (ui_tx, mut ui_rx) = UiSink::channel();
    let (_steer_tx, steer_rx) = tokio::sync::mpsc::channel(4);
    let cancel = ctx.cancel.as_ref().unwrap().clone();
    let mut io = EngineIo {
        ui_tx,
        steer_rx,
        notice_rx,
        cancel: cancel.clone(),
    };
    let mut engine = Engine::new(parent.clone(), Arc::new(registry), ctx, "parent", 0.0);
    engine.set_permissions(PermissionGate::from_mode_str("bypassPermissions"));
    let mut history = Vec::new();
    {
        let waiting = engine.run_turn(&mut io, &mut history, "wait for the review".into());
        tokio::pin!(waiting);
        assert!(
            tokio::time::timeout(Duration::from_millis(30), &mut waiting)
                .await
                .is_err()
        );
        cancel.store(true, Ordering::Relaxed);
        let error = tokio::time::timeout(Duration::from_secs(1), waiting)
            .await
            .expect("cancellation releases the wait")
            .unwrap_err();
        assert_eq!(error, "turn cancelled by user");
    }
    gate.notify_one();
    let report = tokio::time::timeout(Duration::from_secs(3), io.notice_rx.recv())
        .await
        .unwrap()
        .unwrap();
    assert!(report.contains("CHILD-SURVIVED"), "{report}");
    assert!(
        history
            .iter()
            .any(|message| message.content.as_deref() == Some("cancelled while running")),
        "waiting must use the standard dispatch cancellation result"
    );
    assert_eq!(parent.calls.lock().unwrap().len(), 1);
    let mut finished = Vec::new();
    while let Some(event) = ui_rx.try_recv() {
        if let agent_ipc::UiEvent::ToolCallFinished { name, ok, .. } = event {
            finished.push((name, ok));
        }
    }
    assert_eq!(finished, [("wait_child".to_string(), false)]);
}

#[tokio::test]
async fn foreign_and_unknown_ids_expose_no_child_update_and_start_no_work() {
    let registry = ToolRegistry::with_builtins();
    let provider = Arc::new(ScriptedProvider::new());
    provider.script_text("PRIVATE-OWNER-REPORT");
    let gate = Arc::new(tokio::sync::Notify::new());
    *provider.gate.lock().unwrap() = Some(gate.clone());
    let (ctx, mut notices) = context(provider.clone(), 15);
    let started = (registry.spec("delegate").unwrap().exec)(
        serde_json::json!({"task": "private review", "background": true}),
        ctx.clone(),
    )
    .await
    .unwrap();
    let id = child_id(&started.content);
    let mut foreign = ToolCtx::new().with_session(16);
    foreign.install_subagent(ctx.subagent().unwrap().0.clone());
    let foreign = Arc::new(foreign);
    let wait = registry.spec("wait_child").unwrap();
    for attempted_id in [id, u64::MAX] {
        let out = tokio::time::timeout(
            Duration::from_secs(1),
            (wait.exec)(
                serde_json::json!({"child_id": attempted_id}),
                foreign.clone(),
            ),
        )
        .await
        .unwrap()
        .unwrap();
        assert!(out.content.contains("no pending handoff in this session"));
        assert!(!out.content.contains("PRIVATE-OWNER-REPORT"));
    }
    assert!(notices.try_recv().is_err());
    gate.notify_one();
    let report = tokio::time::timeout(Duration::from_secs(3), notices.recv())
        .await
        .unwrap()
        .unwrap();
    assert!(report.contains("PRIVATE-OWNER-REPORT"));
    assert_eq!(provider.calls.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn stopping_a_parked_child_removes_its_pending_handoff() {
    let registry = ToolRegistry::with_builtins();
    let provider = Arc::new(ScriptedProvider::new());
    provider.push_script(ask_script("CANCELLED-QUESTION"));
    let (ctx, mut notices) = context(provider, 17);
    let started = (registry.spec("delegate").unwrap().exec)(
        serde_json::json!({"task": "ask before proceeding", "background": true}),
        ctx.clone(),
    )
    .await
    .unwrap();
    let id = child_id(&started.content);
    (registry.spec("wait_child").unwrap().exec)(serde_json::json!({"child_id": id}), ctx.clone())
        .await
        .unwrap();
    assert!(notices.try_recv().unwrap().contains("CANCELLED-QUESTION"));
    assert_eq!(
        agent_kernel::tools::delegate::abort_children(17),
        1,
        "the question belongs to a live child"
    );
    let out = (registry.spec("wait_child").unwrap().exec)(serde_json::json!({"child_id": id}), ctx)
        .await
        .unwrap();
    assert!(
        out.content.contains("no pending handoff"),
        "a stopped child's question cannot remain a pending handoff: {}",
        out.content
    );
}

#[tokio::test]
async fn a_waiting_parent_round_makes_no_model_or_file_requests_until_the_child_reports() {
    use agent_kernel::channels::UiSink;
    use agent_kernel::engine::{Engine, EngineIo};
    use agent_kernel::permissions::PermissionGate;
    use agent_llm_core::types::{ChatMessage, StreamChunk};

    let registry = ToolRegistry::with_builtins();
    let child_provider = Arc::new(ScriptedProvider::new());
    child_provider.script_text("GATED-REVIEW-REPORT");
    let gate = Arc::new(tokio::sync::Notify::new());
    *child_provider.gate.lock().unwrap() = Some(gate.clone());
    let (ctx, notice_rx) = context(child_provider.clone(), 19);
    let started = (registry.spec("delegate").unwrap().exec)(
        serde_json::json!({"task": "gated review", "background": true}),
        ctx.clone(),
    )
    .await
    .unwrap();
    let parent = Arc::new(ScriptedProvider::new());
    parent.push_script(vec![
        StreamChunk::ToolCallDelta {
            slot: 0,
            id: Some("wait-review".into()),
            name: Some("wait_child".into()),
            args_delta: serde_json::json!({"child_id": child_id(&started.content)}).to_string(),
        },
        StreamChunk::Done {
            prompt_tokens: None,
            completion_tokens: None,
            cached_tokens: None,
        },
    ]);
    parent.script_text("parent handled the review");
    let (ui_tx, mut ui_rx) = UiSink::channel();
    let mut parent_ctx = ctx.with_cancel(ctx.cancel.as_ref().unwrap().clone());
    parent_ctx.install_ui(ui_tx.clone());
    let ctx = Arc::new(parent_ctx);
    let (_steer_tx, steer_rx) = tokio::sync::mpsc::channel(4);
    let mut io = EngineIo {
        ui_tx,
        steer_rx,
        notice_rx,
        cancel: ctx.cancel.as_ref().unwrap().clone(),
    };
    let mut engine = Engine::new(parent.clone(), Arc::new(registry), ctx, "parent", 0.0);
    engine.set_permissions(PermissionGate::from_mode_str("bypassPermissions"));
    let mut history: Vec<ChatMessage> = Vec::new();
    {
        let running = engine.run_turn(&mut io, &mut history, "wait for the review".into());
        tokio::pin!(running);
        assert!(
            tokio::time::timeout(Duration::from_millis(30), &mut running)
                .await
                .is_err(),
            "the same parent round stays open"
        );
        assert_eq!(parent.calls.lock().unwrap().len(), 1);
        assert!(child_provider.calls.lock().unwrap().is_empty());
        gate.notify_one();
        tokio::time::timeout(Duration::from_secs(3), running)
            .await
            .expect("the parent resumes after the report")
            .unwrap();
    }
    assert_eq!(parent.calls.lock().unwrap().len(), 2);
    let seen = parent.seen_messages.lock().unwrap();
    assert_eq!(
        seen[1]
            .iter()
            .filter(|message| message.contains("GATED-REVIEW-REPORT"))
            .count(),
        1,
        "the resumed request sees the report once"
    );
    let mut started_tools = Vec::new();
    while let Some(event) = ui_rx.try_recv() {
        if let agent_ipc::UiEvent::ToolCallStarted { name, .. } = event {
            started_tools.push(name);
        }
    }
    assert_eq!(started_tools, ["wait_child"]);
}
