use std::sync::Arc;

use agent_ipc::events::UiEvent;
use agent_kernel::channels::UiSink;
use agent_kernel::tools::delegate::{
    active_children, spec, wait_child_spec, ChildRoute, SubagentSpawner,
};
use agent_kernel::tools::{ToolCtx, ToolRegistry};
use agent_testkit::ScriptedProvider;

fn spawner(provider: Arc<ScriptedProvider>) -> SubagentSpawner {
    let registry = Arc::new(ToolRegistry::child_builtins());
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
    SubagentSpawner::new(route, registry.clone(), registry)
}

/// A nested event names its child by id, not only by a display label — the
/// label of two unnamed children differs only by index, the id is the identity.
#[tokio::test]
async fn nested_events_carry_the_child_id() {
    let provider = Arc::new(ScriptedProvider::new());
    provider.script_text("child prose");
    let (sink, mut rx) = UiSink::channel();
    let mut ctx = ToolCtx::new();
    ctx.install_subagent(spawner(provider));
    ctx.install_ui(sink);

    (spec().exec)(serde_json::json!({"task": "look"}), Arc::new(ctx))
        .await
        .expect("the child must finish");

    let mut card = None;
    let mut deltas = Vec::new();
    while let Some(ev) = rx.try_recv() {
        match ev {
            UiEvent::ToolCallStarted { child_id, .. } => card = child_id,
            UiEvent::TextDelta {
                parent, child_id, ..
            } => deltas.push((parent, child_id)),
            _ => {}
        }
    }
    let id = card.expect("the child's card names its id");
    assert!(
        !deltas.is_empty(),
        "the child's prose must reach the parent"
    );
    for (parent, child_id) in deltas {
        assert_eq!(parent.as_deref(), Some("subagent"), "nested under the card");
        assert_eq!(child_id, Some(id), "a nested delta must name its child");
    }
}

/// Three children that all finish must leave the registry and must not be
/// reported as still running. A parent that then waits is released by the
/// terminal handoff instead of parking on a child that has already exited.
#[tokio::test]
async fn finished_parallel_children_do_not_leave_the_parent_waiting() {
    let provider = Arc::new(ScriptedProvider::new());
    provider.script_text("alpha");
    provider.script_text("beta");
    provider.script_text("gamma");
    let (sink, mut rx) = UiSink::channel();
    let mut ctx = ToolCtx::new().with_session(7);
    ctx.install_subagent(spawner(provider));
    ctx.install_ui(sink);
    agent_kernel::tools::delegate::install_child_notifier(&mut ctx, |_| {});
    let ctx = Arc::new(ctx);

    let result = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        (spec().exec)(
            serde_json::json!({"tasks": ["a", "b", "c"], "readonly": true}),
            ctx.clone(),
        ),
    )
    .await
    .expect("parallel children must finish")
    .expect("delegation succeeds")
    .content;
    assert!(
        !result.contains("still running"),
        "finished children must not stay in progress: {result}"
    );
    assert!(
        active_children(7).is_empty(),
        "finished children must leave the registry"
    );

    let mut finished = 0;
    while let Some(ev) = rx.try_recv() {
        if let UiEvent::ToolCallFinished {
            parent: Some(parent),
            child_id: Some(_),
            ..
        } = ev
        {
            if parent == "delegate" {
                finished += 1;
            }
        }
    }
    assert_eq!(finished, 3, "each child card closes when its turn returns");

    let waited = tokio::time::timeout(
        std::time::Duration::from_secs(2),
        (wait_child_spec().exec)(serde_json::json!({"child_id": 1}), ctx),
    )
    .await
    .expect("a wait for an already finished child must return");
    assert!(waited.is_ok(), "the parent is not left waiting: {waited:?}");
}

/// The card's task text is the whole task, not a preview cut to fit the card.
#[tokio::test]
async fn the_card_carries_the_whole_task() {
    let provider = Arc::new(ScriptedProvider::new());
    provider.script_text("done");
    let (sink, mut rx) = UiSink::channel();
    let mut ctx = ToolCtx::new();
    ctx.install_subagent(spawner(provider));
    ctx.install_ui(sink);

    let task = "检查鉴权中间件的每一个分支，确认没有遗漏的错误路径。".repeat(6);
    (spec().exec)(serde_json::json!({"task": task}), Arc::new(ctx))
        .await
        .expect("the child must finish");

    let mut preview = None;
    while let Some(ev) = rx.try_recv() {
        if let UiEvent::ToolCallStarted {
            child_id: Some(_),
            args_preview,
            ..
        } = ev
        {
            preview = Some(args_preview);
        }
    }
    assert_eq!(preview.as_deref(), Some(task.as_str()));
}

/// A cancel drops the child's run future mid-poll. The card it already opened
/// must still close as an interruption rather than stay open for good.
#[tokio::test]
async fn a_dropped_child_still_closes_its_card() {
    let provider = Arc::new(ScriptedProvider::new());
    provider.script_text("never answered");
    // The gate holds the child's first sample open for as long as the test likes.
    *provider.gate.lock().unwrap() = Some(Arc::new(tokio::sync::Notify::new()));
    let (sink, mut rx) = UiSink::channel();
    let mut ctx = ToolCtx::new();
    ctx.install_subagent(spawner(provider));
    ctx.install_ui(sink);

    // Nothing ever fires the gate, so the child is still mid-turn when the
    // timeout drops the delegation future the way a cancel does.
    let hung = (spec().exec)(serde_json::json!({"task": "hang"}), Arc::new(ctx));
    let dropped = tokio::time::timeout(std::time::Duration::from_millis(200), hung).await;
    assert!(
        dropped.is_err(),
        "the child must still be running when dropped"
    );

    let mut opened = false;
    let mut interrupted = false;
    while let Some(ev) = rx.try_recv() {
        match ev {
            UiEvent::ToolCallStarted { child_id, .. } => opened = child_id.is_some(),
            UiEvent::ToolCallFinished { ok, content, .. } => {
                interrupted = !ok && content == "已中断";
            }
            _ => {}
        }
    }
    assert!(opened, "the child's card must have opened");
    assert!(
        interrupted,
        "a dropped run must close its card as interrupted"
    );
}
