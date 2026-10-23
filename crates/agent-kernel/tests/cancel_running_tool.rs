//! Cancel has to reach a tool that is ALREADY running — a `bash` waiting on a
//! ten-minute build is the case that matters. Checking the flag between tool
//! rounds is not enough: the turn sits inside the dispatch.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use agent_kernel::channels::UiSink;
use agent_kernel::engine::{Engine, EngineIo};
use agent_kernel::permissions::PermissionGate;
use agent_kernel::tools::registry::{ToolClass, ToolCtx, ToolRegistry, ToolResult, ToolSpec};
use agent_llm_core::types::StreamChunk;
use agent_llm_core::ChatMessage;
use agent_testkit::ScriptedProvider;
use futures::FutureExt;

/// Dropped with the dispatch future — the assertion that the engine let go of
/// the tool instead of waiting it out.
struct DropFlag(Arc<AtomicBool>);

impl Drop for DropFlag {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

/// A tool that only ends when cancelled, standing in for a long `bash`.
fn slow_tool(dropped: Arc<AtomicBool>) -> ToolSpec {
    ToolSpec {
        name: "slow",
        schema: serde_json::json!({ "type": "object", "properties": {} }),
        readonly: false,
        class: ToolClass::Process,
        network: false,
        staged: false,
        review: None,
        cancel_aware: false,
        exec: Arc::new(move |_args, _ctx| {
            let dropped = dropped.clone();
            async move {
                let _guard = DropFlag(dropped);
                tokio::time::sleep(Duration::from_secs(600)).await;
                Ok(ToolResult::text("never"))
            }
            .boxed()
        }),
    }
}

#[tokio::test]
async fn cancel_stops_a_running_tool() {
    let dropped = Arc::new(AtomicBool::new(false));
    let mut registry = ToolRegistry::new();
    registry.register(slow_tool(dropped.clone()));

    let provider = Arc::new(ScriptedProvider::new().with_script(vec![
        StreamChunk::ToolCallDelta {
            slot: 0,
            id: Some("c1".into()),
            name: Some("slow".into()),
            args_delta: "{}".into(),
        },
        StreamChunk::Done {
            prompt_tokens: None,
            completion_tokens: None,
            cached_tokens: None,
        },
    ]));
    // A second request would mean the turn carried on after the cancel.
    provider.script_text("carried on");

    let mut engine = Engine::new(
        provider.clone(),
        Arc::new(registry),
        Arc::new(ToolCtx::new()),
        "scripted",
        0.0,
    );
    engine.set_permissions(PermissionGate::from_mode_str("bypassPermissions"));

    let (ui_tx, mut ui_rx) = UiSink::channel();
    let (_steer_tx, steer_rx) = tokio::sync::mpsc::channel(4);
    let (_notice_tx, notice_rx) = tokio::sync::mpsc::channel(4);
    let cancel = Arc::new(AtomicBool::new(false));
    let mut io = EngineIo {
        ui_tx,
        steer_rx,
        notice_rx,
        cancel: cancel.clone(),
    };
    let mut history: Vec<ChatMessage> = Vec::new();

    let writer = cancel.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(200)).await;
        writer.store(true, Ordering::Relaxed);
    });

    let started = std::time::Instant::now();
    let err = tokio::time::timeout(
        Duration::from_secs(5),
        engine.run_turn(&mut io, &mut history, "run the build".into()),
    )
    .await
    .expect("the cancel must land while the tool runs, not after it")
    .expect_err("a cancelled turn is an error");
    assert_eq!(err, "turn cancelled by user");
    assert!(
        started.elapsed() < Duration::from_secs(2),
        "cancel waited for the tool: {:?}",
        started.elapsed()
    );
    assert!(
        dropped.load(Ordering::Relaxed),
        "the dispatch future must be dropped, not awaited"
    );

    // The call is still answered, so the next request is replayable.
    assert!(
        history
            .iter()
            .any(|m| m.content.as_deref() == Some("cancelled while running")),
        "{history:#?}"
    );
    assert!(
        history
            .iter()
            .any(|m| m.content.as_deref() == Some("已被用户中断")),
        "{history:#?}"
    );
    // One request: the turn never reached the model again.
    assert_eq!(provider.calls.lock().unwrap().len(), 1);

    // The UI saw the running card settle.
    let mut finished = Vec::new();
    while let Some(ev) = ui_rx.try_recv() {
        if let agent_ipc::UiEvent::ToolCallFinished { name, ok, .. } = ev {
            finished.push((name, ok));
        }
    }
    assert_eq!(finished, vec![("slow".to_string(), false)], "{finished:?}");
}

/// A tool that raises the cancel flag itself and returns — the user hitting
/// Cancel just as the first of two calls completes.
fn cancelling_tool(flag: Arc<AtomicBool>) -> ToolSpec {
    ToolSpec {
        name: "first",
        schema: serde_json::json!({ "type": "object", "properties": {} }),
        readonly: true,
        class: ToolClass::Observation,
        network: false,
        staged: false,
        review: None,
        cancel_aware: false,
        exec: Arc::new(move |_args, _ctx| {
            let flag = flag.clone();
            async move {
                flag.store(true, Ordering::Relaxed);
                Ok(ToolResult::text("done"))
            }
            .boxed()
        }),
    }
}

/// Cancel between two calls: the second never starts, so the UI must not get
/// a `ToolCallFinished` for it — there is no card to settle, and a stray one
/// draws a failed call that never ran. History still answers both calls.
#[tokio::test]
async fn cancel_before_dispatch_draws_no_card() {
    let cancel = Arc::new(AtomicBool::new(false));
    let dropped = Arc::new(AtomicBool::new(false));
    let mut registry = ToolRegistry::new();
    registry.register(cancelling_tool(cancel.clone()));
    registry.register(slow_tool(dropped.clone()));

    let provider = Arc::new(ScriptedProvider::new().with_script(vec![
        StreamChunk::ToolCallDelta {
            slot: 0,
            id: Some("c1".into()),
            name: Some("first".into()),
            args_delta: "{}".into(),
        },
        StreamChunk::ToolCallDelta {
            slot: 1,
            id: Some("c2".into()),
            name: Some("slow".into()),
            args_delta: "{}".into(),
        },
        StreamChunk::Done {
            prompt_tokens: None,
            completion_tokens: None,
            cached_tokens: None,
        },
    ]));
    provider.script_text("carried on");

    let mut engine = Engine::new(
        provider.clone(),
        Arc::new(registry),
        Arc::new(ToolCtx::new()),
        "scripted",
        0.0,
    );
    engine.set_permissions(PermissionGate::from_mode_str("bypassPermissions"));

    let (ui_tx, mut ui_rx) = UiSink::channel();
    let (_steer_tx, steer_rx) = tokio::sync::mpsc::channel(4);
    let (_notice_tx, notice_rx) = tokio::sync::mpsc::channel(4);
    let mut io = EngineIo {
        ui_tx,
        steer_rx,
        notice_rx,
        cancel: cancel.clone(),
    };
    let mut history: Vec<ChatMessage> = Vec::new();

    let err = tokio::time::timeout(
        Duration::from_secs(5),
        engine.run_turn(&mut io, &mut history, "two calls".into()),
    )
    .await
    .expect("the turn ends on the cancel")
    .expect_err("a cancelled turn is an error");
    assert_eq!(err, "turn cancelled by user");
    assert!(
        !dropped.load(Ordering::Relaxed),
        "the second call never ran"
    );

    let (mut started, mut finished) = (Vec::new(), Vec::new());
    while let Some(ev) = ui_rx.try_recv() {
        match ev {
            agent_ipc::UiEvent::ToolCallStarted { name, .. } => started.push(name),
            agent_ipc::UiEvent::ToolCallFinished { name, ok, .. } => finished.push((name, ok)),
            _ => {}
        }
    }
    assert_eq!(started, vec!["first".to_string()], "{started:?}");
    assert_eq!(finished, vec![("first".to_string(), true)], "{finished:?}");

    let answered: Vec<_> = history
        .iter()
        .filter_map(|m| m.tool_call_id.as_deref().zip(m.content.as_deref()))
        .collect();
    assert_eq!(
        answered,
        vec![("c1", "done"), ("c2", "cancelled before dispatch")],
        "{history:#?}"
    );
}
