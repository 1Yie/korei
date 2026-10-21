//! A host notice is a tagged host row, never a steer.
//!
//! Background completions and other host-side pushes arrive on `notice_rx`.
//! The text can carry untrusted tool output, so it may never dress itself up
//! as the user's own words (the turn's steer count is the memory distiller's
//! correction seed) nor as a plain system instruction: it lands tagged
//! `NoticeKind::Host`, which adapters fence at user privilege.

use std::sync::Arc;

use agent_ipc::UiEvent;
use agent_kernel::channels::UiSink;
use agent_kernel::engine::{Engine, EngineIo};
use agent_kernel::permissions::PermissionGate;
use agent_kernel::tools::registry::{ToolCtx, ToolRegistry};
use agent_llm_core::types::NoticeKind;
use agent_llm_core::{ChatMessage, Role};
use agent_testkit::ScriptedProvider;

const NOTICE: &str = "[background task bg-1 finished: exited(0)] cargo test";

struct Run {
    history: Vec<ChatMessage>,
    events: Vec<UiEvent>,
    provider: Arc<ScriptedProvider>,
    steers: Vec<String>,
}

async fn run_with_notice(notice: &str) -> Run {
    let provider = Arc::new(ScriptedProvider::new());
    provider.script_text("done");
    let ctx = Arc::new(ToolCtx::new());
    let mut engine = Engine::new(
        provider.clone(),
        Arc::new(ToolRegistry::new()),
        ctx,
        "scripted",
        0.0,
    );
    engine.set_permissions(PermissionGate::from_mode_str("bypassPermissions"));

    let (ui_tx, mut ui_rx) = UiSink::channel();
    let (steer_tx, steer_rx) = tokio::sync::mpsc::channel(4);
    let (notice_tx, notice_rx) = tokio::sync::mpsc::channel(4);
    notice_tx.send(notice.to_string()).await.unwrap();
    let mut io = EngineIo {
        ui_tx,
        steer_rx,
        notice_rx,
        cancel: Arc::new(std::sync::atomic::AtomicBool::new(false)),
    };
    let mut history: Vec<ChatMessage> = Vec::new();

    tokio::time::timeout(
        std::time::Duration::from_secs(10),
        engine.run_turn(&mut io, &mut history, "run the suite".into()),
    )
    .await
    .expect("the turn must not park")
    .expect("the turn should finish");
    drop(steer_tx);

    let steers = engine.steers_this_turn().to_vec();
    drop(io);
    let mut events = Vec::new();
    while let Some(ev) = ui_rx.try_recv() {
        events.push(ev);
    }
    Run {
        history,
        events,
        provider,
        steers,
    }
}

#[tokio::test]
async fn a_notice_lands_as_a_host_row() {
    let run = run_with_notice(NOTICE).await;

    let row = run
        .history
        .iter()
        .find(|m| m.content.as_deref() == Some(NOTICE))
        .expect("the notice row");
    assert_eq!(row.role, Role::System, "{row:#?}");
    assert_eq!(row.notice, Some(NoticeKind::Host));
    assert!(!row.is_steer());
}

#[tokio::test]
async fn a_notice_never_counts_as_a_steer() {
    let run = run_with_notice(NOTICE).await;

    assert!(run.steers.is_empty(), "steers: {:?}", run.steers);
    assert_eq!(
        run.history.iter().filter(|m| m.is_steer()).count(),
        0,
        "{:#?}",
        run.history
    );
}

#[tokio::test]
async fn a_notice_never_echoes_as_a_user_row() {
    let run = run_with_notice(NOTICE).await;

    let steers: Vec<&str> = run
        .events
        .iter()
        .filter_map(|e| match e {
            UiEvent::UserSteer(t) => Some(t.as_str()),
            _ => None,
        })
        .collect();
    assert!(steers.is_empty(), "no steer bubble: {steers:?}");

    let prompts: Vec<&str> = run
        .events
        .iter()
        .filter_map(|e| match e {
            UiEvent::UserPrompt(t) => Some(t.as_str()),
            _ => None,
        })
        .collect();
    assert!(
        !prompts.contains(&NOTICE),
        "a notice must not echo as a prompt: {prompts:?}"
    );
}

/// A job that finishes while the session is idle wakes the model on its own
/// turn: the notices open it, the model answers without a user prompt, and
/// nothing in it counts as a user turn.
#[tokio::test]
async fn an_idle_session_wakes_on_a_notice() {
    let provider = Arc::new(ScriptedProvider::new());
    provider.script_text("the suite passed");
    let ctx = Arc::new(ToolCtx::new());
    let mut engine = Engine::new(
        provider.clone(),
        Arc::new(ToolRegistry::new()),
        ctx,
        "scripted",
        0.0,
    );
    engine.set_permissions(PermissionGate::from_mode_str("bypassPermissions"));

    let (ui_tx, mut ui_rx) = UiSink::channel();
    let (_steer_tx, steer_rx) = tokio::sync::mpsc::channel(4);
    let (notice_tx, notice_rx) = tokio::sync::mpsc::channel(4);
    // One more notice is already queued: the waking call takes it too.
    notice_tx
        .send("[watch] build.log changed".to_string())
        .await
        .unwrap();
    let mut io = EngineIo {
        ui_tx,
        steer_rx,
        notice_rx,
        cancel: Arc::new(std::sync::atomic::AtomicBool::new(false)),
    };
    let mut history: Vec<ChatMessage> = Vec::new();

    let outcome = tokio::time::timeout(
        std::time::Duration::from_secs(10),
        engine.run_notice_turn(&mut io, &mut history, vec![NOTICE.to_string()]),
    )
    .await
    .expect("the notice turn must not park")
    .expect("the notice turn should finish");
    assert_eq!(outcome.text, "the suite passed");

    // Both notices are in the transcript, both tagged as host rows.
    let notices: Vec<&ChatMessage> = history
        .iter()
        .filter(|m| m.notice == Some(NoticeKind::Host))
        .collect();
    assert_eq!(notices.len(), 2, "{history:#?}");
    assert!(notices[0]
        .content
        .as_deref()
        .unwrap()
        .contains("bg-1 finished"));
    assert!(notices[1].content.as_deref().unwrap().contains("build.log"));
    // No user turn was invented for the wake-up.
    assert!(
        !history
            .iter()
            .any(|m| m.role == Role::User && m.notice.is_none()),
        "{history:#?}"
    );
    assert!(outcome.steers.is_empty());

    let seen = provider.seen_messages.lock().unwrap();
    let shown = seen.last().expect("one request");
    assert!(
        shown.iter().any(|m| m.contains("bg-1 finished"))
            && shown.iter().any(|m| m.contains("build.log changed")),
        "the model must see both notices: {shown:?}"
    );

    // The UI draws the answer, not a user bubble.
    let mut events = Vec::new();
    while let Some(ev) = ui_rx.try_recv() {
        events.push(ev);
    }
    assert!(
        !events.iter().any(|e| matches!(e, UiEvent::UserPrompt(_))),
        "{events:?}"
    );
    assert!(
        events
            .iter()
            .any(|e| matches!(e, UiEvent::AssistantMessage(t) if t == "the suite passed")),
        "{events:?}"
    );
}

#[tokio::test]
async fn the_model_still_reads_the_notice() {
    let run = run_with_notice(NOTICE).await;

    let seen = run.provider.seen_messages.lock().unwrap();
    let shown = seen.last().expect("one request");
    assert!(
        shown.iter().any(|m| m.contains("bg-1 finished")),
        "the model must see the completion: {shown:?}"
    );
}
