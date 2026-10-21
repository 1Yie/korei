//! A mid-turn steer is a tagged row and its own event.
//!
//! The model must still read `The user interrupted: …`, but a host replaying
//! history or counting turns reads the tag, never the text. This drives a real
//! turn with a steer already queued and checks all three surfaces: the row in
//! history, the event on the UI stream, and what the provider was shown.

use std::sync::Arc;

use agent_ipc::UiEvent;
use agent_kernel::channels::UiSink;
use agent_kernel::engine::{Engine, EngineIo};
use agent_kernel::permissions::PermissionGate;
use agent_kernel::tools::registry::{ToolCtx, ToolRegistry};
use agent_llm_core::types::NoticeKind;
use agent_llm_core::{ChatMessage, Role};
use agent_testkit::ScriptedProvider;

async fn run_with_steer(steer: &str) -> (Vec<ChatMessage>, Vec<UiEvent>, Arc<ScriptedProvider>) {
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
    steer_tx.send(steer.to_string()).await.unwrap();
    let mut io = EngineIo {
        ui_tx,
        steer_rx,
        notice_rx: tokio::sync::mpsc::channel(4).1,
        cancel: Arc::new(std::sync::atomic::AtomicBool::new(false)),
    };
    let mut history: Vec<ChatMessage> = Vec::new();

    tokio::time::timeout(
        std::time::Duration::from_secs(10),
        engine.run_turn(&mut io, &mut history, "refactor the parser".into()),
    )
    .await
    .expect("the turn must not park")
    .expect("the turn should finish");

    drop(io);
    let mut events = Vec::new();
    while let Some(ev) = ui_rx.try_recv() {
        events.push(ev);
    }
    (history, events, provider)
}

#[tokio::test]
async fn a_steer_lands_as_a_tagged_row() {
    let (history, _events, _provider) = run_with_steer("先别改 config").await;

    let steers: Vec<&ChatMessage> = history.iter().filter(|m| m.is_steer()).collect();
    assert_eq!(steers.len(), 1, "exactly one steer row: {history:#?}");
    let row = steers[0];
    assert_eq!(row.role, Role::User);
    assert_eq!(row.notice, Some(NoticeKind::Steer));
    assert_eq!(row.steer_text(), Some("先别改 config"));

    // The opening prompt is untouched: no tag, not a steer.
    let prompt = history
        .iter()
        .find(|m| m.content.as_deref() == Some("refactor the parser"))
        .expect("the prompt row");
    assert_eq!(prompt.notice, None);
    assert!(!prompt.is_steer());
}

#[tokio::test]
async fn a_steer_is_its_own_event_and_never_a_prompt_echo() {
    let (_history, events, _provider) = run_with_steer("先别改 config").await;

    let steered: Vec<&str> = events
        .iter()
        .filter_map(|e| match e {
            UiEvent::UserSteer(t) => Some(t.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(steered, ["先别改 config"], "the bare words, no prefix");

    let prompts: Vec<&str> = events
        .iter()
        .filter_map(|e| match e {
            UiEvent::UserPrompt(t) => Some(t.as_str()),
            _ => None,
        })
        .collect();
    assert!(
        !prompts.contains(&"先别改 config"),
        "a steer must not echo as a prompt: {prompts:?}"
    );
}

#[tokio::test]
async fn the_model_still_reads_the_interruption_prefix() {
    let (_history, _events, provider) = run_with_steer("先别改 config").await;

    let seen = provider.seen_messages.lock().unwrap();
    let shown = seen.last().expect("one request");
    assert!(
        shown
            .iter()
            .any(|m| m == "The user interrupted: 先别改 config"),
        "the prompt text the model reads must not change: {shown:?}"
    );
}
