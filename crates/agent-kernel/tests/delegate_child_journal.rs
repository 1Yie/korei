//! A child that never finishes its turn — parked on `ask_parent`, aborted, or
//! cancelled while parked — still leaves its history on disk. The host's
//! `persist` is the write this proves, with a host that records instead of
//! writing files.

use std::sync::Arc;

use agent_kernel::tools::delegate::{ChildRoute, SubagentHost, SubagentSpawner};
use agent_kernel::tools::{ToolCtx, ToolRegistry};
use agent_llm_core::types::{ChatMessage, StreamChunk};
use agent_testkit::ScriptedProvider;

/// Records every history the kernel hands to the host, in call order.
#[derive(Default)]
struct RecordingHost {
    persists: std::sync::Mutex<Vec<(u64, Vec<ChatMessage>)>>,
}

impl RecordingHost {
    fn count(&self) -> usize {
        self.persists.lock().unwrap().len()
    }

    /// The most recent write, as (child id, history).
    fn last(&self) -> Option<(u64, Vec<ChatMessage>)> {
        self.persists.lock().unwrap().last().cloned()
    }
}

impl SubagentHost for RecordingHost {
    fn persist(&self, _ctx: &ToolCtx, child_id: u64, _label: &str, history: &[ChatMessage]) {
        self.persists
            .lock()
            .unwrap()
            .push((child_id, history.to_vec()));
    }
}

fn spawner(provider: Arc<ScriptedProvider>, host: Arc<RecordingHost>) -> SubagentSpawner {
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
    SubagentSpawner::new(route, registry.clone(), registry).with_host(host)
}

fn child_asks(provider: &ScriptedProvider, question: &str) {
    provider.push_script(vec![
        StreamChunk::ToolCallDelta {
            slot: 0,
            id: Some("call_ask".into()),
            name: Some("ask_parent".into()),
            args_delta: serde_json::json!({ "question": question }).to_string(),
        },
        StreamChunk::Done {
            prompt_tokens: None,
            completion_tokens: None,
            cached_tokens: None,
        },
    ]);
}

/// The id the waiting notice tells the parent to answer.
fn child_id_of(text: &str) -> u64 {
    const NEEDLE: &str = "child_id ";
    let at = text.find(NEEDLE).expect("the notice names the child") + NEEDLE.len();
    text[at..]
        .chars()
        .take_while(char::is_ascii_digit)
        .collect::<String>()
        .parse()
        .expect("the notice carries a numeric child id")
}

/// The last message of a persisted history, as text.
fn last_text(history: &[ChatMessage]) -> String {
    history
        .last()
        .and_then(|m| m.content.clone())
        .unwrap_or_default()
}

/// Wait for the host to record one more write than `before`.
async fn wait_for_persists(host: &RecordingHost, before: usize) {
    for _ in 0..300 {
        if host.count() > before {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
}

/// A child parked on its parent has already left a transcript, and the last
/// thing in it is the question it is waiting on — otherwise the park is a black
/// hole in the record.
#[tokio::test]
async fn a_parked_child_persists_a_history_ending_with_its_question() {
    let provider = Arc::new(ScriptedProvider::new());
    child_asks(&provider, "which crate should I use?");
    let host = Arc::new(RecordingHost::default());
    let mut ctx = ToolCtx::new();
    ctx.install_subagent(spawner(provider.clone(), host.clone()));
    let ctx = Arc::new(ctx);

    let delegate = agent_kernel::tools::delegate::spec();
    let waiting = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        (delegate.exec)(serde_json::json!({ "task": "inspect" }), ctx),
    )
    .await
    .expect("a parked child must hand its question back")
    .expect("asking is not an error");
    let child_id = child_id_of(&waiting.content);

    let (id, history) = host.last().expect("a parked child must leave a transcript");
    assert_eq!(id, child_id, "the write belongs to the child that parked");
    assert!(
        history.iter().any(|m| m.content.as_deref() == Some("inspect")),
        "the task must be in the transcript: {history:?}"
    );
    assert_eq!(
        last_text(&history),
        "[ask_parent] which crate should I use?",
        "the parked child's history must end with its question: {history:?}"
    );
}

/// A child cancelled while parked writes again once it stops — the first write
/// alone would leave the record looking like a child that vanished mid-turn.
#[tokio::test]
async fn a_child_cancelled_while_parked_persists_again() {
    let provider = Arc::new(ScriptedProvider::new());
    child_asks(&provider, "wait on me");
    let host = Arc::new(RecordingHost::default());
    let cancel = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let mut ctx = ToolCtx::new();
    ctx.cancel = Some(cancel.clone());
    ctx.install_subagent(spawner(provider, host.clone()));
    let ctx = Arc::new(ctx);

    let delegate = agent_kernel::tools::delegate::spec();
    let waiting = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        (delegate.exec)(serde_json::json!({ "task": "wait" }), ctx),
    )
    .await
    .expect("the child must park first")
    .expect("asking is not an error");
    let child_id = child_id_of(&waiting.content);
    let parked = host.count();
    assert!(parked >= 1, "parking must have persisted once already");

    cancel.store(true, std::sync::atomic::Ordering::Relaxed);
    wait_for_persists(&host, parked).await;
    assert!(
        host.count() > parked,
        "a child cancelled while parked must persist its last history"
    );
    let (id, _) = host.last().expect("a write landed just above");
    assert_eq!(id, child_id);
}

/// An abort drops the child's turn future, so the mirror is the only record of
/// what it was running on: the write has to happen before the abort.
#[tokio::test]
async fn an_aborted_child_leaves_the_history_it_was_running_on() {
    let provider = Arc::new(ScriptedProvider::new());
    let gate = Arc::new(tokio::sync::Notify::new());
    *provider.gate.lock().unwrap() = Some(gate);
    provider.script_text("never reached");
    let host = Arc::new(RecordingHost::default());
    let cancel = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let mut ctx = ToolCtx::new();
    ctx.cancel = Some(cancel.clone());
    ctx.install_subagent(spawner(provider.clone(), host.clone()));
    let ctx = Arc::new(ctx);

    let delegate = agent_kernel::tools::delegate::spec();
    let running = tokio::spawn(async move {
        (delegate.exec)(serde_json::json!({ "task": "hold me" }), ctx).await
    });
    // The child's first request is a snapshot: once it reached the provider, the
    // mirror holds the history the abort must save.
    for _ in 0..300 {
        if !provider.reasoning_efforts.lock().unwrap().is_empty() {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    assert!(
        !provider.reasoning_efforts.lock().unwrap().is_empty(),
        "the child must have issued its first request"
    );

    cancel.store(true, std::sync::atomic::Ordering::Relaxed);
    let _ = tokio::time::timeout(std::time::Duration::from_secs(5), running).await;
    wait_for_persists(&host, 0).await;

    let (_, history) = host.last().expect("an aborted child must still be persisted");
    assert!(
        history.iter().any(|m| m.content.as_deref() == Some("hold me")),
        "the aborted child's transcript must hold the task it ran on: {history:?}"
    );
}
