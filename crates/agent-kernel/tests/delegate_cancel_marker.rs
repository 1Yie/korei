//! What a cancelled delegation leaves behind: a tool result that names the
//! child, and a transcript written while the child is still running.
//!
//! Two writers race on a cancel — the parent's `delegate`/`answer_child` poll
//! and the child's own turn — so the recorded text must not depend on which
//! one noticed first. The UI reconnects a child's card to its transcript by
//! `child_id`, and only the persisted tool result survives a reload.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use agent_kernel::channels::UiSink;
use agent_kernel::engine::{Engine, EngineIo};
use agent_kernel::permissions::PermissionGate;
use agent_kernel::tools::delegate::{active_children, ChildRoute, SubagentHost, SubagentSpawner};
use agent_kernel::tools::{ToolCtx, ToolRegistry};
use agent_llm_core::types::{ChatMessage, StreamChunk};
use agent_testkit::ScriptedProvider;

/// A session per test — the child registry is process-wide, so two tests
/// sharing one session would each see the other's children.
const CANCEL_SESSION: i64 = 42;
const JOURNAL_SESSION: i64 = 43;
const PARALLEL_SESSION: i64 = 44;

/// Counts the histories the kernel hands to the host. The real host writes a
/// file; what matters here is that the write happens at all, and when.
#[derive(Default)]
struct CountingHost {
    written: std::sync::Mutex<Vec<u64>>,
}

impl CountingHost {
    fn count(&self) -> usize {
        self.written.lock().unwrap().len()
    }
}

impl SubagentHost for CountingHost {
    fn persist(&self, _ctx: &ToolCtx, child_id: u64, _label: &str, _history: &[ChatMessage]) {
        self.written.lock().unwrap().push(child_id);
    }
}

fn spawner(
    provider: Arc<dyn agent_llm_core::LlmProvider>,
    host: Arc<CountingHost>,
) -> SubagentSpawner {
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

/// A child that never answers: it stands in for any subagent still working
/// when the user cancels, so nothing but the cancel can end the call.
struct StalledProvider;

#[async_trait::async_trait]
impl agent_llm_core::LlmProvider for StalledProvider {
    fn id(&self) -> &'static str {
        "stalled"
    }

    async fn chat_stream(
        &self,
        _model: &str,
        _messages: &[ChatMessage],
        _tools: Option<serde_json::Value>,
        _temperature: f32,
        _reasoning_effort: Option<&str>,
        _params: &agent_llm_core::ModelParams,
    ) -> anyhow::Result<agent_llm_core::provider::BoxStream<StreamChunk>> {
        std::future::pending::<()>().await;
        unreachable!("a stalled child never answers")
    }
}

/// The parent's turn asks for exactly one `delegate` call holding a task.
fn parent_delegates(provider: &ScriptedProvider, task: &str) {
    provider.push_script(vec![
        StreamChunk::ToolCallDelta {
            slot: 0,
            id: Some("call_delegate".into()),
            name: Some("delegate".into()),
            args_delta: serde_json::json!({ "task": task }).to_string(),
        },
        StreamChunk::Done {
            prompt_tokens: None,
            completion_tokens: None,
            cached_tokens: None,
        },
    ]);
}

/// The tool row the parent's history kept for `call_id`.
fn tool_row(history: &[ChatMessage], call_id: &str) -> String {
    history
        .iter()
        .find(|m| m.tool_call_id.as_deref() == Some(call_id))
        .and_then(|m| m.content.clone())
        .unwrap_or_else(|| panic!("no tool result for {call_id}: {history:#?}"))
}

/// The id of `session`'s running child, once its provider has been asked — the
/// signal that the delegate call is really in flight, so the cancel lands
/// mid-turn instead of before the dispatch.
async fn wait_for_running_child(session: i64, provider: &ScriptedProvider) -> u64 {
    for _ in 0..300 {
        let asked = !provider.reasoning_efforts.lock().unwrap().is_empty();
        if let Some(id) = active_children(session).first().copied() {
            if asked {
                return id;
            }
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("the delegate call never reached a running child");
}

/// Cancelling a running child records a tool result the UI can parse: the
/// child's id, and nothing generic, whichever of the two racers noticed the
/// flag first.
#[tokio::test]
async fn a_cancelled_delegate_records_the_child_id() {
    let parent = Arc::new(ScriptedProvider::new());
    parent_delegates(&parent, "hold the line");
    // One script only: a second parent request would mean the turn carried on
    // past the cancel.
    let child = Arc::new(ScriptedProvider::new());
    let gate = Arc::new(tokio::sync::Notify::new());
    *child.gate.lock().unwrap() = Some(gate);
    child.script_text("never reached");

    let host = Arc::new(CountingHost::default());
    let cancel = Arc::new(AtomicBool::new(false));
    let mut ctx = ToolCtx::new().with_session(CANCEL_SESSION);
    ctx.cancel = Some(cancel.clone());
    ctx.install_subagent(spawner(child.clone(), host.clone()));
    let ctx = Arc::new(ctx);

    let mut engine = Engine::new(
        parent.clone(),
        Arc::new(ToolRegistry::with_builtins()),
        ctx,
        "scripted",
        0.0,
    );
    engine.set_permissions(PermissionGate::from_mode_str("bypassPermissions"));

    let (ui_tx, _ui_rx) = UiSink::channel();
    let (_steer_tx, steer_rx) = tokio::sync::mpsc::channel(4);
    let (_notice_tx, notice_rx) = tokio::sync::mpsc::channel(4);
    let mut io = EngineIo {
        ui_tx,
        steer_rx,
        notice_rx,
        cancel: cancel.clone(),
    };
    let mut history: Vec<ChatMessage> = Vec::new();

    let running = tokio::spawn(async move {
        let out = engine
            .run_turn(&mut io, &mut history, "delegate it".into())
            .await;
        (out, history)
    });

    let child_id = wait_for_running_child(CANCEL_SESSION, &child).await;
    // The child is held at its provider gate, so the cancel lands mid-turn.
    let started = std::time::Instant::now();
    cancel.store(true, Ordering::Relaxed);
    let (out, history) = tokio::time::timeout(Duration::from_secs(5), running)
        .await
        .expect("the delegate must return on the cancel, not after the child")
        .expect("the turn task must not panic");
    assert!(
        started.elapsed() < Duration::from_secs(2),
        "the cancel waited for the child: {:?}",
        started.elapsed()
    );
    out.expect_err("a cancelled turn is an error");

    let marker = format!("[subagent cancelled — child_id {child_id}]");
    assert_eq!(tool_row(&history, "call_delegate"), marker, "{history:#?}");
    // The turn's own ending is unchanged: the user is told, and the state is a
    // failure, not a completed turn.
    assert!(
        history
            .iter()
            .any(|m| m.content.as_deref() == Some("已被用户中断")),
        "{history:#?}"
    );
    assert_eq!(parent.calls.lock().unwrap().len(), 1);
}

/// The child's mirror reaches the host on every round, not only when the child
/// stops: an app quit between rounds is exactly the case that would otherwise
/// leave no transcript at all.
#[tokio::test]
async fn every_child_round_persists_while_the_child_runs() {
    let child = Arc::new(ScriptedProvider::new());
    let gate = Arc::new(tokio::sync::Notify::new());
    *child.gate.lock().unwrap() = Some(gate.clone());
    // Round one asks for a tool — the second request it forces is what proves
    // the write is per round rather than per child.
    child.push_script(vec![
        StreamChunk::ToolCallDelta {
            slot: 0,
            id: Some("call_todo".into()),
            name: Some("todo".into()),
            args_delta: serde_json::json!({ "action": "list" }).to_string(),
        },
        StreamChunk::Done {
            prompt_tokens: None,
            completion_tokens: None,
            cached_tokens: None,
        },
    ]);
    child.script_text("done");

    let host = Arc::new(CountingHost::default());
    let mut ctx = ToolCtx::new().with_session(JOURNAL_SESSION);
    ctx.install_subagent(spawner(child.clone(), host.clone()));
    let ctx = Arc::new(ctx);

    let delegate = agent_kernel::tools::delegate::spec();
    let running = tokio::spawn(async move {
        (delegate.exec)(serde_json::json!({ "task": "look around" }), ctx).await
    });

    // Its first request is held open by the gate; the snapshot taken just
    // before it must already have reached the host.
    for _ in 0..300 {
        if host.count() > 0 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert!(
        !running.is_finished(),
        "the write must happen while the child still runs"
    );
    assert!(
        host.count() >= 1,
        "a running child persisted nothing: its transcript only lands when it stops"
    );

    gate.notify_one();
    tokio::time::timeout(Duration::from_secs(5), running)
        .await
        .expect("the released child must finish")
        .expect("the delegate task must not panic")
        .expect("the child finished cleanly");
    assert!(
        host.count() >= 2,
        "the second round must persist too: {} writes",
        host.count()
    );
}

/// In the parallel form every child still gets its own section, headed
/// `cancelled` and carrying the same marker — one place for the UI to read a
/// whole fan-out's child ids from.
#[tokio::test]
async fn a_cancelled_parallel_batch_marks_every_child() {
    let host = Arc::new(CountingHost::default());
    let cancel = Arc::new(AtomicBool::new(false));
    let mut ctx = ToolCtx::new().with_session(PARALLEL_SESSION);
    ctx.cancel = Some(cancel.clone());
    ctx.install_subagent(spawner(Arc::new(StalledProvider), host.clone()));
    let ctx = Arc::new(ctx);

    let delegate = agent_kernel::tools::delegate::spec();
    let running = tokio::spawn(async move {
        (delegate.exec)(
            serde_json::json!({ "tasks": ["first", "second"], "readonly": true }),
            ctx,
        )
        .await
    });

    // Both children have to be up before the cancel, or one section would be a
    // child that never ran.
    for _ in 0..300 {
        if active_children(PARALLEL_SESSION).len() == 2 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    let ids = active_children(PARALLEL_SESSION);
    assert_eq!(ids.len(), 2, "both children must be running: {ids:?}");
    cancel.store(true, Ordering::Relaxed);

    let text = tokio::time::timeout(Duration::from_secs(5), running)
        .await
        .expect("the batch must end on the cancel")
        .expect("the delegate task must not panic")
        .expect("the batch reports its children either way")
        .content;
    // The children start concurrently, so which task got which id is not
    // fixed: every section is cancelled, and each running id is named once.
    for i in 0..ids.len() {
        let head = format!("── [{i}] cancelled ──\n[subagent cancelled — child_id ");
        assert!(text.contains(&head), "missing `{head}`: {text}");
    }
    for id in &ids {
        let marker = format!("[subagent cancelled — child_id {id}]");
        assert_eq!(text.matches(&marker).count(), 1, "`{marker}` once: {text}");
    }
}
