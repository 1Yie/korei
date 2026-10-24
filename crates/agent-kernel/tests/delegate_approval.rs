//! A delegated child that needs permission asks the user directly.
//!
//! The child paints its card on the parent's stream, so the click arrives at
//! the parent's session — but the engine parked on it is the child's. These
//! tests prove the round trip: the card carries the child's id, the verdict
//! reaches that child's gate, and the child carries on with its task.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use agent_ipc::UiEvent;
use agent_kernel::channels::UiSink;
use agent_kernel::permissions::{PermissionGate, PermissionMode};
use agent_kernel::tools::delegate::{ChildRoute, SubagentSpawner};
use agent_kernel::tools::registry::{
    ToolClass, ToolCtx, ToolRegistry, ToolResult, ToolSpec,
};
use agent_tool::{CallRisk, Review};
use agent_llm_core::types::StreamChunk;
use agent_testkit::ScriptedProvider;
use futures::FutureExt;

/// A tool that always needs permission and counts how often it ran.
fn guarded_tool(runs: Arc<AtomicUsize>) -> ToolSpec {
    ToolSpec {
        name: "guarded",
        schema: serde_json::json!({
            "type": "object",
            "properties": { "path": { "type": "string" } }
        }),
        readonly: false,
        class: ToolClass::WorkspaceMutation,
        network: false,
        staged: false,
        review: Some(Arc::new(|_: &serde_json::Value, _: &Arc<ToolCtx>| Review {
            risk: CallRisk::RequiresApproval,
            ..Default::default()
        })),
        cancel_aware: false,
        exec: Arc::new(move |_args, _ctx| {
            let runs = runs.clone();
            async move {
                runs.fetch_add(1, Ordering::Relaxed);
                Ok(ToolResult::text("wrote it"))
            }
            .boxed()
        }),
    }
}

fn spawner(provider: Arc<ScriptedProvider>, runs: Arc<AtomicUsize>) -> SubagentSpawner {
    let mut registry = ToolRegistry::child_builtins();
    registry.register(guarded_tool(runs));
    let registry = Arc::new(registry);
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
    let parent = Arc::new(std::sync::RwLock::new(PermissionGate::new(
        PermissionMode::Auto,
        Default::default(),
    )));
    SubagentSpawner::new(route, registry.clone(), registry)
        // The child is bounded by the parent's mode, and it may ask.
        .with_parent_gate(parent)
        .with_child_approvals(true)
}

/// The child's first round asks to write; the second reports.
fn child_writes_then_reports(provider: &ScriptedProvider) {
    provider.push_script(vec![
        StreamChunk::ToolCallDelta {
            slot: 0,
            id: Some("call_write".into()),
            name: Some("guarded".into()),
            args_delta: serde_json::json!({ "path": "src/lib.rs" }).to_string(),
        },
        StreamChunk::Done {
            prompt_tokens: None,
            completion_tokens: None,
            cached_tokens: None,
        },
    ]);
    provider.script_text("child finished the task");
}

/// The child's approval reaches the user's stream tagged with its id, and the
/// answer resumes the child instead of the parent's turn.
#[tokio::test]
async fn a_childs_approval_is_asked_and_answered_across_engines() {
    let provider = Arc::new(ScriptedProvider::new());
    child_writes_then_reports(&provider);
    let runs = Arc::new(AtomicUsize::new(0));
    let (sink, mut rx) = UiSink::channel();
    let mut ctx = ToolCtx::new();
    ctx.install_subagent(spawner(provider, runs.clone()));
    ctx.install_ui(sink);
    let ctx = Arc::new(ctx);

    let delegate = agent_kernel::tools::delegate::spec();
    let call = tokio::spawn(async move {
        (delegate.exec)(serde_json::json!({ "task": "write the file" }), ctx).await
    });

    // The child is parked on its own card: nothing ran yet, and the card that
    // drew it names the child rather than the parent.
    let (request_id, child_id) = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            while let Some(ev) = rx.try_recv() {
                if let UiEvent::ApprovalRequested {
                    request_id,
                    tool_name,
                    child_id,
                    ..
                } = ev
                {
                    assert_eq!(tool_name, "guarded");
                    return (request_id, child_id);
                }
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("the child's card must reach the parent stream");
    assert!(
        child_id.is_some(),
        "the card must name the child that asked, not the parent's turn"
    );
    assert_eq!(runs.load(Ordering::Relaxed), 0, "nothing ran unapproved");

    assert!(
        agent_kernel::tools::delegate::deliver_child_decision(request_id, true, None),
        "the verdict must find the child that asked"
    );
    let report = tokio::time::timeout(Duration::from_secs(5), call)
        .await
        .expect("the answered child finishes")
        .expect("the task did not panic")
        .expect("the child reported");
    assert!(
        report.content.contains("child finished the task"),
        "{}",
        report.content
    );
    assert_eq!(runs.load(Ordering::Relaxed), 1, "the approved call ran once");
}

/// A refusal is a refusal: the child does not run the call, and it still ends
/// its turn with a report instead of hanging on the card.
#[tokio::test]
async fn a_denied_child_reports_rather_than_hanging() {
    let provider = Arc::new(ScriptedProvider::new());
    child_writes_then_reports(&provider);
    let runs = Arc::new(AtomicUsize::new(0));
    let (sink, mut rx) = UiSink::channel();
    let mut ctx = ToolCtx::new();
    ctx.install_subagent(spawner(provider, runs.clone()));
    ctx.install_ui(sink);
    let ctx = Arc::new(ctx);

    let delegate = agent_kernel::tools::delegate::spec();
    let call = tokio::spawn(async move {
        (delegate.exec)(serde_json::json!({ "task": "write the file" }), ctx).await
    });
    let request_id = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            while let Some(ev) = rx.try_recv() {
                if let UiEvent::ApprovalRequested { request_id, .. } = ev {
                    return request_id;
                }
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("the child's card must reach the parent stream");

    assert!(agent_kernel::tools::delegate::deliver_child_decision(
        request_id,
        false,
        None
    ));
    let _ = tokio::time::timeout(Duration::from_secs(5), call)
        .await
        .expect("a denied child still ends its turn");
    assert_eq!(runs.load(Ordering::Relaxed), 0, "a denied call never runs");
}
