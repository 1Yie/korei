//! A turn that narrates before it answers must hand back the answer.
//!
//! The ReAct loop accumulates every round's text, so a turn that says
//! "I'll start by exploring" and only then answers has two different texts: the
//! narration-plus-answer concatenation and the closing round alone. What the
//! user read is the closing round, and that is what `TurnOutcome::text` — and
//! therefore the subagent report a parent reads — must carry.

use std::sync::Arc;
use std::time::Duration;

use agent_kernel::engine::{Engine, EngineIo};
use agent_kernel::permissions::PermissionGate;
use agent_kernel::tools::delegate::{spec, ChildRoute, SubagentSpawner};
use agent_kernel::tools::{ToolCtx, ToolRegistry};
use agent_llm_core::types::{ChatMessage, StreamChunk};
use agent_testkit::ScriptedProvider;

/// What the model says while it is still working.
const NARRATION: &str = "I'll start by exploring the tool list.";
/// What it says once it is done — the only text the caller should see.
const ANSWER: &str = "The report has 42 lines.";

/// Round one narrates and calls a tool; round two is the answer.
fn script_narration_then_answer(provider: &ScriptedProvider) {
    provider.push_script(vec![
        StreamChunk::ContentDelta(NARRATION.into()),
        StreamChunk::ToolCallDelta {
            slot: 0,
            id: Some("call_1".into()),
            name: Some("todo".into()),
            args_delta: serde_json::json!({ "action": "list" }).to_string(),
        },
        StreamChunk::Done {
            prompt_tokens: Some(5),
            completion_tokens: Some(5),
            cached_tokens: None,
        },
    ]);
    provider.script_text(ANSWER);
}

fn child_route(provider: Arc<ScriptedProvider>) -> Arc<std::sync::RwLock<ChildRoute>> {
    Arc::new(std::sync::RwLock::new(ChildRoute {
        provider,
        model: "test".into(),
        temperature: 0.0,
        context_window: 256_000,
        thinking_level: None,
        thinking_level_map: None,
        model_params: Default::default(),
        model_input: Vec::new(),
        models: Default::default(),
    }))
}

#[tokio::test]
async fn a_multi_round_turn_answers_with_its_last_round() {
    let provider = Arc::new(ScriptedProvider::new());
    script_narration_then_answer(&provider);
    let mut engine = Engine::new(
        provider.clone(),
        Arc::new(ToolRegistry::with_builtins()),
        Arc::new(ToolCtx::new()),
        "test",
        0.0,
    );
    engine.set_permissions(PermissionGate::from_mode_str("bypassPermissions"));

    let (ui_tx, _ui_rx) = agent_kernel::channels::UiSink::channel();
    let (_steer_tx, steer_rx) = tokio::sync::mpsc::channel(4);
    let mut io = EngineIo {
        ui_tx,
        steer_rx,
        notice_rx: tokio::sync::mpsc::channel(4).1,
        cancel: Arc::new(std::sync::atomic::AtomicBool::new(false)),
    };
    let mut history: Vec<ChatMessage> = Vec::new();

    let outcome = tokio::time::timeout(
        Duration::from_secs(10),
        engine.run_turn(&mut io, &mut history, "summarize the log".into()),
    )
    .await
    .expect("the turn must not park")
    .expect("the turn should finish");

    assert_eq!(outcome.tool_calls_run, 1, "the scripted call was dispatched");
    assert_eq!(
        outcome.text, ANSWER,
        "the outcome carries the closing round alone, not the narration"
    );
    assert!(
        !outcome.text.contains("exploring"),
        "narration leaked into the outcome: {}",
        outcome.text
    );

    // The outcome agrees with the row the transcript kept as the answer.
    let last = history
        .iter()
        .rev()
        .find(|m| m.role == agent_llm_core::Role::Assistant && m.tool_calls.is_none())
        .expect("the turn persisted a final assistant row");
    assert_eq!(last.content.as_deref(), Some(ANSWER));
}

#[tokio::test]
async fn a_subagent_report_body_is_the_final_answer_only() {
    let provider = Arc::new(ScriptedProvider::new());
    script_narration_then_answer(&provider);
    let registry = Arc::new(ToolRegistry::child_builtins());
    let mut ctx = ToolCtx::new();
    ctx.install_subagent(SubagentSpawner::new(
        child_route(provider),
        registry.clone(),
        registry,
    ));

    let out = tokio::time::timeout(
        Duration::from_secs(10),
        (spec().exec)(
            serde_json::json!({ "task": "summarize the log" }),
            Arc::new(ctx),
        ),
    )
    .await
    .expect("the child must not park")
    .expect("the child must finish");

    let body = out
        .content
        .split_once('\n')
        .expect("the report carries its header line")
        .1;
    assert_eq!(
        body, ANSWER,
        "the report body is the child's answer alone: {}",
        out.content
    );
    assert!(
        !out.content.contains("exploring"),
        "narration leaked into the report: {}",
        out.content
    );
}

#[tokio::test]
async fn a_silent_closing_round_falls_back_to_what_the_turn_said() {
    let provider = Arc::new(ScriptedProvider::new());
    provider.push_script(vec![
        StreamChunk::ContentDelta(NARRATION.into()),
        StreamChunk::ToolCallDelta {
            slot: 0,
            id: Some("call_1".into()),
            name: Some("todo".into()),
            args_delta: serde_json::json!({ "action": "list" }).to_string(),
        },
        StreamChunk::Done {
            prompt_tokens: Some(5),
            completion_tokens: Some(5),
            cached_tokens: None,
        },
    ]);
    // The closing round streams nothing at all — there is no answer to return.
    provider.push_script(vec![StreamChunk::Done {
        prompt_tokens: Some(5),
        completion_tokens: Some(5),
        cached_tokens: None,
    }]);
    let mut engine = Engine::new(
        provider.clone(),
        Arc::new(ToolRegistry::with_builtins()),
        Arc::new(ToolCtx::new()),
        "test",
        0.0,
    );
    engine.set_permissions(PermissionGate::from_mode_str("bypassPermissions"));

    let (ui_tx, _ui_rx) = agent_kernel::channels::UiSink::channel();
    let (_steer_tx, steer_rx) = tokio::sync::mpsc::channel(4);
    let mut io = EngineIo {
        ui_tx,
        steer_rx,
        notice_rx: tokio::sync::mpsc::channel(4).1,
        cancel: Arc::new(std::sync::atomic::AtomicBool::new(false)),
    };
    let mut history: Vec<ChatMessage> = Vec::new();

    let outcome = tokio::time::timeout(
        Duration::from_secs(10),
        engine.run_turn(&mut io, &mut history, "summarize the log".into()),
    )
    .await
    .expect("the turn must not park")
    .expect("the turn should finish");

    assert_eq!(
        outcome.text, NARRATION,
        "a turn that ends silent still reports the text it did produce"
    );
}
