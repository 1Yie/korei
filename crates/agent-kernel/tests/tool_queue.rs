//! A round's calls are announced before any of them runs, in dispatch order, and
//! each one starts and finishes under the same id.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::Duration;

use agent_ipc::UiEvent;
use agent_kernel::engine::{Engine, EngineIo};
use agent_kernel::permissions::PermissionGate;
use agent_kernel::tools::registry::{ToolClass, ToolCtx, ToolRegistry, ToolResult, ToolSpec};
use agent_llm_core::{ChatMessage, StreamChunk};
use agent_testkit::ScriptedProvider;

fn done() -> StreamChunk {
    StreamChunk::Done {
        prompt_tokens: Some(1),
        completion_tokens: Some(1),
        cached_tokens: None,
    }
}

fn todo_call(slot: usize, id: &str, text: &str) -> StreamChunk {
    StreamChunk::ToolCallDelta {
        slot,
        id: Some(id.into()),
        name: Some("todo".into()),
        args_delta: format!(r#"{{"action":"add","text":"{text}"}}"#),
    }
}

async fn run(rounds: Vec<Vec<StreamChunk>>) -> Vec<UiEvent> {
    let provider = Arc::new(ScriptedProvider::new());
    for round in rounds {
        provider.push_script(round);
    }
    let mut registry = ToolRegistry::with_builtins();
    registry.register(ToolSpec {
        name: "probe",
        schema: serde_json::json!({
            "parameters": {
                "type": "object",
                "properties": {
                    "path": { "type": "string" },
                    "delay_ms": { "type": "integer" }
                },
                "required": ["path", "delay_ms"]
            }
        }),
        readonly: true,
        class: ToolClass::Observation,
        network: false,
        staged: false,
        review: None,
        cancel_aware: false,
        exec: Arc::new(|args, _| {
            Box::pin(async move {
                tokio::time::sleep(Duration::from_millis(args["delay_ms"].as_u64().unwrap())).await;
                Ok(ToolResult::text(args["path"].as_str().unwrap()))
            })
        }),
    });
    let (ui_tx, mut ui_rx) = agent_kernel::channels::UiSink::channel();
    let mut ctx = ToolCtx::new();
    ctx.install_ui(ui_tx.clone());
    let mut engine = Engine::new(provider, Arc::new(registry), Arc::new(ctx), "m", 0.0);
    engine.set_permissions(PermissionGate::from_mode_str("bypassPermissions"));

    let (_steer_tx, steer_rx) = tokio::sync::mpsc::channel(4);
    let mut io = EngineIo {
        ui_tx,
        steer_rx,
        notice_rx: tokio::sync::mpsc::channel(4).1,
        cancel: Arc::new(std::sync::atomic::AtomicBool::new(false)),
    };
    let mut history: Vec<ChatMessage> = Vec::new();
    tokio::time::timeout(
        Duration::from_secs(10),
        engine.run_turn(&mut io, &mut history, "hi".into()),
    )
    .await
    .expect("the turn must finish")
    .expect("the turn should succeed");

    let mut events = Vec::new();
    while let Some(ev) = ui_rx.try_recv() {
        events.push(ev);
    }
    events
}

#[tokio::test]
async fn a_round_is_queued_in_order_and_each_call_starts_under_its_own_id() {
    let events = run(vec![
        vec![
            todo_call(0, "call_1", "a"),
            todo_call(1, "call_2", "b"),
            todo_call(2, "call_3", "c"),
            done(),
        ],
        vec![StreamChunk::ContentDelta("done".into()), done()],
    ])
    .await;

    let queued: Vec<(String, String)> = events
        .iter()
        .filter_map(|ev| match ev {
            UiEvent::ToolCallQueued { name, call_id, .. } => Some((name.clone(), call_id.clone())),
            _ => None,
        })
        .collect();
    assert_eq!(
        queued,
        vec![
            ("todo".to_string(), "call_1".to_string()),
            ("todo".to_string(), "call_2".to_string()),
            ("todo".to_string(), "call_3".to_string()),
        ],
        "the whole round must be announced once, in order"
    );

    let started: Vec<String> = events
        .iter()
        .filter_map(|ev| match ev {
            UiEvent::ToolCallStarted { call_id, .. } => Some(call_id.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(
        started,
        vec![
            "call_1".to_string(),
            "call_2".to_string(),
            "call_3".to_string()
        ],
        "each call must start under the id its queue entry carried"
    );

    let first_started = events
        .iter()
        .position(|ev| matches!(ev, UiEvent::ToolCallStarted { .. }))
        .unwrap();
    assert_eq!(
        events[..first_started]
            .iter()
            .filter(|ev| matches!(ev, UiEvent::ToolCallQueued { .. }))
            .count(),
        3
    );
    let finished: Vec<_> = events
        .iter()
        .filter_map(|ev| match ev {
            UiEvent::ToolCallFinished { call_id, ok, .. } => Some((call_id.as_str(), *ok)),
            _ => None,
        })
        .collect();
    assert_eq!(
        finished,
        vec![("call_1", true), ("call_2", true), ("call_3", true)]
    );
}

#[tokio::test]
async fn a_finished_call_reports_the_same_id_and_repeats_no_card() {
    let events = run(vec![
        vec![todo_call(0, "call_7", "only"), done()],
        vec![StreamChunk::ContentDelta("done".into()), done()],
    ])
    .await;

    let mut queued = 0;
    let mut started = 0;
    let mut finished = 0;
    for ev in &events {
        match ev {
            UiEvent::ToolCallQueued { call_id, .. } => {
                assert_eq!(call_id, "call_7");
                queued += 1;
            }
            UiEvent::ToolCallStarted { call_id, .. } => {
                assert_eq!(call_id, "call_7");
                started += 1;
            }
            UiEvent::ToolCallFinished { call_id, .. } => {
                assert_eq!(call_id, "call_7");
                finished += 1;
            }
            _ => {}
        }
    }
    assert_eq!((queued, started, finished), (1, 1, 1), "one call, one card");
}

#[tokio::test]
async fn parallel_same_named_batch_calls_keep_distinct_result_identities() {
    let mut round = Vec::new();
    for slot in 0..2 {
        round.push(StreamChunk::ToolCallDelta {
            slot,
            id: Some(format!("batch_{slot}")),
            name: Some("batch_execute".into()),
            args_delta: serde_json::json!({
                "calls": [
                    { "tool": "probe", "args": { "path": "slow", "delay_ms": 10 } },
                    { "tool": "probe", "args": { "path": "fast", "delay_ms": 0 } }
                ]
            })
            .to_string(),
        });
    }
    round.push(done());
    let events = run(vec![
        round,
        vec![StreamChunk::ContentDelta("ok".into()), done()],
    ])
    .await;
    let mut pending = BTreeMap::new();
    let mut seen = BTreeSet::new();
    let mut started = 0;
    let mut finished = 0;
    for event in events {
        match event {
            UiEvent::ToolCallFinished {
                name,
                ok,
                content,
                parent: None,
                ..
            } if name == "batch_execute" => assert!(ok, "{content}"),
            UiEvent::ToolCallStarted {
                name,
                call_id,
                args_preview,
                parent: Some(parent),
                ..
            } if name == "probe" && parent == "batch_execute" => {
                assert!(!call_id.is_empty());
                assert!(seen.insert(call_id.clone()));
                assert!(pending.insert(call_id, args_preview).is_none());
                started += 1;
            }
            UiEvent::ToolCallFinished {
                name,
                call_id,
                content,
                ok,
                parent: Some(parent),
                ..
            } if name == "probe" && parent == "batch_execute" => {
                assert!(ok, "{content}");
                assert_eq!(pending.remove(&call_id), Some(content));
                finished += 1;
            }
            _ => {}
        }
    }
    assert_eq!((started, finished), (4, 4));
    assert!(pending.is_empty());
}
