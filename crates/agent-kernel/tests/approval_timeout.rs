//! An approval card must never park the turn. The deadline denies the call on
//! its own, a cancel is not a denial, a click that lands after the deadline
//! approves nothing, and the same call is not asked twice in one turn.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use agent_ipc::UiEvent;
use agent_kernel::channels::{UiReceiver, UiSink};
use agent_kernel::engine::{Engine, EngineIo};
use agent_kernel::permissions::PermissionGate;
use agent_kernel::tools::registry::{
    ToolClass, ToolCtx, ToolRegistry, ToolResult, ToolSpec, ToolVerdict,
};
use agent_llm_core::types::StreamChunk;
use agent_llm_core::ChatMessage;
use agent_testkit::ScriptedProvider;
use futures::FutureExt;

/// The short line the UI shows for an automatically denied card.
const UI_TIMEOUT: &str = "审批超时，已自动拒绝";

/// A write tool that counts how often it actually ran — an unapproved call
/// must leave the counter at zero.
fn write_tool(runs: Arc<AtomicUsize>) -> ToolSpec {
    ToolSpec {
        name: "write",
        schema: serde_json::json!({
            "type": "object",
            "properties": { "path": { "type": "string" } }
        }),
        readonly: false,
        class: ToolClass::WorkspaceMutation,
        network: false,
        staged: false,
        review: None,
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

/// One assistant round that asks to write `path`.
fn write_round(path: &str, id: &str) -> Vec<StreamChunk> {
    vec![
        StreamChunk::ToolCallDelta {
            slot: 0,
            id: Some(id.into()),
            name: Some("write".into()),
            args_delta: serde_json::json!({ "path": path }).to_string(),
        },
        StreamChunk::Done {
            prompt_tokens: None,
            completion_tokens: None,
            cached_tokens: None,
        },
    ]
}

struct Harness {
    engine: Engine,
    io: EngineIo,
    history: Vec<ChatMessage>,
    ui_rx: UiReceiver,
    runs: Arc<AtomicUsize>,
    _steer_tx: tokio::sync::mpsc::Sender<String>,
    _notice_tx: tokio::sync::mpsc::Sender<String>,
}

/// An engine in `default` mode (writes ask) with a scripted model and a short
/// approval deadline.
fn harness(scripts: Vec<Vec<StreamChunk>>, timeout: Duration) -> Harness {
    let runs = Arc::new(AtomicUsize::new(0));
    let mut registry = ToolRegistry::new();
    registry.register(write_tool(runs.clone()));

    let provider = ScriptedProvider::new();
    for script in scripts {
        provider.push_script(script);
    }
    provider.script_text("done");
    let provider = Arc::new(provider);

    let mut engine = Engine::new(
        provider,
        Arc::new(registry),
        Arc::new(ToolCtx::new()),
        "scripted",
        0.0,
    );
    engine.set_permissions(PermissionGate::from_mode_str("default"));
    engine.set_approval_timeout(timeout);

    let (ui_tx, ui_rx) = UiSink::channel();
    let (steer_tx, steer_rx) = tokio::sync::mpsc::channel(4);
    let (notice_tx, notice_rx) = tokio::sync::mpsc::channel(4);
    let io = EngineIo {
        ui_tx,
        steer_rx,
        notice_rx,
        cancel: Arc::new(AtomicBool::new(false)),
    };
    Harness {
        engine,
        io,
        history: Vec::new(),
        ui_rx,
        runs,
        _steer_tx: steer_tx,
        _notice_tx: notice_tx,
    }
}

/// Every tool result the engine recorded, in order.
fn tool_results(history: &[ChatMessage]) -> Vec<&str> {
    history
        .iter()
        .filter(|m| m.tool_call_id.is_some())
        .filter_map(|m| m.content.as_deref())
        .collect()
}

/// Everything the sink emitted so far.
fn drain(rx: &mut UiReceiver) -> Vec<UiEvent> {
    rx.close();
    let mut events = Vec::new();
    while let Some(ev) = rx.try_recv() {
        events.push(ev);
    }
    events
}

fn cards(events: &[UiEvent]) -> Vec<(u64, String)> {
    events
        .iter()
        .filter_map(|ev| match ev {
            UiEvent::ApprovalRequested {
                request_id,
                tool_name,
                ..
            } => Some((*request_id, tool_name.clone())),
            _ => None,
        })
        .collect()
}

fn timed_out_cards(events: &[UiEvent]) -> usize {
    events
        .iter()
        .filter(|ev| matches!(ev, UiEvent::ApprovalTimedOut { .. }))
        .count()
}

/// Answer the first card the way the UI's `ToolDecision` does; the receiver
/// comes back so the test can read what the turn emitted afterwards.
fn approve_first_card(
    slot: Arc<std::sync::Mutex<Option<(u64, ToolVerdict)>>>,
    mut ui_rx: UiReceiver,
) -> tokio::task::JoinHandle<UiReceiver> {
    tokio::spawn(async move {
        while let Some(ev) = ui_rx.recv().await {
            if let UiEvent::ApprovalRequested { request_id, .. } = ev {
                *slot.lock().unwrap() = Some((
                    request_id,
                    ToolVerdict {
                        approved: true,
                        action: None,
                    },
                ));
                break;
            }
        }
        ui_rx
    })
}

#[tokio::test]
async fn a_verdict_before_the_deadline_runs_the_tool() {
    let h = harness(
        vec![write_round("a.txt", "c1")],
        Duration::from_secs(5),
    );
    let Harness {
        mut engine,
        mut io,
        mut history,
        ui_rx,
        runs,
        ..
    } = h;

    let responder = approve_first_card(engine.decision_slot(), ui_rx);
    let outcome = tokio::time::timeout(
        Duration::from_secs(5),
        engine.run_turn(&mut io, &mut history, "write it".into()),
    )
    .await
    .expect("an answered card must not wait out the deadline")
    .expect("the turn finishes");
    let mut ui_rx = responder.await.expect("the responder task");
    let events = drain(&mut ui_rx);

    assert_eq!(outcome.tool_calls_run, 1);
    assert_eq!(runs.load(Ordering::Relaxed), 1, "the approved write ran");
    assert_eq!(tool_results(&history), vec!["wrote it"]);
    // The responder consumed the one card it answered; nothing after it is a
    // second card or a timeout.
    assert_eq!(cards(&events).len(), 0, "{events:#?}");
    assert_eq!(timed_out_cards(&events), 0, "{events:#?}");
}

#[tokio::test]
async fn an_unanswered_card_times_out_into_a_denial() {
    let h = harness(
        vec![write_round("a.txt", "c1")],
        Duration::from_millis(200),
    );
    let Harness {
        mut engine,
        mut io,
        mut history,
        mut ui_rx,
        runs,
        ..
    } = h;

    let started = std::time::Instant::now();
    tokio::time::timeout(
        Duration::from_secs(5),
        engine.run_turn(&mut io, &mut history, "write it".into()),
    )
    .await
    .expect("the turn must not park on an unanswered card")
    .expect("the turn finishes");
    assert!(
        started.elapsed() >= Duration::from_millis(150),
        "the deadline was really waited: {:?}",
        started.elapsed()
    );

    let results = tool_results(&history);
    assert_eq!(results.len(), 1, "{history:#?}");
    assert!(
        results[0].starts_with("[approval timeout]"),
        "{}",
        results[0]
    );
    assert!(results[0].contains("write was NOT run"), "{}", results[0]);
    assert!(
        !results[0].contains("user denied"),
        "a timeout is not a user refusal: {}",
        results[0]
    );
    assert_eq!(runs.load(Ordering::Relaxed), 0, "the write never ran");
    assert_eq!(engine.denials_this_turn(), ["write".to_string()].as_slice());

    let events = drain(&mut ui_rx);
    assert_eq!(timed_out_cards(&events), 1, "{events:#?}");
    assert!(
        events.iter().any(|ev| matches!(
            ev,
            UiEvent::ToolCallFinished {
                name,
                ok: false,
                content,
                ..
            } if name == "write" && content == UI_TIMEOUT
        )),
        "{events:#?}"
    );
}

#[tokio::test]
async fn a_cancel_while_waiting_is_neither_a_timeout_nor_a_denial() {
    let h = harness(
        vec![write_round("a.txt", "c1")],
        Duration::from_secs(30),
    );
    let Harness {
        mut engine,
        mut io,
        mut history,
        mut ui_rx,
        runs,
        ..
    } = h;

    let flag = io.cancel.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(150)).await;
        flag.store(true, Ordering::Relaxed);
    });
    let _ = tokio::time::timeout(
        Duration::from_secs(5),
        engine.run_turn(&mut io, &mut history, "write it".into()),
    )
    .await
    .expect("the cancel lands while the card is open");

    let results = tool_results(&history);
    assert_eq!(results.len(), 1, "{history:#?}");
    assert!(
        results[0].contains("the turn was cancelled while awaiting approval"),
        "{}",
        results[0]
    );
    assert!(
        !results[0].contains("[approval timeout]"),
        "a cancel must not wear the timeout marker: {}",
        results[0]
    );

    let events = drain(&mut ui_rx);
    assert_eq!(timed_out_cards(&events), 0, "{events:#?}");
    assert_eq!(runs.load(Ordering::Relaxed), 0);
    assert!(
        engine.denials_this_turn().is_empty(),
        "cancelling is not a denial: {:?}",
        engine.denials_this_turn()
    );
}

#[tokio::test]
async fn a_click_after_the_deadline_approves_nothing() {
    let h = harness(
        vec![write_round("a.txt", "c1"), write_round("b.txt", "c2")],
        Duration::from_millis(400),
    );
    let Harness {
        mut engine,
        mut io,
        mut history,
        ui_rx,
        runs,
        ..
    } = h;

    let slot = engine.decision_slot();
    let late = tokio::spawn(async move {
        let mut ui_rx = ui_rx;
        let mut first = None;
        while let Some(ev) = ui_rx.recv().await {
            if let UiEvent::ApprovalRequested { request_id, .. } = ev {
                first = Some(request_id);
                break;
            }
        }
        // Long after the first card timed out, while the second one is open.
        tokio::time::sleep(Duration::from_millis(500)).await;
        *slot.lock().unwrap() = Some((
            first.expect("a card was drawn"),
            ToolVerdict {
                approved: true,
                action: None,
            },
        ));
        ui_rx
    });

    tokio::time::timeout(
        Duration::from_secs(10),
        engine.run_turn(&mut io, &mut history, "write both".into()),
    )
    .await
    .expect("both cards must time out on their own")
    .expect("the turn finishes");
    let mut ui_rx = late.await.expect("the late-click task");
    let events = drain(&mut ui_rx);

    let results = tool_results(&history);
    assert_eq!(results.len(), 2, "{history:#?}");
    for result in &results {
        assert!(
            result.starts_with("[approval timeout]"),
            "a stale click must not approve the next request: {result}"
        );
    }
    assert_eq!(runs.load(Ordering::Relaxed), 0);
    // The late-click task consumed the first card while waiting for it; the
    // second card and both timeouts are what is left in the stream.
    assert_eq!(cards(&events).len(), 1, "{events:#?}");
    assert_eq!(timed_out_cards(&events), 2, "{events:#?}");
}

#[tokio::test]
async fn an_identical_repeat_is_not_asked_twice_in_one_turn() {
    let h = harness(
        vec![write_round("a.txt", "c1"), write_round("a.txt", "c2")],
        Duration::from_millis(150),
    );
    let Harness {
        mut engine,
        mut io,
        mut history,
        mut ui_rx,
        runs,
        ..
    } = h;

    tokio::time::timeout(
        Duration::from_secs(10),
        engine.run_turn(&mut io, &mut history, "write it twice".into()),
    )
    .await
    .expect("the repeat must not draw a second card")
    .expect("the turn finishes");

    let results = tool_results(&history);
    assert_eq!(results.len(), 2, "{history:#?}");
    assert!(
        results[0].starts_with("[approval timeout]"),
        "{}",
        results[0]
    );
    assert!(
        results[1].starts_with("[approval timeout]"),
        "{}",
        results[1]
    );
    assert!(
        results[1].contains("The identical request already timed out earlier in this turn."),
        "{}",
        results[1]
    );
    assert_eq!(runs.load(Ordering::Relaxed), 0);

    let events = drain(&mut ui_rx);
    assert_eq!(cards(&events).len(), 1, "only one card: {events:#?}");
    assert_eq!(timed_out_cards(&events), 1, "{events:#?}");
}
