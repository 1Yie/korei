//! Text and reasoning must reach the UI in the order the model streamed them:
//! a switch from one content type to the other flushes the buffered side first,
//! and a tool call's argument stream flushes both before it starts.

use std::sync::Arc;
use std::time::Duration;

use futures::StreamExt;

use agent_ipc::{AgentState, UiEvent};
use agent_kernel::engine::{Engine, EngineIo};
use agent_kernel::permissions::PermissionGate;
use agent_kernel::tools::registry::{ToolCtx, ToolRegistry};
use agent_llm_core::{ChatMessage, LlmProvider, ModelParams, StreamChunk};
use agent_testkit::ScriptedProvider;

fn done() -> StreamChunk {
    StreamChunk::Done {
        prompt_tokens: Some(1),
        completion_tokens: Some(1),
        cached_tokens: None,
    }
}

fn tool_call(args_delta: &str) -> StreamChunk {
    StreamChunk::ToolCallDelta {
        slot: 0,
        id: Some("call_1".into()),
        name: Some("read".into()),
        args_delta: args_delta.into(),
    }
}

fn engine(provider: Arc<dyn LlmProvider>) -> Engine {
    let mut engine = Engine::new(
        provider,
        Arc::new(ToolRegistry::new()),
        Arc::new(ToolCtx::new()),
        "m",
        0.0,
    );
    engine.set_permissions(PermissionGate::from_mode_str("bypassPermissions"));
    engine
}

/// Runs a turn over scripted rounds and returns every UI event it emitted, in order.
async fn run_scripted(rounds: Vec<Vec<StreamChunk>>) -> Vec<UiEvent> {
    let provider = Arc::new(ScriptedProvider::new());
    for round in rounds {
        provider.push_script(round);
    }
    let mut engine = engine(provider);

    let (ui_tx, mut ui_rx) = agent_kernel::channels::UiSink::channel();
    let (_steer_tx, steer_rx) = tokio::sync::mpsc::channel(4);
    let mut io = EngineIo {
        ui_tx,
        steer_rx,
        notice_rx: tokio::sync::mpsc::channel(4).1,
        cancel: Arc::new(std::sync::atomic::AtomicBool::new(false)),
    };
    let mut history: Vec<ChatMessage> = Vec::new();

    tokio::time::timeout(
        Duration::from_secs(5),
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
async fn reasoning_is_delivered_before_the_text_that_follows_it() {
    let events = run_scripted(vec![vec![
        StreamChunk::ReasoningDelta("思考".into()),
        StreamChunk::ContentDelta("答案".into()),
        done(),
    ]])
    .await;

    let first_text = events
        .iter()
        .position(|e| matches!(e, UiEvent::TextDelta { .. }))
        .expect("the answer must be streamed");
    let reasoning: Vec<(usize, &str)> = events
        .iter()
        .enumerate()
        .filter_map(|(i, e)| match e {
            UiEvent::ReasoningDelta { text, .. } => Some((i, text.as_str())),
            _ => None,
        })
        .collect();
    assert!(!reasoning.is_empty(), "the reasoning must be streamed");
    assert!(
        reasoning.iter().all(|(i, _)| *i < first_text),
        "reasoning arrived after the answer began: {events:?}"
    );
    let reasoning_text: String = reasoning.iter().map(|(_, t)| *t).collect();
    assert_eq!(reasoning_text, "思考");
    let text: String = events
        .iter()
        .filter_map(|e| match e {
            UiEvent::TextDelta { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(text, "答案");
}

#[tokio::test]
async fn text_only_turn_emits_no_reasoning_and_finishes() {
    let events = run_scripted(vec![vec![
        StreamChunk::ContentDelta("just text".into()),
        done(),
    ]])
    .await;

    assert!(
        !events
            .iter()
            .any(|e| matches!(e, UiEvent::ReasoningDelta { .. })),
        "no reasoning was streamed: {events:?}"
    );
    let text: String = events
        .iter()
        .filter_map(|e| match e {
            UiEvent::TextDelta { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(text, "just text");
    assert!(events
        .iter()
        .any(|e| matches!(e, UiEvent::StateChanged(AgentState::Finished))));
}

#[tokio::test]
async fn reasoning_tail_is_delivered_before_a_tool_call_starts() {
    let events = run_scripted(vec![
        vec![
            StreamChunk::ReasoningDelta("plan it".into()),
            tool_call("{\"path\":\"a.rs\"}"),
            done(),
        ],
        vec![StreamChunk::ContentDelta("ok".into()), done()],
    ])
    .await;

    let started = events
        .iter()
        .position(|e| matches!(e, UiEvent::ToolCallStarted { .. }))
        .expect("the tool call must start");
    let reasoning_before: String = events[..started]
        .iter()
        .filter_map(|e| match e {
            UiEvent::ReasoningDelta { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(
        reasoning_before, "plan it",
        "the reasoning tail was still buffered when the tool call started: {events:?}"
    );
}

/// Round one streams from `rx` and never finishes on its own; round two answers.
struct Held {
    first: std::sync::Mutex<Option<tokio::sync::mpsc::UnboundedReceiver<StreamChunk>>>,
}

#[async_trait::async_trait]
impl LlmProvider for Held {
    fn id(&self) -> &'static str {
        "held"
    }

    async fn chat_stream(
        &self,
        _model: &str,
        _messages: &[ChatMessage],
        _tools: Option<serde_json::Value>,
        _temperature: f32,
        _reasoning_effort: Option<&str>,
        _params: &ModelParams,
    ) -> anyhow::Result<agent_llm_core::BoxStream<StreamChunk>> {
        let rx = self.first.lock().unwrap().take();
        let Some(mut rx) = rx else {
            let answer =
                futures::stream::iter(vec![Ok(StreamChunk::ContentDelta("ok".into())), Ok(done())]);
            return Ok(Box::pin(answer));
        };
        let streamed = futures::stream::poll_fn(move |cx| rx.poll_recv(cx).map(|c| c.map(Ok)));
        Ok(Box::pin(streamed.chain(futures::stream::pending())))
    }
}

#[tokio::test]
async fn preamble_reaches_the_ui_while_tool_call_arguments_are_still_streaming() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let provider = Arc::new(Held {
        first: std::sync::Mutex::new(Some(rx)),
    });
    let mut engine = engine(provider);

    let (ui_tx, mut ui_rx) = agent_kernel::channels::UiSink::channel();
    let (_steer_tx, steer_rx) = tokio::sync::mpsc::channel(4);
    let cancel = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let mut io = EngineIo {
        ui_tx,
        steer_rx,
        notice_rx: tokio::sync::mpsc::channel(4).1,
        cancel: cancel.clone(),
    };
    let mut history: Vec<ChatMessage> = Vec::new();

    let turn = engine.run_turn(&mut io, &mut history, "hi".into());
    let driver = async {
        tx.send(StreamChunk::ContentDelta("Let me check.".into()))
            .unwrap();
        tx.send(tool_call("{\"pa")).unwrap();
        // Argument fragments 5 ms apart: well inside the 40 ms idle tick, so the
        // sampler never goes quiet and only the tool-call chunk can release the
        // preamble. It must be out before the last fragment is sent.
        let mut preamble_seen = false;
        let mut seen_before_last = false;
        for i in 0..40 {
            tokio::time::sleep(Duration::from_millis(5)).await;
            if i == 39 {
                seen_before_last = preamble_seen;
            }
            tx.send(tool_call("x")).unwrap();
            while let Some(ev) = ui_rx.try_recv() {
                if let UiEvent::TextDelta { text, .. } = ev {
                    preamble_seen |= text.contains("Let me check.");
                }
            }
        }
        seen_before_last
    };
    tokio::pin!(turn);
    let delivered = tokio::select! {
        d = driver => d,
        _ = &mut turn => false,
    };
    cancel.store(true, std::sync::atomic::Ordering::Relaxed);
    assert!(
        delivered,
        "the preamble was held back until the tool call finished streaming"
    );
}
