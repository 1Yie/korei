//! A model that streams a little text and then goes quiet must not leave that
//! text stranded in the engine's coalescing buffer: the user would see the
//! reply freeze mid-sentence and only get the rest when they cancel.

use std::sync::Arc;
use std::time::Duration;

use agent_ipc::UiEvent;
use agent_kernel::engine::{Engine, EngineIo};
use agent_kernel::permissions::PermissionGate;
use agent_kernel::tools::registry::{ToolCtx, ToolRegistry};
use agent_llm_core::{ChatMessage, LlmProvider, ModelParams, StreamChunk};
use futures::StreamExt;

/// Emits `first`, then stalls forever (a hung connection).
struct Stalls;

#[async_trait::async_trait]
impl LlmProvider for Stalls {
    fn id(&self) -> &'static str {
        "stalls"
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
        let head = futures::stream::iter(vec![Ok(StreamChunk::ContentDelta(
            "half a sentence".into(),
        ))]);
        Ok(Box::pin(head.chain(futures::stream::pending())))
    }
}

#[tokio::test]
async fn text_before_a_stall_reaches_the_ui_without_a_cancel() {
    let ctx = Arc::new(ToolCtx::new());
    let mut engine = Engine::new(
        Arc::new(Stalls),
        Arc::new(ToolRegistry::new()),
        ctx,
        "m",
        0.0,
    );
    engine.set_permissions(PermissionGate::from_mode_str("bypassPermissions"));

    let (ui_tx, mut ui_rx) = agent_kernel::channels::UiSink::channel();
    let (_steer_tx, steer_rx) = tokio::sync::mpsc::channel(4);
    let cancel = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let mut io = EngineIo {
        ui_tx,
        steer_rx,
        cancel: cancel.clone(),
    };
    let mut history: Vec<ChatMessage> = Vec::new();

    let turn = engine.run_turn(&mut io, &mut history, "hi".into());
    let watch = async {
        // Generous against the 40 ms coalescing window, far below the 300 s
        // idle timeout: the text must arrive while the stream is still stalled.
        let deadline = tokio::time::sleep(Duration::from_secs(2));
        tokio::pin!(deadline);
        loop {
            tokio::select! {
                _ = &mut deadline => return false,
                ev = ui_rx.recv() => match ev {
                    Some(UiEvent::TextDelta { text, .. }) if text.contains("half a sentence") => return true,
                    Some(_) => {}
                    None => return false,
                },
            }
        }
    };
    tokio::pin!(turn);
    let delivered = tokio::select! {
        d = watch => d,
        _ = &mut turn => false,
    };
    cancel.store(true, std::sync::atomic::Ordering::Relaxed);
    assert!(
        delivered,
        "the streamed text was held in the engine buffer while the stream stalled"
    );
}

/// Streams two bursts with a gap longer than the flush window between them,
/// then finishes. The idle tick must not duplicate, drop or reorder text.
struct Gapped;

#[async_trait::async_trait]
impl LlmProvider for Gapped {
    fn id(&self) -> &'static str {
        "gapped"
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
        let s = async_stream_like();
        Ok(Box::pin(s))
    }
}

fn async_stream_like() -> impl futures::Stream<Item = anyhow::Result<StreamChunk>> {
    futures::stream::unfold(0u8, |step| async move {
        match step {
            0 => Some((Ok(StreamChunk::ContentDelta("alpha ".into())), 1)),
            1 => {
                tokio::time::sleep(Duration::from_millis(150)).await;
                Some((Ok(StreamChunk::ContentDelta("beta".into())), 2))
            }
            2 => Some((
                Ok(StreamChunk::Done {
                    prompt_tokens: Some(1),
                    completion_tokens: Some(1),
                    cached_tokens: None,
                }),
                3,
            )),
            _ => None,
        }
    })
}

#[tokio::test]
async fn idle_flush_keeps_the_text_exact_and_ordered() {
    let ctx = Arc::new(ToolCtx::new());
    let mut engine = Engine::new(
        Arc::new(Gapped),
        Arc::new(ToolRegistry::new()),
        ctx,
        "m",
        0.0,
    );
    engine.set_permissions(PermissionGate::from_mode_str("bypassPermissions"));

    let (ui_tx, mut ui_rx) = agent_kernel::channels::UiSink::channel();
    let (_steer_tx, steer_rx) = tokio::sync::mpsc::channel(4);
    let mut io = EngineIo {
        ui_tx,
        steer_rx,
        cancel: Arc::new(std::sync::atomic::AtomicBool::new(false)),
    };
    let mut history: Vec<ChatMessage> = Vec::new();

    let outcome = tokio::time::timeout(
        Duration::from_secs(5),
        engine.run_turn(&mut io, &mut history, "hi".into()),
    )
    .await
    .expect("the turn must finish")
    .expect("the turn should succeed");
    assert_eq!(outcome.text, "alpha beta");

    let mut streamed = String::new();
    let mut deltas = 0;
    while let Ok(Some(ev)) = tokio::time::timeout(Duration::from_millis(50), ui_rx.recv()).await {
        if let UiEvent::TextDelta { text, .. } = ev {
            streamed.push_str(&text);
            deltas += 1;
        }
    }
    assert_eq!(
        streamed, "alpha beta",
        "no duplicated, lost or reordered text"
    );
    assert!(
        deltas >= 2,
        "the first burst should not wait for the second: {deltas}"
    );
}
