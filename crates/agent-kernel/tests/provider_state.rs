//! Opaque adapter replay data must follow the real engine's tool loop and
//! persisted history without becoming user-visible text.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use agent_kernel::engine::{Engine, EngineIo};
use agent_kernel::permissions::PermissionGate;
use agent_kernel::tools::{ToolCtx, ToolRegistry};
use agent_llm_core::{
    BoxStream, ChatMessage, LlmProvider, ModelParams, ProviderState, Role, StreamChunk,
};
use agent_testkit::ScriptedProvider;
use async_trait::async_trait;
use serde_json::Value;

struct RecordingProvider {
    scripted: ScriptedProvider,
    requests: Mutex<Vec<Vec<ChatMessage>>>,
}

#[async_trait]
impl LlmProvider for RecordingProvider {
    fn id(&self) -> &'static str {
        "fixture"
    }

    async fn chat_stream(
        &self,
        model: &str,
        messages: &[ChatMessage],
        tools: Option<Value>,
        temperature: f32,
        effort: Option<&str>,
        params: &ModelParams,
    ) -> anyhow::Result<BoxStream<StreamChunk>> {
        self.requests.lock().unwrap().push(messages.to_vec());
        self.scripted
            .chat_stream(model, messages, tools, temperature, effort, params)
            .await
    }
}

fn state(marker: &str) -> ProviderState {
    ProviderState {
        origin: "fixture-origin".into(),
        model: "fixture-model".into(),
        data: format!("opaque-private-{marker}"),
    }
}

#[tokio::test]
async fn a_retry_drops_the_dead_attempts_provider_state() {
    let provider = Arc::new(RecordingProvider {
        scripted: ScriptedProvider::new(),
        requests: Mutex::new(Vec::new()),
    });
    provider.scripted.push_script(vec![
        StreamChunk::ContentDelta("partial".into()),
        StreamChunk::ProviderState(state("dead")),
        StreamChunk::Error("503 service unavailable".into()),
    ]);
    // The retry can legitimately be a provider that supplies no replay state.
    provider
        .scripted
        .push_script(vec![StreamChunk::ContentDelta("answer".into()), done()]);
    let mut engine = Engine::new(
        provider.clone(),
        Arc::new(ToolRegistry::with_builtins()),
        Arc::new(ToolCtx::new()),
        "fixture-model",
        0.0,
    );
    let (ui_tx, _ui_rx) = agent_kernel::channels::UiSink::channel();
    let mut io = EngineIo {
        ui_tx,
        steer_rx: tokio::sync::mpsc::channel(4).1,
        notice_rx: tokio::sync::mpsc::channel(4).1,
        cancel: Arc::new(std::sync::atomic::AtomicBool::new(false)),
    };
    let mut history = vec![ChatMessage::system("stable")];
    let outcome = tokio::time::timeout(
        Duration::from_secs(10),
        engine.run_turn(&mut io, &mut history, "answer".into()),
    )
    .await
    .unwrap()
    .unwrap();
    assert_eq!(outcome.text, "answer");
    let answer = history
        .iter()
        .rev()
        .find(|m| m.role == Role::Assistant)
        .unwrap();
    assert_eq!(answer.content.as_deref(), Some("answer"));
    assert_eq!(
        answer.provider_state, None,
        "a failed attempt's opaque state must not attach to the retry's answer"
    );
    assert_eq!(provider.requests.lock().unwrap().len(), 2);
}

fn done() -> StreamChunk {
    StreamChunk::Done {
        prompt_tokens: Some(2048),
        completion_tokens: Some(12),
        cached_tokens: Some(1024),
    }
}

#[tokio::test]
async fn provider_state_survives_a_tool_round_the_next_request_and_a_reload() {
    let provider = Arc::new(RecordingProvider {
        scripted: ScriptedProvider::new(),
        requests: Mutex::new(Vec::new()),
    });
    provider.scripted.push_script(vec![
        StreamChunk::ReasoningDelta("visible thought".into()),
        StreamChunk::ContentDelta("checking".into()),
        StreamChunk::ToolCallDelta {
            slot: 0,
            id: Some("call_1".into()),
            name: Some("todo".into()),
            args_delta: serde_json::json!({"action":"list"}).to_string(),
        },
        StreamChunk::ProviderState(state("first")),
        done(),
    ]);
    provider.scripted.push_script(vec![
        StreamChunk::ContentDelta("done".into()),
        StreamChunk::ProviderState(state("last")),
        done(),
    ]);
    let mut engine = Engine::new(
        provider.clone(),
        Arc::new(ToolRegistry::with_builtins()),
        Arc::new(ToolCtx::new()),
        "fixture-model",
        0.0,
    );
    engine.set_permissions(PermissionGate::from_mode_str("bypassPermissions"));
    let (ui_tx, _ui_rx) = agent_kernel::channels::UiSink::channel();
    let mut io = EngineIo {
        ui_tx,
        steer_rx: tokio::sync::mpsc::channel(4).1,
        notice_rx: tokio::sync::mpsc::channel(4).1,
        cancel: Arc::new(std::sync::atomic::AtomicBool::new(false)),
    };
    let mut history = vec![ChatMessage::system("stable instructions")];
    let outcome = tokio::time::timeout(
        Duration::from_secs(10),
        engine.run_turn(&mut io, &mut history, "check".into()),
    )
    .await
    .unwrap()
    .unwrap();
    assert_eq!(outcome.text, "done");
    assert_eq!(outcome.tool_calls_run, 1);
    let assistants: Vec<_> = history
        .iter()
        .filter(|m| m.role == Role::Assistant)
        .collect();
    assert_eq!(assistants.len(), 2);
    assert_eq!(assistants[0].provider_state, Some(state("first")));
    assert_eq!(assistants[1].provider_state, Some(state("last")));
    let requests = provider.requests.lock().unwrap();
    assert_eq!(requests.len(), 2);
    let previous = requests[1]
        .iter()
        .find(|m| m.role == Role::Assistant)
        .unwrap();
    assert_eq!(
        previous.provider_state,
        Some(state("first")),
        "sanitization must keep the adapter's state"
    );
    assert!(history.iter().all(|m| !m
        .content
        .as_deref()
        .unwrap_or_default()
        .contains("opaque-private")));
    let reloaded: Vec<ChatMessage> =
        serde_json::from_slice(&serde_json::to_vec(&history).unwrap()).unwrap();
    assert_eq!(reloaded, history);
}
